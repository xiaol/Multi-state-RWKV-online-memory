#!/usr/bin/env python3
"""Profile natural-memory per-rank batches in isolated CUDA workers."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback
from typing import Any, Callable, Mapping, Sequence


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

import torch

from deltamem.core import delta as delta_core
from deltamem.core import delta_impl
from deltamem.core.delta import reset_delta_mem_states, snapshot_delta_mem_weights
from experiments.rethinking_rwkv_ms_gemma import run_natural_memory_gate as gate


PROTOCOL_SCHEMA = "rwkv_ms_natural_memory_gate_cuda_profile_protocol.v3"
WORKER_SCHEMA = "rwkv_ms_natural_memory_gate_cuda_profile_worker.v3"
RECEIPT_SCHEMA = "rwkv_ms_natural_memory_gate_cuda_profile_receipt.v3"
PADDED_SELECTION_SCHEMA = "rwkv_ms_natural_memory_gate_padded_selection.v2"
ANSWER_LOGIT_SELECTION_SCHEMA = (
    "rwkv_ms_natural_memory_gate_answer_logit_selection.v2"
)
HF_MIRROR_ENDPOINT = "https://hf-mirror.com"
PROFILED_LOCAL_BATCH_SIZES = (1, 2, 4)
REQUIRED_LOCAL_BATCH_SIZES = (1,)
EXPLORATORY_LOCAL_BATCH_SIZES = (2, 4)
DISTRIBUTED_WORLD_SIZE = gate.distributed.REQUIRED_WORLD_SIZE
DISTRIBUTED_LOCAL_BATCH_SIZE = gate.distributed.REQUIRED_LOCAL_BATCH_SIZE
DISTRIBUTED_GLOBAL_BATCH_SIZE = gate.distributed.REQUIRED_GLOBAL_BATCH_SIZE
DISTRIBUTED_TRAINING_UPDATES = gate.PRODUCTION_UPDATES
PRODUCTION_TRAINING_ROWS = gate.PRODUCTION_TRAINING_ROWS
PRODUCTION_EPOCHS = gate.PRODUCTION_EPOCHS
MINIMUM_HEADROOM_BYTES = 5 * 1024**3
PRODUCTION_RANK = gate.PRODUCTION_ADAPTER_RANK
PRODUCTION_KEY_DIM = gate.PRODUCTION_KEY_DIM
PRODUCTION_TEMPERATURE = gate.PRODUCTION_TEMPERATURE
PRODUCTION_TARGET_LAYERS = gate.DEFAULT_TARGET_LAYERS
PRODUCTION_DTYPE = gate.PRODUCTION_DTYPE
PRODUCTION_ATTN_IMPLEMENTATION = gate.PRODUCTION_ATTN_IMPLEMENTATION
PRODUCTION_PROFILE = "development"
PRODUCTION_OPTIMIZER = "torch.optim.AdamW"
PRODUCTION_OPTIMIZER_FUSED = True
PRODUCTION_LEARNING_RATE = gate.PRODUCTION_LEARNING_RATE
PRODUCTION_WEIGHT_DECAY = 0.0
PRODUCTION_ANSWER_WEIGHT = gate.PRODUCTION_ANSWER_WEIGHT
PRODUCTION_ROUTE_WEIGHT = gate.PRODUCTION_ROUTE_WEIGHT
PRODUCTION_HARD_NEGATIVE_MARGIN = gate.PRODUCTION_HARD_NEGATIVE_MARGIN
PRODUCTION_HARD_NEGATIVE_WEIGHT = gate.PRODUCTION_HARD_NEGATIVE_WEIGHT
PRODUCTION_MAX_GRAD_NORM = gate.PRODUCTION_MAX_GRAD_NORM
PRODUCTION_PROFILE_OPTIMIZER_STEPS = 3
WORKER_TIMEOUT_SECONDS = 30 * 60
PROFILE_STRESS_SEQUENCE = (
    (
        "cold_activation_stress_optimizer_step",
        "activation_max",
        True,
    ),
    (
        "activation_to_answer_logit_rollover_optimizer_step",
        "answer_logit_max",
        True,
    ),
    (
        "answer_logit_to_activation_rollover_optimizer_step",
        "activation_max",
        False,
    ),
)
PROFILE_PHASE_NAMES = (
    "fresh_model_load",
    *(name for name, _, _ in PROFILE_STRESS_SEQUENCE),
)


def _distributed_training_target() -> dict[str, Any]:
    return {
        "world_size": DISTRIBUTED_WORLD_SIZE,
        "local_batch_size": DISTRIBUTED_LOCAL_BATCH_SIZE,
        "global_batch_size": DISTRIBUTED_GLOBAL_BATCH_SIZE,
        "unique_training_rows": PRODUCTION_TRAINING_ROWS,
        "complete_epochs": PRODUCTION_EPOCHS,
        "optimizer_updates": DISTRIBUTED_TRAINING_UPDATES,
        "complete_epoch_schedule": True,
        "replication": "one raw full-model replica per rank",
        "gradient_synchronization": (
            "sum trainable gradients across ranks before global clipping and "
            "identical AdamW steps"
        ),
        "answer_loss_normalization": (
            "local answer-token sum divided by the all-reduced global answer-token "
            "count before gradient summation"
        ),
        "route_loss_normalization": (
            "local route-loss sum divided by the all-reduced global row count before "
            "gradient summation"
        ),
        "online_memory_state": "rank-local and never reduced",
    }


def _production_training_dataset_contract() -> dict[str, Any]:
    return {
        "schema": gate.TRAINING_DATASET_AUDIT_SCHEMA,
        "training_conditions": list(gate.DEFAULT_TRAINING_CONDITIONS),
        "tasks": list(gate.PRODUCTION_TASKS),
        "rows_per_condition_task": {
            condition: {
                task: gate.PRODUCTION_ROWS_PER_CONDITION_TASK
                for task in gate.PRODUCTION_TASKS
            }
            for condition in gate.DEFAULT_TRAINING_CONDITIONS
        },
        "source_query_condition_families": (
            gate.PRODUCTION_ROWS_PER_CONDITION_TASK * len(gate.PRODUCTION_TASKS)
        ),
        "unique_training_rows": PRODUCTION_TRAINING_ROWS,
        "complete_epochs": PRODUCTION_EPOCHS,
        "global_batch_size": DISTRIBUTED_GLOBAL_BATCH_SIZE,
        "optimizer_updates": DISTRIBUTED_TRAINING_UPDATES,
        "row_id_policy": gate.TRAINING_ROW_ID_POLICY,
        "sampling_policy": gate.TRAINING_SAMPLING_POLICY,
    }


@dataclass(frozen=True)
class WorkerProcessResult:
    args: tuple[str, ...]
    returncode: int
    stdout: str
    stderr: str
    pid: int


@dataclass
class OptimizerStepLiveness:
    write_audit: Any | None
    logits: Any | None
    route_logits: Any | None
    answer_loss: Any | None
    route_loss: Any | None
    route_predictions: Any | None
    total_loss: Any | None
    grad_norm: Any | None
    router_audit: Any | None


def canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=True,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def signed_payload(value: Mapping[str, Any], hash_field: str) -> dict[str, Any]:
    if hash_field in value:
        raise ValueError(f"Unsigned payload already contains {hash_field}")
    result = dict(value)
    result[hash_field] = sha256_text(canonical_json(result))
    return result


def verify_signed_payload(value: Mapping[str, Any], hash_field: str) -> bool:
    unsigned = dict(value)
    declared = unsigned.pop(hash_field, None)
    return isinstance(declared, str) and declared == sha256_text(
        canonical_json(unsigned)
    )


def _production_configuration(
    *,
    batch_size: int | None = None,
    delta_config: Any | None = None,
) -> dict[str, Any]:
    if delta_config is None:
        delta_config = gate.build_delta_config(
            target_layers=PRODUCTION_TARGET_LAYERS,
            rank=PRODUCTION_RANK,
            key_dim=PRODUCTION_KEY_DIM,
            temperature=PRODUCTION_TEMPERATURE,
        )
    configuration = {
        "profile": PRODUCTION_PROFILE,
        "training_conditions": list(gate.DEFAULT_TRAINING_CONDITIONS),
        "training_dataset_contract": _production_training_dataset_contract(),
        "rank": PRODUCTION_RANK,
        "target_layers": list(PRODUCTION_TARGET_LAYERS),
        "key_dim": PRODUCTION_KEY_DIM,
        "temperature": PRODUCTION_TEMPERATURE,
        "dtype": PRODUCTION_DTYPE,
        "attn_implementation": PRODUCTION_ATTN_IMPLEMENTATION,
        "optimizer": PRODUCTION_OPTIMIZER,
        "optimizer_fused": PRODUCTION_OPTIMIZER_FUSED,
        "learning_rate": PRODUCTION_LEARNING_RATE,
        "weight_decay": PRODUCTION_WEIGHT_DECAY,
        "answer_weight": PRODUCTION_ANSWER_WEIGHT,
        "route_weight": PRODUCTION_ROUTE_WEIGHT,
        "hard_negative_margin": PRODUCTION_HARD_NEGATIVE_MARGIN,
        "hard_negative_weight": PRODUCTION_HARD_NEGATIVE_WEIGHT,
        "max_grad_norm": PRODUCTION_MAX_GRAD_NORM,
        "profile_optimizer_steps": PRODUCTION_PROFILE_OPTIMIZER_STEPS,
        "distributed_training_target": _distributed_training_target(),
        "profile_stress_sequence": [
            {
                "phase": name,
                "current_batch": current_batch,
                "router_gradient_audit": router_audit,
            }
            for name, current_batch, router_audit in PROFILE_STRESS_SEQUENCE
        ],
        "delta_mem_config": json.loads(canonical_json(delta_config.to_dict())),
    }
    if batch_size is not None:
        if batch_size in REQUIRED_LOCAL_BATCH_SIZES:
            role = "required"
        elif batch_size in EXPLORATORY_LOCAL_BATCH_SIZES:
            role = "exploratory"
        else:
            raise ValueError(f"Unsupported profiled local batch size: {batch_size}")
        configuration["profiled_local_batch_size"] = int(batch_size)
        configuration["profile_batch_role"] = role
    return configuration


def _write_json_exclusive(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=True, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())


def _write_text_exclusive(path: Path, value: str) -> None:
    with path.open("x", encoding="utf-8") as handle:
        handle.write(value)
        handle.flush()
        os.fsync(handle.fileno())


def _resolve_regular_file(path: Path, description: str) -> Path:
    requested = path.expanduser()
    if requested.is_symlink():
        raise ValueError(f"{description} must not be a symbolic link: {requested}")
    resolved = requested.resolve(strict=True)
    if not resolved.is_file():
        raise ValueError(f"{description} is not a regular file: {resolved}")
    return resolved


def _parse_cuda_device(value: str) -> torch.device:
    device = torch.device(value)
    if device.type != "cuda" or device.index is None:
        raise ValueError("Profiler device must be explicit, for example cuda:3")
    if device.index < 0:
        raise ValueError("CUDA device index must be nonnegative")
    return device


def _parse_batch_sizes(value: str | Sequence[int]) -> tuple[int, ...]:
    if isinstance(value, str):
        result = tuple(int(part.strip()) for part in value.split(",") if part.strip())
    else:
        result = tuple(int(item) for item in value)
    if result != PROFILED_LOCAL_BATCH_SIZES:
        raise ValueError(
            "Formal CUDA profile requires local batch sizes 1,2,4 in that order"
        )
    return result


def _example_token_profile(example: Any) -> dict[str, Any]:
    write_lengths = tuple(
        len(record["input_ids"]) for record in example.write_records
    )
    if len(write_lengths) != gate.RECORDS_PER_EPISODE:
        raise ValueError(
            f"Example {example.row_id} is not a four-record correct-state write"
        )
    read_length = len(example.read_input_ids)
    if read_length <= 0 or min(write_lengths) <= 0:
        raise ValueError(f"Example {example.row_id} contains an empty sequence")
    return {
        "row_id": str(example.row_id),
        "episode_id": str(example.episode_id),
        "task": str(example.task),
        "condition": str(example.condition),
        "write_token_lengths": list(write_lengths),
        "read_token_length": read_length,
        "total_unpadded_token_positions": sum(write_lengths) + read_length,
        "maximum_sequence_token_length": max((*write_lengths, read_length)),
    }


def _length_vector(profile: Mapping[str, Any]) -> tuple[int, ...]:
    return tuple(int(value) for value in profile["write_token_lengths"]) + (
        int(profile["read_token_length"]),
    )


def _selection_candidate_corpus(
    profiled: Sequence[tuple[Any, Mapping[str, Any]]],
) -> dict[str, Any]:
    observed_conditions = {profile["condition"] for _, profile in profiled}
    return {
        "rows": len(profiled),
        "conditions": [
            condition
            for condition in gate.DEFAULT_TRAINING_CONDITIONS
            if condition in observed_conditions
        ],
        "row_id_set_sha256": sha256_text(
            canonical_json(sorted(profile["row_id"] for _, profile in profiled))
        ),
    }


def _dimension_partitions(dimensions: tuple[int, ...]) -> tuple[tuple[tuple[int, ...], ...], ...]:
    if not dimensions:
        return ((),)
    first, *remaining = dimensions
    results: set[tuple[tuple[int, ...], ...]] = set()
    for partition in _dimension_partitions(tuple(remaining)):
        results.add(((first,), *partition))
        for index in range(len(partition)):
            blocks = list(partition)
            blocks[index] = tuple(sorted((first, *blocks[index])))
            results.add(tuple(sorted(blocks)))
    return tuple(sorted(results))


def _padded_workload(
    profiles: Sequence[Mapping[str, Any]], batch_size: int
) -> dict[str, Any]:
    if not profiles:
        raise ValueError("Padded workload requires at least one example")
    vectors = [_length_vector(profile) for profile in profiles]
    maxima = tuple(max(vector[index] for vector in vectors) for index in range(5))
    write_positions = batch_size * sum(maxima[:4])
    read_positions = batch_size * maxima[4]
    return {
        "maximum_write_token_lengths": list(maxima[:4]),
        "maximum_read_token_length": maxima[4],
        "padded_write_token_positions": write_positions,
        "padded_read_token_positions": read_positions,
        "total_padded_token_positions": write_positions + read_positions,
    }


def select_padded_workload_examples(
    examples: Sequence[Any], batch_size: int
) -> tuple[list[Any], list[dict[str, Any]], dict[str, Any]]:
    """Exactly maximize batch padding across four writes and the read sequence."""

    if batch_size <= 0 or len(examples) < batch_size:
        raise ValueError("Batch size must be positive and no larger than the corpus")
    profiled = [(example, _example_token_profile(example)) for example in examples]
    allowed_conditions = set(gate.DEFAULT_TRAINING_CONDITIONS)
    if any(profile["condition"] not in allowed_conditions for _, profile in profiled):
        raise ValueError(
            "Padded-workload selection accepts only supervised positive "
            "training conditions"
        )
    profiled.sort(key=lambda item: item[1]["row_id"])
    row_ids = [profile["row_id"] for _, profile in profiled]
    if len(set(row_ids)) != len(row_ids):
        raise ValueError("Training examples contain duplicate row IDs")
    vectors = [_length_vector(profile) for _, profile in profiled]

    best_indices: tuple[int, ...] | None = None
    best_score = -1
    best_signature: tuple[str, ...] | None = None
    for partition in _dimension_partitions(tuple(range(5))):
        if len(partition) > batch_size:
            continue
        anchors: set[int] = set()
        for dimensions in partition:
            scores = [
                (
                    sum(vector[index] for index in dimensions),
                    profile["total_unpadded_token_positions"],
                    profile["read_token_length"],
                    *profile["write_token_lengths"],
                )
                for vector, (_, profile) in zip(vectors, profiled, strict=True)
            ]
            anchors.add(max(range(len(profiled)), key=scores.__getitem__))
        selected = set(anchors)
        while len(selected) < batch_size:
            candidates = [index for index in range(len(profiled)) if index not in selected]
            candidate_scores = []
            for index in candidates:
                profiles = [profiled[item][1] for item in (*selected, index)]
                profile = profiled[index][1]
                candidate_scores.append(
                    (
                        _padded_workload(profiles, batch_size)[
                            "total_padded_token_positions"
                        ],
                        profile["total_unpadded_token_positions"],
                        profile["read_token_length"],
                        *profile["write_token_lengths"],
                    )
                )
            selected.add(candidates[max(range(len(candidates)), key=candidate_scores.__getitem__)])
        ordered = tuple(sorted(selected))
        selected_profiles = [profiled[index][1] for index in ordered]
        score = _padded_workload(selected_profiles, batch_size)[
            "total_padded_token_positions"
        ]
        signature = tuple(profile["row_id"] for profile in selected_profiles)
        if score > best_score or (
            score == best_score
            and (best_signature is None or signature < best_signature)
        ):
            best_indices = ordered
            best_score = score
            best_signature = signature
    if best_indices is None:
        raise RuntimeError("Padded-workload selection produced no candidate batch")

    selected_pairs = [profiled[index] for index in best_indices]
    selected_pairs.sort(
        key=lambda item: (
            -item[1]["total_unpadded_token_positions"],
            -item[1]["read_token_length"],
            tuple(-length for length in item[1]["write_token_lengths"]),
            item[1]["row_id"],
        )
    )
    selected_profiles = [item[1] for item in selected_pairs]
    selected_workload = _padded_workload(selected_profiles, batch_size)
    all_workload = _padded_workload(
        [profile for _, profile in profiled], batch_size
    )
    audit = {
        "schema": PADDED_SELECTION_SCHEMA,
        "candidate_corpus": _selection_candidate_corpus(profiled),
        "method": (
            "exact five-dimension set-partition maximization with deterministic "
            "marginal-workload fill, total/read/write-length secondary ordering, "
            "and row-id final tie breaks"
        ),
        "dimensions": ["write_0", "write_1", "write_2", "write_3", "read"],
        "selected": selected_workload,
        "unconstrained_per_dimension_upper_bound": all_workload,
        "upper_bound_coverage_fraction": (
            selected_workload["total_padded_token_positions"]
            / all_workload["total_padded_token_positions"]
        ),
        "selected_batch_is_exact_constrained_optimum": True,
    }
    return [item[0] for item in selected_pairs], selected_profiles, audit


def select_longest_examples(
    examples: Sequence[Any], batch_size: int
) -> tuple[list[Any], list[dict[str, Any]]]:
    selected, profiles, _ = select_padded_workload_examples(examples, batch_size)
    return selected, profiles


def _answer_predictor_profile(example: Any) -> dict[str, Any]:
    labels = tuple(int(label) for label in example.labels)
    predictor_positions = tuple(
        label_index - 1
        for label_index, label in enumerate(labels[1:], start=1)
        if label != -100
    )
    if not predictor_positions:
        raise ValueError(f"Example {example.row_id} has no answer predictors")
    expected = tuple(
        range(predictor_positions[0], predictor_positions[-1] + 1)
    )
    if predictor_positions != expected:
        raise ValueError(
            f"Example {example.row_id} answer predictors are not contiguous"
        )
    return {
        "answer_predictor_start": predictor_positions[0],
        "answer_predictor_end_exclusive": predictor_positions[-1] + 1,
        "answer_predictor_positions": len(predictor_positions),
    }


def _answer_predictor_union_indices(
    profiles: Sequence[Mapping[str, Any]],
) -> tuple[int, ...]:
    positions: set[int] = set()
    for profile in profiles:
        positions.update(
            range(
                int(profile["answer_predictor_start"]),
                int(profile["answer_predictor_end_exclusive"]),
            )
        )
    if not positions:
        raise ValueError("Answer-logit workload requires supervised positions")
    return tuple(sorted(positions))


def select_answer_logit_examples(
    examples: Sequence[Any], batch_size: int
) -> tuple[list[Any], list[dict[str, Any]], dict[str, Any]]:
    """Exactly maximize compact-logit width for contiguous answer spans."""

    if batch_size <= 0 or len(examples) < batch_size:
        raise ValueError("Batch size must be positive and no larger than the corpus")
    profiled = [
        (
            example,
            {
                **_example_token_profile(example),
                **_answer_predictor_profile(example),
            },
        )
        for example in examples
    ]
    allowed_conditions = set(gate.DEFAULT_TRAINING_CONDITIONS)
    if any(profile["condition"] not in allowed_conditions for _, profile in profiled):
        raise ValueError(
            "Answer-logit selection accepts only supervised positive training "
            "conditions"
        )
    profiled.sort(key=lambda item: item[1]["row_id"])
    row_ids = [profile["row_id"] for _, profile in profiled]
    if len(set(row_ids)) != len(row_ids):
        raise ValueError("Training examples contain duplicate row IDs")

    interval_order = sorted(
        range(len(profiled)),
        key=lambda index: (
            profiled[index][1]["answer_predictor_start"],
            profiled[index][1]["answer_predictor_end_exclusive"],
            profiled[index][1]["row_id"],
        ),
    )
    states: dict[tuple[int, int], tuple[int, tuple[int, ...]]] = {}

    def retain(
        key: tuple[int, int],
        score: int,
        indices: tuple[int, ...],
    ) -> None:
        candidate = (score, tuple(sorted(indices)))
        current = states.get(key)
        if current is None or candidate[0] > current[0] or (
            candidate[0] == current[0] and candidate[1] < current[1]
        ):
            states[key] = candidate

    for position, profile_index in enumerate(interval_order):
        profile = profiled[profile_index][1]
        start = int(profile["answer_predictor_start"])
        end = int(profile["answer_predictor_end_exclusive"])
        retain((position, 1), end - start, (profile_index,))
        for previous_position in range(position):
            previous_profile = profiled[interval_order[previous_position]][1]
            previous_end = int(
                previous_profile["answer_predictor_end_exclusive"]
            )
            if previous_end >= end:
                continue
            added = end - max(start, previous_end)
            for count in range(2, batch_size + 1):
                previous = states.get((previous_position, count - 1))
                if previous is None:
                    continue
                retain(
                    (position, count),
                    previous[0] + added,
                    (*previous[1], profile_index),
                )

    best = max(
        states.values(),
        key=lambda candidate: (candidate[0], tuple(-index for index in candidate[1])),
    )
    selected = set(best[1])
    while len(selected) < batch_size:
        candidates = [index for index in range(len(profiled)) if index not in selected]
        candidate_scores = []
        for index in candidates:
            profiles = [profiled[item][1] for item in (*selected, index)]
            profile = profiled[index][1]
            candidate_scores.append(
                (
                    len(_answer_predictor_union_indices(profiles)),
                    _padded_workload(profiles, batch_size)[
                        "total_padded_token_positions"
                    ],
                    profile["total_unpadded_token_positions"],
                    profile["read_token_length"],
                    *profile["write_token_lengths"],
                )
            )
        selected.add(
            candidates[
                max(range(len(candidates)), key=candidate_scores.__getitem__)
            ]
        )

    selected_pairs = [profiled[index] for index in sorted(selected)]
    selected_pairs.sort(
        key=lambda item: (
            -item[1]["answer_predictor_positions"],
            -item[1]["total_unpadded_token_positions"],
            -item[1]["read_token_length"],
            tuple(-length for length in item[1]["write_token_lengths"]),
            item[1]["row_id"],
        )
    )
    selected_profiles = [item[1] for item in selected_pairs]
    union_indices = _answer_predictor_union_indices(selected_profiles)
    if len(union_indices) != best[0]:
        raise RuntimeError("Answer-logit interval optimization failed exactness audit")
    individual_upper_bound = sum(
        sorted(
            (
                int(profile["answer_predictor_positions"])
                for _, profile in profiled
            ),
            reverse=True,
        )[:batch_size]
    )
    audit = {
        "schema": ANSWER_LOGIT_SELECTION_SCHEMA,
        "candidate_corpus": _selection_candidate_corpus(profiled),
        "method": (
            "exact cardinality-bounded interval-union dynamic programming with "
            "deterministic activation-workload fill and row-id final tie breaks"
        ),
        "selected_answer_predictor_union_indices": list(union_indices),
        "selected_answer_predictor_union_positions": len(union_indices),
        "compact_logit_batch_position_factor": batch_size * len(union_indices),
        "sum_of_top_individual_widths_upper_bound": individual_upper_bound,
        "upper_bound_coverage_fraction": len(union_indices)
        / individual_upper_bound,
        "selected_batch_is_exact_constrained_optimum": True,
        "selected_padded_workload": _padded_workload(
            selected_profiles, batch_size
        ),
    }
    return [item[0] for item in selected_pairs], selected_profiles, audit


def _build_production_training_dataset_audit(
    examples: Sequence[Any],
) -> dict[str, Any]:
    audit = gate.audit_training_dataset(
        examples,
        gate.DEFAULT_TRAINING_CONDITIONS,
    )
    audit = gate.bind_production_training_contract(
        audit,
        epochs=PRODUCTION_EPOCHS,
        global_batch_size=DISTRIBUTED_GLOBAL_BATCH_SIZE,
        requested_max_steps=DISTRIBUTED_TRAINING_UPDATES,
        schedule_mode="complete",
    )
    if not audit["production_contract_passed"]:
        failures = sorted(
            name
            for name, passed in audit["production_dataset_contract_checks"].items()
            if not passed
        )
        failures.extend(
            name
            for name, passed in audit["schedule_contract_checks"].items()
            if not passed
        )
        raise ValueError(
            "CUDA profile training dataset differs from the compositional "
            "production contract: " + ", ".join(failures)
        )
    return audit


def _cuda_snapshot(device: torch.device) -> dict[str, int]:
    free_bytes, total_bytes = torch.cuda.mem_get_info(device)
    return {
        "allocated_bytes": int(torch.cuda.memory_allocated(device)),
        "reserved_bytes": int(torch.cuda.memory_reserved(device)),
        "free_bytes": int(free_bytes),
        "total_bytes": int(total_bytes),
    }


def _phase_result(
    device: torch.device,
    *,
    before: Mapping[str, int],
    elapsed_seconds: float,
) -> dict[str, Any]:
    after = _cuda_snapshot(device)
    if after["total_bytes"] != before["total_bytes"]:
        raise RuntimeError("CUDA total memory changed during a profiling phase")
    return {
        "before": dict(before),
        "after": after,
        "peak_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
        "peak_reserved_bytes": int(torch.cuda.max_memory_reserved(device)),
        "elapsed_seconds": elapsed_seconds,
    }


def build_memory_gate(
    *,
    initial_snapshot: Mapping[str, int],
    phases: Sequence[Mapping[str, Any]],
    minimum_headroom_bytes: int = MINIMUM_HEADROOM_BYTES,
) -> dict[str, Any]:
    if not phases:
        raise ValueError("At least one CUDA phase is required")
    total_bytes = int(initial_snapshot["total_bytes"])
    initial_free = int(initial_snapshot["free_bytes"])
    initial_reserved = int(initial_snapshot["reserved_bytes"])
    peak_allocated = max(int(phase["peak_allocated_bytes"]) for phase in phases)
    peak_reserved = max(int(phase["peak_reserved_bytes"]) for phase in phases)
    if peak_reserved < initial_reserved or peak_reserved > total_bytes:
        raise ValueError("CUDA peak-reserved accounting is invalid")
    isolated_headroom = total_bytes - peak_reserved
    environment_adjusted_headroom = initial_free - (
        peak_reserved - initial_reserved
    )
    observed_free = min(
        int(phase["after"]["free_bytes"])
        for phase in phases
    )
    post_peak_headroom = min(
        isolated_headroom,
        environment_adjusted_headroom,
        observed_free,
    )
    return {
        "device_total_bytes": total_bytes,
        "device_free_before_load_bytes": initial_free,
        "initial_process_reserved_bytes": initial_reserved,
        "peak_allocated_bytes": peak_allocated,
        "peak_reserved_bytes": peak_reserved,
        "isolated_reserved_headroom_bytes": isolated_headroom,
        "environment_adjusted_reserved_headroom_bytes": (
            environment_adjusted_headroom
        ),
        "minimum_observed_free_after_phase_bytes": observed_free,
        "post_peak_reserved_memory_headroom_bytes": post_peak_headroom,
        "minimum_required_headroom_bytes": int(minimum_headroom_bytes),
        "headroom_passed": post_peak_headroom >= minimum_headroom_bytes,
        "headroom_definition": (
            "minimum of total-minus-process-peak-reserved, initial-free-minus-"
            "incremental-process-peak-reserved, and observed post-phase free"
        ),
    }


def _summarize_trainable_audit(audit: Mapping[str, Any]) -> dict[str, Any]:
    actual = list(audit["actual_trainable_names"])
    allowed = list(audit["allowed_delta_mem_trainable_names"])
    expected = list(audit["expected_trainable_names"])
    return {
        "actual_trainable_parameter_tensors": len(actual),
        "allowed_trainable_parameter_tensors": len(allowed),
        "expected_trainable_parameter_tensors": len(expected),
        "actual_trainable_names_sha256": sha256_text(canonical_json(actual)),
        "allowed_trainable_names_sha256": sha256_text(canonical_json(allowed)),
        "expected_trainable_names_sha256": sha256_text(canonical_json(expected)),
        "only_delta_mem_parameters_trainable": audit[
            "only_delta_mem_parameters_trainable"
        ],
        "trainable_name_binding_passed": audit["trainable_name_binding_passed"],
        "nonempty_trainable_set": audit["nonempty_trainable_set"],
        "passed": audit["passed"],
    }


def _execute_optimizer_step(
    model: torch.nn.Module,
    batch: Any,
    optimizer: torch.optim.Optimizer,
    trainable: Sequence[torch.nn.Parameter],
    *,
    dtype: torch.dtype,
    include_router_gradient_audit: bool,
    rollover_liveness_holder: list[OptimizerStepLiveness] | None = None,
) -> tuple[dict[str, Any], OptimizerStepLiveness]:
    prior_liveness = (
        rollover_liveness_holder[0]
        if rollover_liveness_holder is not None and rollover_liveness_holder
        else None
    )
    write_audit = gate._write_episode_batch(model, batch, dtype=dtype)
    if prior_liveness is not None:
        prior_liveness.write_audit = None
    logits, route_logits = gate._read_episode_batch(model, batch, dtype=dtype)
    predictor_positions = gate.runtime._answer_predictor_indices(batch.labels)
    expected_logit_shape = (
        len(batch.examples),
        predictor_positions.numel(),
    )
    if logits.ndim != 3 or logits.shape[:2] != expected_logit_shape:
        raise RuntimeError(
            "Profiler read did not return compact supervised-position logits"
        )
    if prior_liveness is not None:
        prior_liveness.logits = None
        prior_liveness.route_logits = None
    answer_loss = gate.causal_answer_loss(logits, batch.labels)
    if prior_liveness is not None:
        prior_liveness.answer_loss = None
    route_loss, route_predictions = gate.route_loss_and_predictions(
        route_logits,
        batch.query_mask,
        batch.target_slots,
        hard_negative_margin=PRODUCTION_HARD_NEGATIVE_MARGIN,
        hard_negative_weight=PRODUCTION_HARD_NEGATIVE_WEIGHT,
    )
    if prior_liveness is not None:
        prior_liveness.route_loss = None
        prior_liveness.route_predictions = None
    total_loss = (
        PRODUCTION_ANSWER_WEIGHT * answer_loss
        + PRODUCTION_ROUTE_WEIGHT * route_loss
    )
    if prior_liveness is not None:
        prior_liveness.total_loss = None
    if not bool(torch.isfinite(total_loss).item()):
        raise RuntimeError("Profiler encountered a non-finite training loss")
    router_audit = None
    if include_router_gradient_audit:
        router_audit = gate.runtime._router_gradient_audit(model, route_loss)
    total_loss.backward()
    grad_norm = torch.nn.utils.clip_grad_norm_(
        trainable, PRODUCTION_MAX_GRAD_NORM
    )
    if prior_liveness is not None:
        prior_liveness.grad_norm = None
        prior_liveness.router_audit = None
    if rollover_liveness_holder is not None:
        rollover_liveness_holder.clear()
    if not bool(torch.isfinite(grad_norm).item()):
        raise RuntimeError("Profiler encountered a non-finite gradient norm")
    reset_delta_mem_states(model)
    optimizer.step()
    route_correct = sum(
        int(prediction.eq(batch.target_slots).sum().item())
        for prediction in route_predictions.values()
    )
    route_total = len(route_predictions) * len(batch.examples)
    result = {
        "rows": len(batch.examples),
        "answer_predictor_positions": int(predictor_positions.numel()),
        "answer_logit_shape": list(logits.shape),
        "answer_logit_dtype": str(logits.dtype),
        "answer_loss": float(answer_loss.detach().float().item()),
        "route_loss": float(route_loss.detach().float().item()),
        "total_loss": float(total_loss.detach().float().item()),
        "gradient_norm": float(grad_norm.detach().float().item()),
        "route_correct": route_correct,
        "route_total": route_total,
        "full_occupancy_count": int(write_audit["full_occupancy_count"]),
        "full_occupancy_total": int(write_audit["full_occupancy_total"]),
        "forced_write_route_match_count": int(
            write_audit["forced_write_route_match_count"]
        ),
        "forced_write_route_total": int(
            write_audit["forced_write_route_total"]
        ),
        "router_gradient_audit": router_audit,
    }
    if (
        result["full_occupancy_total"] <= 0
        or result["full_occupancy_count"] != result["full_occupancy_total"]
        or result["forced_write_route_total"] <= 0
        or result["forced_write_route_match_count"]
        != result["forced_write_route_total"]
    ):
        raise RuntimeError("Profiler step did not execute valid four-slot writes")
    if router_audit is not None and not router_audit["all_modules_finite_nonzero"]:
        raise RuntimeError("Profiler router-gradient audit failed")
    rollover_liveness = OptimizerStepLiveness(
        write_audit=write_audit,
        logits=logits,
        route_logits=route_logits,
        answer_loss=answer_loss,
        route_loss=route_loss,
        route_predictions=route_predictions,
        total_loss=total_loss,
        grad_norm=grad_norm,
        router_audit=router_audit,
    )
    return result, rollover_liveness


def _measure_optimizer_step(
    model: torch.nn.Module,
    examples: Sequence[Any],
    optimizer: torch.optim.Optimizer,
    trainable: Sequence[torch.nn.Parameter],
    *,
    pad_token_id: int,
    device: torch.device,
    dtype: torch.dtype,
    include_router_gradient_audit: bool,
    rollover_liveness_holder: list[OptimizerStepLiveness] | None = None,
    prior_batch_holder: list[Any] | None = None,
) -> tuple[dict[str, Any], OptimizerStepLiveness | None, Any | None]:
    torch.cuda.synchronize(device)
    torch.cuda.reset_peak_memory_stats(device)
    before = _cuda_snapshot(device)
    started = time.perf_counter()
    batch = None
    try:
        batch = gate.collate_examples(
            examples, pad_token_id=pad_token_id, device=device
        )
        if prior_batch_holder is not None:
            prior_batch_holder.clear()
        optimizer.zero_grad(set_to_none=True)
        step, rollover_liveness = _execute_optimizer_step(
            model,
            batch,
            optimizer,
            trainable,
            dtype=dtype,
            include_router_gradient_audit=include_router_gradient_audit,
            rollover_liveness_holder=rollover_liveness_holder,
        )
    except torch.cuda.OutOfMemoryError as error:
        error_traceback = traceback.format_exc()
        torch.cuda.synchronize(device)
        phase = _phase_result(
            device,
            before=before,
            elapsed_seconds=time.perf_counter() - started,
        )
        phase.update(
            {
                "status": "cuda_out_of_memory",
                "step": None,
                "error": {
                    "type": type(error).__name__,
                    "message": str(error),
                    "traceback_sha256": sha256_text(error_traceback),
                },
                "includes_first_step_router_gradient_audit": (
                    include_router_gradient_audit
                ),
            }
        )
        return phase, None, batch
    torch.cuda.synchronize(device)
    phase = _phase_result(
        device,
        before=before,
        elapsed_seconds=time.perf_counter() - started,
    )
    phase["status"] = "passed"
    phase["step"] = step
    phase["includes_first_step_router_gradient_audit"] = (
        include_router_gradient_audit
    )
    return phase, rollover_liveness, batch


def _measure_stress_sequence(
    model: torch.nn.Module,
    *,
    activation_examples: Sequence[Any],
    answer_logit_examples: Sequence[Any],
    optimizer: torch.optim.Optimizer,
    trainable: Sequence[torch.nn.Parameter],
    pad_token_id: int,
    device: torch.device,
    dtype: torch.dtype,
    measure_step: Callable[..., tuple[dict[str, Any], Any, Any]] = (
        _measure_optimizer_step
    ),
) -> tuple[dict[str, dict[str, Any] | None], int]:
    example_sets = {
        "activation_max": activation_examples,
        "answer_logit_max": answer_logit_examples,
    }
    specifications = tuple(
        (name, example_sets[current_batch], router_audit)
        for name, current_batch, router_audit in PROFILE_STRESS_SEQUENCE
    )
    phases: dict[str, dict[str, Any] | None] = {
        name: None for name, _, _ in specifications
    }
    completed_optimizer_steps = 0
    prior_liveness: OptimizerStepLiveness | None = None
    prior_batch: Any | None = None
    for name, examples, include_router_gradient_audit in specifications:
        liveness_holder = (
            [prior_liveness] if prior_liveness is not None else None
        )
        prior_batch_holder = [prior_batch] if prior_batch is not None else None
        prior_liveness = None
        prior_batch = None
        phase, current_liveness, current_batch = measure_step(
            model,
            examples,
            optimizer,
            trainable,
            pad_token_id=pad_token_id,
            device=device,
            dtype=dtype,
            include_router_gradient_audit=include_router_gradient_audit,
            rollover_liveness_holder=liveness_holder,
            prior_batch_holder=prior_batch_holder,
        )
        phases[name] = phase
        if liveness_holder is not None:
            liveness_holder.clear()
        if prior_batch_holder is not None:
            prior_batch_holder.clear()
        if phase.get("status") != "passed":
            current_liveness = None
            current_batch = None
            break
        completed_optimizer_steps += 1
        prior_liveness = current_liveness
        prior_batch = current_batch
    prior_liveness = None
    prior_batch = None
    return phases, completed_optimizer_steps


def _device_evidence(device: torch.device) -> dict[str, Any]:
    properties = torch.cuda.get_device_properties(device)
    return {
        "requested": str(device),
        "index": int(device.index),
        "name": properties.name,
        "capability": list(torch.cuda.get_device_capability(device)),
        "reported_total_memory_bytes": int(properties.total_memory),
        "torch_version": torch.__version__,
        "torch_cuda_version": torch.version.cuda,
    }


def _profile_worker(
    *,
    source_manifest: Path,
    batch_size: int,
    device_name: str,
) -> dict[str, Any]:
    if batch_size not in PROFILED_LOCAL_BATCH_SIZES:
        raise ValueError(f"Unsupported profiled local batch size: {batch_size}")
    os.environ["HF_ENDPOINT"] = HF_MIRROR_ENDPOINT
    gate.configure_hf_mirror(HF_MIRROR_ENDPOINT)
    device = _parse_cuda_device(device_name)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")
    if device.index >= torch.cuda.device_count():
        raise ValueError(f"CUDA device does not exist: {device}")
    torch.cuda.set_device(device)
    torch.cuda.empty_cache()
    torch.cuda.synchronize(device)
    torch.cuda.reset_peak_memory_stats(device)
    initial_snapshot = _cuda_snapshot(device)
    started = time.perf_counter()
    try:
        return _profile_worker_on_initialized_device(
            source_manifest=source_manifest,
            batch_size=batch_size,
            device=device,
            initial_snapshot=initial_snapshot,
        )
    except torch.cuda.OutOfMemoryError as error:
        error_traceback = traceback.format_exc()
        torch.cuda.synchronize(device)
        phase = _phase_result(
            device,
            before=initial_snapshot,
            elapsed_seconds=time.perf_counter() - started,
        )
        phase.update(
            {
                "status": "cuda_out_of_memory",
                "error": {
                    "type": type(error).__name__,
                    "message": str(error),
                    "traceback_sha256": sha256_text(error_traceback),
                },
            }
        )
        failure = _worker_failure_payload(
            source_manifest=source_manifest,
            batch_size=batch_size,
            device_name=device_name,
            error=error,
        )
        unsigned = dict(failure)
        unsigned.pop("worker_receipt_sha256")
        unsigned.update(
            {
                "device": _device_evidence(device),
                "cuda_oom_telemetry": {
                    "initial_snapshot": initial_snapshot,
                    "active_phase": phase,
                },
                "memory_gate": build_memory_gate(
                    initial_snapshot=initial_snapshot,
                    phases=[phase],
                ),
            }
        )
        return signed_payload(unsigned, "worker_receipt_sha256")


def _profile_worker_on_initialized_device(
    *,
    source_manifest: Path,
    batch_size: int,
    device: torch.device,
    initial_snapshot: Mapping[str, int],
) -> dict[str, Any]:

    manifest_path = _resolve_regular_file(
        source_manifest, "Natural development manifest"
    )
    bundle = gate.load_profile_bundle(manifest_path, profile=PRODUCTION_PROFILE)
    if not bundle.eligibility.get("passed"):
        raise ValueError("Formal development source eligibility failed")
    model_root, model_artifact_paths = gate.resolve_model_artifacts(
        bundle.model_binding
    )
    source_before = gate.snapshot_files(bundle.source_paths)
    model_before = gate.snapshot_files(model_artifact_paths)
    manifest_payload_sha256 = bundle.development_manifest["manifest_receipt"][
        "payload_sha256"
    ]
    delta_config = gate.build_delta_config(
        target_layers=PRODUCTION_TARGET_LAYERS,
        rank=PRODUCTION_RANK,
        key_dim=PRODUCTION_KEY_DIM,
        temperature=PRODUCTION_TEMPERATURE,
    )
    dtype = gate._dtype(PRODUCTION_DTYPE)

    torch.cuda.reset_peak_memory_stats(device)
    load_before = _cuda_snapshot(device)
    load_started = time.perf_counter()
    model, tokenizer, replaced_layers, trainable_names, checkpointed_mlps = (
        gate._load_model_and_tokenizer(
            {"model": {"path": str(model_root)}},
            device=device,
            dtype=dtype,
            attn_implementation=PRODUCTION_ATTN_IMPLEMENTATION,
            delta_config=delta_config,
        )
    )
    torch.cuda.synchronize(device)
    load_phase = _phase_result(
        device,
        before=load_before,
        elapsed_seconds=time.perf_counter() - load_started,
    )
    load_phase["status"] = "passed"
    audit = gate.audit_trainable_parameters(
        model, expected_trainable_names=trainable_names
    )
    if not audit["passed"]:
        raise RuntimeError("Only the expected Delta-Mem parameters may be trainable")

    training_examples = gate.build_training_examples(
        bundle.train_episodes,
        tokenizer,
        gate.DEFAULT_TRAINING_CONDITIONS,
    )
    training_dataset_audit = _build_production_training_dataset_audit(
        training_examples
    )
    training_row_id_set_sha256 = sha256_text(
        canonical_json(sorted(example.row_id for example in training_examples))
    )
    activation_examples, activation_profiles, activation_selection_audit = (
        select_padded_workload_examples(training_examples, batch_size)
    )
    activation_profiles = [
        {
            **profile,
            **_answer_predictor_profile(example),
        }
        for example, profile in zip(
            activation_examples, activation_profiles, strict=True
        )
    ]
    answer_logit_examples, answer_logit_profiles, answer_logit_selection_audit = (
        select_answer_logit_examples(training_examples, batch_size)
    )
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(
        trainable,
        lr=PRODUCTION_LEARNING_RATE,
        weight_decay=PRODUCTION_WEIGHT_DECAY,
        fused=PRODUCTION_OPTIMIZER_FUSED,
    )
    model.train()
    adapter_before = gate._state_dict_sha256(snapshot_delta_mem_weights(model))
    optimizer_phases, completed_optimizer_steps = _measure_stress_sequence(
        model,
        activation_examples=activation_examples,
        answer_logit_examples=answer_logit_examples,
        optimizer=optimizer,
        trainable=trainable,
        pad_token_id=int(tokenizer.pad_token_id),
        device=device,
        dtype=dtype,
    )
    reset_delta_mem_states(model)
    optimizer.zero_grad(set_to_none=True)
    gc.collect()
    torch.cuda.empty_cache()
    adapter_after = gate._state_dict_sha256(snapshot_delta_mem_weights(model))
    adapter_changed = adapter_after != adapter_before
    if completed_optimizer_steps > 0 and not adapter_changed:
        raise RuntimeError("Representative optimizer steps changed no Delta-Mem weights")

    measured_phases = [
        load_phase,
        *(
            phase
            for phase in optimizer_phases.values()
            if phase is not None
        ),
    ]
    memory_gate = build_memory_gate(
        initial_snapshot=load_before,
        phases=measured_phases,
    )
    execution_passed = (
        completed_optimizer_steps == PRODUCTION_PROFILE_OPTIMIZER_STEPS
    )
    execution_error = next(
        (
            phase.get("error")
            for phase in optimizer_phases.values()
            if phase is not None and phase.get("status") != "passed"
        ),
        None,
    )
    gate_passed = execution_passed and memory_gate["headroom_passed"]
    source_after = gate.assert_snapshot_unchanged(
        source_before, description="Natural profiling source"
    )
    model_after = gate.assert_snapshot_unchanged(
        model_before, description="Natural profiling model"
    )
    configuration = _production_configuration(
        batch_size=batch_size,
        delta_config=delta_config,
    )
    profiler_path = Path(__file__).resolve(strict=True)
    runner_path = Path(gate.__file__).resolve(strict=True)
    shared_runtime_path = Path(gate.runtime.__file__).resolve(strict=True)
    delta_api_path = Path(delta_core.__file__).resolve(strict=True)
    delta_impl_path = Path(delta_impl.__file__).resolve(strict=True)
    result = {
        "schema": WORKER_SCHEMA,
        "status": "passed" if gate_passed else "failed",
        "pid": os.getpid(),
        "hf_endpoint": os.environ["HF_ENDPOINT"],
        "device": _device_evidence(device),
        "source_manifest_path": str(manifest_path),
        "source_manifest_file_sha256": sha256_file(manifest_path),
        "source_manifest_payload_sha256": manifest_payload_sha256,
        "model_path": str(model_root),
        "model_binding_sha256": bundle.model_binding["binding_sha256"],
        "source_files_before": source_before,
        "source_files_after": source_after,
        "model_files_before": model_before,
        "model_files_after": model_after,
        "profiler_file_sha256": sha256_file(profiler_path),
        "natural_runner_file_sha256": sha256_file(runner_path),
        "shared_runtime_file_sha256": sha256_file(shared_runtime_path),
        "delta_api_file_sha256": sha256_file(delta_api_path),
        "delta_impl_file_sha256": sha256_file(delta_impl_path),
        "configuration": configuration,
        "replaced_layers": list(replaced_layers),
        "checkpointed_frozen_mlps": list(checkpointed_mlps),
        "trainable_audit": _summarize_trainable_audit(audit),
        "selection_policy": {
            "activation_stress": (
                "exactly maximize total batch-padded token positions across the four "
                "write invocations and one read invocation"
            ),
            "answer_logit_stress": (
                "exactly maximize the union of supervised causal answer-predictor "
                "positions projected through the vocabulary head"
            ),
        },
        "activation_selection_audit": activation_selection_audit,
        "answer_logit_selection_audit": answer_logit_selection_audit,
        "training_examples_considered": len(training_examples),
        "training_row_id_set_sha256": training_row_id_set_sha256,
        "training_dataset_audit": training_dataset_audit,
        "training_dataset_audit_sha256": sha256_text(
            canonical_json(training_dataset_audit)
        ),
        "activation_stress_examples": activation_profiles,
        "activation_stress_examples_sha256": sha256_text(
            canonical_json(activation_profiles)
        ),
        "answer_logit_stress_examples": answer_logit_profiles,
        "answer_logit_stress_examples_sha256": sha256_text(
            canonical_json(answer_logit_profiles)
        ),
        "adapter_state_sha256_before": adapter_before,
        "adapter_state_sha256_after": adapter_after,
        "adapter_changed": adapter_changed,
        "execution_gate": {
            "required_optimizer_steps": PRODUCTION_PROFILE_OPTIMIZER_STEPS,
            "completed_optimizer_steps": completed_optimizer_steps,
            "phase_completion": {
                name: phase is not None and phase.get("status") == "passed"
                for name, phase in optimizer_phases.items()
            },
            "error": execution_error,
            "passed": execution_passed,
        },
        "phases": {
            "fresh_model_load": load_phase,
            **optimizer_phases,
        },
        "memory_gate": memory_gate,
        "gate_passed": gate_passed,
    }
    del (
        optimizer,
        activation_examples,
        answer_logit_examples,
        training_examples,
        tokenizer,
        model,
    )
    gc.collect()
    torch.cuda.empty_cache()
    return signed_payload(result, "worker_receipt_sha256")


def _worker_failure_payload(
    *,
    source_manifest: Path,
    batch_size: int,
    device_name: str,
    error: BaseException,
) -> dict[str, Any]:
    profiler_path = Path(__file__).resolve(strict=True)
    runner_path = Path(gate.__file__).resolve(strict=True)
    shared_runtime_path = Path(gate.runtime.__file__).resolve(strict=True)
    delta_api_path = Path(delta_core.__file__).resolve(strict=True)
    delta_impl_path = Path(delta_impl.__file__).resolve(strict=True)
    manifest_path = source_manifest.expanduser().resolve()
    payload = {
        "schema": WORKER_SCHEMA,
        "status": "failed",
        "pid": os.getpid(),
        "hf_endpoint": os.environ.get("HF_ENDPOINT"),
        "device_requested": device_name,
        "profiled_local_batch_size": batch_size,
        "source_manifest_path": str(manifest_path),
        "source_manifest_file_sha256": (
            sha256_file(manifest_path) if manifest_path.is_file() else None
        ),
        "profiler_file_sha256": sha256_file(profiler_path),
        "natural_runner_file_sha256": sha256_file(runner_path),
        "shared_runtime_file_sha256": sha256_file(shared_runtime_path),
        "delta_api_file_sha256": sha256_file(delta_api_path),
        "delta_impl_file_sha256": sha256_file(delta_impl_path),
        "error": {
            "type": type(error).__name__,
            "message": str(error),
            "traceback_sha256": sha256_text(traceback.format_exc()),
        },
        "gate_passed": False,
    }
    return signed_payload(payload, "worker_receipt_sha256")


def build_worker_command(
    *,
    source_manifest: Path,
    worker_output: Path,
    batch_size: int,
    device_name: str,
) -> list[str]:
    return [
        sys.executable,
        str(Path(__file__).resolve(strict=True)),
        "--source-manifest",
        str(source_manifest),
        "--device",
        device_name,
        "--worker-local-batch-size",
        str(batch_size),
        "--worker-output",
        str(worker_output),
    ]


def _run_worker_process(
    *,
    source_manifest: Path,
    worker_output: Path,
    batch_size: int,
    device_name: str,
) -> WorkerProcessResult:
    environment = dict(os.environ)
    environment["HF_ENDPOINT"] = HF_MIRROR_ENDPOINT
    command = build_worker_command(
        source_manifest=source_manifest,
        worker_output=worker_output,
        batch_size=batch_size,
        device_name=device_name,
    )
    process = subprocess.Popen(
        command,
        cwd=Path(__file__).resolve(strict=True).parents[2],
        env=environment,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    timed_out = False
    try:
        stdout, stderr = process.communicate(timeout=WORKER_TIMEOUT_SECONDS)
    except subprocess.TimeoutExpired:
        timed_out = True
        process.kill()
        stdout, stderr = process.communicate()
        stderr = (
            stderr
            + f"\nProfiler worker exceeded {WORKER_TIMEOUT_SECONDS} seconds.\n"
        )
    return WorkerProcessResult(
        args=tuple(command),
        returncode=124 if timed_out else int(process.returncode),
        stdout=stdout,
        stderr=stderr,
        pid=int(process.pid),
    )


def _read_worker_result(path: Path, expected_batch_size: int) -> Mapping[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"Worker result is absent: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, Mapping) or value.get("schema") != WORKER_SCHEMA:
        raise ValueError(f"Worker result schema differs: {path}")
    if not verify_signed_payload(value, "worker_receipt_sha256"):
        raise ValueError(f"Worker receipt signature differs: {path}")
    configuration = value.get("configuration")
    configured_batch = (
        configuration.get("profiled_local_batch_size")
        if isinstance(configuration, Mapping)
        else value.get("profiled_local_batch_size")
    )
    if configured_batch != expected_batch_size:
        raise ValueError(f"Worker batch binding differs: {path}")
    return value


def _is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _is_finite_number(value: Any) -> bool:
    return type(value) in (int, float) and math.isfinite(float(value))


def _valid_file_snapshot(value: Any) -> bool:
    if not isinstance(value, Mapping) or not value:
        return False
    return all(
        isinstance(path, str)
        and bool(path)
        and isinstance(evidence, Mapping)
        and type(evidence.get("bytes")) is int
        and evidence["bytes"] >= 0
        and _is_sha256(evidence.get("sha256"))
        for path, evidence in value.items()
    )


def _valid_cuda_snapshot(value: Any, *, total_bytes: int) -> bool:
    if not isinstance(value, Mapping):
        return False
    fields = (
        "allocated_bytes",
        "reserved_bytes",
        "free_bytes",
        "total_bytes",
    )
    if any(type(value.get(field)) is not int for field in fields):
        return False
    allocated = value["allocated_bytes"]
    reserved = value["reserved_bytes"]
    free = value["free_bytes"]
    return (
        value["total_bytes"] == total_bytes
        and 0 <= allocated <= reserved <= total_bytes
        and 0 <= free <= total_bytes
    )


def _valid_phase_memory(value: Any, *, total_bytes: int) -> bool:
    if not isinstance(value, Mapping):
        return False
    if not _valid_cuda_snapshot(value.get("before"), total_bytes=total_bytes):
        return False
    if not _valid_cuda_snapshot(value.get("after"), total_bytes=total_bytes):
        return False
    peak_allocated = value.get("peak_allocated_bytes")
    peak_reserved = value.get("peak_reserved_bytes")
    elapsed = value.get("elapsed_seconds")
    return (
        type(peak_allocated) is int
        and type(peak_reserved) is int
        and 0 <= peak_allocated <= peak_reserved <= total_bytes
        and peak_allocated >= value["before"]["allocated_bytes"]
        and peak_allocated >= value["after"]["allocated_bytes"]
        and peak_reserved >= value["before"]["reserved_bytes"]
        and peak_reserved >= value["after"]["reserved_bytes"]
        and _is_finite_number(elapsed)
        and elapsed >= 0
    )


def _valid_optimizer_step(
    value: Any,
    *,
    batch_size: int,
    predictor_positions: int,
    include_router_gradient_audit: bool,
) -> bool:
    if not isinstance(value, Mapping):
        return False
    shape = value.get("answer_logit_shape")
    if not (
        type(value.get("rows")) is int
        and value["rows"] == batch_size
        and type(value.get("answer_predictor_positions")) is int
        and value["answer_predictor_positions"] == predictor_positions
        and isinstance(shape, list)
        and len(shape) == 3
        and all(type(dimension) is int for dimension in shape)
        and shape[0] == batch_size
        and shape[1] == predictor_positions
        and shape[2] > 0
        and value.get("answer_logit_dtype") == str(torch.bfloat16)
    ):
        return False
    loss_fields = ("answer_loss", "route_loss", "total_loss", "gradient_norm")
    if not all(
        _is_finite_number(value.get(field)) and value[field] >= 0
        for field in loss_fields
    ):
        return False
    expected_total = (
        PRODUCTION_ANSWER_WEIGHT * value["answer_loss"]
        + PRODUCTION_ROUTE_WEIGHT * value["route_loss"]
    )
    if not math.isclose(
        value["total_loss"], expected_total, rel_tol=1e-5, abs_tol=1e-4
    ):
        return False
    integer_fields = (
        "route_correct",
        "route_total",
        "full_occupancy_count",
        "full_occupancy_total",
        "forced_write_route_match_count",
        "forced_write_route_total",
    )
    if any(type(value.get(field)) is not int for field in integer_fields):
        return False
    if not (
        value["route_total"] > 0
        and 0 <= value["route_correct"] <= value["route_total"]
        and value["full_occupancy_total"] > 0
        and value["full_occupancy_count"] == value["full_occupancy_total"]
        and value["forced_write_route_total"] > 0
        and value["forced_write_route_match_count"]
        == value["forced_write_route_total"]
    ):
        return False
    router_audit = value.get("router_gradient_audit")
    if include_router_gradient_audit:
        return (
            isinstance(router_audit, Mapping)
            and router_audit.get("all_modules_finite_nonzero") is True
        )
    return router_audit is None


def _training_dataset_evidence_passed(result: Mapping[str, Any]) -> bool:
    audit = result.get("training_dataset_audit")
    if not isinstance(audit, Mapping):
        return False
    try:
        contract_passed = gate.validate_production_training_contract(
            audit,
            schedule_mode="complete",
        )
    except (KeyError, TypeError, ValueError, ZeroDivisionError):
        return False
    return (
        result.get("training_examples_considered") == PRODUCTION_TRAINING_ROWS
        and _is_sha256(result.get("training_row_id_set_sha256"))
        and audit.get("training_row_id_set_sha256")
        == result.get("training_row_id_set_sha256")
        and result.get("training_dataset_audit_sha256")
        == sha256_text(canonical_json(audit))
        and contract_passed
    )


def _selection_evidence_passed(
    result: Mapping[str, Any],
    *,
    batch_size: int,
    phases: Mapping[str, Any],
) -> bool:
    activation_profiles = result.get("activation_stress_examples")
    answer_profiles = result.get("answer_logit_stress_examples")
    activation_audit = result.get("activation_selection_audit")
    answer_audit = result.get("answer_logit_selection_audit")
    candidate_corpus = {
        "rows": PRODUCTION_TRAINING_ROWS,
        "conditions": list(gate.DEFAULT_TRAINING_CONDITIONS),
        "row_id_set_sha256": result.get("training_row_id_set_sha256"),
    }
    if not (
        isinstance(activation_profiles, list)
        and isinstance(answer_profiles, list)
        and len(activation_profiles) == batch_size
        and len(answer_profiles) == batch_size
        and isinstance(activation_audit, Mapping)
        and isinstance(answer_audit, Mapping)
        and result.get("training_examples_considered") == PRODUCTION_TRAINING_ROWS
        and _is_sha256(result.get("training_row_id_set_sha256"))
        and activation_audit.get("candidate_corpus") == candidate_corpus
        and answer_audit.get("candidate_corpus") == candidate_corpus
        and result.get("activation_stress_examples_sha256")
        == sha256_text(canonical_json(activation_profiles))
        and result.get("answer_logit_stress_examples_sha256")
        == sha256_text(canonical_json(answer_profiles))
    ):
        return False
    for profiles in (activation_profiles, answer_profiles):
        if any(
            not isinstance(profile, Mapping)
            or profile.get("condition") not in gate.DEFAULT_TRAINING_CONDITIONS
            or not isinstance(profile.get("row_id"), str)
            or not profile["row_id"].endswith(
                f"::training-condition={profile.get('condition')}"
            )
            for profile in profiles
        ):
            return False
        row_ids = [profile.get("row_id") for profile in profiles]
        if any(not isinstance(row_id, str) or not row_id for row_id in row_ids):
            return False
        if len(set(row_ids)) != batch_size:
            return False
    try:
        activation_workload = _padded_workload(activation_profiles, batch_size)
        activation_predictors = len(
            _answer_predictor_union_indices(activation_profiles)
        )
        answer_workload = _padded_workload(answer_profiles, batch_size)
        answer_union = _answer_predictor_union_indices(answer_profiles)
    except (KeyError, TypeError, ValueError):
        return False
    if not (
        activation_audit.get("schema") == PADDED_SELECTION_SCHEMA
        and activation_audit.get("selected_batch_is_exact_constrained_optimum")
        is True
        and activation_audit.get("selected") == activation_workload
        and answer_audit.get("schema") == ANSWER_LOGIT_SELECTION_SCHEMA
        and answer_audit.get("selected_batch_is_exact_constrained_optimum")
        is True
        and answer_audit.get("selected_answer_predictor_union_indices")
        == list(answer_union)
        and answer_audit.get("selected_answer_predictor_union_positions")
        == len(answer_union)
        and answer_audit.get("compact_logit_batch_position_factor")
        == batch_size * len(answer_union)
        and answer_audit.get("selected_padded_workload") == answer_workload
    ):
        return False
    predictor_counts = {
        "activation_max": activation_predictors,
        "answer_logit_max": len(answer_union),
    }
    vocab_sizes: set[int] = set()
    for name, current_batch, router_audit in PROFILE_STRESS_SEQUENCE:
        phase = phases.get(name)
        if not isinstance(phase, Mapping):
            return False
        if phase.get("includes_first_step_router_gradient_audit") is not router_audit:
            return False
        step = phase.get("step")
        if not _valid_optimizer_step(
            step,
            batch_size=batch_size,
            predictor_positions=predictor_counts[current_batch],
            include_router_gradient_audit=router_audit,
        ):
            return False
        vocab_sizes.add(step["answer_logit_shape"][2])
    return len(vocab_sizes) == 1


def _trainable_evidence_passed(value: Any) -> bool:
    if not isinstance(value, Mapping):
        return False
    counts = (
        value.get("actual_trainable_parameter_tensors"),
        value.get("allowed_trainable_parameter_tensors"),
        value.get("expected_trainable_parameter_tensors"),
    )
    return (
        all(type(count) is int for count in counts)
        and counts[0] > 0
        and counts[0] == counts[2]
        and counts[0] <= counts[1]
        and _is_sha256(value.get("actual_trainable_names_sha256"))
        and _is_sha256(value.get("allowed_trainable_names_sha256"))
        and _is_sha256(value.get("expected_trainable_names_sha256"))
        and value["actual_trainable_names_sha256"]
        == value["expected_trainable_names_sha256"]
        and value.get("only_delta_mem_parameters_trainable") is True
        and value.get("trainable_name_binding_passed") is True
        and value.get("nonempty_trainable_set") is True
        and value.get("passed") is True
    )


def _device_evidence_passed(
    result: Mapping[str, Any], *, device_name: str
) -> bool:
    device = result.get("device")
    memory_gate = result.get("memory_gate")
    expected_device = _parse_cuda_device(device_name)
    if not isinstance(device, Mapping) or not isinstance(memory_gate, Mapping):
        return False
    total_bytes = device.get("reported_total_memory_bytes")
    return (
        device.get("requested") == device_name
        and type(device.get("index")) is int
        and device["index"] == expected_device.index
        and isinstance(device.get("name"), str)
        and bool(device["name"])
        and type(total_bytes) is int
        and total_bytes > 0
        and memory_gate.get("device_total_bytes") == total_bytes
    )


def _memory_gate_matches_phases(result: Mapping[str, Any]) -> bool:
    phases = result.get("phases")
    if not isinstance(phases, Mapping):
        return False
    if set(phases) != set(PROFILE_PHASE_NAMES):
        return False
    measured_phases = [phases[name] for name in PROFILE_PHASE_NAMES]
    if any(
        not isinstance(phase, Mapping) or phase.get("status") != "passed"
        for phase in measured_phases
    ):
        return False
    return _memory_gate_matches_measured_phases(result, measured_phases)


def _memory_gate_matches_measured_phases(
    result: Mapping[str, Any],
    measured_phases: Sequence[Mapping[str, Any]],
) -> bool:
    memory_gate = result.get("memory_gate")
    if not isinstance(memory_gate, Mapping) or not measured_phases:
        return False
    load_before = measured_phases[0].get("before")
    if not isinstance(load_before, Mapping):
        return False
    try:
        recomputed = build_memory_gate(
            initial_snapshot=load_before,
            phases=measured_phases,
            minimum_headroom_bytes=MINIMUM_HEADROOM_BYTES,
        )
    except (KeyError, TypeError, ValueError):
        return False
    return dict(memory_gate) == recomputed


def _passing_worker_evidence(
    result: Mapping[str, Any], *, batch_size: int, device_name: str
) -> dict[str, Any]:
    phases = result.get("phases")
    exact_phase_set = (
        isinstance(phases, Mapping)
        and set(phases) == set(PROFILE_PHASE_NAMES)
    )
    phase_status = exact_phase_set and all(
        isinstance(phases[name], Mapping)
        and phases[name].get("status") == "passed"
        for name in PROFILE_PHASE_NAMES
    )
    device_passed = _device_evidence_passed(result, device_name=device_name)
    total_bytes = (
        result.get("device", {}).get("reported_total_memory_bytes")
        if isinstance(result.get("device"), Mapping)
        else None
    )
    phase_memory = (
        phase_status
        and type(total_bytes) is int
        and all(
            _valid_phase_memory(phases[name], total_bytes=total_bytes)
            for name in PROFILE_PHASE_NAMES
        )
    )
    execution_gate = result.get("execution_gate")
    optimizer_names = [name for name, _, _ in PROFILE_STRESS_SEQUENCE]
    execution_passed = (
        isinstance(execution_gate, Mapping)
        and execution_gate.get("required_optimizer_steps")
        == PRODUCTION_PROFILE_OPTIMIZER_STEPS
        and execution_gate.get("completed_optimizer_steps")
        == PRODUCTION_PROFILE_OPTIMIZER_STEPS
        and execution_gate.get("phase_completion")
        == {name: True for name in optimizer_names}
        and execution_gate.get("error") is None
        and execution_gate.get("passed") is True
    )
    training_dataset_passed = _training_dataset_evidence_passed(result)
    selection_passed = exact_phase_set and _selection_evidence_passed(
        result, batch_size=batch_size, phases=phases
    )
    optimizer_steps_passed = selection_passed
    adapter_passed = (
        _is_sha256(result.get("adapter_state_sha256_before"))
        and _is_sha256(result.get("adapter_state_sha256_after"))
        and result["adapter_state_sha256_before"]
        != result["adapter_state_sha256_after"]
        and result.get("adapter_changed") is True
    )
    snapshots_passed = all(
        _valid_file_snapshot(result.get(f"{prefix}_files_before"))
        and result.get(f"{prefix}_files_before")
        == result.get(f"{prefix}_files_after")
        for prefix in ("source", "model")
    )
    memory_consistent = _memory_gate_matches_phases(result)
    memory_gate = result.get("memory_gate")
    memory_headroom = (
        memory_consistent
        and isinstance(memory_gate, Mapping)
        and memory_gate.get("headroom_passed") is True
        and memory_gate.get("minimum_required_headroom_bytes")
        == MINIMUM_HEADROOM_BYTES
    )
    checks = {
        "worker_status": (
            result.get("status") == "passed"
            and result.get("gate_passed") is True
        ),
        "phase_set": exact_phase_set,
        "phase_status": phase_status,
        "phase_memory_evidence": phase_memory,
        "optimizer_step_evidence": optimizer_steps_passed,
        "execution_gate": execution_passed,
        "training_dataset_contract": training_dataset_passed,
        "selection_evidence": selection_passed,
        "adapter_change": adapter_passed,
        "trainable_boundary": _trainable_evidence_passed(
            result.get("trainable_audit")
        ),
        "immutable_snapshots": snapshots_passed,
        "device_evidence": device_passed,
        "memory_gate_recomputation": memory_consistent,
        "memory_headroom": memory_headroom,
    }
    failed_checks = [name for name, passed in checks.items() if not passed]
    return {
        "checks": checks,
        "failed_checks": failed_checks,
        "device_binding_passed": device_passed,
        "memory_gate_recomputation_passed": memory_consistent,
        "memory_headroom_passed": memory_headroom,
        "execution_passed": execution_passed and optimizer_steps_passed,
        "passed": not failed_checks,
    }


def _cuda_oom_observed(result: Mapping[str, Any]) -> bool:
    phases = result.get("phases")
    if isinstance(phases, Mapping) and any(
        isinstance(phase, Mapping)
        and phase.get("status") == "cuda_out_of_memory"
        for phase in phases.values()
    ):
        return True
    telemetry = result.get("cuda_oom_telemetry")
    return (
        isinstance(telemetry, Mapping)
        and isinstance(telemetry.get("active_phase"), Mapping)
        and telemetry["active_phase"].get("status") == "cuda_out_of_memory"
    )


def _valid_cuda_oom_error(value: Any) -> bool:
    return (
        isinstance(value, Mapping)
        and value.get("type") == "OutOfMemoryError"
        and isinstance(value.get("message"), str)
        and bool(value["message"])
        and _is_sha256(value.get("traceback_sha256"))
    )


def _cuda_oom_evidence_passed(result: Mapping[str, Any]) -> bool:
    phases = result.get("phases")
    if not isinstance(phases, Mapping) or set(phases) != set(PROFILE_PHASE_NAMES):
        return False
    load_phase = phases[PROFILE_PHASE_NAMES[0]]
    if not isinstance(load_phase, Mapping) or load_phase.get("status") != "passed":
        return False

    optimizer_names = tuple(name for name, _, _ in PROFILE_STRESS_SEQUENCE)
    oom_indices = [
        index
        for index, name in enumerate(optimizer_names)
        if isinstance(phases[name], Mapping)
        and phases[name].get("status") == "cuda_out_of_memory"
    ]
    if len(oom_indices) != 1:
        return False
    oom_index = oom_indices[0]
    measured_phases: list[Mapping[str, Any]] = [load_phase]
    expected_completion: dict[str, bool] = {}
    oom_error: Mapping[str, Any] | None = None
    for index, (name, _, router_audit) in enumerate(PROFILE_STRESS_SEQUENCE):
        phase = phases[name]
        expected_completion[name] = index < oom_index
        if index < oom_index:
            if not (
                isinstance(phase, Mapping)
                and phase.get("status") == "passed"
                and isinstance(phase.get("step"), Mapping)
                and phase.get("error") is None
                and phase.get("includes_first_step_router_gradient_audit")
                is router_audit
            ):
                return False
            measured_phases.append(phase)
        elif index == oom_index:
            if not (
                isinstance(phase, Mapping)
                and phase.get("status") == "cuda_out_of_memory"
                and phase.get("step") is None
                and phase.get("includes_first_step_router_gradient_audit")
                is router_audit
                and _valid_cuda_oom_error(phase.get("error"))
            ):
                return False
            oom_error = phase["error"]
            measured_phases.append(phase)
        elif phase is not None:
            return False

    execution_gate = result.get("execution_gate")
    phase_completion = (
        execution_gate.get("phase_completion")
        if isinstance(execution_gate, Mapping)
        else None
    )
    if not (
        isinstance(execution_gate, Mapping)
        and type(execution_gate.get("required_optimizer_steps")) is int
        and execution_gate["required_optimizer_steps"]
        == PRODUCTION_PROFILE_OPTIMIZER_STEPS
        and type(execution_gate.get("completed_optimizer_steps")) is int
        and execution_gate["completed_optimizer_steps"] == oom_index
        and isinstance(phase_completion, Mapping)
        and dict(phase_completion) == expected_completion
        and execution_gate.get("error") == oom_error
        and execution_gate.get("passed") is False
    ):
        return False

    device = result.get("device")
    total_bytes = (
        device.get("reported_total_memory_bytes")
        if isinstance(device, Mapping)
        else None
    )
    return (
        type(total_bytes) is int
        and total_bytes > 0
        and all(
            _valid_phase_memory(phase, total_bytes=total_bytes)
            for phase in measured_phases
        )
        and _memory_gate_matches_measured_phases(result, measured_phases)
    )


def _worker_invocation_evidence(
    *,
    batch_size: int,
    result_path: Path,
    stdout_path: Path,
    stderr_path: Path,
    completed: Any,
    parent_pid: int,
    device_name: str,
    expected_command: Sequence[str],
) -> dict[str, Any]:
    entry: dict[str, Any] = {
        "profiled_local_batch_size": batch_size,
        "subprocess_returncode": int(completed.returncode),
        "stdout": {
            "path": str(stdout_path),
            "bytes": stdout_path.stat().st_size,
            "sha256": sha256_file(stdout_path),
        },
        "stderr": {
            "path": str(stderr_path),
            "bytes": stderr_path.stat().st_size,
            "sha256": sha256_file(stderr_path),
        },
    }
    try:
        result = _read_worker_result(result_path, batch_size)
    except (OSError, ValueError, json.JSONDecodeError) as error:
        entry.update(
            {
                "result_valid": False,
                "error": {"type": type(error).__name__, "message": str(error)},
                "gate_passed": False,
            }
        )
        return entry
    reported_worker_pid = result.get("pid")
    subprocess_pid = getattr(completed, "pid", None)
    isolated = (
        type(reported_worker_pid) is int
        and type(subprocess_pid) is int
        and reported_worker_pid == subprocess_pid
        and subprocess_pid != parent_pid
    )
    memory_gate = result.get("memory_gate")
    execution_gate = result.get("execution_gate")
    configuration = result.get("configuration")
    command_passed = tuple(getattr(completed, "args", ())) == tuple(
        expected_command
    )
    manifest_argument_index = expected_command.index("--source-manifest") + 1
    source_manifest_path_passed = result.get("source_manifest_path") == str(
        Path(expected_command[manifest_argument_index]).resolve()
    )
    endpoint_passed = result.get("hf_endpoint") == HF_MIRROR_ENDPOINT
    worker_evidence = _passing_worker_evidence(
        result,
        batch_size=batch_size,
        device_name=device_name,
    )
    cuda_oom_observed = _cuda_oom_observed(result)
    cuda_oom_evidence_passed = _cuda_oom_evidence_passed(result)
    device_passed = worker_evidence["device_binding_passed"]
    memory_gate_consistent = worker_evidence[
        "memory_gate_recomputation_passed"
    ]
    memory_headroom_passed = worker_evidence["memory_headroom_passed"]
    execution_passed = worker_evidence["execution_passed"]
    expected_configuration = _production_configuration(batch_size=batch_size)
    configuration_passed = (
        isinstance(configuration, Mapping)
        and dict(configuration) == expected_configuration
    )
    entry.update(
        {
            "result_valid": True,
            "result": {
                "path": str(result_path),
                "bytes": result_path.stat().st_size,
                "sha256": sha256_file(result_path),
            },
            "worker_receipt_sha256": result["worker_receipt_sha256"],
            "reported_worker_pid": reported_worker_pid,
            "subprocess_pid": subprocess_pid,
            "fresh_process_isolation_passed": isolated,
            "worker_command_passed": command_passed,
            "source_manifest_path_passed": source_manifest_path_passed,
            "hf_endpoint_passed": endpoint_passed,
            "device_binding_passed": device_passed,
            "status": result.get("status"),
            "cuda_out_of_memory_observed": cuda_oom_observed,
            "cuda_oom_evidence_passed": cuda_oom_evidence_passed,
            "model_binding_sha256": result.get("model_binding_sha256"),
            "source_manifest_payload_sha256": result.get(
                "source_manifest_payload_sha256"
            ),
            "source_manifest_file_sha256": result.get(
                "source_manifest_file_sha256"
            ),
            "profiler_file_sha256": result.get("profiler_file_sha256"),
            "natural_runner_file_sha256": result.get(
                "natural_runner_file_sha256"
            ),
            "shared_runtime_file_sha256": result.get(
                "shared_runtime_file_sha256"
            ),
            "delta_api_file_sha256": result.get("delta_api_file_sha256"),
            "delta_impl_file_sha256": result.get("delta_impl_file_sha256"),
            "memory_gate": memory_gate,
            "memory_gate_recomputation_passed": memory_gate_consistent,
            "execution_gate": execution_gate,
            "worker_evidence_checks": worker_evidence["checks"],
            "worker_evidence_failed_checks": worker_evidence["failed_checks"],
            "worker_evidence_passed": worker_evidence["passed"],
            "configuration": configuration,
            "memory_headroom_passed": memory_headroom_passed,
            "execution_passed": execution_passed,
            "configuration_passed": configuration_passed,
            "gate_passed": bool(
                completed.returncode == 0
                and isolated
                and command_passed
                and source_manifest_path_passed
                and endpoint_passed
                and device_passed
                and worker_evidence["passed"]
                and configuration_passed
            ),
        }
    )
    return entry


def build_profile_gate(
    workers: Sequence[Mapping[str, Any]],
    *,
    expected_bindings: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    profiled = list(PROFILED_LOCAL_BATCH_SIZES)
    required = list(REQUIRED_LOCAL_BATCH_SIZES)
    exploratory = list(EXPLORATORY_LOCAL_BATCH_SIZES)
    actual = [worker.get("profiled_local_batch_size") for worker in workers]
    profile_set_complete = actual == profiled
    workers_by_batch = {
        worker.get("profiled_local_batch_size"): worker for worker in workers
    }
    required_present = all(actual.count(batch_size) == 1 for batch_size in required)
    required_workers = [
        workers_by_batch[batch_size]
        for batch_size in required
        if batch_size in workers_by_batch
    ]
    common_fields = (
        "model_binding_sha256",
        "source_manifest_payload_sha256",
        "source_manifest_file_sha256",
        "profiler_file_sha256",
        "natural_runner_file_sha256",
        "shared_runtime_file_sha256",
        "delta_api_file_sha256",
        "delta_impl_file_sha256",
    )
    bindings = {
        field: sorted(
            {
                str(worker.get(field))
                for worker in workers
                if worker.get(field) is not None
            }
        )
        for field in common_fields
    }

    def artifact_binding_passed(worker: Mapping[str, Any]) -> bool:
        presence = all(_is_sha256(worker.get(field)) for field in common_fields)
        expected = expected_bindings is None or all(
            worker.get(field) == expected_value
            for field, expected_value in expected_bindings.items()
        )
        return presence and expected

    required_valid = required_present and all(
        worker.get("result_valid") is True for worker in required_workers
    )
    required_isolated = required_valid and all(
        worker.get("fresh_process_isolation_passed") is True
        for worker in required_workers
    )
    required_invocation = required_valid and all(
        worker.get("subprocess_returncode") == 0
        and worker.get("worker_command_passed") is True
        and worker.get("source_manifest_path_passed") is True
        and worker.get("hf_endpoint_passed") is True
        and worker.get("device_binding_passed") is True
        for worker in required_workers
    )
    required_binding = required_valid and all(
        artifact_binding_passed(worker) for worker in required_workers
    )
    required_headroom = required_valid and all(
        worker.get("memory_headroom_passed") is True
        for worker in required_workers
    )
    required_execution = required_valid and all(
        worker.get("execution_passed") is True for worker in required_workers
    )
    required_worker_gate = required_valid and all(
        worker.get("gate_passed") is True for worker in required_workers
    )
    launch_checks = {
        "required_local_batch_present": required_present,
        "required_worker_receipt_valid": required_valid,
        "required_fresh_process_isolation": required_isolated,
        "required_worker_invocation_binding": required_invocation,
        "required_artifact_binding": required_binding,
        "required_memory_headroom": required_headroom,
        "required_optimizer_execution": required_execution,
        "required_worker_gate": required_worker_gate,
    }
    launch_failed_checks = [
        name for name, passed in launch_checks.items() if not passed
    ]
    launch_gate = {
        "selected_world_size": DISTRIBUTED_WORLD_SIZE,
        "selected_local_batch_size": DISTRIBUTED_LOCAL_BATCH_SIZE,
        "selected_global_batch_size": DISTRIBUTED_GLOBAL_BATCH_SIZE,
        "checks": launch_checks,
        "failed_checks": launch_failed_checks,
        "passed": not launch_failed_checks,
    }

    valid_process_ids = [
        worker.get("subprocess_pid")
        for worker in workers
        if worker.get("result_valid") is True
    ]
    distinct_processes = all(
        type(pid) is int for pid in valid_process_ids
    ) and len(valid_process_ids) == len(set(valid_process_ids))
    exploratory_outcomes: dict[str, str] = {}
    exploratory_passing: list[int] = []
    exploratory_oom: list[int] = []
    exploratory_insufficient_headroom: list[int] = []
    exploratory_unclassified: list[int] = []
    for batch_size in exploratory:
        worker = workers_by_batch.get(batch_size)
        worker_checks = (
            worker.get("worker_evidence_checks")
            if isinstance(worker, Mapping)
            else None
        )
        bound = (
            isinstance(worker, Mapping)
            and worker.get("result_valid") is True
            and worker.get("fresh_process_isolation_passed") is True
            and distinct_processes
            and worker.get("worker_command_passed") is True
            and worker.get("source_manifest_path_passed") is True
            and worker.get("hf_endpoint_passed") is True
            and worker.get("device_binding_passed") is True
            and worker.get("configuration_passed") is True
            and isinstance(worker_checks, Mapping)
            and worker_checks.get("immutable_snapshots") is True
            and worker_checks.get("trainable_boundary") is True
            and artifact_binding_passed(worker)
        )
        if bound and worker.get("gate_passed") is True:
            outcome = "passed"
            exploratory_passing.append(batch_size)
        elif (
            bound
            and worker.get("status") == "failed"
            and worker.get("cuda_out_of_memory_observed") is True
            and worker.get("cuda_oom_evidence_passed") is True
            and worker.get("subprocess_returncode") == 1
            and worker.get("gate_passed") is False
        ):
            outcome = "cuda_out_of_memory"
            exploratory_oom.append(batch_size)
        elif (
            bound
            and worker.get("status") == "failed"
            and worker.get("cuda_out_of_memory_observed") is False
            and worker.get("subprocess_returncode") == 1
            and worker.get("execution_passed") is True
            and worker.get("memory_gate_recomputation_passed") is True
            and worker.get("memory_headroom_passed") is False
            and worker.get("gate_passed") is False
        ):
            outcome = "insufficient_headroom"
            exploratory_insufficient_headroom.append(batch_size)
        else:
            outcome = "malformed_or_unclassified"
            exploratory_unclassified.append(batch_size)
        exploratory_outcomes[str(batch_size)] = outcome
    exploration_complete = (
        profile_set_complete
        and distinct_processes
        and not exploratory_unclassified
    )
    exploration = {
        "complete": exploration_complete,
        "outcomes_by_local_batch_size": exploratory_outcomes,
        "passing_local_batch_sizes": exploratory_passing,
        "cuda_oom_local_batch_sizes": exploratory_oom,
        "insufficient_headroom_local_batch_sizes": (
            exploratory_insufficient_headroom
        ),
        "malformed_or_unclassified_local_batch_sizes": exploratory_unclassified,
    }
    failed_checks = list(launch_failed_checks)
    if not exploration_complete:
        failed_checks.append("exploration_complete")
    return {
        "profiled_local_batch_sizes": profiled,
        "required_local_batch_sizes": required,
        "exploratory_local_batch_sizes": exploratory,
        "observed_local_batch_sizes": actual,
        "profile_set_complete": profile_set_complete,
        "required_worker_receipts_valid": required_valid,
        "required_fresh_process_isolation_passed": required_isolated,
        "required_worker_invocation_binding_passed": required_invocation,
        "required_artifact_binding_passed": required_binding,
        "required_memory_headroom_passed": required_headroom,
        "required_optimizer_execution_passed": required_execution,
        "common_bindings": bindings,
        "expected_bindings": dict(expected_bindings or {}),
        "launch_gate": launch_gate,
        "exploration": exploration,
        "failed_checks": failed_checks,
        "passed": launch_gate["passed"] and exploration_complete,
    }


def _parent_protocol_bindings(manifest_path: Path) -> dict[str, str]:
    bundle = gate.load_profile_bundle(
        manifest_path, profile=PRODUCTION_PROFILE
    )
    if bundle.eligibility.get("passed") is not True:
        raise ValueError("Formal development source eligibility failed in parent")
    manifest_receipt = bundle.development_manifest.get("manifest_receipt")
    if not isinstance(manifest_receipt, Mapping):
        raise ValueError("Development manifest receipt is absent")
    bindings = {
        "source_manifest_payload_sha256": manifest_receipt.get("payload_sha256"),
        "model_binding_sha256": bundle.model_binding.get("binding_sha256"),
    }
    if not all(_is_sha256(value) for value in bindings.values()):
        raise ValueError("Development source/model binding is invalid")
    return bindings


def run_orchestrator(
    *,
    source_manifest: Path,
    output_dir: Path,
    device_name: str,
    batch_sizes: Sequence[int] = PROFILED_LOCAL_BATCH_SIZES,
    worker_runner: Callable[..., Any] = (
        _run_worker_process
    ),
) -> dict[str, Any]:
    os.environ["HF_ENDPOINT"] = HF_MIRROR_ENDPOINT
    device = _parse_cuda_device(device_name)
    selected_batch_sizes = _parse_batch_sizes(batch_sizes)
    manifest_path = _resolve_regular_file(
        source_manifest, "Natural development manifest"
    )
    parent_bindings = _parent_protocol_bindings(manifest_path)
    requested_output = output_dir.expanduser()
    if requested_output.is_symlink():
        raise ValueError(f"Profiler output must not be a symbolic link: {requested_output}")
    resolved_output = requested_output.resolve()
    if resolved_output.exists():
        raise ValueError(f"Profiler output must be fresh: {resolved_output}")
    resolved_output.mkdir(parents=True, exist_ok=False)
    workers_dir = resolved_output / "workers"
    workers_dir.mkdir()

    profiler_path = Path(__file__).resolve(strict=True)
    runner_path = Path(gate.__file__).resolve(strict=True)
    shared_runtime_path = Path(gate.runtime.__file__).resolve(strict=True)
    delta_api_path = Path(delta_core.__file__).resolve(strict=True)
    delta_impl_path = Path(delta_impl.__file__).resolve(strict=True)
    protocol = signed_payload(
        {
            "schema": PROTOCOL_SCHEMA,
            "hf_endpoint": HF_MIRROR_ENDPOINT,
            "source_manifest_path": str(manifest_path),
            "source_manifest_file_sha256": sha256_file(manifest_path),
            **parent_bindings,
            "device": str(device),
            "profiled_local_batch_sizes": list(selected_batch_sizes),
            "required_local_batch_sizes": list(REQUIRED_LOCAL_BATCH_SIZES),
            "exploratory_local_batch_sizes": list(
                EXPLORATORY_LOCAL_BATCH_SIZES
            ),
            "distributed_training_target": _distributed_training_target(),
            "minimum_headroom_bytes": MINIMUM_HEADROOM_BYTES,
            "worker_timeout_seconds": WORKER_TIMEOUT_SECONDS,
            "configuration": _production_configuration(),
            "selection_policy": (
                "separate exact constrained maxima for padded activation positions "
                "and compact supervised answer-logit predictor-union width"
            ),
            "measurement_policy": (
                "each local batch runs sequentially in a fresh Python process and fresh "
                "model; local batch 1 is launch-critical while local batches 2 and 4 "
                "are exploratory observations; "
                "measure fresh load, activation-max cold AdamW with router audit, "
                "answer-logit-max AdamW with activation-max prior liveness and router "
                "audit, then activation-max AdamW with answer-logit-max prior liveness; "
                "prior locals are released at their production overwrite points"
            ),
            "profiler_file_sha256": sha256_file(profiler_path),
            "natural_runner_file_sha256": sha256_file(runner_path),
            "shared_runtime_file_sha256": sha256_file(shared_runtime_path),
            "delta_api_file_sha256": sha256_file(delta_api_path),
            "delta_impl_file_sha256": sha256_file(delta_impl_path),
        },
        "protocol_payload_sha256",
    )
    protocol_path = resolved_output / "protocol.json"
    _write_json_exclusive(protocol_path, protocol)

    workers: list[dict[str, Any]] = []
    parent_pid = os.getpid()
    for batch_size in selected_batch_sizes:
        result_path = workers_dir / f"local_batch_{batch_size}.json"
        stdout_path = workers_dir / f"local_batch_{batch_size}.stdout.log"
        stderr_path = workers_dir / f"local_batch_{batch_size}.stderr.log"
        expected_command = build_worker_command(
            source_manifest=manifest_path,
            worker_output=result_path,
            batch_size=batch_size,
            device_name=str(device),
        )
        try:
            completed = worker_runner(
                source_manifest=manifest_path,
                worker_output=result_path,
                batch_size=batch_size,
                device_name=str(device),
            )
        except Exception as error:
            completed = subprocess.CompletedProcess(
                args=expected_command,
                returncode=127,
                stdout="",
                stderr=f"{type(error).__name__}: {error}\n",
            )
        _write_text_exclusive(stdout_path, completed.stdout or "")
        _write_text_exclusive(stderr_path, completed.stderr or "")
        workers.append(
            _worker_invocation_evidence(
                batch_size=batch_size,
                result_path=result_path,
                stdout_path=stdout_path,
                stderr_path=stderr_path,
                completed=completed,
                parent_pid=parent_pid,
                device_name=str(device),
                expected_command=expected_command,
            )
        )

    profile_gate = build_profile_gate(
        workers,
        expected_bindings={
            "source_manifest_payload_sha256": protocol[
                "source_manifest_payload_sha256"
            ],
            "model_binding_sha256": protocol["model_binding_sha256"],
            "source_manifest_file_sha256": protocol[
                "source_manifest_file_sha256"
            ],
            "profiler_file_sha256": protocol["profiler_file_sha256"],
            "natural_runner_file_sha256": protocol[
                "natural_runner_file_sha256"
            ],
            "shared_runtime_file_sha256": protocol[
                "shared_runtime_file_sha256"
            ],
            "delta_api_file_sha256": protocol["delta_api_file_sha256"],
            "delta_impl_file_sha256": protocol["delta_impl_file_sha256"],
        },
    )
    receipt = signed_payload(
        {
            "schema": RECEIPT_SCHEMA,
            "status": "passed" if profile_gate["passed"] else "failed",
            "parent_pid": parent_pid,
            "protocol": {
                "path": str(protocol_path),
                "bytes": protocol_path.stat().st_size,
                "sha256": sha256_file(protocol_path),
                "protocol_payload_sha256": protocol["protocol_payload_sha256"],
            },
            "workers": workers,
            "gate": profile_gate,
            "launch_gate_passed": profile_gate["launch_gate"]["passed"],
            "exploration_complete": profile_gate["exploration"]["complete"],
            "gate_passed": profile_gate["passed"],
        },
        "profile_receipt_sha256",
    )
    receipt_path = resolved_output / "profile_receipt.json"
    _write_json_exclusive(receipt_path, receipt)
    return {
        "output_dir": str(resolved_output),
        "receipt_path": str(receipt_path),
        "receipt": receipt,
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--device", required=True)
    parser.add_argument("--local-batch-sizes", default="1,2,4")
    parser.add_argument(
        "--worker-local-batch-size", type=int, help=argparse.SUPPRESS
    )
    parser.add_argument("--worker-output", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    worker_mode = (
        args.worker_local_batch_size is not None or args.worker_output is not None
    )
    if worker_mode:
        if args.worker_local_batch_size is None or args.worker_output is None:
            parser.error("worker local batch size and output must be supplied together")
        if args.output_dir is not None:
            parser.error("worker mode does not accept --output-dir")
    elif args.output_dir is None:
        parser.error("orchestrator mode requires --output-dir")
    return args


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    os.environ["HF_ENDPOINT"] = HF_MIRROR_ENDPOINT
    if args.worker_local_batch_size is not None:
        try:
            result = _profile_worker(
                source_manifest=args.source_manifest,
                batch_size=args.worker_local_batch_size,
                device_name=args.device,
            )
        except Exception as error:
            result = _worker_failure_payload(
                source_manifest=args.source_manifest,
                batch_size=args.worker_local_batch_size,
                device_name=args.device,
                error=error,
            )
            traceback.print_exc()
        _write_json_exclusive(args.worker_output.resolve(), result)
        return 0 if result.get("gate_passed") is True else 1

    result = run_orchestrator(
        source_manifest=args.source_manifest,
        output_dir=args.output_dir,
        device_name=args.device,
        batch_sizes=_parse_batch_sizes(args.local_batch_sizes),
    )
    print(
        json.dumps(
            {
                "output_dir": result["output_dir"],
                "receipt_path": result["receipt_path"],
                "gate": result["receipt"]["gate"],
            },
            ensure_ascii=True,
            sort_keys=True,
        )
    )
    return 0 if result["receipt"]["gate_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
