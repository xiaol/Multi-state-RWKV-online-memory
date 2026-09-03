#!/usr/bin/env python3
"""Evaluate a direct native-PLE RWKV adapter without benchmark-time routing."""

from __future__ import annotations

import argparse
from contextlib import contextmanager, nullcontext
import gc
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import time
from typing import Any, Iterator, Mapping, Sequence

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

import torch
from torch.distributed.elastic.multiprocessing.errors import record


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[1]
for import_root in (SCRIPT_DIR, PROJECT_ROOT):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

from deltamem.core.delta import (  # noqa: E402
    get_delta_mem_online_state,
    iter_delta_mem_modules,
    load_delta_mem_adapter,
    load_delta_mem_online_state,
    reset_delta_mem_states,
    set_delta_mem_write_enabled,
)
from experiments.rethinking_rwkv_ms_gemma import (  # noqa: E402
    analyze_novel_agent_eval as recovery,
)
from experiments.rethinking_rwkv_ms_gemma import (  # noqa: E402
    natural_memory_distributed as distributed,
)
from experiments.rethinking_rwkv_ms_gemma import (  # noqa: E402
    recurrent_routed_posttrain_common as routed_common,
)
from experiments.rethinking_rwkv_ms_gemma import (  # noqa: E402
    run_natural_memory_native_projected_rwkv_hybrid_benchmark_eval as state_eval,
)
from experiments.rethinking_rwkv_ms_gemma import (  # noqa: E402
    run_novel_agent_eval as generation,
)
from experiments.rethinking_rwkv_ms_gemma import (  # noqa: E402
    run_recurrent_routed_development_generation as development,
)
from experiments.rethinking_rwkv_ms_gemma import (  # noqa: E402
    run_recurrent_routed_final as final,
)


SCHEMA = "rwkv_ms_natural_memory_native_rwkv_direct_ple_development.v1"
ROW_SCHEMA = "rwkv_ms_natural_memory_native_rwkv_direct_ple_development_row.v2"
CONTROL_SCHEMA = "rwkv_ms_natural_memory_native_rwkv_direct_ple_development_control.v1"
INPUT_SCHEMA = "rwkv_ms_natural_memory_native_rwkv_direct_ple_development_input.v1"
BASELINE_COMMIT_SCHEMA = "rwkv_ms_natural_memory_native_rwkv_direct_ple_baseline_commit.v1"
BASELINE_COMMIT_STATUS = "development_baselines_committed_candidate_not_evaluated"
BASELINE_SOURCE_RUNNER_SHA256 = "5b0a3ee4da7169dc0f7f8c597bb019f3caf877a6b6755454bc0a4980660d4b5d"
WORLD_SIZE = 4
TASKS = ("attribution", "narrative", "scene")
SYSTEMS = (
    "frozen_gemma_base",
    "v9_projected_slot_baseline",
    "direct_ple_candidate",
)
CONDITIONS = (
    "correct_recurrent_state",
    "zero_recurrent_state",
    "matched_donor_recurrent_state",
    "slot_shuffled_recurrent_state",
    "layer_permuted_recurrent_state",
    "projected_only_bypass",
)
STATE_CONDITIONS = CONDITIONS[:-1]
TASK_METRICS = {
    "attribution": "accuracy",
    "narrative": "unit_label_accuracy",
    "scene": "micro_f1",
}
MAX_NEW_TOKENS = {"attribution": 1024, "narrative": 1024, "scene": 128}
HF_MIRROR_ENDPOINT = "https://hf-mirror.com"
DEFAULT_BASE_MODEL = routed_common.BASE_MODEL
DEFAULT_V9_ADAPTER = SCRIPT_DIR / "local_artifacts/natural_memory_native_shared_qo_gate_stage1_v9/adapter"
DEFAULT_BASELINE_COMMIT = SCRIPT_DIR / "local_artifacts/natural_memory_native_rwkv_direct_ple_development_baselines_v1.json"
DEFAULT_OUTPUT = SCRIPT_DIR / "local_artifacts/natural_memory_native_rwkv_direct_ple_development_v1"


def canonical_sha256(value: Any) -> str:
    return distributed.canonical_sha256(value)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=True, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def append_jsonl(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(value, ensure_ascii=True, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=True, sort_keys=True) + "\n")


def validate_training_result(result_path: Path, adapter_dir: Path) -> Mapping[str, Any]:
    result_path = result_path.expanduser().resolve(strict=True)
    adapter_dir = adapter_dir.expanduser().resolve(strict=True)
    result = json.loads(result_path.read_text(encoding="utf-8"))
    receipt = result.get("receipt")
    unsigned = dict(result)
    unsigned.pop("receipt", None)
    allowed_schemas = {
        "rwkv_ms_natural_memory_native_rwkv_direct_ple_causal_train.v1",
        "rwkv_ms_natural_memory_native_rwkv_direct_ple_continuation.v5",
    }
    training = result.get("training", {})
    if (
        not isinstance(receipt, Mapping)
        or receipt.get("payload_sha256") != canonical_sha256(unsigned)
        or result.get("schema") not in allowed_schemas
        or result.get("status") != "training_complete_development_evaluation_authorized"
        or result.get("passed") is not True
        or result.get("updates") != 32
        or result.get("open_development_evaluation_authorized") is not True
        or result.get("final_rows_opened") is not False
        or result.get("publisher_validation_opened") is not False
        or result.get("publisher_test_opened") is not False
        or result.get("protected_splits_opened") not in (None, [])
    ):
        raise ValueError("Direct PLE training result does not authorize development")
    if result.get("schema") == "rwkv_ms_natural_memory_native_rwkv_direct_ple_continuation.v5":
        binding = training.get("step_binding_audit")
        if (
            not isinstance(binding, Mapping)
            or binding.get("passed") is not True
            or binding.get("rows") != 32
            or binding.get("learning_rate") != 2.5e-5
            or binding.get("max_gradient_norm") != 0.1
            or binding.get("contrast_margin") != 0.05
            or binding.get("protocol_payload_sha256")
            != "29657c8d7172a402e808276890d464de8051450023b4d511f11ae22f815cafdc"
        ):
            raise ValueError("Direct PLE continuation step binding does not authorize development")
    expected_files = result.get("adapter_files")
    if not isinstance(expected_files, Mapping):
        raise ValueError("Direct PLE adapter manifest is missing")
    for filename, metadata in expected_files.items():
        path = adapter_dir / str(filename)
        if not path.is_file() or sha256_file(path) != metadata.get("sha256"):
            raise ValueError(f"Direct PLE adapter file differs: {path}")
    return result


def baseline_commit_binding(
    commit_path: Path,
    commit: Mapping[str, Any],
) -> Mapping[str, Any]:
    resolved = commit_path.expanduser().resolve(strict=True)
    return {
        "path": str(resolved),
        "sha256": sha256_file(resolved),
        "receipt": commit["receipt"]["payload_sha256"],
        "files": {
            system: {
                "rows": int(commit["files"][system]["rows"]),
                "sha256": str(commit["files"][system]["sha256"]),
            }
            for system in SYSTEMS[:2]
        },
    }


def validate_baseline_commit(
    commit_path: Path,
    *,
    base_model: Path,
    v9_adapter: Path,
) -> tuple[Mapping[str, Any], Mapping[str, list[dict[str, Any]]]]:
    resolved = commit_path.expanduser().resolve(strict=True)
    commit = json.loads(resolved.read_text(encoding="utf-8"))
    receipt = commit.get("receipt")
    unsigned = dict(commit)
    unsigned.pop("receipt", None)
    source = commit.get("source", {})
    bindings = commit.get("bindings", {})
    rows_by_task = development.read_v2_rows()
    expected_rows_by_task = {task: len(rows_by_task[task]) for task in TASKS}
    expected_rows = {
        (task, int(row["line_index"])): row
        for task in TASKS
        for row in rows_by_task[task]
    }
    resolved_v9 = v9_adapter.expanduser().resolve(strict=True)
    expected_bindings = {
        "base_model": str(base_model.expanduser().resolve(strict=True)),
        "base_model_revision": routed_common.BASE_MODEL_REVISION,
        "base_model_weights_sha256": routed_common.BASE_MODEL_WEIGHTS_SHA256,
        "v9_adapter": str(resolved_v9),
        "v9_adapter_weights_sha256": sha256_file(resolved_v9 / "delta_mem_adapter.pt"),
        "v9_adapter_config_sha256": sha256_file(resolved_v9 / "delta_mem_config.json"),
        "development_manifest_receipt": development.V2_MANIFEST_RECEIPT,
        "prompt_variants_sha256": canonical_sha256(routed_common.PROMPT_VARIANTS),
        "systems": list(SYSTEMS[:2]),
        "rows_per_task": expected_rows_by_task,
        "world_size": WORLD_SIZE,
    }
    if (
        not isinstance(receipt, Mapping)
        or receipt.get("algorithm") != "sha256"
        or receipt.get("payload_scope") != "canonical_result_without_receipt"
        or receipt.get("payload_sha256") != canonical_sha256(unsigned)
        or commit.get("schema") != BASELINE_COMMIT_SCHEMA
        or commit.get("status") != BASELINE_COMMIT_STATUS
        or commit.get("passed") is not True
        or bindings != expected_bindings
        or source.get("source_runner_sha256") != BASELINE_SOURCE_RUNNER_SHA256
        or source.get("candidate_prediction_files") != []
        or source.get("control_files") != []
        or commit.get("task_router") is not False
        or commit.get("template_matcher") is not False
        or commit.get("dual_pass_selector") is not False
        or commit.get("benchmark_specific_decoder") is not False
        or commit.get("final_rows_opened") is not False
        or commit.get("publisher_validation_opened") is not False
        or commit.get("publisher_test_opened") is not False
        or set(commit.get("files", {})) != set(SYSTEMS[:2])
        or set(commit.get("summaries", {})) != set(SYSTEMS[:2])
    ):
        raise ValueError("Direct PLE baseline commit contract differs")
    source_root = Path(str(source.get("root", ""))).expanduser().resolve(strict=True)
    failure_log = Path(str(source.get("failure_log", ""))).expanduser().resolve(strict=True)
    if (
        not source_root.is_dir()
        or sha256_file(failure_log) != source.get("failure_log_sha256")
        or source.get("failure")
        != "candidate_fp32_parameter_storage_without_bf16_autocast"
    ):
        raise ValueError("Direct PLE baseline source evidence differs")
    records_by_system: dict[str, list[dict[str, Any]]] = {}
    for system in SYSTEMS[:2]:
        metadata = commit["files"][system]
        path = Path(str(metadata["path"])).expanduser().resolve(strict=True)
        records = read_jsonl(path)
        actual = {
            (str(row["task"]), int(row["line_index"])): row for row in records
        }
        if (
            len(records) != len(expected_rows)
            or len(actual) != len(expected_rows)
            or set(actual) != set(expected_rows)
            or len(records) != int(metadata["rows"])
            or sha256_file(path) != metadata["sha256"]
        ):
            raise ValueError(f"Direct PLE committed baseline rows differ: {system}")
        for key, row in actual.items():
            expected = expected_rows[key]
            if (
                row.get("system") != system
                or row.get("row_sha256") != expected["row_sha256"]
                or int(row["prompt_variant"]) != int(expected["prompt_variant"])
            ):
                raise ValueError(f"Direct PLE baseline row binding differs: {system} {key}")
        shards = metadata.get("shards", {})
        if set(shards) != {str(rank) for rank in range(WORLD_SIZE)}:
            raise ValueError(f"Direct PLE baseline shard set differs: {system}")
        shard_records: list[dict[str, Any]] = []
        for rank in range(WORLD_SIZE):
            shard_metadata = shards[str(rank)]
            shard_path = Path(str(shard_metadata["path"])).expanduser().resolve(strict=True)
            shard_rows = read_jsonl(shard_path)
            if (
                len(shard_rows) != int(shard_metadata["rows"])
                or sha256_file(shard_path) != shard_metadata["sha256"]
            ):
                raise ValueError(f"Direct PLE baseline shard differs: {system} rank={rank}")
            shard_records.extend(shard_rows)
        shard_records.sort(
            key=lambda row: (TASKS.index(str(row["task"])), int(row["line_index"]))
        )
        if canonical_sha256(shard_records) != canonical_sha256(records):
            raise ValueError(f"Direct PLE baseline consolidation differs: {system}")
        summary = summarize_records(records)
        if canonical_sha256(summary) != canonical_sha256(commit["summaries"][system]):
            raise ValueError(f"Direct PLE baseline summary differs: {system}")
        records_by_system[system] = records
    return commit, records_by_system


def preserve_candidate_precision(model: torch.nn.Module, result: Mapping[str, Any]) -> Mapping[str, Any]:
    trainables = result.get("input_binding", {}).get("model_audit", {}).get("trainables", {})
    suffixes = tuple(str(value) for value in trainables.get("trainable_suffixes", ()))
    core_family = ".hrm_rwkv7_core."
    expected = int(trainables.get("parameter_tensors", 0))
    if not suffixes or expected <= 0:
        raise ValueError("Direct PLE trainable precision binding is missing")
    selected = []
    for name, parameter in model.named_parameters():
        is_core = core_family in name and not name.endswith(".ln_x.bias")
        if name.endswith(suffixes) or is_core:
            parameter.data = parameter.data.float()
            parameter.requires_grad_(False)
            selected.append(name)
    if len(selected) != expected:
        raise ValueError(
            f"Direct PLE trainable precision count differs: expected={expected} actual={len(selected)}"
        )
    return {
        "parameter_tensors": len(selected),
        "parameter_suffixes": list(suffixes),
        "parameter_families": [core_family],
        "parameter_names_sha256": canonical_sha256(selected),
        "dtype": "float32",
    }


@contextmanager
def projected_only_bypass(model: torch.nn.Module) -> Iterator[None]:
    modules = tuple(iter_delta_mem_modules(model))
    saved = [
        (module, module.memory_readout_mode, module.rwkv_ms_hybrid_mode)
        for _, module in modules
    ]
    for module, _, _ in saved:
        module.memory_readout_mode = "projected_kv_slots"
        module.rwkv_ms_hybrid_mode = "addressed_moe_controller"
    try:
        yield
    finally:
        for module, readout_mode, hybrid_mode in saved:
            module.memory_readout_mode = readout_mode
            module.rwkv_ms_hybrid_mode = hybrid_mode


def stack_states(states: Mapping[str, Mapping[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    if not states:
        raise ValueError("Cannot stack an empty state mapping")
    keys = {frozenset(value) for value in states.values()}
    if len(keys) != 1:
        raise ValueError("Condition state keys differ")
    return {
        key: torch.cat([state[key] for state in states.values()], dim=0)
        for key in next(iter(keys))
    }


def slot_shuffled_recurrent_state(
    recurrent: Mapping[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    shuffled = {
        key: value.detach().cpu().clone() for key, value in recurrent.items()
    }
    for key, value in recurrent.items():
        if key.endswith(".__rwkv_ms_positions") or key.endswith(".__rwkv_ms_previous_source"):
            continue
        shuffled[key] = value.roll(shifts=1, dims=2).detach().cpu().clone()
    return shuffled


def generate_from_states(
    model: torch.nn.Module,
    tokenizer: Any,
    messages: Sequence[Mapping[str, str]],
    states: Mapping[str, Mapping[str, torch.Tensor]],
    *,
    task: str,
    device: str,
    bypass: bool = False,
) -> Mapping[str, Mapping[str, Any]]:
    ordered_conditions = tuple(states)
    reset_delta_mem_states(model)
    load_delta_mem_online_state(model, stack_states(states))
    set_delta_mem_write_enabled(model, False)
    context = projected_only_bypass(model) if bypass else nullcontext()
    try:
        encoded = state_eval.causal.encode_prompt(tokenizer, messages, generation=True)
        input_ids = encoded.input_ids.to(device).expand(len(ordered_conditions), -1).clone()
        attention_mask = encoded.attention_mask.to(device).expand(len(ordered_conditions), -1).clone()
        if device.startswith("cuda"):
            torch.cuda.reset_peak_memory_stats(device)
        started = time.perf_counter()
        with context, torch.inference_mode(), generation.inference_autocast_context(model, device):
            output_ids = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                generation_config=generation.model_generation_config(
                    model, tokenizer, MAX_NEW_TOKENS[task]
                ),
            )
        elapsed = time.perf_counter() - started
        generated_ids = output_ids[:, input_ids.size(1) :]
        raw = [
            tokenizer.decode(value, skip_special_tokens=True).strip()
            for value in generated_ids
        ]
        peak = (
            int(torch.cuda.max_memory_allocated(device))
            if device.startswith("cuda")
            else None
        )
        return {
            condition: {
                "raw_generation": value,
                "parsed_json": generation.extract_json(value),
                "input_tokens": int(input_ids.size(1)),
                "output_tokens": int(generated_ids[index].size(0)),
                "hit_max_new_tokens": int(generated_ids[index].size(0)) >= MAX_NEW_TOKENS[task],
                "elapsed_seconds": elapsed,
                "peak_cuda_memory_bytes": peak,
                "generation_batch_conditions": len(ordered_conditions),
            }
            for index, (condition, value) in enumerate(zip(ordered_conditions, raw, strict=True))
        }
    finally:
        reset_delta_mem_states(model)
        set_delta_mem_write_enabled(model, True)


def recovered_score(task: str, prediction: Any, gold: Mapping[str, Any]) -> Mapping[str, Any]:
    if task == "attribution":
        return {
            "covered": prediction is not None,
            "correct": bool(prediction is not None and prediction == gold.get("best_candidate")),
        }
    if task == "narrative":
        labels = recovery.gold_label_map(gold)
        return {
            "covered": prediction is not None,
            "correct_units": (
                sum(prediction.get(key) == value for key, value in labels.items())
                if isinstance(prediction, Mapping)
                else 0
            ),
            "gold_units": len(labels),
        }
    gold_boundaries = recovery.strict_gold_boundaries(gold)
    predicted = set(prediction if isinstance(prediction, list) else [])
    return {
        "covered": prediction is not None,
        "tp": len(predicted & gold_boundaries),
        "fp": len(predicted - gold_boundaries),
        "fn": len(gold_boundaries - predicted),
    }


def candidate_record(
    task: str,
    row: Mapping[str, Any],
    condition: str,
    generated: Mapping[str, Any],
    *,
    state: Mapping[str, torch.Tensor],
    correct_state: Mapping[str, torch.Tensor],
    projected_digest: str,
    donor: Mapping[str, Any],
    module_names: Sequence[str],
    zero_projected_output_identical: bool,
) -> Mapping[str, Any]:
    raw = str(generated["raw_generation"])
    parsed = generated["parsed_json"]
    strict = final.strict_task_json(task, raw)
    prediction = final.prediction_for(task, parsed, row["messages"])
    score = generation.score_prediction(task, strict, row["gold"])
    recurrent, projected = state_eval.split_state(state, module_names)
    return {
        "schema": ROW_SCHEMA,
        "system": "direct_ple_candidate",
        "task": task,
        "line_index": int(row["line_index"]),
        "source_ordinal": int(row["source_ordinal"]),
        "prompt_variant": int(row["prompt_variant"]),
        "row_sha256": row["row_sha256"],
        "gold": row["gold"],
        "prediction": prediction,
        "score": score,
        "recovered_score": recovered_score(task, prediction, row["gold"]),
        "max_new_tokens": MAX_NEW_TOKENS[task],
        "online_memory_protocol": "write_then_read_state_intervention",
        "raw_generation": raw,
        "parsed_json": parsed,
        "strict_parsed_json": strict,
        "input_tokens": generated["input_tokens"],
        "output_tokens": generated["output_tokens"],
        "hit_max_new_tokens": generated["hit_max_new_tokens"],
        "elapsed_seconds": generated["elapsed_seconds"],
        "peak_cuda_memory_bytes": generated["peak_cuda_memory_bytes"],
        "condition": condition,
        "state_sha256": state_eval.tensor_digest(state),
        "correct_state_sha256": state_eval.tensor_digest(correct_state),
        "condition_recurrent_sha256": state_eval.tensor_digest(recurrent),
        "projected_carrier_sha256": state_eval.tensor_digest(projected),
        "correct_projected_carrier_sha256": projected_digest,
        "projected_carrier_byte_identical": state_eval.tensor_digest(projected) == projected_digest,
        "projected_carrier_fixed": state_eval.tensor_digest(projected) == projected_digest,
        "zero_vs_projected_output_identical": zero_projected_output_identical,
        "donor_source_ordinal": int(donor["source_ordinal"]),
        "donor_row_sha256": donor["row_sha256"],
        "benchmark_time_task_router": False,
        "benchmark_time_template_matcher": False,
        "benchmark_time_dual_pass_selector": False,
        "benchmark_specific_decoder": False,
    }


def candidate_row_conditions(
    model: torch.nn.Module,
    tokenizer: Any,
    row: Mapping[str, Any],
    donor: Mapping[str, Any],
    *,
    module_names: Sequence[str],
    device: str,
) -> list[Mapping[str, Any]]:
    with torch.inference_mode(), generation.inference_autocast_context(model, device):
        correct_state = state_eval.prime_state(model, tokenizer, row, device=device)
        donor_state = state_eval.prime_state(model, tokenizer, donor, device=device)
    correct_recurrent, correct_projected = state_eval.split_state(correct_state, module_names)
    donor_recurrent, donor_projected = state_eval.split_state(donor_state, module_names)
    projected_digest = state_eval.tensor_digest(correct_projected)
    if state_eval.tensor_digest(donor_projected) == projected_digest:
        raise RuntimeError("Direct PLE donor projected carrier unexpectedly matches target")
    states = {
        "correct_recurrent_state": state_eval.merge_state(correct_recurrent, correct_projected),
        "zero_recurrent_state": state_eval.merge_state(
            state_eval.zero_recurrent_state(correct_recurrent), correct_projected
        ),
        "matched_donor_recurrent_state": state_eval.merge_state(donor_recurrent, correct_projected),
        "slot_shuffled_recurrent_state": state_eval.merge_state(
            slot_shuffled_recurrent_state(correct_recurrent), correct_projected
        ),
        "layer_permuted_recurrent_state": state_eval.merge_state(
            state_eval.permute_recurrent_state(correct_recurrent, module_names), correct_projected
        ),
    }
    generated = generate_from_states(
        model,
        tokenizer,
        row["messages"],
        states,
        task=row["task"],
        device=device,
    )
    bypass = generate_from_states(
        model,
        tokenizer,
        row["messages"],
        {"projected_only_bypass": states["correct_recurrent_state"]},
        task=row["task"],
        device=device,
        bypass=True,
    )["projected_only_bypass"]
    zero_projected_output_identical = (
        generated["zero_recurrent_state"]["raw_generation"] == bypass["raw_generation"]
        and generated["zero_recurrent_state"]["parsed_json"] == bypass["parsed_json"]
    )
    records = []
    for condition in STATE_CONDITIONS:
        records.append(
            candidate_record(
                row["task"],
                row,
                condition,
                generated[condition],
                state=states[condition],
                correct_state=correct_state,
                projected_digest=projected_digest,
                donor=donor,
                module_names=module_names,
                zero_projected_output_identical=zero_projected_output_identical,
            )
        )
    records.append(
        candidate_record(
            row["task"],
            row,
            "projected_only_bypass",
            bypass,
            state=states["correct_recurrent_state"],
            correct_state=correct_state,
            projected_digest=projected_digest,
            donor=donor,
            module_names=module_names,
            zero_projected_output_identical=zero_projected_output_identical,
        )
    )
    return records


def summarize_records(records: Sequence[Mapping[str, Any]]) -> Mapping[str, Any]:
    return development.summarize(list(records))


def metric_map(summary: Mapping[str, Any]) -> Mapping[str, float]:
    return {
        task: float(summary["by_task"][task][TASK_METRICS[task]]) for task in TASKS
    }


def weighted_metric(metrics: Mapping[str, float]) -> float:
    return sum(float(metrics[task]) for task in TASKS) / len(TASKS)


def promotion(
    systems: Mapping[str, Mapping[str, Any]],
    condition_summaries: Mapping[str, Mapping[str, Any]],
) -> Mapping[str, Any]:
    if set(systems) != set(SYSTEMS):
        return {"passed": False, "status": "all_systems_required"}
    metrics = {system: metric_map(summary) for system, summary in systems.items()}
    stronger = {
        task: max(
            metrics["frozen_gemma_base"][task],
            metrics["v9_projected_slot_baseline"][task],
        )
        for task in TASKS
    }
    candidate = metrics["direct_ple_candidate"]
    candidate_not_worse = all(candidate[task] >= stronger[task] for task in TASKS)
    strict_improvement = any(
        candidate[task] > metrics["frozen_gemma_base"][task]
        and candidate[task] > metrics["v9_projected_slot_baseline"][task]
        for task in TASKS
    )
    correct = metric_map(condition_summaries["correct_recurrent_state"])
    causal_task_nonworse = {
        condition: all(
            correct[task] >= metric_map(condition_summaries[condition])[task]
            for task in TASKS
        )
        for condition in CONDITIONS[1:]
    }
    causal_aggregate_strict = {
        condition: weighted_metric(correct)
        > weighted_metric(metric_map(condition_summaries[condition]))
        for condition in CONDITIONS[1:]
    }
    schema_non_regression = all(
        float(systems["direct_ple_candidate"]["by_task"][task]["strict_schema_valid"])
        / float(systems["direct_ple_candidate"]["by_task"][task]["rows"])
        >= max(
            float(systems["frozen_gemma_base"]["by_task"][task]["strict_schema_valid"])
            / float(systems["frozen_gemma_base"]["by_task"][task]["rows"]),
            float(systems["v9_projected_slot_baseline"]["by_task"][task]["strict_schema_valid"])
            / float(systems["v9_projected_slot_baseline"]["by_task"][task]["rows"]),
        )
        for task in TASKS
    )
    causal_pass = all(causal_task_nonworse.values()) and all(
        causal_aggregate_strict.values()
    )
    passed = bool(
        candidate_not_worse
        and strict_improvement
        and schema_non_regression
        and causal_pass
    )
    return {
        "metrics": metrics,
        "candidate_at_least_stronger_baseline_every_task": candidate_not_worse,
        "candidate_strictly_better_than_both_one_task": strict_improvement,
        "schema_non_regression": schema_non_regression,
        "correct_state_task_nonworse_vs_controls": causal_task_nonworse,
        "correct_state_aggregate_strict_vs_controls": causal_aggregate_strict,
        "causal_pass": causal_pass,
        "passed": passed,
    }


def load_system(
    system: str,
    *,
    base_model: Path,
    v9_adapter: Path,
    candidate_adapter: Path,
    candidate_result: Mapping[str, Any],
    device: str,
) -> tuple[torch.nn.Module, Any, Mapping[str, Any] | None]:
    adapter = None if system == "frozen_gemma_base" else v9_adapter
    if system == "direct_ple_candidate":
        adapter = candidate_adapter
    model, tokenizer = final.load_model(base_model, adapter, device)
    precision = None
    if system == "direct_ple_candidate":
        precision = preserve_candidate_precision(model, candidate_result)
        loaded = load_delta_mem_adapter(model, candidate_adapter)
        if loaded.rwkv_ms_hybrid_mode != "address_keyed_moe_ple":
            raise ValueError("Direct PLE adapter mode differs after load")
        if not all(
            module._ple_input_projection_hook_handle is not None
            and module.rwkv_ms_hybrid_mode == "address_keyed_moe_ple"
            for _, module in iter_delta_mem_modules(model)
        ):
            raise ValueError("Direct PLE adapter did not bind native Gemma PLE hooks")
    return model, tokenizer, precision


def main_run(args: argparse.Namespace) -> int:
    if os.environ.get("HF_ENDPOINT") != HF_MIRROR_ENDPOINT:
        raise ValueError(f"HF_ENDPOINT must be exactly {HF_MIRROR_ENDPOINT}")
    candidate_result = validate_training_result(args.training_result, args.candidate_adapter)
    baseline_commit, baseline_records = validate_baseline_commit(
        args.baseline_commit,
        base_model=args.base_model,
        v9_adapter=args.v9_adapter,
    )
    context = distributed.initialize_distributed_training(args.device, timeout_seconds=7200)
    if context is None or context.world_size != WORLD_SIZE:
        raise ValueError("Direct PLE development evaluation requires exactly four ranks")
    try:
        rows_by_task = development.read_v2_rows()
        flat_rows = [(task, row) for task in TASKS for row in rows_by_task[task]]
        source_by_task = {
            task: tuple(
                final.as_source_row(task, row)
                for row in rows_by_task[task]
                if int(row["prompt_variant"]) == 0
            )
            for task in TASKS
        }
        source_lookup = {
            task: {row.source_ordinal: row for row in source_by_task[task]}
            for task in TASKS
        }
        donor_lookup = {
            task: {
                ordinal: final.choose_control_donor(target, source_by_task[task]).source_ordinal
                for ordinal, target in source_lookup[task].items()
            }
            for task in TASKS
        }
        row_lookup = {
            task: {
                (int(row["source_ordinal"]), int(row["prompt_variant"])): row
                for row in rows_by_task[task]
            }
            for task in TASKS
        }
        output = args.output_dir.expanduser().resolve()
        creation_error = None
        if context.is_primary:
            try:
                output.mkdir(parents=True, exist_ok=False)
            except BaseException as error:
                creation_error = error
        distributed.phase_consensus(context, phase="direct-ple-development-output", error=creation_error)
        copy_error = None
        if context.is_primary:
            try:
                for system in SYSTEMS[:2]:
                    source = Path(str(baseline_commit["files"][system]["path"]))
                    destination = output / f"{system}.jsonl"
                    shutil.copyfile(source, destination)
                    if sha256_file(destination) != baseline_commit["files"][system]["sha256"]:
                        raise ValueError(f"Copied direct PLE baseline differs: {system}")
            except BaseException as error:
                copy_error = error
        distributed.phase_consensus(
            context,
            phase="direct-ple-development-baseline-copy",
            error=copy_error,
        )
        all_system_records: dict[str, list[Mapping[str, Any]]] = {
            system: list(records) for system, records in baseline_records.items()
        }
        candidate_controls: list[Mapping[str, Any]] = []
        candidate_precision = None
        for system in SYSTEMS[2:]:
            model, tokenizer, precision = load_system(
                system,
                base_model=args.base_model.expanduser().resolve(strict=True),
                v9_adapter=args.v9_adapter.expanduser().resolve(strict=True),
                candidate_adapter=args.candidate_adapter.expanduser().resolve(strict=True),
                candidate_result=candidate_result,
                device=str(context.device),
            )
            if precision is not None:
                candidate_precision = precision
            model.eval()
            row_path = output / f"shard-{context.process_rank}" / f"{system}.jsonl"
            local_records: list[Mapping[str, Any]] = []
            module_names = (
                tuple(name for name, _ in routed_common.ordered_modules(model))
                if system == "direct_ple_candidate"
                else ()
            )
            for flat_index, (task, row) in enumerate(flat_rows):
                if flat_index % WORLD_SIZE != context.process_rank:
                    continue
                if system != "direct_ple_candidate":
                    evaluated = dict(
                        final.evaluate_row(
                            model,
                            tokenizer,
                            task,
                            row,
                            str(context.device),
                            system,
                        )
                    )
                    evaluated["prompt_variant"] = int(row["prompt_variant"])
                    local_records.append(evaluated)
                else:
                    donor_ordinal = donor_lookup[task][int(row["source_ordinal"])]
                    donor = row_lookup[task][(donor_ordinal, int(row["prompt_variant"]))]
                    records = candidate_row_conditions(
                        model,
                        tokenizer,
                        row,
                        donor,
                        module_names=module_names,
                        device=str(context.device),
                    )
                    candidate_controls.extend(records)
                    local_records.append(
                        next(record for record in records if record["condition"] == "correct_recurrent_state")
                    )
                append_jsonl(row_path, local_records[-1])
            torch.distributed.barrier(group=context.control_group)
            if context.is_primary:
                records = []
                for rank in range(WORLD_SIZE):
                    records.extend(read_jsonl(output / f"shard-{rank}" / f"{system}.jsonl"))
                records.sort(key=lambda value: (TASKS.index(str(value["task"])), int(value["line_index"])))
                write_jsonl(output / f"{system}.jsonl", records)
                all_system_records[system] = records
            if system == "direct_ple_candidate":
                control_path = output / f"shard-{context.process_rank}" / "direct_ple_controls.jsonl"
                write_jsonl(control_path, candidate_controls)
                torch.distributed.barrier(group=context.control_group)
                if context.is_primary:
                    candidate_controls = []
                    for rank in range(WORLD_SIZE):
                        candidate_controls.extend(read_jsonl(output / f"shard-{rank}" / "direct_ple_controls.jsonl"))
                    candidate_controls.sort(
                        key=lambda value: (
                            TASKS.index(str(value["task"])),
                            int(value["line_index"]),
                            CONDITIONS.index(str(value["condition"])),
                        )
                    )
                    write_jsonl(output / "direct_ple_controls.jsonl", candidate_controls)
            del model, tokenizer
            gc.collect()
            if context.device.type == "cuda":
                torch.cuda.empty_cache()
            distributed.require_consensus(context, True, description=f"completed direct PLE system {system}")
        if context.is_primary:
            system_summaries = {
                system: summarize_records(records)
                for system, records in all_system_records.items()
            }
            condition_summaries = {
                condition: summarize_records(
                    [record for record in candidate_controls if record["condition"] == condition]
                )
                for condition in CONDITIONS
            }
            promotion_result = promotion(system_summaries, condition_summaries)
            input_binding = {
                "schema": INPUT_SCHEMA,
                "training_result": str(args.training_result.expanduser().resolve()),
                "training_result_sha256": sha256_file(args.training_result.expanduser().resolve()),
                "training_result_receipt": candidate_result["receipt"]["payload_sha256"],
                "candidate_adapter": str(args.candidate_adapter.expanduser().resolve()),
                "candidate_adapter_files": candidate_result["adapter_files"],
                "base_model": str(args.base_model.expanduser().resolve()),
                "base_model_revision": routed_common.BASE_MODEL_REVISION,
                "base_model_weights_sha256": routed_common.BASE_MODEL_WEIGHTS_SHA256,
                "v9_adapter": str(args.v9_adapter.expanduser().resolve()),
                "baseline_commit": baseline_commit_binding(
                    args.baseline_commit,
                    baseline_commit,
                ),
                "development_manifest_receipt": development.V2_MANIFEST_RECEIPT,
                "rows_per_task": {task: len(rows_by_task[task]) for task in TASKS},
                "prompt_variants_sha256": canonical_sha256(routed_common.PROMPT_VARIANTS),
                "systems": list(SYSTEMS),
                "conditions": list(CONDITIONS),
                "world_size": WORLD_SIZE,
                "rank_devices": list(context.rank_devices),
                "generation": "single-pass greedy write-then-read with captured state interventions",
                "projected_carrier_fixed": True,
                "task_router": False,
                "template_matcher": False,
                "dual_pass_selector": False,
                "benchmark_specific_decoder": False,
                "final_rows_opened": False,
                "publisher_validation_opened": False,
                "publisher_test_opened": False,
                "hf_endpoint": os.environ.get("HF_ENDPOINT"),
                "candidate_precision": candidate_precision,
                "runner_sha256": sha256_file(Path(__file__)),
            }
            result = {
                "schema": SCHEMA,
                "status": "development_generation_passed" if promotion_result["passed"] else "development_generation_failed",
                "passed": bool(promotion_result["passed"]),
                "input_binding": input_binding,
                "summary": {
                    "systems": system_summaries,
                    "conditions": condition_summaries,
                    "promotion": promotion_result,
                },
                "raw_prediction_files": {
                    system: {
                        "path": str((output / f"{system}.jsonl").resolve()),
                        "rows": len(records),
                        "sha256": sha256_file(output / f"{system}.jsonl"),
                    }
                    for system, records in all_system_records.items()
                },
                "control_file": {
                    "path": str((output / "direct_ple_controls.jsonl").resolve()),
                    "rows": len(candidate_controls),
                    "sha256": sha256_file(output / "direct_ple_controls.jsonl"),
                },
                "final_rows_opened": False,
                "publisher_validation_opened": False,
                "publisher_test_opened": False,
            }
            result["receipt"] = {
                "algorithm": "sha256",
                "payload_scope": "canonical_result_without_receipt",
                "payload_sha256": canonical_sha256(result),
            }
            write_json(output / "result.json", result)
            write_json(output / "summary.json", result["summary"])
        codes = distributed.gather_objects(context, 0 if not context.is_primary or promotion_result["passed"] else 1)
        return int(codes[0])
    finally:
        distributed.destroy_distributed_training(context)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training-result", type=Path, required=True)
    parser.add_argument("--candidate-adapter", type=Path, required=True)
    parser.add_argument("--base-model", type=Path, default=DEFAULT_BASE_MODEL)
    parser.add_argument("--v9-adapter", type=Path, default=DEFAULT_V9_ADAPTER)
    parser.add_argument("--baseline-commit", type=Path, default=DEFAULT_BASELINE_COMMIT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args(argv)


@record
def main(argv: Sequence[str] | None = None) -> int:
    return main_run(parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
