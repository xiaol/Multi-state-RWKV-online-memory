#!/usr/bin/env python3
"""Evaluate Gemma and RWKV-MS on Novel Agent author-compatible splits.

This is a transfer evaluation: the RWKV-MS adapter was trained on base-SFT
novel-writing data, not on the structured agent-task train splits.
"""

from __future__ import annotations

import argparse
import copy
import gc
import hashlib
import json
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
import subprocess
import sys
import time
from typing import Any, Iterable


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = Path(__file__).resolve().parents[2]
for import_root in (SCRIPT_DIR, PROJECT_ROOT):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

from common import (  # noqa: E402
    collect_rwkv_trace,
    load_model_and_tokenizer,
    reset_delta_state,
)
from deltamem.chat_templates import apply_chat_template  # noqa: E402
from deltamem.core.delta import iter_delta_mem_modules  # noqa: E402


NORMAL_FUSION_PROFILES = (
    "native",
    "native_gate_open",
    "gate_open_gamma_0",
    "gate_open_gamma_0p01",
    "post_attention_norm_gate_open_0p01",
)
RUNTIME_NORM_HOOK_PLACEMENTS = frozenset(
    {
        "normalized_residual_correction",
        "post_attention_norm",
        "post_attention_residual_hybrid",
    }
)


@dataclass(frozen=True)
class TaskSpec:
    name: str
    relative_dir: str
    kind: str
    max_new_tokens: int

    def relative_path(self, split: str) -> str:
        return f"{self.relative_dir}/{split}.jsonl"


TASK_SPECS = (
    TaskSpec(
        name="attribution-v3.2",
        relative_dir="v3.2-attribution-best-candidate",
        kind="attribution",
        max_new_tokens=1024,
    ),
    TaskSpec(
        name="narrative-v3.2",
        relative_dir="v3.2-narrative-type-classification",
        kind="narrative",
        max_new_tokens=1024,
    ),
    TaskSpec(
        name="scene-v3.2",
        relative_dir="v3.2-scene-boundary-detection",
        kind="scene",
        max_new_tokens=1024,
    ),
    TaskSpec(
        name="scene-v4-current",
        relative_dir="v4-scene-boundary-detection",
        kind="scene",
        max_new_tokens=128,
    ),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-model", required=True)
    parser.add_argument("--memory-dir", required=True)
    parser.add_argument("--delta-mem-root", default=str(PROJECT_ROOT))
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--reference-results-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--split", default="val", choices=("val", "test"))
    parser.add_argument("--conditions", default="base,normal")
    parser.add_argument(
        "--normal-fusion-profile",
        default="native",
        choices=NORMAL_FUSION_PROFILES,
        help="Runtime fusion profile applied after the normal adapter is loaded natively.",
    )
    parser.add_argument("--expected-memory-layer-count", type=int, default=42)
    parser.add_argument("--tasks", default=",".join(spec.name for spec in TASK_SPECS))
    parser.add_argument("--limit-per-task", type=int)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", default="bfloat16", choices=("bfloat16", "float16", "float32"))
    parser.add_argument("--attn-implementation", default="sdpa")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_text(value: str) -> str:
    return sha256_bytes(value.encode("utf-8"))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def normal_fusion_profile_definition(name: str) -> dict[str, Any]:
    definitions = {
        "native": {
            "name": "native",
            "description": "Checkpoint-native placement, content gate, and learned gains.",
        },
        "native_gate_open": {
            "name": "native_gate_open",
            "description": "Checkpoint-native placement/gains with fusion gate forced open.",
            "memory_fusion_mode": "add",
        },
        "gate_open_gamma_0": {
            "name": "gate_open_gamma_0",
            "description": "Gate open; hybrid direct residual gain fixed to zero.",
            "memory_fusion_mode": "add",
            "memory_fusion_placement": "post_attention_residual_hybrid",
            "memory_fusion_residual_scale": 0.0,
            "memory_fusion_residual_gain": 0.0,
        },
        "gate_open_gamma_0p01": {
            "name": "gate_open_gamma_0p01",
            "description": "Gate open; uniform hybrid direct residual gain 0.01.",
            "memory_fusion_mode": "add",
            "memory_fusion_placement": "post_attention_residual_hybrid",
            "memory_fusion_residual_scale": 0.01,
            "memory_fusion_residual_gain": 0.01,
        },
        "post_attention_norm_gate_open_0p01": {
            "name": "post_attention_norm_gate_open_0p01",
            "description": "Gate open; direct post-attention-norm memory scale 0.01.",
            "memory_fusion_mode": "add",
            "memory_fusion_placement": "post_attention_norm",
            "memory_fusion_residual_scale": 0.01,
        },
    }
    if name not in definitions:
        raise ValueError(
            f"Unknown normal fusion profile {name!r}; expected {NORMAL_FUSION_PROFILES}"
        )
    return dict(definitions[name])


def normal_fusion_fingerprint_fields(
    name: str,
    expected_layer_count: int,
) -> dict[str, Any]:
    if expected_layer_count <= 0:
        raise ValueError("Expected memory layer count must be positive")
    definition = normal_fusion_profile_definition(name)
    return {
        "normal_fusion_profile": name,
        "normal_fusion_profile_definition": definition,
        "expected_memory_layer_count": expected_layer_count,
        "profile_definition_sha256": sha256_text(
            json.dumps(definition, sort_keys=True, separators=(",", ":"))
        ),
    }


def apply_normal_fusion_profile(
    model,
    *,
    profile_name: str,
    expected_layer_count: int,
) -> dict[str, Any]:
    import torch

    definition = normal_fusion_profile_definition(profile_name)
    modules = sorted(
        list(iter_delta_mem_modules(model)),
        key=lambda item: int(item[1].layer_idx),
    )
    layer_indices = [int(module.layer_idx) for _, module in modules]
    if layer_indices != list(range(expected_layer_count)):
        raise RuntimeError(
            "Normal fusion profile requires the complete zero-based memory layer range: "
            f"expected={list(range(expected_layer_count))} actual={layer_indices}"
        )
    unsupported = {
        name: {
            "placement": module.memory_fusion_placement,
            "hook_bound": module._post_attention_norm_hook_handle is not None,
        }
        for name, module in modules
        if module.memory_fusion_placement not in RUNTIME_NORM_HOOK_PLACEMENTS
        or module._post_attention_norm_hook_handle is None
    }
    if unsupported:
        raise ValueError(
            "Normal fusion profile requires existing Gemma post-attention norm hooks: "
            f"{unsupported}"
        )

    for name, module in modules:
        if "memory_fusion_mode" in definition:
            module.memory_fusion_mode = str(definition["memory_fusion_mode"])
        if "memory_fusion_placement" in definition:
            module.memory_fusion_placement = str(
                definition["memory_fusion_placement"]
            )
        if "memory_fusion_residual_scale" in definition:
            module.memory_fusion_residual_scale = float(
                definition["memory_fusion_residual_scale"]
            )
        if "memory_fusion_residual_gain" in definition:
            if module.memory_fusion_placement != "post_attention_residual_hybrid":
                raise RuntimeError(
                    f"Fusion profile {profile_name} requests a hybrid gain for {name}"
                )
            if not hasattr(module, "memory_fusion_residual_gain_raw"):
                raise RuntimeError(f"Fusion profile {profile_name} has no gain at {name}")
            module.set_memory_fusion_residual_gain(
                float(definition["memory_fusion_residual_gain"])
            )

    settings = []
    for name, module in modules:
        raw = getattr(module, "memory_fusion_residual_gain_raw", None)
        effective_gain = None
        if module.memory_fusion_placement == "post_attention_residual_hybrid":
            if raw is None:
                raise RuntimeError(f"Hybrid fusion profile has no gain at {name}")
            effective_gain = float(
                module._resolved_memory_fusion_residual_gain(
                    device=raw.device,
                    dtype=torch.float32,
                )
                .detach()
                .item()
            )
        settings.append(
            {
                "module_name": name,
                "layer_index": int(module.layer_idx),
                "memory_fusion_mode": str(module.memory_fusion_mode),
                "memory_fusion_placement": str(module.memory_fusion_placement),
                "memory_fusion_residual_scale": float(
                    module.memory_fusion_residual_scale
                ),
                "memory_fusion_residual_scale_max": float(
                    module.memory_fusion_residual_scale_max
                ),
                "memory_fusion_residual_gain_raw": (
                    None if raw is None else float(raw.detach().float().item())
                ),
                "memory_fusion_residual_gain_effective": effective_gain,
                "post_attention_norm_hook_bound": (
                    module._post_attention_norm_hook_handle is not None
                ),
            }
        )
    settings_sha256 = sha256_text(
        json.dumps(settings, sort_keys=True, separators=(",", ":"))
    )
    return {
        "profile": profile_name,
        "definition": definition,
        "layer_count": len(settings),
        "layer_indices": layer_indices,
        "effective_settings_sha256": settings_sha256,
        "effective_settings": settings,
    }


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json_atomic(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def git_revision(path: Path) -> dict[str, Any]:
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=path, text=True, stderr=subprocess.DEVNULL
        ).strip()
        dirty = bool(
            subprocess.check_output(
                ["git", "status", "--porcelain"],
                cwd=path,
                text=True,
                stderr=subprocess.DEVNULL,
            ).strip()
        )
        return {"commit": commit, "dirty": dirty}
    except (OSError, subprocess.CalledProcessError):
        return {"commit": None, "dirty": None}


def extract_json(text: str) -> Any | None:
    """Extract the first complete JSON object or array from model text."""

    stripped = text.strip()
    try:
        return json.loads(stripped)
    except json.JSONDecodeError:
        pass
    decoder = json.JSONDecoder()
    for index, character in enumerate(stripped):
        if character not in "[{":
            continue
        try:
            value, _ = decoder.raw_decode(stripped[index:])
        except json.JSONDecodeError:
            continue
        if isinstance(value, (dict, list)):
            return value
    return None


def normalized_label_rows(value: Any) -> list[dict[str, Any]] | None:
    if isinstance(value, dict):
        value = value.get("labels")
    if not isinstance(value, list):
        return None
    return [item for item in value if isinstance(item, dict)]


def normalized_boundaries(value: Any) -> set[int] | None:
    if not isinstance(value, dict) or not isinstance(value.get("boundaries"), list):
        return None
    boundaries: set[int] = set()
    for item in value["boundaries"]:
        if isinstance(item, bool):
            continue
        if isinstance(item, int):
            boundaries.add(item)
        elif isinstance(item, str) and item.strip().isdigit():
            boundaries.add(int(item.strip()))
    return boundaries


def score_prediction(kind: str, prediction: Any, gold: Any) -> dict[str, Any]:
    json_extracted = prediction is not None
    official_json_ok = isinstance(prediction, dict) and bool(prediction)
    if kind == "attribution":
        schema_valid = isinstance(prediction, dict) and "best_candidate" in prediction
        correct = bool(
            schema_valid
            and isinstance(gold, dict)
            and prediction.get("best_candidate") == gold.get("best_candidate")
        )
        return {
            "json_extracted": json_extracted,
            "official_json_ok": official_json_ok,
            "schema_valid": schema_valid,
            "correct": correct,
        }

    if kind == "narrative":
        gold_rows = normalized_label_rows(gold) or []
        prediction_rows = normalized_label_rows(prediction)
        schema_valid = prediction_rows is not None
        predicted = {
            str(item.get("unit_id")): item.get("type")
            for item in (prediction_rows or [])
            if item.get("unit_id") is not None
        }
        correct_units = sum(
            1
            for item in gold_rows
            if predicted.get(str(item.get("unit_id"))) == item.get("type")
        )
        return {
            "json_extracted": json_extracted,
            "official_json_ok": official_json_ok,
            "schema_valid": schema_valid,
            "gold_units": len(gold_rows),
            "predicted_units": len(predicted),
            "correct_units": correct_units,
        }

    if kind == "scene":
        gold_boundaries = normalized_boundaries(gold) or set()
        prediction_boundaries = normalized_boundaries(prediction)
        schema_valid = prediction_boundaries is not None
        predicted = prediction_boundaries or set()
        tp = len(gold_boundaries & predicted)
        fp = len(predicted - gold_boundaries)
        fn = len(gold_boundaries - predicted)
        denominator = 2 * tp + fp + fn
        return {
            "json_extracted": json_extracted,
            "official_json_ok": official_json_ok,
            "schema_valid": schema_valid,
            "gold_boundaries": sorted(gold_boundaries),
            "predicted_boundaries": sorted(predicted),
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "sample_f1": 0.0 if denominator == 0 else 2 * tp / denominator,
        }

    raise ValueError(f"Unsupported task kind: {kind}")


def load_task_rows(
    spec: TaskSpec,
    dataset_root: Path,
    split: str,
    limit_per_task: int | None,
) -> tuple[Path, list[dict[str, Any]]]:
    path = dataset_root / spec.relative_path(split)
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            if not raw_line.strip():
                continue
            row = json.loads(raw_line)
            messages = row.get("messages")
            if not isinstance(messages, list) or len(messages) < 3:
                raise ValueError(f"Invalid messages at {path}:{line_number}")
            gold = extract_json(str(messages[-1].get("content", "")))
            if gold is None:
                raise ValueError(f"Invalid gold JSON at {path}:{line_number}")
            rows.append(
                {
                    "line_index": len(rows),
                    "messages": messages[:-1],
                    "gold": gold,
                    "row_sha256": sha256_text(raw_line.rstrip("\n")),
                }
            )
            if limit_per_task is not None and len(rows) >= limit_per_task:
                break
    return path, rows


def ratio(numerator: int | float, denominator: int | float) -> float:
    return 0.0 if denominator == 0 else numerator / denominator


def f1_from_counts(tp: int, fp: int, fn: int) -> float:
    return ratio(2 * tp, 2 * tp + fp + fn)


def summarize_task(spec: TaskSpec, records: Iterable[dict[str, Any]]) -> dict[str, Any]:
    rows = list(records)
    extracted = sum(bool(row["score"]["json_extracted"]) for row in rows)
    official_json_ok = sum(bool(row["score"]["official_json_ok"]) for row in rows)
    schema_valid = sum(bool(row["score"]["schema_valid"]) for row in rows)
    common: dict[str, Any] = {
        "samples": len(rows),
        "json_extracted": extracted,
        "json_extracted_rate": ratio(extracted, len(rows)),
        "official_json_ok": official_json_ok,
        "official_json_rate": ratio(official_json_ok, len(rows)),
        "schema_valid": schema_valid,
        "schema_valid_rate": ratio(schema_valid, len(rows)),
        "hit_max_new_tokens": sum(bool(row["hit_max_new_tokens"]) for row in rows),
        "input_tokens": sum(int(row["input_tokens"]) for row in rows),
        "output_tokens": sum(int(row["output_tokens"]) for row in rows),
        "elapsed_seconds": sum(float(row["elapsed_seconds"]) for row in rows),
    }
    if spec.kind == "attribution":
        correct = sum(bool(row["score"]["correct"]) for row in rows)
        common.update(
            {
                "correct": correct,
                "strict_accuracy": ratio(correct, len(rows)),
                "official_parsed_only_accuracy": ratio(correct, official_json_ok),
                "primary_metric": ratio(correct, len(rows)),
                "primary_metric_name": "strict_accuracy",
            }
        )
        return common
    if spec.kind == "narrative":
        correct_units = sum(int(row["score"]["correct_units"]) for row in rows)
        gold_units = sum(int(row["score"]["gold_units"]) for row in rows)
        parsed_gold_units = sum(
            int(row["score"]["gold_units"])
            for row in rows
            if row["score"]["official_json_ok"]
        )
        common.update(
            {
                "correct_units": correct_units,
                "gold_units": gold_units,
                "strict_accuracy": ratio(correct_units, gold_units),
                "official_parsed_only_accuracy": ratio(correct_units, parsed_gold_units),
                "primary_metric": ratio(correct_units, gold_units),
                "primary_metric_name": "strict_accuracy",
            }
        )
        return common
    tp = sum(int(row["score"]["tp"]) for row in rows)
    fp = sum(int(row["score"]["fp"]) for row in rows)
    fn = sum(int(row["score"]["fn"]) for row in rows)
    valid_rows = [row for row in rows if row["score"]["schema_valid"]]
    valid_tp = sum(int(row["score"]["tp"]) for row in valid_rows)
    valid_fp = sum(int(row["score"]["fp"]) for row in valid_rows)
    valid_fn = sum(int(row["score"]["fn"]) for row in valid_rows)
    common.update(
        {
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "strict_precision": ratio(tp, tp + fp),
            "strict_recall": ratio(tp, tp + fn),
            "strict_f1": f1_from_counts(tp, fp, fn),
            "macro_sample_f1": ratio(
                sum(float(row["score"]["sample_f1"]) for row in rows), len(rows)
            ),
            "schema_valid_only_f1": f1_from_counts(valid_tp, valid_fp, valid_fn),
            "primary_metric": f1_from_counts(tp, fp, fn),
            "primary_metric_name": "strict_f1",
        }
    )
    return common


def reference_metrics(reference_dir: Path, evaluation_split: str) -> dict[str, dict[str, Any]]:
    compare_path = reference_dir / "eval_compare_results.json"
    scene_path = reference_dir / "scene_boundary_final.json"
    compare = read_json(compare_path)
    stage32 = compare["+Stage2 v3.2"]
    scene_results = read_json(scene_path)["v4-590"]["per_sample"]
    scene_tp = sum(int(row["tp"]) for row in scene_results)
    scene_fp = sum(int(row["fp"]) for row in scene_results)
    scene_fn = sum(int(row["fn"]) for row in scene_results)
    split_caveat = (
        " The artifact reports the test split while this run evaluates validation."
        if evaluation_split == "val"
        else ""
    )
    common = {
        "reference_model": "Qwen3-8B Novel Base SFT plus task-specific LoRA",
        "comparison_caveat": (
            "Different base model and task-specific training; not an apples-to-apples model delta."
            f"{split_caveat}"
        ),
    }
    return {
        "attribution-v3.2": {
            **common,
            "metric_name": "best_accuracy",
            "artifact_metric": float(
                stage32["attribution-best-candidate"]["best_accuracy"]
            ),
            "artifact_source": str(compare_path),
        },
        "narrative-v3.2": {
            **common,
            "metric_name": "accuracy",
            "artifact_metric": float(
                stage32["narrative-type-classification"]["accuracy"]
            ),
            "artifact_source": str(compare_path),
        },
        "scene-v3.2": {
            **common,
            "metric_name": "f1",
            "artifact_metric": float(stage32["scene-boundary-detection"]["f1"]),
            "artifact_source": str(compare_path),
        },
        "scene-v4-current": {
            **common,
            "metric_name": "micro_f1",
            "artifact_metric": f1_from_counts(scene_tp, scene_fp, scene_fn),
            "artifact_counts": {"samples": len(scene_results), "tp": scene_tp, "fp": scene_fp, "fn": scene_fn},
            "artifact_source": str(scene_path),
            "published_historical_metric": 0.305,
            "published_historical_note": (
                "The reported 30.5% used a historical 49-row v4 test. The downloaded "
                "current v4 test and raw result artifact contain 149 rows, so 30.5% is context only."
            ),
        },
        "source_hashes": {
            "eval_compare_results.json": sha256_file(compare_path),
            "scene_boundary_final.json": sha256_file(scene_path),
        },
    }


def validate_metrics(task_data: dict[str, tuple[Path, list[dict[str, Any]]]]) -> None:
    for spec in TASK_SPECS:
        rows = task_data.get(spec.name, (None, []))[1]
        if not rows:
            continue
        perfect = score_prediction(spec.kind, rows[0]["gold"], rows[0]["gold"])
        if not perfect["schema_valid"]:
            raise AssertionError(f"Gold self-parse failed for {spec.name}")
        if spec.kind == "attribution" and not perfect["correct"]:
            raise AssertionError(f"Gold self-score failed for {spec.name}")
        if spec.kind == "narrative" and perfect["correct_units"] != perfect["gold_units"]:
            raise AssertionError(f"Gold self-score failed for {spec.name}")
        if spec.kind == "scene" and (perfect["fp"] != 0 or perfect["fn"] != 0):
            raise AssertionError(f"Gold self-score failed for {spec.name}")


def model_generation_config(model, tokenizer, max_new_tokens: int):
    generation_config = copy.deepcopy(model.generation_config)
    generation_config.do_sample = False
    generation_config.max_new_tokens = max_new_tokens
    generation_config.use_cache = True
    generation_config.temperature = None
    generation_config.top_p = None
    generation_config.top_k = None
    if tokenizer.pad_token_id is not None:
        generation_config.pad_token_id = tokenizer.pad_token_id
    return generation_config


def generate_one(
    *,
    model,
    tokenizer,
    messages: list[dict[str, str]],
    max_new_tokens: int,
    device: str,
) -> dict[str, Any]:
    import torch

    reset_delta_state(model)
    try:
        rendered = apply_chat_template(
            tokenizer,
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        encoded = tokenizer(
            rendered,
            return_tensors="pt",
            add_special_tokens=False,
        )
        input_ids = encoded.input_ids.to(device)
        attention_mask = encoded.attention_mask.to(device)
        if device.startswith("cuda"):
            torch.cuda.reset_peak_memory_stats(device)
        started_at = time.perf_counter()
        with torch.inference_mode():
            output_ids = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                generation_config=model_generation_config(model, tokenizer, max_new_tokens),
            )
        elapsed = time.perf_counter() - started_at
        generated_ids = output_ids[:, input_ids.size(1) :]
        response = tokenizer.decode(generated_ids[0], skip_special_tokens=True).strip()
        peak_memory = None
        if device.startswith("cuda"):
            peak_memory = int(torch.cuda.max_memory_allocated(device))
        return {
            "status": "ok",
            "raw_generation": response,
            "parsed_json": extract_json(response),
            "input_tokens": int(input_ids.size(1)),
            "output_tokens": int(generated_ids.size(1)),
            "hit_max_new_tokens": int(generated_ids.size(1)) >= max_new_tokens,
            "elapsed_seconds": elapsed,
            "peak_cuda_memory_bytes": peak_memory,
            "memory_trace": collect_rwkv_trace(model),
        }
    finally:
        reset_delta_state(model)


def read_records(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    rows: list[dict[str, Any]] = []
    raw_lines = path.read_text(encoding="utf-8").splitlines()
    for line_index, raw_line in enumerate(raw_lines):
        if not raw_line.strip():
            continue
        try:
            rows.append(json.loads(raw_line))
        except json.JSONDecodeError:
            if line_index == len(raw_lines) - 1:
                break
            raise
    return rows


def append_record(path: Path, record: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def record_key(task_name: str, line_index: int) -> str:
    return f"{task_name}:{line_index}"


def clear_model_memory() -> None:
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass


def load_normal_model(args: argparse.Namespace):
    model, tokenizer = load_model_and_tokenizer(
        base_model=args.base_model,
        device=args.device,
        dtype=args.dtype,
        attn_implementation=args.attn_implementation,
        delta_mem_root=args.delta_mem_root,
        memory_dir=args.memory_dir,
        memory_repo=None,
    )
    runtime_profile = apply_normal_fusion_profile(
        model,
        profile_name=args.normal_fusion_profile,
        expected_layer_count=args.expected_memory_layer_count,
    )
    return model, tokenizer, runtime_profile


def selected_specs(raw: str) -> list[TaskSpec]:
    requested = [item.strip() for item in raw.split(",") if item.strip()]
    by_name = {spec.name: spec for spec in TASK_SPECS}
    unknown = [name for name in requested if name not in by_name]
    if unknown:
        raise ValueError(f"Unknown tasks: {', '.join(unknown)}")
    return [by_name[name] for name in requested]


def selected_conditions(raw: str) -> list[str]:
    conditions = [item.strip() for item in raw.split(",") if item.strip()]
    unknown = [name for name in conditions if name not in {"base", "normal"}]
    if unknown:
        raise ValueError(f"Unknown conditions: {', '.join(unknown)}")
    if not conditions:
        raise ValueError("At least one condition is required")
    return conditions


def resolve_training_root(dataset_root: Path, specs: list[TaskSpec], split: str) -> Path:
    candidates = (dataset_root, dataset_root / "training")
    for candidate in candidates:
        if all((candidate / spec.relative_path(split)).is_file() for spec in specs):
            return candidate.resolve()
    expected = ", ".join(spec.relative_path(split) for spec in specs)
    raise FileNotFoundError(
        f"Could not find all selected task files below {dataset_root}; expected {expected}"
    )


def memory_training_metadata(memory_dir: Path) -> dict[str, Any]:
    protocol_path = memory_dir / "training_protocol.json"
    lineage_path = memory_dir / "ablation_lineage_manifest.json"
    metadata: dict[str, Any] = {"memory_dir": str(memory_dir.resolve())}
    if protocol_path.is_file():
        protocol = read_json(protocol_path)
        metadata["training_protocol"] = {
            "path": str(protocol_path.resolve()),
            "sha256": sha256_file(protocol_path),
            "schema_version": protocol.get("schema_version"),
            "objective": protocol.get("memory_objective_version"),
            "loss_mode": protocol.get("memory_loss_mode"),
            "train_file": protocol.get("train_file"),
            "train_samples": protocol.get("train_samples"),
            "max_steps": protocol.get("max_steps"),
        }
    else:
        metadata["training_protocol"] = None
    if lineage_path.is_file():
        metadata["lineage"] = {
            "path": str(lineage_path.resolve()),
            "sha256": sha256_file(lineage_path),
        }
    return metadata


def main() -> None:
    args = parse_args()
    specs = selected_specs(args.tasks)
    conditions = selected_conditions(args.conditions)
    if args.limit_per_task is not None and args.limit_per_task <= 0:
        raise ValueError("--limit-per-task must be positive")
    fusion_fingerprint_fields = normal_fusion_fingerprint_fields(
        args.normal_fusion_profile,
        args.expected_memory_layer_count,
    )

    training_root = resolve_training_root(args.dataset_root, specs, args.split)
    task_data = {
        spec.name: load_task_rows(spec, training_root, args.split, args.limit_per_task)
        for spec in specs
    }
    validate_metrics(task_data)
    references = reference_metrics(args.reference_results_dir, args.split)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.overwrite:
        for path in [
            *(args.output_dir / f"{condition}.jsonl" for condition in conditions),
            args.output_dir / "manifest.json",
            args.output_dir / "progress.json",
            args.output_dir / "summary.json",
        ]:
            path.unlink(missing_ok=True)

    base_config_path = Path(args.base_model) / "config.json"
    memory_dir = Path(args.memory_dir)
    fingerprint_payload = {
        "evaluator_sha256": sha256_file(Path(__file__)),
        "base_model": str(Path(args.base_model).resolve()),
        "base_config_sha256": sha256_file(base_config_path),
        "memory_dir": str(memory_dir.resolve()),
        "memory_adapter_sha256": sha256_file(memory_dir / "delta_mem_adapter.pt"),
        "memory_config_sha256": sha256_file(memory_dir / "delta_mem_config.json"),
        "split": args.split,
        "datasets": {
            spec.name: {
                "path": str(task_data[spec.name][0].resolve()),
                "sha256": sha256_file(task_data[spec.name][0]),
                "selected_rows": len(task_data[spec.name][1]),
                "max_new_tokens": spec.max_new_tokens,
            }
            for spec in specs
        },
        "conditions": conditions,
        "device": args.device,
        "dtype": args.dtype,
        "attn_implementation": args.attn_implementation,
        **fusion_fingerprint_fields,
    }
    fingerprint = sha256_text(json.dumps(fingerprint_payload, sort_keys=True))
    manifest_path = args.output_dir / "manifest.json"
    if manifest_path.is_file():
        previous = read_json(manifest_path)
        if previous.get("fingerprint") != fingerprint:
            raise RuntimeError(
                f"Output manifest fingerprint differs at {manifest_path}; use --overwrite or a new output directory"
            )
    else:
        write_json_atomic(
            manifest_path,
            {
                "created_at": utc_now(),
                "fingerprint": fingerprint,
                "fingerprint_payload": fingerprint_payload,
                "evaluation_kind": "structured-task transfer evaluation",
                "split": args.split,
                "memory_training": memory_training_metadata(memory_dir),
                "training_scope": "Checkpoint-specific training provenance is recorded in memory_training.",
                "overlap_audit": {
                    "status": "not_performed_for_this_checkpoint",
                    "interpretation": "Do not infer passage-level independence from the task split alone.",
                },
                "generation_protocol": (
                    "Greedy decoding with author-compatible per-task token caps, Gemma chat template, "
                    "batch size 1, RWKV state reset before every example, "
                    f"split={args.split}, normal_fusion_profile="
                    f"{args.normal_fusion_profile}."
                ),
                "code": {
                    "rwkv_repo": git_revision(PROJECT_ROOT),
                    "novel_agent_repo": git_revision(args.reference_results_dir.parents[1]),
                },
                "references": references,
            },
        )

    expected_total = sum(len(task_data[spec.name][1]) for spec in specs) * len(conditions)
    completed_total = 0
    for condition in conditions:
        records_path = args.output_dir / f"{condition}.jsonl"
        existing_rows = read_records(records_path)
        existing: dict[str, dict[str, Any]] = {}
        for row in existing_rows:
            if row.get("condition") != condition:
                raise ValueError(f"Unexpected condition in {records_path}: {row.get('condition')}")
            if row.get("status") == "ok":
                existing[str(row["key"])] = row
        completed_total += len(existing)

        pending = []
        for spec in specs:
            for sample in task_data[spec.name][1]:
                key = record_key(spec.name, int(sample["line_index"]))
                prior = existing.get(key)
                if prior is not None:
                    if prior.get("row_sha256") != sample["row_sha256"]:
                        raise ValueError(f"Dataset row changed for resumed key {key}")
                    continue
                pending.append((spec, sample))
        if not pending:
            print(f"EVAL_CONDITION_COMPLETE condition={condition} resumed=true", flush=True)
            continue

        print(f"EVAL_LOAD condition={condition} pending={len(pending)}", flush=True)
        normal_fusion_runtime = None
        if condition == "base":
            model, tokenizer = load_model_and_tokenizer(
                base_model=args.base_model,
                device=args.device,
                dtype=args.dtype,
                attn_implementation=args.attn_implementation,
            )
        else:
            model, tokenizer, normal_fusion_runtime = load_normal_model(args)
            manifest = read_json(manifest_path)
            recorded_runtime = manifest.get("normal_fusion_runtime")
            if (
                recorded_runtime is not None
                and recorded_runtime != normal_fusion_runtime
            ):
                raise RuntimeError(
                    "Normal fusion runtime differs from the existing output manifest"
                )
            if recorded_runtime is None:
                manifest["normal_fusion_runtime"] = normal_fusion_runtime
                write_json_atomic(manifest_path, manifest)

        for spec, sample in pending:
            result = generate_one(
                model=model,
                tokenizer=tokenizer,
                messages=sample["messages"],
                max_new_tokens=spec.max_new_tokens,
                device=args.device,
            )
            score = score_prediction(spec.kind, result["parsed_json"], sample["gold"])
            key = record_key(spec.name, int(sample["line_index"]))
            record = {
                "key": key,
                "condition": condition,
                "normal_fusion_profile": (
                    None if condition == "base" else args.normal_fusion_profile
                ),
                "task": spec.name,
                "task_kind": spec.kind,
                "split": args.split,
                "line_index": sample["line_index"],
                "row_sha256": sample["row_sha256"],
                "gold": sample["gold"],
                "max_new_tokens": spec.max_new_tokens,
                "score": score,
                "completed_at": utc_now(),
                **result,
            }
            append_record(records_path, record)
            existing[key] = record
            completed_total += 1
            write_json_atomic(
                args.output_dir / "progress.json",
                {
                    "updated_at": utc_now(),
                    "fingerprint": fingerprint,
                    "completed": completed_total,
                    "expected": expected_total,
                    "last_key": f"{condition}:{key}",
                },
            )
            print(
                "EVAL_PROGRESS "
                f"condition={condition} task={spec.name} sample={int(sample['line_index']) + 1}/"
                f"{len(task_data[spec.name][1])} schema_valid={score['schema_valid']} "
                f"input_tokens={result['input_tokens']} output_tokens={result['output_tokens']} "
                f"seconds={result['elapsed_seconds']:.2f} completed={completed_total}/{expected_total}",
                flush=True,
            )

        del model, tokenizer
        clear_model_memory()
        print(f"EVAL_CONDITION_COMPLETE condition={condition} resumed=false", flush=True)

    all_records = {
        condition: read_records(args.output_dir / f"{condition}.jsonl")
        for condition in conditions
    }
    summaries: dict[str, dict[str, Any]] = {}
    for condition, records in all_records.items():
        summaries[condition] = {}
        for spec in specs:
            task_records = sorted(
                (row for row in records if row["task"] == spec.name),
                key=lambda row: int(row["line_index"]),
            )
            task_summary = summarize_task(spec, task_records)
            reference = references[spec.name]
            task_summary["reference"] = reference
            if args.split == "test":
                task_summary["delta_vs_reference_artifact"] = (
                    task_summary["primary_metric"] - float(reference["artifact_metric"])
                )
            summaries[condition][spec.name] = task_summary

    comparisons: dict[str, Any] = {}
    if "base" in summaries and "normal" in summaries:
        for spec in specs:
            base_metric = float(summaries["base"][spec.name]["primary_metric"])
            normal_metric = float(summaries["normal"][spec.name]["primary_metric"])
            comparisons[spec.name] = {
                "metric_name": summaries["normal"][spec.name]["primary_metric_name"],
                "base": base_metric,
                "normal": normal_metric,
                "normal_minus_base": normal_metric - base_metric,
            }

    summary = {
        "completed_at": utc_now(),
        "fingerprint": fingerprint,
        "split": args.split,
        "complete": all(
            len(all_records[condition])
            == sum(len(task_data[spec.name][1]) for spec in specs)
            for condition in conditions
        ),
        "normal_fusion_profile": args.normal_fusion_profile,
        "normal_fusion_runtime": read_json(manifest_path).get(
            "normal_fusion_runtime"
        ),
        "conditions": summaries,
        "normal_vs_base": comparisons,
        "references": references,
    }
    write_json_atomic(args.output_dir / "summary.json", summary)
    write_json_atomic(
        args.output_dir / "progress.json",
        {
            "updated_at": utc_now(),
            "fingerprint": fingerprint,
            "completed": sum(len(rows) for rows in all_records.values()),
            "expected": expected_total,
            "state": "complete" if summary["complete"] else "incomplete",
        },
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
