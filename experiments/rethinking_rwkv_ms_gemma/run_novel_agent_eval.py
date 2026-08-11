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
from importlib import metadata as importlib_metadata
import json
import math
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
import re
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
    memory_condition,
    reset_delta_state,
    set_delta_write_enabled,
)
from deltamem.chat_templates import apply_chat_template  # noqa: E402
from deltamem.scene_boundary import (  # noqa: E402
    extract_json as _shared_extract_json,
    literal_boundaries as _shared_literal_boundaries,
)
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
EVALUATION_CONTRACTS = (
    "generic",
    "scene_v6_validation",
    "scene_v6_final_test",
)
ONLINE_MEMORY_PROTOCOLS = (
    "legacy_write_only",
    "write_then_read",
)
DATASET_SPLITS = ("val", "test", "train_derived_development")
SCENE_V6_CONTRACT_ROWS = {
    "scene_v6_validation": ("val", 170),
    "scene_v6_final_test": ("test", 149),
}
OFFICIAL_SCENE_V4_DATASET_REVISION = "5d3040d21f51b3ce90b9396b058e552c47f43cd5"
OFFICIAL_SCENE_V4_SHA256 = {
    "val": "61e94bcc536a124b07aef2c38ba285d7073d94a223866b58ddc7e5e1f509d513",
    "test": "d8b50ca3862bd40f023155bd14aa7b25d9d5dd3db4ea1c4d5a7e6f4f79cdfd6d",
}
SCENE_V6_TARGET_LAYERS = tuple(range(42))
SCENE_V6_DELTA_HEADS = ("q", "o")
SCENE_V6_TRAIN_SHA256 = (
    "785fe54c0a4e5c64e33f64f9bc88d64719576407c21eb0d520f9dec5a59b8e22"
)
SCENE_V6_TRAINING_PROTOCOL_FILE_SHA256 = (
    "b13968ee8ca21f1bc73afed7f48b2845f6cf1a62dc5d94e537638483f0051966"
)
SCENE_V6_TRAINING_PROTOCOL_CANONICAL_SHA256 = (
    "5f43306d3849d1097ffff3f96db0b257c83db300444357e845b156f928cdce29"
)
SCENE_V6_IDENTITY_CHECKPOINT_RECEIPT_SCHEMA = (
    "rwkv_ms_scene_v6_identity_checkpoint.v1"
)
SCENE_V6_IDENTITY_OBJECTIVE_VERSION = "scene_state_identity_ce_v2"
SCENE_V6_IDENTITY_OBJECTIVE_EXPECTED = {
    "memory_loss_mode": "scene_state_identity_ce",
    "objective_version": SCENE_V6_IDENTITY_OBJECTIVE_VERSION,
    "margin": 0.5,
    "margin_mode": "per_row_hinge_relu_v1",
    "objective_formula": (
        "full_correct_ce + correct_all_semantic_ce + "
        "mean(relu(margin - (donor_pair_semantic_ce - "
        "correct_pair_semantic_ce)))"
    ),
    "backward_mode": (
        "sequential_replayed_donor_single_zero_diagnostic_exact_first_order_v2"
    ),
    "read_protocol": "state_only_same_read_correct_donor_zero_adapter_active_v1",
    "zero_protocol": "adapter_active_reset_state_writes_disabled_v1",
    "semantic_mask_mode": "top_level_boundaries_nonwhitespace_offset_overlap_v1",
    "semantic_loss_normalization": "selected_tokens_per_row_then_batch_mean_v1",
    "target_mode": "first_pair_distinguishing_semantic_token_v1",
    "causal_prefix_mode": "exact_input_ids_and_attention_before_pair_target_v1",
    "full_correct_ce_weight": 1.0,
    "correct_all_semantic_ce_weight": 1.0,
    "donor_margin_weight": 1.0,
    "zero_diagnostic_weight": 0.0,
    "zero_diagnostic_gradient": False,
    "read_time_positions_observable": False,
    "correct_all_semantic_scope": "all_semantic_tokens_v1",
    "pair_semantic_scope": "first_pair_distinguishing_semantic_token_v1",
    "donor_margin_scope": "first_pair_distinguishing_semantic_token_v1",
    "zero_diagnostic_scope": "all_semantic_tokens_v1",
    "target_strata": [
        "presence",
        "same_cardinality_value",
        "cross_cardinality_value",
    ],
    "pairing_version": "nearest_write_token_length_label_distinct_symmetric_pair_v2",
    "pairing_refinement": (
        "maximize_nonempty_same_cardinality_within_nearest_length_budget_v1"
    ),
    "pairing_length_control": (
        "nearest_feasible_symmetric_absolute_write_token_delta_v1"
    ),
    "pairing_limitation": (
        "nonzero_write_token_length_deltas_retained_without_truncation_"
        "or_hybrid_carriers_v1"
    ),
    "auxiliary_regularizers": "all_zero",
}
SCENE_V6_IDENTITY_PAIR_MANIFEST_SHA256 = (
    "2ceb291b9c21063164e30ca0b8b052798f8ba42d9a089a5abc78d1cb321dc008"
)
SCENE_V6_IDENTITY_TRAIN_SHA256 = (
    "5f35f6ed41a2edaf88afee83626f17c34da38f5cb61cf4b6796a03eaae38f897"
)
SCENE_V6_IDENTITY_HARD32_SELECTION_SHA256 = (
    "76d510e5f02c30f2cf3a0262cc4c97d69ef8f52861e9ced855d530b233d916db"
)
SCENE_V6_IDENTITY_HARD32_HOLDOUT_SHA256 = (
    "b5b1137de89f82eee4b3ae3e3c7b5305240699ec7b65e84b61cb415a7a000d4a"
)
SCENE_V6_IDENTITY_HARD32_RECEIPT_SCHEMA = "scene_v6_identity_hard32_receipt.v2"


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
    parser.add_argument("--split", default="val", choices=DATASET_SPLITS)
    parser.add_argument("--conditions", default="base,normal")
    parser.add_argument(
        "--online-memory-protocol",
        choices=ONLINE_MEMORY_PROTOCOLS,
        default="legacy_write_only",
        help=(
            "Projected-KV runtime protocol for the normal condition. "
            "write_then_read primes state from the prompt, disables writes, then "
            "decodes from a read-only replay of the same prompt."
        ),
    )
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
    parser.add_argument(
        "--evaluation-contract",
        choices=EVALUATION_CONTRACTS,
        default="generic",
    )
    parser.add_argument(
        "--hard32-receipt",
        type=Path,
        help="Passed hard32 identity receipt required before scene_v6_validation.",
    )
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


def fingerprint_payload_sha256(payload: Any) -> str:
    if not isinstance(payload, dict):
        raise ValueError("Manifest fingerprint_payload must be an object")
    return sha256_text(json.dumps(payload, sort_keys=True))


def artifact_identity(root: Path, relative_names: Iterable[str]) -> dict[str, Any]:
    root = root.expanduser().resolve()
    files: list[dict[str, Any]] = []
    for relative_name in sorted(set(relative_names)):
        relative_path = Path(relative_name)
        if relative_path.is_absolute() or ".." in relative_path.parts:
            raise ValueError(f"Unsafe model artifact path: {relative_name!r}")
        path = root / relative_path
        if not path.is_file():
            raise FileNotFoundError(f"Missing model artifact referenced by config: {path}")
        files.append(
            {
                "relative_path": relative_path.as_posix(),
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    if not files:
        raise ValueError(f"No model artifacts selected under {root}")
    return {
        "files": files,
        "combined_sha256": sha256_text(
            json.dumps(files, sort_keys=True, separators=(",", ":"))
        ),
    }


def base_model_weight_identity(base_model: Path) -> dict[str, Any]:
    root = base_model.expanduser().resolve()
    for index_name in (
        "model.safetensors.index.json",
        "pytorch_model.bin.index.json",
    ):
        index_path = root / index_name
        if not index_path.is_file():
            continue
        index = read_json(index_path)
        weight_map = index.get("weight_map")
        if not isinstance(weight_map, dict) or not weight_map:
            raise ValueError(f"Invalid model weight map: {index_path}")
        shard_names = list(weight_map.values())
        if not all(isinstance(name, str) and name for name in shard_names):
            raise ValueError(f"Invalid model shard name in {index_path}")
        return {
            "layout": "sharded",
            "index": index_name,
            **artifact_identity(root, [index_name, *set(shard_names)]),
        }
    safetensors = sorted(path.name for path in root.glob("*.safetensors"))
    weight_files = safetensors or sorted(
        path.name for path in root.glob("pytorch_model*.bin")
    )
    if not weight_files:
        raise FileNotFoundError(f"No local model weight files found under {root}")
    return {"layout": "unsharded", **artifact_identity(root, weight_files)}


def base_model_prompt_identity(base_model: Path) -> dict[str, Any]:
    root = base_model.expanduser().resolve()
    exact_names = {
        "config.json",
        "generation_config.json",
        "tokenizer_config.json",
        "tokenizer.json",
        "chat_template.jinja",
        "special_tokens_map.json",
        "added_tokens.json",
        "vocab.json",
        "merges.txt",
        "tokenizer.model",
    }
    return artifact_identity(
        root,
        [name for name in exact_names if (root / name).is_file()],
    )


def runtime_package_versions() -> dict[str, str | None]:
    versions: dict[str, str | None] = {"python": sys.version.split()[0]}
    for distribution in (
        "torch",
        "transformers",
        "tokenizers",
        "accelerate",
        "safetensors",
        "huggingface-hub",
    ):
        try:
            versions[distribution] = importlib_metadata.version(distribution)
        except importlib_metadata.PackageNotFoundError:
            versions[distribution] = None
    return versions


def protected_code_identity() -> dict[str, str]:
    paths = {
        "evaluator": Path(__file__),
        "common": SCRIPT_DIR / "common.py",
        "chat_templates": PROJECT_ROOT / "deltamem" / "chat_templates.py",
        "delta_impl": PROJECT_ROOT / "deltamem" / "core" / "delta_impl.py",
        "rwkv_ms_core": PROJECT_ROOT / "deltamem" / "core" / "hrm_rwkv7.py",
        "backbone_compatibility": (
            PROJECT_ROOT / "deltamem" / "core" / "backbone_compat.py"
        ),
        "model_loading": PROJECT_ROOT / "deltamem" / "model_loading.py",
        "runtime_session": PROJECT_ROOT / "deltamem" / "runtime" / "session.py",
    }
    return {f"{name}_sha256": sha256_file(path) for name, path in paths.items()}


def validate_existing_manifest(
    manifest: Any,
    *,
    expected_fingerprint: str,
) -> dict[str, Any]:
    if not isinstance(manifest, dict):
        raise ValueError("Existing output manifest must be an object")
    recorded_fingerprint = manifest.get("fingerprint")
    if not isinstance(recorded_fingerprint, str):
        raise ValueError("Existing output manifest fingerprint is invalid")
    if fingerprint_payload_sha256(manifest.get("fingerprint_payload")) != recorded_fingerprint:
        raise ValueError(
            "Existing output manifest fingerprint_payload does not hash to its fingerprint"
        )
    fingerprint_contract = manifest["fingerprint_payload"].get(
        "evaluation_contract", {"name": "generic"}
    )
    payload_references = manifest["fingerprint_payload"].get("references")
    if (
        fingerprint_contract.get("name") != "generic"
        and manifest.get("references") != payload_references
    ):
        raise ValueError(
            "Existing output manifest references differ from its fingerprint payload"
        )
    if recorded_fingerprint != expected_fingerprint:
        raise ValueError("Existing output manifest fingerprint differs from this run")
    return manifest


def validate_protected_output_manifest_presence(
    *,
    contract_name: str,
    manifest_path: Path,
    output_paths: Iterable[Path],
) -> None:
    if (
        contract_name != "generic"
        and not manifest_path.is_file()
        and any(path.exists() for path in output_paths)
    ):
        raise RuntimeError(
            "Protected scene V6 output artifacts exist without their locked manifest; "
            "refusing to create a replacement manifest"
        )


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
    if profile_name != "native":
        unsupported = {
            name: {
                "placement": module.memory_fusion_placement,
                "hook_bound": (
                    getattr(module, "_post_attention_norm_hook_handle", None)
                    is not None
                ),
            }
            for name, module in modules
            if module.memory_fusion_placement not in RUNTIME_NORM_HOOK_PLACEMENTS
            or getattr(module, "_post_attention_norm_hook_handle", None) is None
        }
        if unsupported:
            raise ValueError(
                "Non-native fusion profile requires existing Gemma post-attention "
                f"norm hooks: {unsupported}"
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
                    getattr(module, "_post_attention_norm_hook_handle", None) is not None
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

    return _shared_extract_json(text)


def normalized_label_rows(value: Any) -> list[dict[str, Any]] | None:
    if isinstance(value, dict):
        value = value.get("labels")
    if not isinstance(value, list):
        return None
    return [item for item in value if isinstance(item, dict)]


def literal_boundaries(value: Any) -> dict[tuple[str, Any], Any] | None:
    return _shared_literal_boundaries(value)


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
        gold_boundaries = literal_boundaries(gold) or {}
        prediction_boundaries = literal_boundaries(prediction)
        schema_valid = prediction_boundaries is not None
        predicted = prediction_boundaries or {}
        gold_keys = set(gold_boundaries)
        predicted_keys = set(predicted)
        tp = len(gold_keys & predicted_keys)
        fp = len(predicted_keys - gold_keys)
        fn = len(gold_keys - predicted_keys)
        denominator = 2 * tp + fp + fn
        sort_key = lambda item: json.dumps(
            item,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        return {
            "json_extracted": json_extracted,
            "official_json_ok": official_json_ok,
            "schema_valid": schema_valid,
            "gold_boundaries": sorted(gold_boundaries.values(), key=sort_key),
            "predicted_boundaries": sorted(predicted.values(), key=sort_key),
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
        else (
            " The artifact reports the test split while this run evaluates a "
            "publisher-TRAIN-derived development partition."
            if evaluation_split == "train_derived_development"
            else ""
        )
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
    online_memory_protocol: str = "legacy_write_only",
) -> dict[str, Any]:
    import torch

    if online_memory_protocol not in ONLINE_MEMORY_PROTOCOLS:
        raise ValueError(
            f"Unknown online memory protocol: {online_memory_protocol!r}"
        )
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
            if online_memory_protocol == "write_then_read":
                set_delta_write_enabled(model, True)
                model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    use_cache=False,
                    return_dict=True,
                    logits_to_keep=1,
                )
                set_delta_write_enabled(model, False)
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
            "online_memory_protocol": online_memory_protocol,
        }
    finally:
        reset_delta_state(model)
        set_delta_write_enabled(model, True)


def read_records(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    rows: list[dict[str, Any]] = []
    with path.open("rb+") as handle:
        file_size = os.fstat(handle.fileno()).st_size
        while True:
            record_offset = handle.tell()
            raw_line = handle.readline()
            if not raw_line:
                break
            if not raw_line.strip():
                continue
            try:
                decoded = json.loads(raw_line.decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError):
                if handle.tell() != file_size or raw_line.endswith(b"\n"):
                    raise
                handle.seek(record_offset)
                handle.truncate()
                handle.flush()
                os.fsync(handle.fileno())
                break
            if not isinstance(decoded, dict):
                raise ValueError(
                    f"Evaluation record at byte {record_offset} is not an object"
                )
            rows.append(decoded)
            if handle.tell() == file_size and not raw_line.endswith(b"\n"):
                handle.seek(0, os.SEEK_END)
                handle.write(b"\n")
                handle.flush()
                os.fsync(handle.fileno())
    return rows


def validate_resume_record_contract(
    record: dict[str, Any],
    *,
    condition: str,
    spec: TaskSpec,
    sample: dict[str, Any],
    split: str,
    fingerprint: str,
    normal_fusion_profile: str,
) -> None:
    line_index = int(sample["line_index"])
    key = record_key(spec.name, line_index)
    expected_fields = {
        "status": "ok",
        "fingerprint": fingerprint,
        "key": key,
        "condition": condition,
        "normal_fusion_profile": (
            None if condition == "base" else normal_fusion_profile
        ),
        "task": spec.name,
        "task_kind": spec.kind,
        "split": split,
        "line_index": line_index,
        "row_sha256": sample["row_sha256"],
        "gold": sample["gold"],
        "max_new_tokens": spec.max_new_tokens,
    }
    for field, expected in expected_fields.items():
        if record.get(field) != expected:
            raise ValueError(f"Resume record {field} differs for {condition}:{key}")
    raw_generation = record.get("raw_generation")
    if not isinstance(raw_generation, str):
        raise ValueError(f"Resume record raw_generation is invalid for {condition}:{key}")
    if extract_json(raw_generation) != record.get("parsed_json"):
        raise ValueError(
            f"Resume record raw_generation does not reproduce parsed_json for "
            f"{condition}:{key}"
        )
    expected_score = score_prediction(
        spec.kind,
        record.get("parsed_json"),
        sample["gold"],
    )
    if record.get("score") != expected_score:
        raise ValueError(f"Resume record score differs for {condition}:{key}")
    input_tokens = record.get("input_tokens")
    if (
        isinstance(input_tokens, bool)
        or not isinstance(input_tokens, int)
        or input_tokens <= 0
    ):
        raise ValueError(f"Resume record input_tokens is invalid for {condition}:{key}")
    output_tokens = record.get("output_tokens")
    if (
        isinstance(output_tokens, bool)
        or not isinstance(output_tokens, int)
        or not 0 <= output_tokens <= spec.max_new_tokens
    ):
        raise ValueError(f"Resume record output_tokens is invalid for {condition}:{key}")
    hit_max_new_tokens = record.get("hit_max_new_tokens")
    if (
        not isinstance(hit_max_new_tokens, bool)
        or hit_max_new_tokens != (output_tokens >= spec.max_new_tokens)
    ):
        raise ValueError(
            f"Resume record hit_max_new_tokens is inconsistent for {condition}:{key}"
        )
    elapsed_seconds = record.get("elapsed_seconds")
    if (
        isinstance(elapsed_seconds, bool)
        or not isinstance(elapsed_seconds, (int, float))
        or not math.isfinite(float(elapsed_seconds))
        or float(elapsed_seconds) < 0.0
    ):
        raise ValueError(
            f"Resume record elapsed_seconds is invalid for {condition}:{key}"
        )


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
    unknown = [name for name in conditions if name not in {"base", "normal", "no_write"}]
    if unknown:
        raise ValueError(f"Unknown conditions: {', '.join(unknown)}")
    if not conditions:
        raise ValueError("At least one condition is required")
    if len(set(conditions)) != len(conditions):
        raise ValueError("Conditions must not contain duplicates")
    return conditions


def memory_architecture_contract(memory_dir: Path) -> dict[str, Any]:
    config_path = memory_dir.expanduser().resolve() / "delta_mem_config.json"
    config = read_json(config_path)
    target_layers = config.get("target_layers")
    delta_heads = config.get("delta_heads")
    rank = config.get("rank")
    semantics_version = config.get("rwkv_ms_semantics_version")
    memory_backend = config.get("memory_backend")
    if not isinstance(target_layers, list) or any(
        isinstance(layer, bool) or not isinstance(layer, int)
        for layer in target_layers
    ):
        raise ValueError(f"Invalid target_layers in {config_path}")
    if not isinstance(delta_heads, list) or any(
        not isinstance(head, str) for head in delta_heads
    ):
        raise ValueError(f"Invalid delta_heads in {config_path}")
    if isinstance(rank, bool) or not isinstance(rank, int):
        raise ValueError(f"Invalid rank in {config_path}")
    if isinstance(semantics_version, bool) or not isinstance(semantics_version, int):
        raise ValueError(f"Invalid rwkv_ms_semantics_version in {config_path}")
    if not isinstance(memory_backend, str):
        raise ValueError(f"Invalid memory_backend in {config_path}")
    return {
        "target_layers": list(target_layers),
        "delta_heads": list(delta_heads),
        "rank": rank,
        "rwkv_ms_semantics_version": semantics_version,
        "memory_backend": memory_backend,
    }


def canonical_object_sha256(payload: Any) -> str:
    return sha256_text(json.dumps(payload, sort_keys=True, separators=(",", ":")))


def _validated_receipt_file(
    record: Any,
    *,
    description: str,
    expected_path: Path | None = None,
) -> dict[str, Any]:
    if not isinstance(record, dict):
        raise ValueError(f"Scene V6 checkpoint receipt {description} is missing")
    raw_path = record.get("path")
    digest = record.get("file_sha256", record.get("sha256"))
    if not isinstance(raw_path, str) or not raw_path:
        raise ValueError(f"Scene V6 checkpoint receipt {description} path is invalid")
    if not isinstance(digest, str) or re.fullmatch(r"[0-9a-f]{64}", digest) is None:
        raise ValueError(f"Scene V6 checkpoint receipt {description} SHA-256 is invalid")
    path = Path(raw_path).expanduser().resolve()
    if expected_path is not None and path != expected_path.expanduser().resolve():
        raise ValueError(f"Scene V6 checkpoint receipt {description} path differs")
    if not path.is_file() or sha256_file(path) != digest:
        raise ValueError(f"Scene V6 checkpoint receipt {description} file differs")
    return {"path": str(path), "file_sha256": digest}


def scene_v6_identity_checkpoint_lineage(checkpoint_dir: Path) -> dict[str, Any]:
    checkpoint_dir = checkpoint_dir.expanduser().resolve()
    match = re.fullmatch(r"checkpoint-([1-9][0-9]*)", checkpoint_dir.name)
    if match is None or checkpoint_dir.parent.name != "trainer":
        raise ValueError(
            "Scene V6 identity evaluation requires RUN_ROOT/trainer/checkpoint-N"
        )
    checkpoint_step = int(match.group(1))
    receipt_path = checkpoint_dir / "checkpoint_receipt.json"
    if not receipt_path.is_file():
        raise ValueError(
            f"Scene V6 identity checkpoint receipt is missing: {receipt_path}"
        )
    receipt = read_json(receipt_path)
    if not isinstance(receipt, dict):
        raise ValueError("Scene V6 identity checkpoint receipt must be an object")
    unsigned_receipt = dict(receipt)
    recorded_receipt_sha = unsigned_receipt.pop("receipt_sha256", None)
    if recorded_receipt_sha != canonical_object_sha256(unsigned_receipt):
        raise ValueError("Scene V6 identity checkpoint receipt checksum differs")
    run_root = checkpoint_dir.parent.parent
    expected_literals = {
        "schema": SCENE_V6_IDENTITY_CHECKPOINT_RECEIPT_SCHEMA,
        "experiment": "scene_memory_v6_identity_proof",
        "run_mode": "proof",
        "checkpoint_step": checkpoint_step,
        "complete": True,
        "run_root": str(run_root),
        "checkpoint_dir": str(checkpoint_dir),
    }
    for field, expected in expected_literals.items():
        if receipt.get(field) != expected:
            raise ValueError(f"Scene V6 identity checkpoint receipt {field} differs")

    launch = _validated_receipt_file(
        receipt.get("launch"),
        description="launch manifest",
        expected_path=run_root / "launch_manifest.json",
    )
    data_contract = _validated_receipt_file(
        receipt.get("data_contract"),
        description="data contract",
    )
    source_lock = _validated_receipt_file(
        receipt.get("source_lock"),
        description="source lock",
    )
    pair_manifest = _validated_receipt_file(
        receipt.get("pair_manifest"),
        description="pair manifest",
    )
    if pair_manifest["file_sha256"] != SCENE_V6_IDENTITY_PAIR_MANIFEST_SHA256:
        raise ValueError("Scene V6 identity checkpoint pair manifest differs")

    objective = receipt.get("objective")
    if not isinstance(objective, dict):
        raise ValueError("Scene V6 identity checkpoint objective is missing")
    if objective != SCENE_V6_IDENTITY_OBJECTIVE_EXPECTED:
        missing = object()
        objective_fields = set(objective) | set(
            SCENE_V6_IDENTITY_OBJECTIVE_EXPECTED
        )
        differing_field = next(
            field
            for field in sorted(objective_fields)
            if objective.get(field, missing)
            != SCENE_V6_IDENTITY_OBJECTIVE_EXPECTED.get(field, missing)
        )
        raise ValueError(
            f"Scene V6 identity checkpoint objective {differing_field} differs"
        )

    train_partition = receipt.get("train_partition")
    if not isinstance(train_partition, dict):
        raise ValueError("Scene V6 identity checkpoint train partition is missing")
    train_file = _validated_receipt_file(
        train_partition,
        description="train partition",
    )
    if (
        train_file["file_sha256"] != SCENE_V6_IDENTITY_TRAIN_SHA256
        or train_partition.get("rows") != 32
        or train_partition.get("source_split") != "train"
    ):
        raise ValueError("Scene V6 identity checkpoint train partition differs")

    hard32 = receipt.get("hard32_selection")
    if not isinstance(hard32, dict):
        raise ValueError("Scene V6 identity checkpoint hard32 selection is missing")
    indices_record = hard32.get("indices", hard32.get("indices_file"))
    holdout_record = hard32.get("holdout", hard32.get("pair_holdout"))
    indices = _validated_receipt_file(
        indices_record,
        description="hard32 indices",
    )
    holdout = _validated_receipt_file(
        holdout_record,
        description="hard32 holdout",
    )
    if (
        indices["file_sha256"] != SCENE_V6_IDENTITY_HARD32_SELECTION_SHA256
        or holdout["file_sha256"] != SCENE_V6_IDENTITY_HARD32_HOLDOUT_SHA256
        or hard32.get("rows") != 32
        or hard32.get("source_split") != "val"
        or hard32.get("test_rows") != 0
    ):
        raise ValueError("Scene V6 identity checkpoint hard32 selection differs")

    trainer_state_record = receipt.get("trainer_state")
    if (
        not isinstance(trainer_state_record, dict)
        or trainer_state_record.get("global_step") != checkpoint_step
    ):
        raise ValueError("Scene V6 identity checkpoint trainer state step differs")

    artifacts = receipt.get("checkpoint_artifacts")
    if not isinstance(artifacts, dict):
        raise ValueError("Scene V6 identity checkpoint artifacts are missing")
    required_artifacts = {
        "adapter": checkpoint_dir / "delta_mem_adapter.pt",
        "config": checkpoint_dir / "delta_mem_config.json",
        "protocol": checkpoint_dir / "training_protocol.json",
        "trainer_state": checkpoint_dir / "trainer_state.json",
        "optimizer": checkpoint_dir / "optimizer.pt",
        "scheduler": checkpoint_dir / "scheduler.pt",
    }
    validated_artifacts: dict[str, Any] = {}
    for name, expected_path in required_artifacts.items():
        validated_artifacts[name] = _validated_receipt_file(
            artifacts.get(name),
            description=f"checkpoint {name}",
            expected_path=expected_path,
        )
    rng_records = artifacts.get("rng", artifacts.get("rng_files"))
    if isinstance(rng_records, dict):
        rng_records = [rng_records]
    if not isinstance(rng_records, list) or not rng_records:
        raise ValueError("Scene V6 identity checkpoint RNG artifacts are missing")
    validated_rng = [
        _validated_receipt_file(record, description="checkpoint RNG")
        for record in rng_records
    ]

    history = receipt.get("history")
    if (
        not isinstance(history, dict)
        or history.get("identity_metrics_finite") is not True
        or "finite" in history
        or history.get("last_step") != checkpoint_step
        or not isinstance(history.get("records"), int)
        or history["records"] < checkpoint_step
    ):
        raise ValueError("Scene V6 identity checkpoint history differs")
    if "adapter_change_from_step_zero" in receipt:
        raise ValueError("Scene V6 identity checkpoint adapter-change proof differs")
    adapter_change = receipt.get("adapter_change")
    if not isinstance(adapter_change, dict):
        raise ValueError("Scene V6 identity checkpoint adapter-change proof is missing")

    return {
        "lineage_kind": "identity_checkpoint_receipt",
        "run_root": str(run_root),
        "checkpoint_dir": str(checkpoint_dir),
        "checkpoint_step": checkpoint_step,
        "checkpoint_receipt": {
            "path": str(receipt_path),
            "file_sha256": sha256_file(receipt_path),
            "payload_sha256": recorded_receipt_sha,
            "schema": SCENE_V6_IDENTITY_CHECKPOINT_RECEIPT_SCHEMA,
        },
        "launch": launch,
        "data_contract": data_contract,
        "source_lock": source_lock,
        "pair_manifest": pair_manifest,
        "objective": objective,
        "train_partition": train_file,
        "hard32_selection": {"indices": indices, "holdout": holdout},
        "checkpoint_artifacts": {
            **validated_artifacts,
            "rng": validated_rng,
        },
        "history": history,
        "adapter_change": adapter_change,
    }


def scene_v6_training_lineage(memory_dir: Path) -> dict[str, Any]:
    checkpoint_dir = memory_dir.expanduser().resolve()
    if (checkpoint_dir / "checkpoint_receipt.json").is_file():
        return scene_v6_identity_checkpoint_lineage(checkpoint_dir)
    match = re.fullmatch(r"checkpoint-(128|256|384|512)", checkpoint_dir.name)
    if match is None or checkpoint_dir.parent.name != "trainer":
        raise ValueError(
            "Scene V6 evaluation requires a Stage1 checkpoint at "
            "RUN_ROOT/trainer/checkpoint-{128,256,384,512}"
        )
    checkpoint_step = int(match.group(1))
    run_root = checkpoint_dir.parent.parent
    launch_path = run_root / "launch_manifest.json"
    summary_path = run_root / "training_summary.json"
    initial_manifest_path = run_root / "initial_adapter" / "initial_adapter_manifest.json"
    protocol_path = checkpoint_dir / "training_protocol.json"
    trainer_state_path = checkpoint_dir / "trainer_state.json"
    for path in (
        launch_path,
        summary_path,
        initial_manifest_path,
        protocol_path,
        trainer_state_path,
    ):
        if not path.is_file():
            raise ValueError(f"Scene V6 checkpoint provenance file is missing: {path}")

    launch = read_json(launch_path)
    unsigned_launch = dict(launch)
    recorded_launch_sha = unsigned_launch.pop("manifest_sha256", None)
    if recorded_launch_sha != canonical_object_sha256(unsigned_launch):
        raise ValueError("Scene V6 launch manifest checksum differs")
    expected_launch_fields = {
        "schema": "rwkv_ms_scene_memory_v6_launch.v1",
        "experiment": "scene_memory_v6",
        "run_mode": "stage1",
        "fresh_run": True,
        "resume_from_checkpoint": None,
        "warm_start_from_checkpoint": None,
    }
    for field, expected in expected_launch_fields.items():
        if launch.get(field) != expected:
            raise ValueError(f"Scene V6 launch manifest {field} differs")
    if launch.get("paths", {}).get("output_dir") != str(run_root):
        raise ValueError("Scene V6 launch output path differs from checkpoint lineage")
    if launch.get("stage", {}).get("checkpoint_steps") != [128, 256, 384, 512]:
        raise ValueError("Scene V6 Stage1 checkpoint schedule differs")
    if launch.get("stage", {}).get("max_steps") != 512:
        raise ValueError("Scene V6 Stage1 max_steps differs")
    topology = launch.get("topology", {})
    if (
        topology.get("target_layers") != list(SCENE_V6_TARGET_LAYERS)
        or topology.get("delta_heads") != list(SCENE_V6_DELTA_HEADS)
        or topology.get("rank") != 4
        or topology.get("rwkv_ms_semantics_version") != 2
        or topology.get("memory_backend") != "rwkv_ms"
    ):
        raise ValueError("Scene V6 launch topology differs")
    training_partition = launch.get("data_contract", {}).get("training_partition", {})
    if (
        training_partition.get("source_split") != "train"
        or training_partition.get("rows") != 1804
        or training_partition.get("sha256") != SCENE_V6_TRAIN_SHA256
        or training_partition.get("val_or_test_rows_emitted_for_training") != 0
    ):
        raise ValueError("Scene V6 launch training partition differs")
    sampling = launch.get("sampling_audit", {})
    if (
        sampling.get("seed") != 42
        or sampling.get("data_seed") != 42
        or sampling.get("stage1_updates") != 512
        or sampling.get("sample_without_replacement") is not True
        or sampling.get("extension_beyond_update_512_allowed") is not False
    ):
        raise ValueError("Scene V6 launch sampling contract differs")
    source_lock_path = SCRIPT_DIR / "scene_memory_v6_source_lock.json"
    source_lock = launch.get("source_lock", {})
    if (
        source_lock.get("file_sha256") != sha256_file(source_lock_path)
        or source_lock.get("payload") != read_json(source_lock_path)
    ):
        raise ValueError("Scene V6 launch source lock differs from the frozen source")

    protocol = read_json(protocol_path)
    if sha256_file(protocol_path) != SCENE_V6_TRAINING_PROTOCOL_FILE_SHA256:
        raise ValueError("Scene V6 checkpoint training protocol file hash differs")
    if canonical_object_sha256(protocol) != SCENE_V6_TRAINING_PROTOCOL_CANONICAL_SHA256:
        raise ValueError("Scene V6 checkpoint training protocol payload differs")
    trainer_state = read_json(trainer_state_path)
    if trainer_state.get("global_step") != checkpoint_step:
        raise ValueError("Scene V6 trainer state step differs from checkpoint path")

    initial_manifest = read_json(initial_manifest_path)
    unsigned_initial = dict(initial_manifest)
    recorded_initial_sha = unsigned_initial.pop("manifest_sha256", None)
    if recorded_initial_sha != canonical_object_sha256(unsigned_initial):
        raise ValueError("Scene V6 initial-adapter manifest checksum differs")
    if (
        initial_manifest.get("schema") != "deltamem.seeded_initial_adapter.v1"
        or initial_manifest.get("fresh_run") is not True
        or initial_manifest.get("global_step") != 0
        or initial_manifest.get("training_started") is not False
        or initial_manifest.get("launch_manifest", {}).get("path")
        != str(launch_path)
        or initial_manifest.get("launch_manifest", {}).get("sha256")
        != sha256_file(launch_path)
    ):
        raise ValueError("Scene V6 initial-adapter lineage differs")
    training_summary = read_json(summary_path)
    if (
        training_summary.get("resume_from_checkpoint") is not None
        or training_summary.get("warm_start_from_checkpoint") is not None
        or training_summary.get("initial_adapter_output_dir")
        != str(run_root / "initial_adapter")
        or training_summary.get("initial_adapter_manifest_sha256")
        != recorded_initial_sha
        or training_summary.get("train_samples") != 1804
        or training_summary.get("seed") != 42
        or training_summary.get("data_seed") != 42
        or training_summary.get("train_sampler_seed") != 42
    ):
        raise ValueError("Scene V6 completed training summary lineage differs")
    return {
        "lineage_kind": "legacy_stage1_completed_summary",
        "run_root": str(run_root),
        "checkpoint_step": checkpoint_step,
        "launch_manifest_sha256": sha256_file(launch_path),
        "launch_manifest_payload_sha256": recorded_launch_sha,
        "training_summary_sha256": sha256_file(summary_path),
        "initial_adapter_manifest_sha256": sha256_file(initial_manifest_path),
        "initial_adapter_manifest_payload_sha256": recorded_initial_sha,
        "training_protocol_file_sha256": SCENE_V6_TRAINING_PROTOCOL_FILE_SHA256,
        "training_protocol_canonical_sha256": (
            SCENE_V6_TRAINING_PROTOCOL_CANONICAL_SHA256
        ),
        "trainer_state_sha256": sha256_file(trainer_state_path),
        "source_lock_sha256": sha256_file(source_lock_path),
    }


def has_scene_v6_lineage_marker(memory_dir: Path) -> bool:
    checkpoint_dir = memory_dir.expanduser().resolve()
    protocol_path = checkpoint_dir / "training_protocol.json"
    if protocol_path.is_file():
        try:
            if (
                sha256_file(protocol_path)
                == SCENE_V6_TRAINING_PROTOCOL_FILE_SHA256
                or canonical_object_sha256(read_json(protocol_path))
                == SCENE_V6_TRAINING_PROTOCOL_CANONICAL_SHA256
            ):
                return True
        except (OSError, json.JSONDecodeError):
            pass
    if (
        re.fullmatch(r"checkpoint-[1-9][0-9]*", checkpoint_dir.name) is None
        or checkpoint_dir.parent.name != "trainer"
    ):
        return False
    run_root = checkpoint_dir.parent.parent
    launch_path = run_root / "launch_manifest.json"
    if not launch_path.is_file():
        return False
    try:
        launch = read_json(launch_path)
    except (OSError, json.JSONDecodeError):
        return False
    return bool(
        isinstance(launch, dict)
        and launch.get("schema") == "rwkv_ms_scene_memory_v6_launch.v1"
        and launch.get("experiment") == "scene_memory_v6"
        and launch.get("paths", {}).get("output_dir") == str(run_root)
    )


def _validate_hard32_bound_file(record: Any, *, description: str) -> None:
    if not isinstance(record, dict):
        raise ValueError(f"Hard32 receipt {description} binding is missing")
    raw_path = record.get("path")
    digest = record.get("sha256")
    if not isinstance(raw_path, str) or not isinstance(digest, str):
        raise ValueError(f"Hard32 receipt {description} binding is invalid")
    path = Path(raw_path).expanduser().resolve()
    if not path.is_file() or sha256_file(path) != digest:
        raise ValueError(f"Hard32 receipt {description} artifact differs")


def validate_scene_v6_hard32_pass_receipt(
    receipt_path: Path,
    *,
    memory_dir: Path,
    candidate_lineage: dict[str, Any],
) -> dict[str, Any]:
    receipt_path = receipt_path.expanduser().resolve()
    receipt = read_json(receipt_path)
    if not isinstance(receipt, dict):
        raise ValueError("Hard32 receipt must contain an object")
    unsigned = dict(receipt)
    recorded_sha = unsigned.pop("receipt_sha256", None)
    if recorded_sha != canonical_object_sha256(unsigned):
        raise ValueError("Hard32 receipt checksum differs")
    if (
        receipt.get("schema") != SCENE_V6_IDENTITY_HARD32_RECEIPT_SCHEMA
        or receipt.get("status") != "pass"
    ):
        raise ValueError("Scene V6 validation requires a passed hard32 receipt")
    contract = receipt.get("contract")
    if (
        not isinstance(contract, dict)
        or contract.get("name") != "scene_v6_identity_hard32"
        or contract.get("rows") != 32
    ):
        raise ValueError("Hard32 receipt contract differs")
    memory_dir = memory_dir.expanduser().resolve()
    checkpoint = receipt.get("checkpoint")
    if (
        not isinstance(checkpoint, dict)
        or checkpoint.get("memory_dir") != str(memory_dir)
        or checkpoint.get("adapter_sha256")
        != sha256_file(memory_dir / "delta_mem_adapter.pt")
        or checkpoint.get("config_sha256")
        != sha256_file(memory_dir / "delta_mem_config.json")
        or checkpoint.get("candidate_lineage") != candidate_lineage
    ):
        raise ValueError("Hard32 receipt checkpoint binding differs")
    selection = receipt.get("selection")
    if (
        not isinstance(selection, dict)
        or selection.get("sha256")
        != SCENE_V6_IDENTITY_HARD32_SELECTION_SHA256
        or selection.get("holdout_sha256")
        != SCENE_V6_IDENTITY_HARD32_HOLDOUT_SHA256
        or selection.get("pair_manifest_sha256")
        != SCENE_V6_IDENTITY_PAIR_MANIFEST_SHA256
    ):
        raise ValueError("Hard32 receipt selection binding differs")
    _validate_hard32_bound_file(selection, description="selection")
    outputs = receipt.get("outputs")
    if not isinstance(outputs, dict):
        raise ValueError("Hard32 receipt output bindings are missing")
    _validate_hard32_bound_file(outputs.get("manifest"), description="manifest")
    _validate_hard32_bound_file(outputs.get("summary"), description="summary")
    condition_outputs = outputs.get("conditions")
    expected_conditions = {
        "base_full",
        "normal_full",
        "no_write_full",
        "state_only",
        "state_only_donor",
        "state_only_no_write",
    }
    if not isinstance(condition_outputs, dict) or set(condition_outputs) != expected_conditions:
        raise ValueError("Hard32 receipt condition output bindings differ")
    for condition, binding in condition_outputs.items():
        _validate_hard32_bound_file(binding, description=f"{condition} output")
    gate = receipt.get("gate")
    if (
        not isinstance(gate, dict)
        or gate.get("status") != "pass"
        or gate.get("all_gates_passed") is not True
        or gate.get("full170_authorized_for_bound_checkpoint") is not True
    ):
        raise ValueError("Hard32 receipt gate differs")
    return {
        "path": str(receipt_path),
        "file_sha256": sha256_file(receipt_path),
        "payload_sha256": recorded_sha,
        "evaluation_fingerprint": receipt.get("evaluation_fingerprint"),
    }


def validate_contract_lineage_mode(*, contract: str, memory_dir: Path) -> None:
    if contract == "generic" and has_scene_v6_lineage_marker(memory_dir):
        raise ValueError(
            "A recognized scene_memory_v6 checkpoint cannot use the generic evaluation "
            "contract; use scene_v6_validation or the receipt-gated final-test contract"
        )


def validate_evaluation_contract(
    *,
    contract: str,
    split: str,
    specs: list[TaskSpec],
    conditions: list[str],
    task_data: dict[str, tuple[Path, list[dict[str, Any]]]],
    limit_per_task: int | None,
    overwrite: bool,
    normal_fusion_profile: str,
    expected_memory_layer_count: int,
    memory_target_layers: list[int],
    memory_delta_heads: list[str],
    memory_rank: int,
    rwkv_ms_semantics_version: int,
    memory_backend: str,
    hard32_receipt_authorization: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if contract not in EVALUATION_CONTRACTS:
        raise ValueError(f"Unsupported evaluation contract: {contract}")
    if contract == "generic":
        return {"name": contract, "phase": "generic"}
    if contract == "scene_v6_final_test":
        raise ValueError(
            "scene_v6_final_test is unavailable until a passed validation-selection "
            "receipt binds the chosen checkpoint; Qwen remains unpaired positional "
            "context unless a genuine row-hash alignment manifest is provided"
        )
    if contract == "scene_v6_validation" and hard32_receipt_authorization is None:
        raise ValueError(
            "scene_v6_validation requires a passed hard32 receipt binding this checkpoint"
        )
    expected_split, expected_rows = SCENE_V6_CONTRACT_ROWS[contract]
    if split != expected_split:
        raise ValueError(
            f"{contract} requires split={expected_split}, received split={split}"
        )
    if [spec.name for spec in specs] != ["scene-v4-current"]:
        raise ValueError(f"{contract} requires exactly the scene-v4-current task")
    if conditions != ["base", "normal", "no_write"]:
        raise ValueError(
            f"{contract} requires conditions in exact order: base,normal,no_write"
        )
    if limit_per_task is not None:
        raise ValueError(f"{contract} forbids --limit-per-task")
    actual_rows = len(task_data["scene-v4-current"][1])
    if actual_rows != expected_rows:
        raise ValueError(
            f"{contract} requires exactly {expected_rows} official rows, found {actual_rows}"
        )
    dataset_path = task_data["scene-v4-current"][0]
    dataset_sha256 = sha256_file(dataset_path)
    expected_dataset_sha256 = OFFICIAL_SCENE_V4_SHA256[expected_split]
    if dataset_sha256 != expected_dataset_sha256:
        raise ValueError(
            f"{contract} requires the official scene-v4 {expected_split} file at "
            f"revision {OFFICIAL_SCENE_V4_DATASET_REVISION}"
        )
    if contract == "scene_v6_final_test" and overwrite:
        raise ValueError("scene_v6_final_test forbids --overwrite")
    if normal_fusion_profile != "native":
        raise ValueError(f"{contract} requires --normal-fusion-profile native")
    if expected_memory_layer_count != 42:
        raise ValueError(f"{contract} requires --expected-memory-layer-count 42")
    if memory_target_layers != list(SCENE_V6_TARGET_LAYERS):
        raise ValueError(f"{contract} requires checkpoint target_layers=0..41")
    if memory_delta_heads != list(SCENE_V6_DELTA_HEADS):
        raise ValueError(f"{contract} requires checkpoint delta_heads=q,o")
    if memory_rank != 4:
        raise ValueError(f"{contract} requires checkpoint rank=4")
    if rwkv_ms_semantics_version != 2:
        raise ValueError(f"{contract} requires checkpoint rwkv_ms_semantics_version=2")
    if memory_backend != "rwkv_ms":
        raise ValueError(f"{contract} requires checkpoint memory_backend=rwkv_ms")
    result = {
        "name": contract,
        "phase": "validation_selection" if expected_split == "val" else "final_test",
        "split": expected_split,
        "task": "scene-v4-current",
        "rows": expected_rows,
        "conditions": list(conditions),
        "normal_fusion_profile": "native",
        "expected_memory_layer_count": 42,
        "memory_target_layers": list(SCENE_V6_TARGET_LAYERS),
        "memory_delta_heads": list(SCENE_V6_DELTA_HEADS),
        "memory_rank": 4,
        "rwkv_ms_semantics_version": 2,
        "memory_backend": "rwkv_ms",
        "official_dataset_revision": OFFICIAL_SCENE_V4_DATASET_REVISION,
        "official_dataset_sha256": expected_dataset_sha256,
        "overwrite_allowed": contract != "scene_v6_final_test",
        "generation_policy": (
            "Append-only resumable records; completed keys are never regenerated."
        ),
        "hard32_receipt_authorization": hard32_receipt_authorization,
    }
    if contract == "scene_v6_final_test":
        result.update(
            {
                "checkpoint_selection_forbidden": True,
                "test_once_enforcement_scope": (
                    "per_output_directory_and_fingerprint"
                ),
                "test_once_enforcement_caveat": (
                    "A new output directory can rerun inference; global single-use "
                    "enforcement is not provided. Checkpoint selection on test remains "
                    "forbidden."
                ),
            }
        )
    return result


def generate_for_condition(
    condition: str,
    *,
    model,
    tokenizer,
    messages: list[dict[str, str]],
    max_new_tokens: int,
    device: str,
    online_memory_protocol: str = "legacy_write_only",
) -> dict[str, Any]:
    if condition == "no_write":
        with memory_condition(model, "no_write"):
            return generate_one(
                model=model,
                tokenizer=tokenizer,
                messages=messages,
                max_new_tokens=max_new_tokens,
                device=device,
                online_memory_protocol="legacy_write_only",
            )
    return generate_one(
        model=model,
        tokenizer=tokenizer,
        messages=messages,
        max_new_tokens=max_new_tokens,
        device=device,
        online_memory_protocol=(
            online_memory_protocol
            if condition == "normal"
            else "legacy_write_only"
        ),
    )


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
    if args.hard32_receipt is not None:
        args.hard32_receipt = args.hard32_receipt.expanduser().resolve()
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
    memory_dir = Path(args.memory_dir).expanduser().resolve()
    validate_contract_lineage_mode(
        contract=args.evaluation_contract,
        memory_dir=memory_dir,
    )
    memory_architecture = memory_architecture_contract(memory_dir)
    candidate_lineage = (
        None
        if args.evaluation_contract == "generic"
        else scene_v6_training_lineage(memory_dir)
    )
    hard32_receipt_authorization = None
    if args.evaluation_contract == "scene_v6_validation":
        if args.hard32_receipt is None:
            raise ValueError("scene_v6_validation requires --hard32-receipt")
        hard32_receipt_authorization = validate_scene_v6_hard32_pass_receipt(
            args.hard32_receipt,
            memory_dir=memory_dir,
            candidate_lineage=candidate_lineage,
        )
    elif args.hard32_receipt is not None:
        raise ValueError("--hard32-receipt is accepted only by scene_v6_validation")
    evaluation_contract = validate_evaluation_contract(
        contract=args.evaluation_contract,
        split=args.split,
        specs=specs,
        conditions=conditions,
        task_data=task_data,
        limit_per_task=args.limit_per_task,
        overwrite=args.overwrite,
        normal_fusion_profile=args.normal_fusion_profile,
        expected_memory_layer_count=args.expected_memory_layer_count,
        memory_target_layers=memory_architecture["target_layers"],
        memory_delta_heads=memory_architecture["delta_heads"],
        memory_rank=memory_architecture["rank"],
        rwkv_ms_semantics_version=memory_architecture[
            "rwkv_ms_semantics_version"
        ],
        memory_backend=memory_architecture["memory_backend"],
        hard32_receipt_authorization=hard32_receipt_authorization,
    )
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

    base_model_path = Path(args.base_model).expanduser().resolve()
    base_config_path = base_model_path / "config.json"
    protected_identity: dict[str, Any] = {}
    if evaluation_contract["name"] != "generic":
        protected_identity = {
            "base_model_weights": base_model_weight_identity(base_model_path),
            "base_model_prompt_artifacts": base_model_prompt_identity(
                base_model_path
            ),
            "runtime_packages": runtime_package_versions(),
            "code": protected_code_identity(),
            "reference_repository_revision": git_revision(
                args.reference_results_dir.parents[1]
            ),
        }
    fingerprint_payload = {
        "evaluator_sha256": sha256_file(Path(__file__)),
        "base_model": str(base_model_path),
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
        "evaluation_contract": evaluation_contract,
        "candidate_lineage": candidate_lineage,
        "hard32_receipt_authorization": hard32_receipt_authorization,
        "references": references,
        "device": args.device,
        "dtype": args.dtype,
        "attn_implementation": args.attn_implementation,
        "online_memory_protocol": args.online_memory_protocol,
        **fusion_fingerprint_fields,
        **protected_identity,
    }
    fingerprint = fingerprint_payload_sha256(fingerprint_payload)
    manifest_path = args.output_dir / "manifest.json"
    protected_output_paths = [
        *(args.output_dir / f"{condition}.jsonl" for condition in conditions),
        args.output_dir / "progress.json",
        args.output_dir / "summary.json",
    ]
    validate_protected_output_manifest_presence(
        contract_name=evaluation_contract["name"],
        manifest_path=manifest_path,
        output_paths=protected_output_paths,
    )
    if manifest_path.is_file():
        previous = read_json(manifest_path)
        try:
            validate_existing_manifest(
                previous,
                expected_fingerprint=fingerprint,
            )
        except ValueError as exc:
            raise RuntimeError(
                f"Cannot resume from {manifest_path}: {exc}; use --overwrite or a new "
                "output directory"
            ) from exc
    else:
        write_json_atomic(
            manifest_path,
            {
                "created_at": utc_now(),
                "fingerprint": fingerprint,
                "fingerprint_payload": fingerprint_payload,
                "evaluation_kind": "structured-task transfer evaluation",
                "evaluation_contract": evaluation_contract,
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
                    f"online_memory_protocol={args.online_memory_protocol}, "
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

    protected_contract = evaluation_contract["name"] != "generic"
    expected_records = {
        record_key(spec.name, int(sample["line_index"])): (spec, sample)
        for spec in specs
        for sample in task_data[spec.name][1]
    }
    expected_total = sum(len(task_data[spec.name][1]) for spec in specs) * len(conditions)
    completed_total = 0
    for condition in conditions:
        records_path = args.output_dir / f"{condition}.jsonl"
        existing_rows = read_records(records_path)
        existing: dict[str, dict[str, Any]] = {}
        for row in existing_rows:
            if row.get("condition") != condition:
                raise ValueError(f"Unexpected condition in {records_path}: {row.get('condition')}")
            if protected_contract:
                key = str(row.get("key"))
                expected = expected_records.get(key)
                if expected is None:
                    raise ValueError(
                        f"Unexpected resume key in {records_path}: {key}"
                    )
                spec, sample = expected
                validate_resume_record_contract(
                    row,
                    condition=condition,
                    spec=spec,
                    sample=sample,
                    split=args.split,
                    fingerprint=fingerprint,
                    normal_fusion_profile=args.normal_fusion_profile,
                )
            if row.get("status") == "ok":
                key = str(row["key"])
                if key in existing:
                    raise ValueError(f"Duplicate completed key in {records_path}: {key}")
                existing[key] = row
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
            result = generate_for_condition(
                condition,
                model=model,
                tokenizer=tokenizer,
                messages=sample["messages"],
                max_new_tokens=spec.max_new_tokens,
                device=args.device,
                online_memory_protocol=args.online_memory_protocol,
            )
            score = score_prediction(spec.kind, result["parsed_json"], sample["gold"])
            key = record_key(spec.name, int(sample["line_index"]))
            record = {
                "fingerprint": fingerprint,
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
    if "normal" in summaries and "no_write" in summaries:
        for spec in specs:
            normal_metric = float(summaries["normal"][spec.name]["primary_metric"])
            no_write_metric = float(
                summaries["no_write"][spec.name]["primary_metric"]
            )
            task_comparison = comparisons.setdefault(
                spec.name,
                {
                    "metric_name": summaries["normal"][spec.name][
                        "primary_metric_name"
                    ],
                    "normal": normal_metric,
                },
            )
            task_comparison.update(
                {
                    "no_write": no_write_metric,
                    "normal_minus_no_write": normal_metric - no_write_metric,
                }
            )

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
        "evaluation_contract": evaluation_contract,
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
