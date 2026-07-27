#!/usr/bin/env python3
"""Screen Gemma attention-residual memory hybrids on exact W1/W8 pairs."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import statistics
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
from datasets import Dataset, load_from_disk

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from deltamem.core.delta import iter_delta_mem_modules
from experiments.rethinking_rwkv_ms_gemma.common import (
    load_model_and_tokenizer,
    read_jsonl,
    write_json,
)
from experiments.rethinking_rwkv_ms_gemma.diagnose_layer_state_swaps import (
    baseline_gap_summary,
    checkpoint_file_provenance,
    condition_metrics,
    directional_row_effect,
    history_token_selection,
    prime_writer_snapshots,
    replay_with_token_nll,
    summarize_condition_rows,
)
from experiments.rethinking_rwkv_ms_gemma.diagnose_memory_representation import (
    load_pairing_donors,
)
from experiments.rethinking_rwkv_ms_gemma.eval_episode_memory_ce import (
    load_protocol,
    reset_runtime,
    sha256_file,
    source_identity,
    validate_artifacts,
)


SUPPORTED_CONDITIONS = (
    "native",
    "native_gate_open",
    "gate_open_gamma_0",
    "gate_open_gamma_0p01",
    "post_attention_norm_gate_open_0p01",
)
SUPPORTED_RUNTIME_PLACEMENTS = frozenset(
    {
        "normalized_residual_correction",
        "post_attention_norm",
        "post_attention_residual_hybrid",
    }
)
CORRECT_FULL_CE_NATIVE_RATIO_CEILING = 1.25
CORRECT_W1_CE_NATIVE_DELTA_CEILING = 0.5


def parse_condition_names(raw: str) -> list[str]:
    names: list[str] = []
    for item in raw.split(","):
        name = item.strip().lower().replace("-", "_")
        if not name:
            continue
        if name not in SUPPORTED_CONDITIONS:
            raise ValueError(
                f"Unsupported condition {name!r}; expected one of {SUPPORTED_CONDITIONS}"
            )
        if name not in names:
            names.append(name)
    if not names:
        raise ValueError("At least one hybrid condition is required")
    return names


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-model", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--tokenized-dataset", type=Path, required=True)
    parser.add_argument("--source-jsonl", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--conditions",
        default=",".join(SUPPORTED_CONDITIONS),
        help=(
            "Comma-separated bounded screen. Every condition receives freshly primed "
            "writer states."
        ),
    )
    parser.add_argument("--history-span-tokens", type=int, default=8, choices=(1, 8, 16, 32))
    parser.add_argument("--expected-layer-count", type=int, default=42)
    parser.add_argument("--expected-row-count", type=int, default=32)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--dtype",
        default="bfloat16",
        choices=("bfloat16", "float16", "float32"),
    )
    parser.add_argument("--attn-implementation", default="sdpa")
    parser.add_argument("--delta-mem-root", type=Path, default=REPO_ROOT)
    return parser.parse_args()


def default_output_path(checkpoint: Path) -> Path:
    if checkpoint.parent.name != "trainer":
        raise ValueError("--output is required unless checkpoint is RUN_ROOT/trainer/checkpoint-N")
    return (
        checkpoint.parent.parent
        / "residual_hybrid_diagnostic"
        / f"{checkpoint.name}_fresh_prime_hybrid_screen.json"
    )


def validate_modules(
    modules: list[tuple[str, torch.nn.Module]],
    expected_layer_count: int,
) -> list[tuple[str, torch.nn.Module]]:
    ordered = sorted(modules, key=lambda item: int(item[1].layer_idx))
    layer_indices = [int(module.layer_idx) for _, module in ordered]
    if layer_indices != list(range(expected_layer_count)):
        raise RuntimeError(
            "Residual hybrid diagnostic requires all layers in order: "
            f"expected={list(range(expected_layer_count))} actual={layer_indices}"
        )
    unsupported = {
        name: {
            "backend": module.memory_backend,
            "delta_heads": sorted(module.active_delta_heads),
            "placement": module.memory_fusion_placement,
            "post_attention_norm_hook_bound": (
                module._post_attention_norm_hook_handle is not None
            ),
        }
        for name, module in ordered
        if module.memory_backend != "rwkv_ms"
        or module.active_delta_heads != frozenset({"o"})
        or module.memory_fusion_placement not in SUPPORTED_RUNTIME_PLACEMENTS
        or module._post_attention_norm_hook_handle is None
    }
    if unsupported:
        raise ValueError(
            "Residual hybrid diagnostic requires RWKV-MS O-only modules: "
            f"{unsupported}"
        )
    return ordered


def capture_fusion_settings(
    modules: list[tuple[str, torch.nn.Module]],
) -> dict[str, dict[str, Any]]:
    settings: dict[str, dict[str, Any]] = {}
    for name, module in modules:
        raw_gain = getattr(module, "memory_fusion_residual_gain_raw", None)
        settings[name] = {
            "layer_index": int(module.layer_idx),
            "memory_fusion_placement": str(module.memory_fusion_placement),
            "memory_fusion_mode": str(module.memory_fusion_mode),
            "memory_fusion_residual_scale": float(module.memory_fusion_residual_scale),
            "memory_fusion_residual_scale_max": float(
                module.memory_fusion_residual_scale_max
            ),
            "memory_fusion_residual_gain_raw": (
                None if raw_gain is None else raw_gain.detach().cpu().clone()
            ),
        }
    return settings


def restore_fusion_settings(
    modules: list[tuple[str, torch.nn.Module]],
    saved: dict[str, dict[str, Any]],
) -> None:
    actual_names = {name for name, _ in modules}
    if actual_names != set(saved):
        raise RuntimeError(
            "Fusion setting module mismatch: "
            f"missing={sorted(set(saved).difference(actual_names))} "
            f"extra={sorted(actual_names.difference(saved))}"
        )
    with torch.no_grad():
        for name, module in modules:
            state = saved[name]
            module.memory_fusion_placement = state["memory_fusion_placement"]
            module.memory_fusion_mode = state["memory_fusion_mode"]
            module.memory_fusion_residual_scale = state[
                "memory_fusion_residual_scale"
            ]
            module.memory_fusion_residual_scale_max = state[
                "memory_fusion_residual_scale_max"
            ]
            saved_raw = state["memory_fusion_residual_gain_raw"]
            current_raw = getattr(module, "memory_fusion_residual_gain_raw", None)
            if (saved_raw is None) != (current_raw is None):
                raise RuntimeError(
                    f"Residual hybrid gain topology changed for {name}: "
                    f"saved={saved_raw is not None} current={current_raw is not None}"
                )
            if saved_raw is not None:
                current_raw.copy_(
                    saved_raw.to(device=current_raw.device, dtype=current_raw.dtype)
                )


def serializable_fusion_settings(
    saved: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    rows = []
    for name, state in sorted(saved.items(), key=lambda item: item[1]["layer_index"]):
        raw = state["memory_fusion_residual_gain_raw"]
        rows.append(
            {
                "module_name": name,
                "layer_index": int(state["layer_index"]),
                "memory_fusion_placement": state["memory_fusion_placement"],
                "memory_fusion_mode": state["memory_fusion_mode"],
                "memory_fusion_residual_scale": float(
                    state["memory_fusion_residual_scale"]
                ),
                "memory_fusion_residual_scale_max": float(
                    state["memory_fusion_residual_scale_max"]
                ),
                "memory_fusion_residual_gain_raw": (
                    None if raw is None else float(raw.float().item())
                ),
            }
        )
    return rows


def effective_fusion_settings(
    modules: list[tuple[str, torch.nn.Module]],
) -> list[dict[str, Any]]:
    rows = []
    for name, module in modules:
        raw = getattr(module, "memory_fusion_residual_gain_raw", None)
        effective_gain = None
        if module.memory_fusion_placement == "post_attention_residual_hybrid":
            if raw is None:
                raise RuntimeError(f"Hybrid condition has no residual gain parameter: {name}")
            effective_gain = float(
                module._resolved_memory_fusion_residual_gain(
                    device=raw.device,
                    dtype=torch.float32,
                )
                .detach()
                .item()
            )
        rows.append(
            {
                "module_name": name,
                "layer_index": int(module.layer_idx),
                "memory_fusion_placement": str(module.memory_fusion_placement),
                "memory_fusion_mode": str(module.memory_fusion_mode),
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
    return rows


def initial_condition_screen(names: list[str]) -> list[dict[str, Any]]:
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
    return [dict(definitions[name]) for name in names]


def apply_condition(
    modules: list[tuple[str, torch.nn.Module]],
    condition: dict[str, Any],
) -> None:
    for name, module in modules:
        if "memory_fusion_mode" in condition:
            module.memory_fusion_mode = str(condition["memory_fusion_mode"])
        if "memory_fusion_placement" in condition:
            module.memory_fusion_placement = str(
                condition["memory_fusion_placement"]
            )
        if "memory_fusion_residual_scale" in condition:
            module.memory_fusion_residual_scale = float(
                condition["memory_fusion_residual_scale"]
            )
        if "memory_fusion_residual_gain" in condition:
            if module.memory_fusion_placement != "post_attention_residual_hybrid":
                raise RuntimeError(
                    f"Condition {condition['name']} requests a residual gain for "
                    f"non-hybrid module {name}"
                )
            module.set_memory_fusion_residual_gain(
                float(condition["memory_fusion_residual_gain"])
            )


def snapshot_bank_sha256(
    snapshots: list[dict[str, torch.Tensor]],
) -> str:
    digest = hashlib.sha256()
    digest.update(b"rwkv_ms_writer_snapshot_bank.v1\0")
    for row_index, snapshot in enumerate(snapshots):
        digest.update(f"row:{row_index}\0".encode("ascii"))
        for key in sorted(snapshot):
            tensor = snapshot[key].detach().cpu().contiguous()
            digest.update(key.encode("utf-8") + b"\0")
            digest.update(str(tensor.dtype).encode("ascii") + b"\0")
            digest.update(json.dumps(list(tensor.shape)).encode("ascii") + b"\0")
            digest.update(tensor.view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def canonical_json_sha256(payload: Any) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def pairing_protocol_summary(manifest: dict[str, Any]) -> dict[str, Any]:
    splits = manifest.get("splits")
    if not isinstance(splits, dict):
        raise ValueError("Pairing manifest has no split mapping")
    split_summaries = {}
    for split_name, split in splits.items():
        if not isinstance(split, dict):
            raise ValueError(f"Pairing manifest split is invalid: {split_name}")
        required = (
            "sample_count",
            "rotation",
            "target_mode",
            "target_span_tokens",
            "target_token_count",
            "source_fingerprint",
            "paired_fingerprint",
            "pairs_sha256",
            "manifest_sha256",
        )
        missing = [key for key in required if key not in split]
        if missing:
            raise ValueError(
                f"Pairing manifest split {split_name} is missing: {missing}"
            )
        split_summaries[split_name] = {key: split[key] for key in required}
    required_top = (
        "pairing_version",
        "pairing_scope",
        "target_mode",
        "target_span_tokens",
        "target_token_count",
        "data_seed",
        "tokenized_fingerprint",
        "manifest_sha256",
    )
    missing_top = [key for key in required_top if key not in manifest]
    if missing_top:
        raise ValueError(f"Pairing manifest is missing: {missing_top}")
    return {
        **{key: manifest[key] for key in required_top},
        "splits": split_summaries,
    }


def validate_pairing_manifest_integrity(
    *,
    checkpoint: Path,
    protocol: dict[str, Any],
    tokenized: Dataset,
    split_name: str,
    donors: list[int],
) -> dict[str, Any]:
    manifest_path = checkpoint / "content_contrast_pairing_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(manifest, dict):
        raise ValueError("Pairing manifest must be a JSON object")
    recorded_manifest_hash = manifest.get("manifest_sha256")
    unhashed_manifest = dict(manifest)
    unhashed_manifest.pop("manifest_sha256", None)
    computed_manifest_hash = canonical_json_sha256(unhashed_manifest)
    if recorded_manifest_hash != computed_manifest_hash:
        raise ValueError(
            "Pairing manifest top-level self hash does not match: "
            f"recorded={recorded_manifest_hash} computed={computed_manifest_hash}"
        )

    splits = manifest.get("splits")
    split = None if not isinstance(splits, dict) else splits.get(split_name)
    if not isinstance(split, dict):
        raise ValueError(f"Pairing manifest has no valid {split_name!r} split")
    pairs = split.get("pairs")
    if not isinstance(pairs, list):
        raise ValueError(f"Pairing manifest split {split_name!r} has no pair list")
    recorded_pairs_hash = split.get("pairs_sha256")
    computed_pairs_hash = canonical_json_sha256(pairs)
    if recorded_pairs_hash != computed_pairs_hash:
        raise ValueError(
            f"Pairing manifest split {split_name!r} pair hash does not match: "
            f"recorded={recorded_pairs_hash} computed={computed_pairs_hash}"
        )
    recorded_split_hash = split.get("manifest_sha256")
    unhashed_split = dict(split)
    unhashed_split.pop("manifest_sha256", None)
    computed_split_hash = canonical_json_sha256(unhashed_split)
    if recorded_split_hash != computed_split_hash:
        raise ValueError(
            f"Pairing manifest split {split_name!r} self hash does not match: "
            f"recorded={recorded_split_hash} computed={computed_split_hash}"
        )

    expected_protocol_summary = protocol.get("content_contrast_pairing")
    actual_protocol_summary = pairing_protocol_summary(manifest)
    if expected_protocol_summary != actual_protocol_summary:
        raise ValueError(
            "Pairing manifest does not match training_protocol content_contrast_pairing"
        )
    expected_fingerprint = protocol.get("tokenized_fingerprint")
    if manifest.get("tokenized_fingerprint") != expected_fingerprint:
        raise ValueError("Pairing manifest tokenized fingerprint does not match protocol")
    if split.get("source_fingerprint") != expected_fingerprint:
        raise ValueError(
            f"Pairing manifest split {split_name!r} source fingerprint does not match"
        )
    if int(split.get("sample_count", -1)) != len(tokenized):
        raise ValueError(
            f"Pairing manifest split {split_name!r} sample count does not match dataset"
        )
    if len(pairs) != len(tokenized) or len(donors) != len(tokenized):
        raise ValueError("Pairing manifest pair or donor count does not match dataset")

    target_span_tokens = int(split["target_span_tokens"])
    target_mode = str(split["target_mode"])
    checked_write_hashes = 0
    checked_target_spans = 0
    for source_index, pair in enumerate(pairs):
        if not isinstance(pair, dict):
            raise ValueError(f"Pairing manifest row {source_index} is not an object")
        if int(pair.get("source_index", -1)) != source_index:
            raise ValueError(f"Pairing manifest source order differs at row {source_index}")
        partner_index = donors[source_index]
        if int(pair.get("partner_index", -1)) != partner_index:
            raise ValueError(f"Pairing manifest donor differs at row {source_index}")
        if pair.get("source_id") != f"{split_name}:{source_index}" or pair.get(
            "partner_id"
        ) != f"{split_name}:{partner_index}":
            raise ValueError(f"Pairing manifest source/donor IDs differ at row {source_index}")

        def write_hash(row: dict[str, Any]) -> str:
            payload = {
                field: [int(value) for value in row[f"write_{field}"]]
                for field in (
                    "input_ids",
                    "attention_mask",
                    "message_ids",
                    "sentence_ids",
                )
            }
            return canonical_json_sha256(payload)

        source_write_hash = write_hash(tokenized[source_index])
        partner_write_hash = write_hash(tokenized[partner_index])
        if pair.get("source_write_sha256") != source_write_hash:
            raise ValueError(f"Pairing manifest source write differs at row {source_index}")
        if pair.get("partner_write_sha256") != partner_write_hash:
            raise ValueError(f"Pairing manifest donor write differs at row {source_index}")
        if pair.get("negative_write_sha256") != partner_write_hash:
            raise ValueError(f"Pairing manifest negative write differs at row {source_index}")
        checked_write_hashes += 2

        selection = history_token_selection(
            tokenized[source_index],
            tokenized[partner_index],
            primary_span_tokens=target_span_tokens,
            unaligned_policy="error",
        )
        window = selection["windows"][str(target_span_tokens)]
        ordinals = window["supervised_ordinals"]
        donor_token_ids = [
            selection["donor_supervised_token_ids"][ordinal]
            for ordinal in ordinals
        ]
        target_mask = [False] * len(tokenized[source_index]["labels"])
        for label_position in window["target_label_positions"]:
            target_mask[int(label_position)] = True
        expected_pair_fields = {
            "target_mode": target_mode,
            "target_span_tokens": target_span_tokens,
            "first_differing_supervised_ordinal": selection[
                "first_history_ordinal"
            ],
            "first_target_label_position": selection[
                "first_history_label_position"
            ],
            "first_target_predictor_position": selection[
                "first_history_predictor_position"
            ],
            "target_label_positions": window["target_label_positions"],
            "target_token_ids": window["target_token_ids"],
            "donor_token_ids": donor_token_ids,
            "target_mask_sha256": canonical_json_sha256(target_mask),
        }
        mismatches = [
            key
            for key, expected in expected_pair_fields.items()
            if pair.get(key) != expected
        ]
        if mismatches:
            raise ValueError(
                f"Pairing manifest target span differs at row {source_index}: "
                f"{mismatches}"
            )
        checked_target_spans += 1

    return {
        "status": "verified",
        "path": str(manifest_path),
        "file_sha256": sha256_file(manifest_path),
        "manifest_sha256": computed_manifest_hash,
        "split_manifest_sha256": computed_split_hash,
        "pairs_sha256": computed_pairs_hash,
        "split": split_name,
        "pair_count": len(pairs),
        "checked_write_hash_count": checked_write_hashes,
        "checked_target_span_count": checked_target_spans,
        "protocol_summary_exact_match": True,
    }


def evaluate_condition(
    *,
    model,
    tokenized: Dataset,
    source_rows: list[dict[str, Any]],
    donors: list[int],
    snapshots: list[dict[str, torch.Tensor]],
    device: str,
    primary_span_tokens: int,
) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, int]]:
    rows: list[dict[str, Any]] = []
    correct_replay_count = 0
    donor_replay_count = 0
    with torch.inference_mode():
        for row_index in range(len(tokenized)):
            donor_index = donors[row_index]
            target_row = tokenized[row_index]
            selection = history_token_selection(
                target_row,
                tokenized[donor_index],
                primary_span_tokens=primary_span_tokens,
                unaligned_policy="error",
            )
            expected_tokens = int(selection["target_supervised_token_count"])
            correct_replay = replay_with_token_nll(
                model=model,
                target_row=target_row,
                online_state=snapshots[row_index],
                device=device,
                expected_token_count=expected_tokens,
            )
            correct_replay_count += 1
            donor_replay = replay_with_token_nll(
                model=model,
                target_row=target_row,
                online_state=snapshots[donor_index],
                device=device,
                expected_token_count=expected_tokens,
            )
            donor_replay_count += 1
            correct = condition_metrics(correct_replay, selection)
            donor = condition_metrics(donor_replay, selection)
            gap = directional_row_effect(
                correct,
                donor,
                intervention_improves_when_lower=False,
            )
            rows.append(
                {
                    "row_index": row_index,
                    "donor_row_index": donor_index,
                    "target": source_identity(source_rows[row_index], row_index),
                    "exact_donor": source_identity(source_rows[donor_index], donor_index),
                    "history_token_selection": selection,
                    "correct_memory": correct,
                    "exact_donor_memory": donor,
                    "donor_minus_correct": gap,
                }
            )
            primary_gap = gap["history_windows"][str(primary_span_tokens)]["ce_effect"]
            print(
                f"row {row_index + 1:02d}/{len(tokenized)} "
                f"history{primary_span_tokens}_gap={primary_gap:.6f}",
                flush=True,
            )
    summary = {
        "correct_memory": summarize_condition_rows(rows, "correct_memory"),
        "exact_donor_memory": summarize_condition_rows(rows, "exact_donor_memory"),
        "donor_minus_correct": baseline_gap_summary(rows),
    }
    counts = {
        "correct_replay_count": correct_replay_count,
        "donor_replay_count": donor_replay_count,
        "replay_count": correct_replay_count + donor_replay_count,
    }
    return rows, summary, counts


def effect_distribution(values: list[float]) -> dict[str, Any]:
    if not values:
        raise ValueError("Cannot summarize an empty effect distribution")
    ordered = sorted(float(value) for value in values)
    trim_count = int(len(ordered) * 0.1)
    trimmed = (
        ordered[trim_count:-trim_count]
        if trim_count > 0 and trim_count * 2 < len(ordered)
        else ordered
    )
    return {
        "count": len(ordered),
        "mean": statistics.fmean(ordered),
        "median": statistics.median(ordered),
        "trim_fraction": 0.1,
        "trim_count_each_tail": trim_count,
        "trimmed_mean": statistics.fmean(trimmed),
        "population_std": statistics.pstdev(ordered),
        "min": ordered[0],
        "max": ordered[-1],
        "absolute_gap_gt_0p2_count": sum(abs(value) > 0.2 for value in ordered),
    }


def condition_window_effects(
    condition: dict[str, Any],
    span_tokens: int,
) -> list[float]:
    key = str(span_tokens)
    return [
        float(row["donor_minus_correct"]["history_windows"][key]["ce_effect"])
        for row in condition["rows"]
    ]


def rank_conditions(conditions: list[dict[str, Any]], span_tokens: int) -> list[dict[str, Any]]:
    native_condition = next(
        (condition for condition in conditions if condition["name"] == "native"),
        None,
    )
    if native_condition is None:
        raise ValueError("Hybrid condition ranking requires the native condition")
    native_correct = native_condition["summary"]["correct_memory"]
    native_correct_full_ce = float(
        native_correct["full_answer"]["token_weighted_ce"]
    )
    native_correct_w1_ce = float(
        native_correct["history_windows"]["1"]["token_weighted_ce"]
    )
    correct_full_ce_ceiling = (
        native_correct_full_ce * CORRECT_FULL_CE_NATIVE_RATIO_CEILING
    )
    correct_w1_ce_ceiling = (
        native_correct_w1_ce + CORRECT_W1_CE_NATIVE_DELTA_CEILING
    )
    span_key = str(span_tokens)
    ranked = []
    for condition in conditions:
        primary = condition["summary"]["donor_minus_correct"]["history_windows"][span_key]
        full = condition["summary"]["donor_minus_correct"]["full_answer"]
        w1 = condition["summary"]["donor_minus_correct"]["history_windows"]["1"]
        correct_full = condition["summary"]["correct_memory"]["full_answer"]
        correct_w1 = condition["summary"]["correct_memory"]["history_windows"]["1"]
        w1_distribution = effect_distribution(condition_window_effects(condition, 1))
        primary_distribution = effect_distribution(
            condition_window_effects(condition, span_tokens)
        )
        stable_memory_signal = bool(
            w1_distribution["median"] > 0.0
            and w1_distribution["trimmed_mean"] > 0.0
            and float(w1["positive_row_fraction"]) >= 0.625
        )
        correct_history_quality_passed = bool(
            float(correct_full["token_weighted_ce"]) <= correct_full_ce_ceiling
            and float(correct_w1["token_weighted_ce"]) <= correct_w1_ce_ceiling
        )
        selection_eligible = bool(
            stable_memory_signal and correct_history_quality_passed
        )
        settings = condition["effective_fusion_settings"]
        effective_gains = [
            float(item["memory_fusion_residual_gain_effective"])
            for item in settings
            if item["memory_fusion_residual_gain_effective"] is not None
        ]
        ranked.append(
            {
                "condition": condition["name"],
                "placements": sorted(
                    {item["memory_fusion_placement"] for item in settings}
                ),
                "fusion_modes": sorted(
                    {item["memory_fusion_mode"] for item in settings}
                ),
                "mean_effective_residual_gain": (
                    None
                    if not effective_gains
                    else statistics.fmean(effective_gains)
                ),
                "primary_history_span_tokens": span_tokens,
                "primary_gap": primary["token_weighted_ce_effect"],
                "primary_positive_fraction": primary["positive_row_fraction"],
                "primary_distribution": primary_distribution,
                "w1_gap": w1["token_weighted_ce_effect"],
                "w1_positive_fraction": w1["positive_row_fraction"],
                "w1_distribution": w1_distribution,
                "full_answer_gap": full["token_weighted_ce_effect"],
                "correct_full_answer_ce": correct_full["token_weighted_ce"],
                "correct_w1_ce": correct_w1["token_weighted_ce"],
                "correct_history_quality_constraint": {
                    "native_correct_full_answer_ce": native_correct_full_ce,
                    "correct_full_answer_ce_ratio_ceiling": (
                        CORRECT_FULL_CE_NATIVE_RATIO_CEILING
                    ),
                    "correct_full_answer_ce_ceiling": correct_full_ce_ceiling,
                    "native_correct_w1_ce": native_correct_w1_ce,
                    "correct_w1_ce_delta_ceiling": (
                        CORRECT_W1_CE_NATIVE_DELTA_CEILING
                    ),
                    "correct_w1_ce_ceiling": correct_w1_ce_ceiling,
                    "passed": correct_history_quality_passed,
                },
                "stable_memory_signal": stable_memory_signal,
                "selection_eligible": selection_eligible,
            }
        )
    ordered = sorted(
        ranked,
        key=lambda item: (
            not bool(item["selection_eligible"]),
            not bool(item["correct_history_quality_constraint"]["passed"]),
            not bool(item["stable_memory_signal"]),
            -float(item["w1_distribution"]["median"]),
            -float(item["w1_distribution"]["trimmed_mean"]),
            -float(item["w1_positive_fraction"]),
            -float(item["w1_gap"]),
            float(item["correct_full_answer_ce"]),
            str(item["condition"]),
        ),
    )
    return [{"rank": rank, **item} for rank, item in enumerate(ordered, start=1)]


def run_condition_sweep(
    *,
    model,
    modules: list[tuple[str, torch.nn.Module]],
    native_settings: dict[str, dict[str, Any]],
    conditions: list[dict[str, Any]],
    tokenized: Dataset,
    source_rows: list[dict[str, Any]],
    donors: list[int],
    device: str,
    primary_span_tokens: int,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    results: list[dict[str, Any]] = []
    seen_snapshot_banks: list[list[dict[str, torch.Tensor]]] = []
    totals = {
        "condition_count": 0,
        "writer_prime_count": 0,
        "correct_replay_count": 0,
        "donor_replay_count": 0,
        "replay_count": 0,
    }
    expected_rows = len(tokenized)
    try:
        for condition in conditions:
            condition_started = time.time()
            print(f"condition {condition['name']}", flush=True)
            restore_fusion_settings(modules, native_settings)
            apply_condition(modules, condition)
            reset_runtime(model, write_enabled=True)
            current_settings = effective_fusion_settings(modules)

            snapshots = prime_writer_snapshots(
                model=model,
                tokenized=tokenized,
                modules=modules,
                device=device,
            )
            if any(snapshots is prior for prior in seen_snapshot_banks):
                raise RuntimeError(
                    f"Condition {condition['name']} reused a writer snapshot bank object"
                )
            seen_snapshot_banks.append(snapshots)
            snapshot_digest = snapshot_bank_sha256(snapshots)

            rows, summary, replay_counts = evaluate_condition(
                model=model,
                tokenized=tokenized,
                source_rows=source_rows,
                donors=donors,
                snapshots=snapshots,
                device=device,
                primary_span_tokens=primary_span_tokens,
            )
            actual_prime_count = len(snapshots)
            expected_replay_count = expected_rows * 2
            if actual_prime_count != expected_rows:
                raise RuntimeError(
                    f"Condition {condition['name']} prime count mismatch: "
                    f"expected={expected_rows} actual={actual_prime_count}"
                )
            if replay_counts != {
                "correct_replay_count": expected_rows,
                "donor_replay_count": expected_rows,
                "replay_count": expected_replay_count,
            }:
                raise RuntimeError(
                    f"Condition {condition['name']} replay count mismatch: "
                    f"expected={expected_replay_count} actual={replay_counts}"
                )
            results.append(
                {
                    **condition,
                    "elapsed_seconds": time.time() - condition_started,
                    "effective_fusion_settings": current_settings,
                    "writer_snapshot_scope": "condition_local_fresh_prime",
                    "writer_snapshot_count": actual_prime_count,
                    "writer_prime_count": actual_prime_count,
                    "writer_snapshot_bank_sha256": snapshot_digest,
                    **replay_counts,
                    "summary": summary,
                    "rows": rows,
                }
            )
            totals["condition_count"] += 1
            totals["writer_prime_count"] += actual_prime_count
            totals["correct_replay_count"] += replay_counts[
                "correct_replay_count"
            ]
            totals["donor_replay_count"] += replay_counts["donor_replay_count"]
            totals["replay_count"] += replay_counts["replay_count"]
    finally:
        restore_fusion_settings(modules, native_settings)
        reset_runtime(model, write_enabled=True)

    expected_totals = {
        "condition_count": len(conditions),
        "writer_prime_count": len(conditions) * expected_rows,
        "correct_replay_count": len(conditions) * expected_rows,
        "donor_replay_count": len(conditions) * expected_rows,
        "replay_count": len(conditions) * expected_rows * 2,
    }
    if totals != expected_totals:
        raise RuntimeError(
            f"Condition sweep totals mismatch: expected={expected_totals} actual={totals}"
        )
    return results, totals


def main() -> None:
    args = parse_args()
    started_at = time.time()
    checkpoint = args.checkpoint.expanduser().resolve()
    tokenized_path = args.tokenized_dataset.expanduser().resolve()
    source_path = args.source_jsonl.expanduser().resolve()
    output_path = (
        default_output_path(checkpoint)
        if args.output is None
        else args.output.expanduser().resolve()
    )
    condition_names = parse_condition_names(args.conditions)
    if "native" not in condition_names:
        raise ValueError("--conditions must include native as the quality reference")
    conditions = initial_condition_screen(condition_names)
    if args.expected_row_count <= 0:
        raise ValueError("--expected-row-count must be positive")

    protocol = load_protocol(checkpoint)
    tokenized: Dataset = load_from_disk(str(tokenized_path))
    if len(tokenized) != args.expected_row_count:
        raise RuntimeError(
            f"Expected {args.expected_row_count} rows, found {len(tokenized)}"
        )
    source_rows = read_jsonl(source_path)
    ready_metadata = validate_artifacts(
        checkpoint=checkpoint,
        tokenized=tokenized,
        tokenized_path=tokenized_path,
        source_path=source_path,
        source_rows=source_rows,
        protocol=protocol,
    )
    donors, pairing = load_pairing_donors(
        checkpoint,
        split_name=str(protocol.get("dataset_split", "train")),
        row_count=len(tokenized),
        fallback_seed=20260724,
        tokenized=tokenized,
    )
    if pairing["source"] != "checkpoint_pairing_manifest":
        raise FileNotFoundError("Exact checkpoint pairing manifest is required")
    pairing_integrity = validate_pairing_manifest_integrity(
        checkpoint=checkpoint,
        protocol=protocol,
        tokenized=tokenized,
        split_name=str(protocol.get("dataset_split", "train")),
        donors=donors,
    )

    model, _ = load_model_and_tokenizer(
        base_model=str(args.base_model.expanduser().resolve()),
        device=args.device,
        dtype=args.dtype,
        attn_implementation=args.attn_implementation,
        delta_mem_root=args.delta_mem_root,
        memory_dir=checkpoint,
    )
    model.eval()
    modules = validate_modules(list(iter_delta_mem_modules(model)), args.expected_layer_count)
    native_settings = capture_fusion_settings(modules)
    saved_adapter_config = json.loads(
        (checkpoint / "delta_mem_config.json").read_text(encoding="utf-8")
    )
    results, execution_counts = run_condition_sweep(
        model=model,
        modules=modules,
        native_settings=native_settings,
        conditions=conditions,
        tokenized=tokenized,
        source_rows=source_rows,
        donors=donors,
        device=args.device,
        primary_span_tokens=args.history_span_tokens,
    )
    digests_to_conditions: dict[str, list[str]] = {}
    for condition in results:
        digests_to_conditions.setdefault(
            condition["writer_snapshot_bank_sha256"], []
        ).append(condition["name"])
    digest_collisions = {
        digest: names
        for digest, names in digests_to_conditions.items()
        if len(names) > 1
    }
    ranking = rank_conditions(results, args.history_span_tokens)

    payload = {
        "schema": "rwkv_ms_fresh_prime_residual_hybrid_screen.v2",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "elapsed_seconds": time.time() - started_at,
        "provenance": {
            "script": str(Path(__file__).resolve()),
            "script_sha256": sha256_file(Path(__file__).resolve()),
            "base_model": str(args.base_model.expanduser().resolve()),
            "checkpoint": str(checkpoint),
            "checkpoint_files": checkpoint_file_provenance(checkpoint),
            "tokenized_dataset": str(tokenized_path),
            "tokenized_fingerprint": getattr(tokenized, "_fingerprint", None),
            "tokenized_ready_metadata": ready_metadata,
            "source_jsonl": str(source_path),
            "source_jsonl_sha256": sha256_file(source_path),
            "pairing": pairing,
            "pairing_integrity": pairing_integrity,
            "exact_donor_indices": donors,
            "training_protocol": protocol,
            "saved_adapter_config": saved_adapter_config,
            "native_fusion_settings": serializable_fusion_settings(
                native_settings
            ),
            "layer_module_names": [name for name, _ in modules],
            "layer_indices": [int(module.layer_idx) for _, module in modules],
            "expected_layer_count": args.expected_layer_count,
            "expected_row_count": args.expected_row_count,
            "model_load_count": 1,
            "writer_snapshot_policy": "fresh_condition_local_prime_no_cache",
            "snapshot_digest_collisions": digest_collisions,
            "execution_counts": execution_counts,
            "device": args.device,
            "dtype": args.dtype,
            "attn_implementation": args.attn_implementation,
            "torch_version": torch.__version__,
            "python_version": platform.python_version(),
            "command_arguments": vars(args)
            | {
                "base_model": str(args.base_model),
                "checkpoint": str(args.checkpoint),
                "tokenized_dataset": str(args.tokenized_dataset),
                "source_jsonl": str(args.source_jsonl),
                "output": None if args.output is None else str(args.output),
                "delta_mem_root": str(args.delta_mem_root),
            },
        },
        "selection_protocol": {
            "alignment": "supervised_token_ordinal_v1",
            "primary_history_span_tokens": args.history_span_tokens,
            "ranking_priority": (
                "selection eligibility (stable W1 plus correct-history quality), then "
                "quality pass, stable W1 signal, W1 median, W1 10%-trimmed mean, "
                "positive fraction, mean W1 gap, and lower correct-history full CE"
            ),
            "stable_memory_signal": (
                "W1 median > 0, W1 10%-trimmed mean > 0, and W1 positive row "
                "fraction >= 0.625"
            ),
            "correct_history_quality_constraint": {
                "full_answer_ce": (
                    "Must be no more than 1.25 times native correct-history full CE."
                ),
                "w1_ce": (
                    "Must be no more than 0.5 CE above native correct-history W1 CE."
                ),
                "interpretation": (
                    "This rejects noisy donor separation that destroys the answer under "
                    "correct memory; it is not a KL-to-base preservation constraint."
                ),
            },
            "note": (
                "W1 is the pure pre-teacher-forcing memory token. Correct-history full CE "
                "is a task-quality metric, not a KL-to-base preservation objective. Each "
                "condition writes its own histories because fusion changes downstream writer "
                "hidden states."
            ),
        },
        "condition_definitions": conditions,
        "conditions": results,
        "ranking": ranking,
    }
    write_json(output_path, payload)
    print(json.dumps(ranking, indent=2, sort_keys=True), flush=True)
    print(f"wrote {output_path}", flush=True)


if __name__ == "__main__":
    main()
