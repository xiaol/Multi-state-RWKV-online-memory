#!/usr/bin/env python3
"""Rank RWKV-MS layers by dataset-wide causal online-state swaps."""

from __future__ import annotations

import argparse
import json
import math
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

from deltamem.core.delta import get_delta_mem_online_state, iter_delta_mem_modules
from experiments.rethinking_rwkv_ms_gemma.common import (
    load_model_and_tokenizer,
    read_jsonl,
    write_json,
)
from experiments.rethinking_rwkv_ms_gemma.diagnose_memory_representation import (
    load_pairing_donors,
    module_online_state_keys,
    replace_module_online_state,
    replay_fixed_target,
)
from experiments.rethinking_rwkv_ms_gemma.eval_episode_memory_ce import (
    load_protocol,
    prime_write,
    reset_runtime,
    sha256_file,
    source_identity,
    validate_artifacts,
)


HISTORY_SPAN_TOKEN_COUNTS = (1, 8, 16, 32)
DEFAULT_PRIMARY_HISTORY_SPAN_TOKENS = 1
DEFAULT_PROGRESS_LAYER_INTERVAL = 6
SUPPORTED_STATE_SWAP_PLACEMENTS = frozenset(
    {"attention_output", "post_attention_residual_hybrid"}
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-model", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--tokenized-dataset", type=Path, required=True)
    parser.add_argument("--source-jsonl", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--shuffle-seed", type=int, default=20260724)
    parser.add_argument("--symmetric-top-k", type=int, default=6)
    parser.add_argument(
        "--baseline-only",
        action="store_true",
        help=(
            "Evaluate exact correct/donor full-answer and W1/W8 gaps without "
            "running any per-layer state swaps."
        ),
    )
    parser.add_argument(
        "--history-span-tokens",
        type=int,
        choices=HISTORY_SPAN_TOKEN_COUNTS,
        default=DEFAULT_PRIMARY_HISTORY_SPAN_TOKENS,
    )
    parser.add_argument(
        "--unaligned-token-policy",
        choices=("error", "full_answer"),
        default="error",
    )
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
        raise ValueError(
            "--output is required unless checkpoint is RUN_ROOT/trainer/checkpoint-N"
        )
    return (
        checkpoint.parent.parent
        / "layer_state_swap_diagnostic"
        / f"{checkpoint.name}_all_rows_layer_state_swaps.json"
    )


def supervised_token_trace(row: dict[str, Any]) -> list[dict[str, int]]:
    input_ids = row["input_ids"]
    labels = row["labels"]
    attention_mask = row["attention_mask"]
    if not (len(input_ids) == len(labels) == len(attention_mask)):
        raise ValueError("Input IDs, labels, and attention mask must have equal lengths")
    if labels and labels[0] != -100 and attention_mask[0] != 0:
        raise ValueError("A supervised token at position zero has no causal predictor")

    trace: list[dict[str, int]] = []
    for label_position in range(1, len(labels)):
        label = int(labels[label_position])
        if label == -100 or int(attention_mask[label_position]) == 0:
            continue
        predictor_position = label_position - 1
        if int(attention_mask[predictor_position]) == 0:
            raise ValueError(
                f"Supervised label at {label_position} has a masked predictor position"
            )
        if int(input_ids[label_position]) != label:
            raise ValueError(
                "Supervised labels must match their input token IDs: "
                f"position={label_position} input={input_ids[label_position]} label={label}"
            )
        trace.append(
            {
                "ordinal": len(trace),
                "token_id": label,
                "label_position": label_position,
                "predictor_position": predictor_position,
                "predictor_token_id": int(input_ids[predictor_position]),
            }
        )
    if not trace:
        raise ValueError("A state-swap target must have supervised answer tokens")
    return trace


def history_token_selection(
    target_row: dict[str, Any],
    donor_row: dict[str, Any],
    *,
    primary_span_tokens: int,
    unaligned_policy: str,
) -> dict[str, Any]:
    if primary_span_tokens not in HISTORY_SPAN_TOKEN_COUNTS:
        raise ValueError(
            f"primary_span_tokens must be one of {HISTORY_SPAN_TOKEN_COUNTS}"
        )
    if unaligned_policy not in {"error", "full_answer"}:
        raise ValueError(f"Unsupported unaligned token policy: {unaligned_policy}")

    target_trace = supervised_token_trace(target_row)
    donor_trace = supervised_token_trace(donor_row)
    target_ids = [item["token_id"] for item in target_trace]
    donor_ids = [item["token_id"] for item in donor_trace]
    alignment_error: str | None = None
    first_history_ordinal: int | None = None
    causal_prefix_identical: bool | None = None
    if len(target_ids) != len(donor_ids):
        alignment_error = (
            "supervised token counts differ: "
            f"target={len(target_ids)} donor={len(donor_ids)}"
        )
    elif [item["label_position"] for item in target_trace] != [
        item["label_position"] for item in donor_trace
    ] or [item["predictor_position"] for item in target_trace] != [
        item["predictor_position"] for item in donor_trace
    ]:
        alignment_error = "supervised label or causal predictor positions differ"
    else:
        first_history_ordinal = next(
            (
                ordinal
                for ordinal, (target_id, donor_id) in enumerate(zip(target_ids, donor_ids))
                if target_id != donor_id
            ),
            None,
        )
        if first_history_ordinal is None:
            alignment_error = "target and donor supervised token IDs are identical"
        else:
            first_target = target_trace[first_history_ordinal]
            first_donor = donor_trace[first_history_ordinal]
            target_prefix = [
                int(token)
                for token in target_row["input_ids"][: first_target["label_position"]]
            ]
            donor_prefix = [
                int(token)
                for token in donor_row["input_ids"][: first_donor["label_position"]]
            ]
            causal_prefix_identical = target_prefix == donor_prefix
            if not causal_prefix_identical:
                alignment_error = (
                    "causal source prefixes differ before the first history token"
                )
                first_history_ordinal = None

    fallback_used = alignment_error is not None
    if fallback_used and unaligned_policy == "error":
        raise ValueError(f"Cannot identify history-dependent target tokens: {alignment_error}")

    windows: dict[str, dict[str, Any]] = {}
    for requested_count in HISTORY_SPAN_TOKEN_COUNTS:
        if fallback_used:
            ordinals = list(range(len(target_trace)))
        else:
            assert first_history_ordinal is not None
            end = min(first_history_ordinal + requested_count, len(target_trace))
            ordinals = list(range(first_history_ordinal, end))
        windows[str(requested_count)] = {
            "requested_token_count": requested_count,
            "actual_token_count": len(ordinals),
            "supervised_ordinals": ordinals,
            "target_token_ids": [target_ids[ordinal] for ordinal in ordinals],
            "target_label_positions": [
                target_trace[ordinal]["label_position"] for ordinal in ordinals
            ],
            "target_predictor_positions": [
                target_trace[ordinal]["predictor_position"] for ordinal in ordinals
            ],
            "selection_kind": (
                "full_answer_fallback" if fallback_used else "first_history_window"
            ),
        }

    return {
        "alignment": "supervised_token_ordinal_v1",
        "status": "full_answer_fallback" if fallback_used else "aligned",
        "fallback_used": fallback_used,
        "fallback_reason": alignment_error,
        "first_history_ordinal": first_history_ordinal,
        "first_history_label_position": (
            None
            if first_history_ordinal is None
            else target_trace[first_history_ordinal]["label_position"]
        ),
        "first_history_predictor_position": (
            None
            if first_history_ordinal is None
            else target_trace[first_history_ordinal]["predictor_position"]
        ),
        "causal_prefix_identical": causal_prefix_identical,
        "primary_history_span_tokens": primary_span_tokens,
        "primary_window_key": str(primary_span_tokens),
        "target_supervised_token_count": len(target_trace),
        "donor_supervised_token_count": len(donor_trace),
        "target_supervised_token_ids": target_ids,
        "donor_supervised_token_ids": donor_ids,
        "target_trace": target_trace,
        "donor_trace": donor_trace,
        "windows": windows,
    }


def validate_layer_modules(
    modules: list[tuple[str, torch.nn.Module]],
    expected_layer_count: int,
) -> list[tuple[str, torch.nn.Module]]:
    if expected_layer_count <= 0:
        raise ValueError("--expected-layer-count must be positive")
    if len(modules) != expected_layer_count:
        raise RuntimeError(
            "State-swap diagnostic requires every expected layer: "
            f"expected={expected_layer_count} actual={len(modules)}"
        )
    ordered = sorted(modules, key=lambda item: int(item[1].layer_idx))
    layer_indices = [int(module.layer_idx) for _, module in ordered]
    if layer_indices != list(range(expected_layer_count)):
        raise RuntimeError(
            "Attached Delta-Mem layers must cover the complete zero-based layer range: "
            f"{layer_indices}"
        )
    unsupported = {
        name: {
            "memory_backend": module.memory_backend,
            "memory_fusion_placement": module.memory_fusion_placement,
        }
        for name, module in ordered
        if module.memory_backend != "rwkv_ms"
        or module.memory_fusion_placement not in SUPPORTED_STATE_SWAP_PLACEMENTS
    }
    if unsupported:
        raise ValueError(
            "State swaps require RWKV-MS modules with a supported saved fusion placement: "
            f"{unsupported}"
        )
    return ordered


def expected_snapshot_keys(
    modules: list[tuple[str, torch.nn.Module]],
) -> set[str]:
    return {
        key
        for name, _ in modules
        for key in module_online_state_keys(name)
    }


def validate_writer_snapshots(
    snapshots: list[dict[str, torch.Tensor]],
    modules: list[tuple[str, torch.nn.Module]],
) -> None:
    required_keys = expected_snapshot_keys(modules)
    for row_index, snapshot in enumerate(snapshots):
        actual_keys = set(snapshot)
        if actual_keys != required_keys:
            raise RuntimeError(
                f"Writer snapshot {row_index} has unexpected keys: "
                f"missing={sorted(required_keys.difference(actual_keys))} "
                f"extra={sorted(actual_keys.difference(required_keys))}"
            )


def prime_writer_snapshots(
    *,
    model,
    tokenized: Dataset,
    modules: list[tuple[str, torch.nn.Module]],
    device: str,
) -> list[dict[str, torch.Tensor]]:
    snapshots: list[dict[str, torch.Tensor]] = []
    with torch.inference_mode():
        for row_index in range(len(tokenized)):
            reset_runtime(model, write_enabled=True)
            prime_write(model, tokenized[row_index], device)
            snapshots.append(get_delta_mem_online_state(model))
            print(
                f"writer {row_index + 1:02d}/{len(tokenized)} "
                f"tokens={len(tokenized[row_index]['write_input_ids'])}",
                flush=True,
            )
    validate_writer_snapshots(snapshots, modules)
    return snapshots


def replay_with_token_nll(
    *,
    model,
    target_row: dict[str, Any],
    online_state: dict[str, torch.Tensor],
    device: str,
    expected_token_count: int,
) -> dict[str, Any]:
    metrics, _ = replay_fixed_target(
        model=model,
        target_row=target_row,
        online_state=online_state,
        device=device,
        capture=None,
        include_token_nll=True,
    )
    token_nll = [float(value) for value in metrics.pop("token_nll")]
    if len(token_nll) != expected_token_count:
        raise RuntimeError(
            "Replay token NLL count does not match supervised-token alignment: "
            f"expected={expected_token_count} actual={len(token_nll)}"
        )
    metrics["token_nll"] = token_nll
    return metrics


def condition_metrics(
    replay: dict[str, Any],
    selection: dict[str, Any],
) -> dict[str, Any]:
    token_nll = replay["token_nll"]
    history_windows: dict[str, dict[str, Any]] = {}
    for key, window in selection["windows"].items():
        ordinals = window["supervised_ordinals"]
        selected = [token_nll[ordinal] for ordinal in ordinals]
        nll_sum = math.fsum(selected)
        history_windows[key] = {
            "requested_token_count": int(window["requested_token_count"]),
            "token_count": len(selected),
            "nll_sum": nll_sum,
            "ce": nll_sum / len(selected),
            "token_nll": selected,
        }
    return {
        "full_answer": {
            "token_count": int(replay["token_count"]),
            "nll_sum": float(replay["nll_sum"]),
            "ce": float(replay["ce"]),
            "token_nll": list(token_nll),
        },
        "history_windows": history_windows,
    }


def _scope_effect(
    baseline: dict[str, Any],
    intervention: dict[str, Any],
    *,
    intervention_improves_when_lower: bool,
) -> dict[str, Any]:
    if int(baseline["token_count"]) != int(intervention["token_count"]):
        raise RuntimeError("Baseline and intervention token counts differ")
    sign = 1.0 if intervention_improves_when_lower else -1.0
    ce_effect = sign * (float(baseline["ce"]) - float(intervention["ce"]))
    nll_effect = sign * (
        float(baseline["nll_sum"]) - float(intervention["nll_sum"])
    )
    return {
        "token_count": int(baseline["token_count"]),
        "baseline_ce": float(baseline["ce"]),
        "intervention_ce": float(intervention["ce"]),
        "ce_effect": ce_effect,
        "nll_effect": nll_effect,
        "positive": bool(ce_effect > 0.0),
    }


def directional_row_effect(
    baseline: dict[str, Any],
    intervention: dict[str, Any],
    *,
    intervention_improves_when_lower: bool,
) -> dict[str, Any]:
    return {
        "full_answer": _scope_effect(
            baseline["full_answer"],
            intervention["full_answer"],
            intervention_improves_when_lower=intervention_improves_when_lower,
        ),
        "history_windows": {
            key: _scope_effect(
                baseline["history_windows"][key],
                intervention["history_windows"][key],
                intervention_improves_when_lower=intervention_improves_when_lower,
            )
            for key in baseline["history_windows"]
        },
    }


def summarize_effect_scopes(scopes: list[dict[str, Any]]) -> dict[str, Any]:
    if not scopes:
        raise ValueError("Cannot summarize empty state-swap effects")
    ce_effects = [float(scope["ce_effect"]) for scope in scopes]
    token_count = sum(int(scope["token_count"]) for scope in scopes)
    nll_effect = math.fsum(float(scope["nll_effect"]) for scope in scopes)
    return {
        "row_count": len(scopes),
        "token_count": token_count,
        "nll_effect_sum": nll_effect,
        "mean_ce_effect": statistics.fmean(ce_effects),
        "median_ce_effect": statistics.median(ce_effects),
        "min_ce_effect": min(ce_effects),
        "max_ce_effect": max(ce_effects),
        "token_weighted_ce_effect": nll_effect / token_count,
        "positive_row_count": sum(bool(scope["positive"]) for scope in scopes),
        "positive_row_fraction": (
            sum(bool(scope["positive"]) for scope in scopes) / len(scopes)
        ),
    }


def summarize_direction(row_effects: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "full_answer": summarize_effect_scopes(
            [row["full_answer"] for row in row_effects]
        ),
        "history_windows": {
            key: summarize_effect_scopes(
                [row["history_windows"][key] for row in row_effects]
            )
            for key in row_effects[0]["history_windows"]
        },
    }


def summarize_condition_rows(
    baselines: list[dict[str, Any]],
    condition_name: str,
) -> dict[str, Any]:
    def summarize(scopes: list[dict[str, Any]]) -> dict[str, Any]:
        values = [float(scope["ce"]) for scope in scopes]
        token_count = sum(int(scope["token_count"]) for scope in scopes)
        nll_sum = math.fsum(float(scope["nll_sum"]) for scope in scopes)
        return {
            "row_count": len(scopes),
            "token_count": token_count,
            "nll_sum": nll_sum,
            "token_weighted_ce": nll_sum / token_count,
            "mean_row_ce": statistics.fmean(values),
            "median_row_ce": statistics.median(values),
            "min_row_ce": min(values),
            "max_row_ce": max(values),
        }

    return {
        "full_answer": summarize(
            [baseline[condition_name]["full_answer"] for baseline in baselines]
        ),
        "history_windows": {
            key: summarize(
                [
                    baseline[condition_name]["history_windows"][key]
                    for baseline in baselines
                ]
            )
            for key in baselines[0][condition_name]["history_windows"]
        },
    }


def baseline_gap_summary(baselines: list[dict[str, Any]]) -> dict[str, Any]:
    return summarize_direction([baseline["donor_minus_correct"] for baseline in baselines])


def evaluate_exact_pair_baselines(
    *,
    model,
    tokenized: Dataset,
    source_rows: list[dict[str, Any]],
    donors: list[int],
    snapshots: list[dict[str, torch.Tensor]],
    device: str,
    primary_span_tokens: int,
    unaligned_policy: str,
) -> list[dict[str, Any]]:
    baselines: list[dict[str, Any]] = []
    with torch.inference_mode():
        for row_index in range(len(tokenized)):
            donor_index = donors[row_index]
            target_row = tokenized[row_index]
            selection = history_token_selection(
                target_row,
                tokenized[donor_index],
                primary_span_tokens=primary_span_tokens,
                unaligned_policy=unaligned_policy,
            )
            expected_tokens = int(selection["target_supervised_token_count"])
            correct_replay = replay_with_token_nll(
                model=model,
                target_row=target_row,
                online_state=snapshots[row_index],
                device=device,
                expected_token_count=expected_tokens,
            )
            donor_replay = replay_with_token_nll(
                model=model,
                target_row=target_row,
                online_state=snapshots[donor_index],
                device=device,
                expected_token_count=expected_tokens,
            )
            correct = condition_metrics(correct_replay, selection)
            donor = condition_metrics(donor_replay, selection)
            donor_minus_correct = directional_row_effect(
                correct,
                donor,
                intervention_improves_when_lower=False,
            )
            baselines.append(
                {
                    "row_index": row_index,
                    "target": source_identity(source_rows[row_index], row_index),
                    "exact_donor": source_identity(source_rows[donor_index], donor_index),
                    "donor_row_index": donor_index,
                    "target_write_tokens": len(target_row["write_input_ids"]),
                    "donor_write_tokens": len(tokenized[donor_index]["write_input_ids"]),
                    "read_tokens": len(target_row["input_ids"]),
                    "history_token_selection": selection,
                    "correct_memory": correct,
                    "exact_donor_memory": donor,
                    "donor_minus_correct": donor_minus_correct,
                }
            )
            primary_key = str(primary_span_tokens)
            primary_gap = donor_minus_correct["history_windows"][primary_key]["ce_effect"]
            print(
                f"row {row_index + 1:02d}/{len(tokenized)} baseline "
                f"correct_full={correct['full_answer']['ce']:.6f} "
                f"donor_full={donor['full_answer']['ce']:.6f} "
                f"history{primary_key}_gap={primary_gap:.6f} donor={donor_index}",
                flush=True,
            )
    return baselines


def evaluate_donor_to_correct_swaps(
    *,
    model,
    tokenized: Dataset,
    source_rows: list[dict[str, Any]],
    donors: list[int],
    snapshots: list[dict[str, torch.Tensor]],
    modules: list[tuple[str, torch.nn.Module]],
    device: str,
    primary_span_tokens: int,
    unaligned_policy: str,
    progress_layer_interval: int = DEFAULT_PROGRESS_LAYER_INTERVAL,
) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
    layer_results = {
        name: {
            "module_name": name,
            "layer_index": int(module.layer_idx),
            "donor_to_correct": {
                "definition": (
                    "Exact donor state baseline minus CE after replacing this layer's "
                    "three RWKV-MS online-state keys with the target writer state."
                ),
                "row_effects": [],
            },
            "correct_to_donor": None,
            "bidirectional": None,
        }
        for name, module in modules
    }
    baselines = evaluate_exact_pair_baselines(
        model=model,
        tokenized=tokenized,
        source_rows=source_rows,
        donors=donors,
        snapshots=snapshots,
        device=device,
        primary_span_tokens=primary_span_tokens,
        unaligned_policy=unaligned_policy,
    )
    with torch.inference_mode():
        for row_index, baseline in enumerate(baselines):
            donor_index = int(baseline["donor_row_index"])
            target_row = tokenized[row_index]
            selection = baseline["history_token_selection"]
            expected_tokens = int(selection["target_supervised_token_count"])
            donor = baseline["exact_donor_memory"]
            primary_key = str(primary_span_tokens)
            for module_position, (name, _) in enumerate(modules, start=1):
                swapped_replay = replay_with_token_nll(
                    model=model,
                    target_row=target_row,
                    online_state=replace_module_online_state(
                        snapshots[donor_index],
                        snapshots[row_index],
                        name,
                    ),
                    device=device,
                    expected_token_count=expected_tokens,
                )
                swapped = condition_metrics(swapped_replay, selection)
                effect = directional_row_effect(
                    donor,
                    swapped,
                    intervention_improves_when_lower=True,
                )
                layer_results[name]["donor_to_correct"]["row_effects"].append(
                    {
                        "row_index": row_index,
                        "donor_row_index": donor_index,
                        "intervention_token_nll": swapped["full_answer"]["token_nll"],
                        **effect,
                    }
                )
                if (
                    module_position == 1
                    or module_position % progress_layer_interval == 0
                    or module_position == len(modules)
                ):
                    primary_effect = effect["history_windows"][primary_key]["ce_effect"]
                    print(
                        f"row {row_index + 1:02d}/{len(tokenized)} "
                        f"forward_layer {module_position:02d}/{len(modules)} "
                        f"layer={layer_results[name]['layer_index']} "
                        f"history{primary_key}_gain={primary_effect:.6f} "
                        f"full_gain={effect['full_answer']['ce_effect']:.6f}",
                        flush=True,
                    )

    for layer in layer_results.values():
        direction = layer["donor_to_correct"]
        direction["aggregate"] = summarize_direction(direction["row_effects"])
    return baselines, layer_results


def empty_forward_rankings() -> dict[str, list[dict[str, Any]]]:
    return {
        "by_primary_token_weighted_ce_gain": [],
        "by_primary_positive_row_fraction": [],
    }


def run_diagnostic_evaluation(
    *,
    model,
    tokenized: Dataset,
    source_rows: list[dict[str, Any]],
    donors: list[int],
    snapshots: list[dict[str, torch.Tensor]],
    modules: list[tuple[str, torch.nn.Module]],
    device: str,
    primary_span_tokens: int,
    unaligned_policy: str,
    symmetric_top_k: int,
    baseline_only: bool,
) -> tuple[
    list[dict[str, Any]],
    dict[str, dict[str, Any]],
    dict[str, list[dict[str, Any]]],
    list[str],
    list[dict[str, Any]],
]:
    if baseline_only:
        baselines = evaluate_exact_pair_baselines(
            model=model,
            tokenized=tokenized,
            source_rows=source_rows,
            donors=donors,
            snapshots=snapshots,
            device=device,
            primary_span_tokens=primary_span_tokens,
            unaligned_policy=unaligned_policy,
        )
        return baselines, {}, empty_forward_rankings(), [], []

    baselines, layer_results = evaluate_donor_to_correct_swaps(
        model=model,
        tokenized=tokenized,
        source_rows=source_rows,
        donors=donors,
        snapshots=snapshots,
        modules=modules,
        device=device,
        primary_span_tokens=primary_span_tokens,
        unaligned_policy=unaligned_policy,
    )
    forward_rankings = rank_forward_layers(layer_results, primary_span_tokens)
    selected_module_names = [
        entry["module_name"]
        for entry in forward_rankings["by_primary_token_weighted_ce_gain"][
            : min(symmetric_top_k, len(modules))
        ]
    ]
    evaluate_correct_to_donor_swaps(
        model=model,
        tokenized=tokenized,
        donors=donors,
        snapshots=snapshots,
        baselines=baselines,
        layer_results=layer_results,
        selected_module_names=selected_module_names,
        device=device,
        primary_span_tokens=primary_span_tokens,
    )
    bidirectional_ranking = rank_bidirectional_layers(
        layer_results,
        selected_module_names,
        primary_span_tokens,
    )
    return (
        baselines,
        layer_results,
        forward_rankings,
        selected_module_names,
        bidirectional_ranking,
    )


def rank_forward_layers(
    layer_results: dict[str, dict[str, Any]],
    primary_span_tokens: int,
) -> dict[str, list[dict[str, Any]]]:
    primary_key = str(primary_span_tokens)
    entries = []
    for name, layer in layer_results.items():
        primary = layer["donor_to_correct"]["aggregate"]["history_windows"][primary_key]
        full = layer["donor_to_correct"]["aggregate"]["full_answer"]
        entries.append(
            {
                "module_name": name,
                "layer_index": int(layer["layer_index"]),
                "primary_history_span_tokens": primary_span_tokens,
                "token_weighted_ce_gain": float(primary["token_weighted_ce_effect"]),
                "positive_row_count": int(primary["positive_row_count"]),
                "positive_row_fraction": float(primary["positive_row_fraction"]),
                "mean_row_ce_gain": float(primary["mean_ce_effect"]),
                "full_answer_token_weighted_ce_gain": float(
                    full["token_weighted_ce_effect"]
                ),
            }
        )

    def add_ranks(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
        return [{"rank": rank, **item} for rank, item in enumerate(items, start=1)]

    by_gain = sorted(
        entries,
        key=lambda item: (
            -item["token_weighted_ce_gain"],
            -item["positive_row_fraction"],
            item["layer_index"],
            item["module_name"],
        ),
    )
    by_positive = sorted(
        entries,
        key=lambda item: (
            -item["positive_row_fraction"],
            -item["token_weighted_ce_gain"],
            item["layer_index"],
            item["module_name"],
        ),
    )
    return {
        "by_primary_token_weighted_ce_gain": add_ranks(by_gain),
        "by_primary_positive_row_fraction": add_ranks(by_positive),
    }


def _bidirectional_scope(
    forward: dict[str, Any],
    reverse: dict[str, Any],
) -> dict[str, Any]:
    if int(forward["token_count"]) != int(reverse["token_count"]):
        raise RuntimeError("Forward and reverse swap token counts differ")
    ce_effect = (float(forward["ce_effect"]) + float(reverse["ce_effect"])) * 0.5
    nll_effect = (float(forward["nll_effect"]) + float(reverse["nll_effect"])) * 0.5
    return {
        "token_count": int(forward["token_count"]),
        "donor_to_correct_ce_gain": float(forward["ce_effect"]),
        "correct_to_donor_ce_damage": float(reverse["ce_effect"]),
        "ce_effect": ce_effect,
        "nll_effect": nll_effect,
        "positive": bool(forward["ce_effect"] > 0.0 and reverse["ce_effect"] > 0.0),
    }


def bidirectional_row_effect(
    forward: dict[str, Any],
    reverse: dict[str, Any],
) -> dict[str, Any]:
    return {
        "full_answer": _bidirectional_scope(
            forward["full_answer"],
            reverse["full_answer"],
        ),
        "history_windows": {
            key: _bidirectional_scope(
                forward["history_windows"][key],
                reverse["history_windows"][key],
            )
            for key in forward["history_windows"]
        },
    }


def evaluate_correct_to_donor_swaps(
    *,
    model,
    tokenized: Dataset,
    donors: list[int],
    snapshots: list[dict[str, torch.Tensor]],
    baselines: list[dict[str, Any]],
    layer_results: dict[str, dict[str, Any]],
    selected_module_names: list[str],
    device: str,
    primary_span_tokens: int,
) -> None:
    primary_key = str(primary_span_tokens)
    with torch.inference_mode():
        for row_index in range(len(tokenized)):
            baseline = baselines[row_index]
            donor_index = donors[row_index]
            selection = baseline["history_token_selection"]
            expected_tokens = int(selection["target_supervised_token_count"])
            for selected_position, name in enumerate(selected_module_names, start=1):
                ablated_replay = replay_with_token_nll(
                    model=model,
                    target_row=tokenized[row_index],
                    online_state=replace_module_online_state(
                        snapshots[row_index],
                        snapshots[donor_index],
                        name,
                    ),
                    device=device,
                    expected_token_count=expected_tokens,
                )
                ablated = condition_metrics(ablated_replay, selection)
                reverse = directional_row_effect(
                    baseline["correct_memory"],
                    ablated,
                    intervention_improves_when_lower=False,
                )
                direction = layer_results[name]["correct_to_donor"]
                if direction is None:
                    direction = {
                        "definition": (
                            "Correct state baseline to CE after replacing this layer's three "
                            "RWKV-MS online-state keys with the exact donor writer state; "
                            "positive means damage."
                        ),
                        "row_effects": [],
                    }
                    layer_results[name]["correct_to_donor"] = direction
                direction["row_effects"].append(
                    {
                        "row_index": row_index,
                        "donor_row_index": donor_index,
                        "intervention_token_nll": ablated["full_answer"]["token_nll"],
                        **reverse,
                    }
                )
                if selected_position == 1 or selected_position == len(selected_module_names):
                    effect = reverse["history_windows"][primary_key]["ce_effect"]
                    print(
                        f"row {row_index + 1:02d}/{len(tokenized)} "
                        f"reverse_layer {selected_position:02d}/{len(selected_module_names)} "
                        f"layer={layer_results[name]['layer_index']} "
                        f"history{primary_key}_damage={effect:.6f}",
                        flush=True,
                    )

    for name in selected_module_names:
        reverse_direction = layer_results[name]["correct_to_donor"]
        assert reverse_direction is not None
        reverse_direction["aggregate"] = summarize_direction(
            reverse_direction["row_effects"]
        )
        forward_rows = {
            int(row["row_index"]): row
            for row in layer_results[name]["donor_to_correct"]["row_effects"]
        }
        reverse_rows = {
            int(row["row_index"]): row for row in reverse_direction["row_effects"]
        }
        if set(forward_rows) != set(reverse_rows):
            raise RuntimeError(f"Bidirectional rows do not match for {name}")
        combined_rows = []
        for row_index in sorted(forward_rows):
            forward = forward_rows[row_index]
            reverse = reverse_rows[row_index]
            combined = bidirectional_row_effect(forward, reverse)
            combined_rows.append(
                {
                    "row_index": row_index,
                    "donor_row_index": int(forward["donor_row_index"]),
                    **combined,
                }
            )
        layer_results[name]["bidirectional"] = {
            "definition": (
                "Mean of donor-to-correct CE gain and correct-to-donor CE damage. A row "
                "is bidirectionally positive only when both directional effects are positive."
            ),
            "row_effects": combined_rows,
            "aggregate": summarize_direction(combined_rows),
        }


def rank_bidirectional_layers(
    layer_results: dict[str, dict[str, Any]],
    selected_module_names: list[str],
    primary_span_tokens: int,
) -> list[dict[str, Any]]:
    primary_key = str(primary_span_tokens)
    entries = []
    for name in selected_module_names:
        layer = layer_results[name]
        combined = layer["bidirectional"]["aggregate"]["history_windows"][primary_key]
        forward = layer["donor_to_correct"]["aggregate"]["history_windows"][primary_key]
        reverse = layer["correct_to_donor"]["aggregate"]["history_windows"][primary_key]
        entries.append(
            {
                "module_name": name,
                "layer_index": int(layer["layer_index"]),
                "primary_history_span_tokens": primary_span_tokens,
                "bidirectional_token_weighted_ce_effect": float(
                    combined["token_weighted_ce_effect"]
                ),
                "bidirectional_positive_row_count": int(
                    combined["positive_row_count"]
                ),
                "bidirectional_positive_row_fraction": float(
                    combined["positive_row_fraction"]
                ),
                "donor_to_correct_token_weighted_ce_gain": float(
                    forward["token_weighted_ce_effect"]
                ),
                "correct_to_donor_token_weighted_ce_damage": float(
                    reverse["token_weighted_ce_effect"]
                ),
            }
        )
    ranked = sorted(
        entries,
        key=lambda item: (
            -item["bidirectional_token_weighted_ce_effect"],
            -item["bidirectional_positive_row_fraction"],
            item["layer_index"],
            item["module_name"],
        ),
    )
    return [{"rank": rank, **item} for rank, item in enumerate(ranked, start=1)]


def checkpoint_file_provenance(checkpoint: Path) -> dict[str, dict[str, Any]]:
    provenance = {}
    for filename in (
        "delta_mem_adapter.pt",
        "delta_mem_config.json",
        "training_protocol.json",
        "content_contrast_pairing_manifest.json",
    ):
        path = checkpoint / filename
        if path.is_file():
            provenance[filename] = {
                "path": str(path),
                "size_bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
    return provenance


def main() -> None:
    args = parse_args()
    if not args.baseline_only and args.symmetric_top_k <= 0:
        raise ValueError("--symmetric-top-k must be positive")
    if args.expected_row_count <= 0:
        raise ValueError("--expected-row-count must be positive")
    checkpoint = args.checkpoint.expanduser().resolve()
    tokenized_path = args.tokenized_dataset.expanduser().resolve()
    source_path = args.source_jsonl.expanduser().resolve()
    output_path = (
        default_output_path(checkpoint)
        if args.output is None
        else args.output.expanduser().resolve()
    )
    started_at = time.time()

    protocol = load_protocol(checkpoint)
    tokenized: Dataset = load_from_disk(str(tokenized_path))
    if len(tokenized) != args.expected_row_count:
        raise RuntimeError(
            "State-swap diagnostic requires every expected dataset row: "
            f"expected={args.expected_row_count} actual={len(tokenized)}"
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
    donors, pairing_provenance = load_pairing_donors(
        checkpoint,
        split_name=str(protocol.get("dataset_split", "train")),
        row_count=len(tokenized),
        fallback_seed=args.shuffle_seed,
        tokenized=tokenized,
    )
    if pairing_provenance["source"] != "checkpoint_pairing_manifest":
        raise FileNotFoundError(
            "This diagnostic requires the checkpoint's exact content-contrast pairing manifest"
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
    modules = validate_layer_modules(
        list(iter_delta_mem_modules(model)),
        args.expected_layer_count,
    )
    effective_configs = {
        name: module.delta_config.to_dict() for name, module in modules
    }
    effective_residual_hybrid_gains = {
        name: float(
            module._resolved_memory_fusion_residual_gain(
                device=module.memory_fusion_residual_gain_raw.device,
                dtype=torch.float32,
            )
            .detach()
            .item()
        )
        for name, module in modules
        if module.memory_fusion_placement == "post_attention_residual_hybrid"
    }
    saved_adapter_config = json.loads(
        (checkpoint / "delta_mem_config.json").read_text(encoding="utf-8")
    )

    try:
        snapshots = prime_writer_snapshots(
            model=model,
            tokenized=tokenized,
            modules=modules,
            device=args.device,
        )
        (
            baselines,
            layer_results,
            forward_rankings,
            selected_module_names,
            bidirectional_ranking,
        ) = run_diagnostic_evaluation(
            model=model,
            tokenized=tokenized,
            source_rows=source_rows,
            donors=donors,
            snapshots=snapshots,
            modules=modules,
            device=args.device,
            primary_span_tokens=args.history_span_tokens,
            unaligned_policy=args.unaligned_token_policy,
            symmetric_top_k=args.symmetric_top_k,
            baseline_only=args.baseline_only,
        )
    finally:
        reset_runtime(model, write_enabled=True)

    baseline_replay_count = len(tokenized) * 2
    forward_layer_swap_replay_count = 0 if args.baseline_only else len(tokenized) * len(modules)
    result = {
        "schema": "rwkv_ms_dataset_layer_state_swaps.v1",
        "diagnostic_mode": (
            "exact_pair_baseline_only" if args.baseline_only else "layer_state_swaps"
        ),
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
            "training_protocol": protocol,
            "pairing": pairing_provenance,
            "exact_donor_indices": donors,
            "saved_adapter_config": saved_adapter_config,
            "effective_adapter_configs": effective_configs,
            "effective_residual_hybrid_gains": effective_residual_hybrid_gains,
            "layer_module_names": [name for name, _ in modules],
            "layer_indices": [int(module.layer_idx) for _, module in modules],
            "expected_layer_count": args.expected_layer_count,
            "expected_row_count": args.expected_row_count,
            "writer_snapshot_count": len(snapshots),
            "writer_prime_count": len(snapshots),
            "model_load_count": 1,
            "baseline_replay_count": baseline_replay_count,
            "forward_layer_swap_replay_count": forward_layer_swap_replay_count,
            "forward_replay_count": (
                baseline_replay_count + forward_layer_swap_replay_count
            ),
            "symmetric_replay_count": len(tokenized) * len(selected_module_names),
            "device": args.device,
            "dtype": args.dtype,
            "attn_implementation": args.attn_implementation,
            "torch_version": torch.__version__,
            "python_version": platform.python_version(),
            "command_arguments": vars(args) | {
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
            "definition": (
                "Compare target and exact-donor supervised token IDs at equal ordinals. "
                "The first unequal ordinal starts fixed target windows of 1, 8, 16, and "
                "32 tokens. Causal predictor positions are validated for every target."
            ),
            "history_span_token_counts": list(HISTORY_SPAN_TOKEN_COUNTS),
            "primary_history_span_tokens": args.history_span_tokens,
            "unaligned_token_policy": args.unaligned_token_policy,
            "fallback_row_count": sum(
                bool(row["history_token_selection"]["fallback_used"])
                for row in baselines
            ),
        },
        "metric_definitions": {
            "full_answer_ce": (
                "Float32 causal CE over every supervised answer token; retained as the "
                "secondary quality metric."
            ),
            "primary_history_ce": (
                f"Float32 causal CE over {args.history_span_tokens} target tokens beginning "
                "at the first target/donor supervised-token mismatch. This drives ranking."
            ),
            "donor_to_correct_ce_gain": (
                "Exact donor-state baseline CE minus CE after one target-layer state insertion."
            ),
            "correct_to_donor_ce_damage": (
                "CE after one exact-donor-layer state insertion minus correct-state baseline CE."
            ),
            "bidirectional_ce_effect": (
                "Mean of donor-to-correct gain and correct-to-donor damage. Bidirectional "
                "positive-row counts require both directional effects to be positive."
            ),
            "token_weighted_ce_effect": (
                "Sum of per-row NLL effects divided by the corresponding selected token count."
            ),
        },
        "baseline_summary": {
            "correct_memory": summarize_condition_rows(baselines, "correct_memory"),
            "exact_donor_memory": summarize_condition_rows(
                baselines,
                "exact_donor_memory",
            ),
            "donor_minus_correct": baseline_gap_summary(baselines),
        },
        "rows": baselines,
        "layers": layer_results,
        "rankings": {
            "primary_metric": (
                None
                if args.baseline_only
                else f"history_windows.{args.history_span_tokens}.token_weighted_ce_effect"
            ),
            "symmetric_selection_rule": (
                None
                if args.baseline_only
                else (
                    "Top token-weighted donor-to-correct primary-window CE gain, then "
                    "positive-row fraction, then layer index."
                )
            ),
            **forward_rankings,
            "symmetric_top_k_requested": args.symmetric_top_k,
            "symmetric_layer_names": selected_module_names,
            "selected_layers_by_bidirectional_primary_effect": bidirectional_ranking,
        },
    }
    write_json(output_path, result)
    console_summary = {"baseline_summary": result["baseline_summary"]}
    if not args.baseline_only:
        console_summary.update(
            {
                "forward_top": forward_rankings[
                    "by_primary_token_weighted_ce_gain"
                ][: args.symmetric_top_k],
                "bidirectional_top": bidirectional_ranking,
            }
        )
    print(json.dumps(console_summary, indent=2, sort_keys=True), flush=True)
    print(f"wrote {output_path}", flush=True)


if __name__ == "__main__":
    main()
