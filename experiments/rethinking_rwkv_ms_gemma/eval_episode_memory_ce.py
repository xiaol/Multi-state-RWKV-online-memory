#!/usr/bin/env python3
"""Evaluate supervised episode CE under matched, absent, and mismatched memory."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import statistics
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from datasets import Dataset, load_from_disk

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.rethinking_rwkv_ms_gemma.common import (
    load_model_and_tokenizer,
    logits_to_keep_kwargs,
    read_jsonl,
    write_json,
)
from deltamem.core.delta import (
    collect_delta_mem_output_ratio_stats,
    collect_delta_mem_state_stats,
    iter_delta_mem_modules,
    reset_delta_mem_states,
    set_delta_mem_read_context_mask,
    set_delta_mem_write_enabled,
    set_delta_mem_write_message_ids,
    set_delta_mem_write_sentence_ids,
)


CONDITION_CORRECT = "correct_memory"
CONDITION_NO_WRITE = "no_write_zero_state"
CONDITION_SHUFFLED = "shuffled_write_memory"
CONDITIONS = (CONDITION_CORRECT, CONDITION_NO_WRITE, CONDITION_SHUFFLED)
PREFIX_TOKEN_COUNTS = (1, 8, 16, 32)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-model", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--tokenized-dataset", type=Path, required=True)
    parser.add_argument("--source-jsonl", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--shuffle-seed", type=int, default=20260724)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--dtype",
        default="bfloat16",
        choices=("bfloat16", "float16", "float32"),
    )
    parser.add_argument("--attn-implementation", default="sdpa")
    parser.add_argument("--delta-mem-root", type=Path, default=REPO_ROOT)
    parser.add_argument(
        "--memory-fusion-placement",
        choices=("attention_output", "post_attention_norm", "normalized_residual_correction"),
        default=None,
    )
    parser.add_argument("--memory-fusion-residual-scale", type=float, default=None)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def default_output_path(
    checkpoint: Path,
    *,
    memory_fusion_placement: str | None = None,
    memory_fusion_residual_scale: float | None = None,
) -> Path:
    if checkpoint.parent.name != "trainer":
        raise ValueError(
            "--output is required unless checkpoint is RUN_ROOT/trainer/checkpoint-N"
        )
    suffix = ""
    if memory_fusion_placement is not None:
        suffix += f"_{memory_fusion_placement}"
    if memory_fusion_residual_scale is not None:
        scale = format(memory_fusion_residual_scale, ".12g").replace("-", "m").replace(".", "p")
        suffix += f"_scale{scale}"
    return checkpoint.parent.parent / f"{checkpoint.name}_answer_token_ce_ablation{suffix}.json"


def load_protocol(checkpoint: Path) -> dict[str, Any]:
    protocol_path = checkpoint / "training_protocol.json"
    if not protocol_path.is_file():
        raise FileNotFoundError(f"Missing training protocol: {protocol_path}")
    return json.loads(protocol_path.read_text(encoding="utf-8"))


def load_ready_metadata(dataset_path: Path) -> dict[str, Any] | None:
    ready_path = dataset_path / "_READY"
    if not ready_path.is_file():
        return None
    return json.loads(ready_path.read_text(encoding="utf-8"))


def validate_tokenized_fingerprint(
    *,
    expected: str | None,
    actual: str | None,
    tokenized_path: Path,
    ready_metadata: dict[str, Any] | None,
) -> dict[str, Any]:
    if actual == expected:
        return {
            "status": "exact",
            "protocol_fingerprint": expected,
            "loaded_fingerprint": actual,
        }
    if (
        ready_metadata is None
        or ready_metadata.get("cache_key") != tokenized_path.name
    ):
        raise ValueError(
            "Tokenized dataset fingerprint does not match checkpoint protocol: "
            f"expected={expected} actual={actual}"
        )
    return {
        "status": "legacy_presave_fingerprint_mismatch",
        "protocol_fingerprint": expected,
        "loaded_fingerprint": actual,
        "cache_key": ready_metadata["cache_key"],
    }


def validate_artifacts(
    *,
    checkpoint: Path,
    tokenized: Dataset,
    tokenized_path: Path,
    source_path: Path,
    source_rows: list[dict],
    protocol: dict[str, Any],
) -> dict[str, Any] | None:
    for filename in ("delta_mem_adapter.pt", "delta_mem_config.json"):
        if not (checkpoint / filename).is_file():
            raise FileNotFoundError(f"Missing checkpoint file: {checkpoint / filename}")

    required_columns = {
        "input_ids",
        "attention_mask",
        "labels",
        "write_input_ids",
        "write_attention_mask",
        "write_message_ids",
        "write_sentence_ids",
    }
    missing_columns = sorted(required_columns.difference(tokenized.column_names))
    if missing_columns:
        raise ValueError(f"Tokenized dataset is missing columns: {', '.join(missing_columns)}")

    ready_metadata = load_ready_metadata(tokenized_path)
    expected_fingerprint = protocol.get("tokenized_fingerprint")
    actual_fingerprint = getattr(tokenized, "_fingerprint", None)
    fingerprint_validation = validate_tokenized_fingerprint(
        expected=expected_fingerprint,
        actual=actual_fingerprint,
        tokenized_path=tokenized_path,
        ready_metadata=ready_metadata,
    )
    expected_samples = int(protocol["tokenized_samples"])
    if len(tokenized) != expected_samples:
        raise ValueError(
            f"Tokenized row count mismatch: expected={expected_samples} actual={len(tokenized)}"
        )
    if len(source_rows) != len(tokenized):
        raise ValueError(
            "This evaluator requires one tokenized episode per source row: "
            f"source={len(source_rows)} tokenized={len(tokenized)}"
        )
    expected_source = Path(protocol["train_file"]).resolve()
    if source_path.resolve() != expected_source:
        raise ValueError(
            f"Source dataset does not match protocol: expected={expected_source} actual={source_path.resolve()}"
        )
    expected_protocol = {
        "training_mode": "episode",
        "assistant_loss_mode": "final_assistant_only",
        "episode_recent_messages": 1,
        "memory_base_kl_weight": 0.0,
    }
    for key, expected_value in expected_protocol.items():
        if protocol.get(key) != expected_value:
            raise ValueError(
                f"Checkpoint protocol {key} must be {expected_value!r}, got {protocol.get(key)!r}"
            )

    if ready_metadata is not None:
        ready_expectations = {
            "training_mode": protocol["training_mode"],
            "assistant_loss_mode": protocol["assistant_loss_mode"],
            "episode_recent_messages": protocol["episode_recent_messages"],
            "max_write_length": protocol["max_write_length"],
            "max_length": protocol["max_length"],
            "memory_write_granularity": protocol["memory_write_granularity"],
        }
        for key, expected_value in ready_expectations.items():
            if ready_metadata.get(key) != expected_value:
                raise ValueError(
                    f"Tokenized _READY metadata {key} mismatch: "
                    f"expected={expected_value!r} actual={ready_metadata.get(key)!r}"
                )

    for row_index, row in enumerate(tokenized):
        input_length = len(row["input_ids"])
        if input_length == 0 or input_length > int(protocol["max_length"]):
            raise ValueError(f"Invalid read length at row {row_index}: {input_length}")
        if not row["write_input_ids"]:
            raise ValueError(f"Empty write sequence at row {row_index}")
        if len(row["write_input_ids"]) > int(protocol["max_write_length"]):
            raise ValueError(f"Write sequence exceeds protocol limit at row {row_index}")
        supervised_count = sum(
            label != -100 and attention != 0
            for label, attention in zip(row["labels"][1:], row["attention_mask"][1:])
        )
        if supervised_count == 0:
            raise ValueError(f"No supervised next-token targets at row {row_index}")
        if supervised_count < max(PREFIX_TOKEN_COUNTS):
            raise ValueError(
                f"Row {row_index} has {supervised_count} supervised targets; "
                f"at least {max(PREFIX_TOKEN_COUNTS)} are required for prefix evaluation"
            )
    validated_metadata = {} if ready_metadata is None else dict(ready_metadata)
    validated_metadata["fingerprint_validation"] = fingerprint_validation
    return validated_metadata


def make_mismatched_donors(tokenized: Dataset, seed: int) -> list[int]:
    if len(tokenized) < 2:
        raise ValueError("Shuffled-write evaluation requires at least two rows")
    target_writes = [row["write_input_ids"] for row in tokenized]
    rng = random.Random(seed)
    donors = list(range(len(tokenized)))
    for _ in range(10_000):
        rng.shuffle(donors)
        if all(
            donor_index != target_index
            and target_writes[donor_index] != target_writes[target_index]
            for target_index, donor_index in enumerate(donors)
        ):
            return donors
    raise RuntimeError("Could not construct a mismatched write-memory permutation")


def tensor_row(row: dict[str, Any], key: str, device: str) -> torch.Tensor:
    return torch.tensor([row[key]], dtype=torch.long, device=device)


def reset_runtime(model, *, write_enabled: bool) -> None:
    reset_delta_mem_states(model)
    set_delta_mem_read_context_mask(model, None)
    set_delta_mem_write_message_ids(model, None)
    set_delta_mem_write_sentence_ids(model, None)
    set_delta_mem_write_enabled(model, write_enabled)


def prime_write(model, row: dict[str, Any], device: str) -> dict[str, float]:
    write_input_ids = tensor_row(row, "write_input_ids", device)
    write_attention_mask = tensor_row(row, "write_attention_mask", device)
    write_message_ids = tensor_row(row, "write_message_ids", device)
    write_sentence_ids = tensor_row(row, "write_sentence_ids", device)
    set_delta_mem_read_context_mask(model, None)
    set_delta_mem_write_message_ids(model, write_message_ids)
    set_delta_mem_write_sentence_ids(model, write_sentence_ids)
    set_delta_mem_write_enabled(model, True)
    model(
        input_ids=write_input_ids,
        attention_mask=write_attention_mask,
        use_cache=False,
        return_dict=True,
        **logits_to_keep_kwargs(model, 1),
    )
    set_delta_mem_write_message_ids(model, None)
    set_delta_mem_write_sentence_ids(model, None)
    return collect_delta_mem_state_stats(model)


def supervised_token_nll(
    logits: torch.Tensor,
    labels: torch.Tensor,
    attention_mask: torch.Tensor,
) -> torch.Tensor:
    supervised_mask = labels[:, 1:].ne(-100) & attention_mask[:, 1:].ne(0)
    selected_logits = logits[:, :-1, :].masked_select(
        supervised_mask.unsqueeze(-1)
    ).view(-1, logits.size(-1))
    targets = labels[:, 1:].masked_select(supervised_mask)
    if targets.numel() == 0:
        raise ValueError("Read sequence has no supervised next-token targets")
    return F.cross_entropy(selected_logits.float(), targets, reduction="none")


def summarize_token_nll(token_nll: torch.Tensor) -> dict[str, Any]:
    token_count = int(token_nll.numel())
    if token_count == 0:
        raise ValueError("Cannot summarize an empty supervised-token NLL tensor")
    nll_sum = float(token_nll.sum().item())
    prefixes: dict[str, dict[str, float | int]] = {}
    for requested_count in PREFIX_TOKEN_COUNTS:
        prefix_count = min(requested_count, token_count)
        prefix_sum = float(token_nll[:prefix_count].sum().item())
        prefixes[str(requested_count)] = {
            "requested_token_count": requested_count,
            "token_count": prefix_count,
            "nll_sum": prefix_sum,
            "ce": prefix_sum / prefix_count,
        }
    return {
        "token_count": token_count,
        "nll_sum": nll_sum,
        "ce": nll_sum / token_count,
        "prefixes": prefixes,
    }


def evaluate_condition(
    *,
    model,
    target_row: dict[str, Any],
    write_row: dict[str, Any] | None,
    device: str,
    read_write_enabled: bool,
) -> dict[str, Any]:
    reset_runtime(model, write_enabled=False)
    if write_row is None:
        pre_read_state = collect_delta_mem_state_stats(model)
    else:
        pre_read_state = prime_write(model, write_row, device)

    input_ids = tensor_row(target_row, "input_ids", device)
    attention_mask = tensor_row(target_row, "attention_mask", device)
    labels = tensor_row(target_row, "labels", device)
    read_context_mask = labels.eq(-100) & attention_mask.ne(0)
    set_delta_mem_write_enabled(model, read_write_enabled)
    set_delta_mem_read_context_mask(model, read_context_mask)
    outputs = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        use_cache=False,
        return_dict=True,
    )
    token_nll = supervised_token_nll(outputs.logits, labels, attention_mask)
    if not torch.isfinite(token_nll).all():
        raise RuntimeError("Evaluation produced non-finite supervised token NLL values")
    result = {
        **summarize_token_nll(token_nll),
        "pre_read_state": pre_read_state,
        "post_read_state": collect_delta_mem_state_stats(model),
        "output_ratio_stats": collect_delta_mem_output_ratio_stats(model),
    }
    for stats_name in ("pre_read_state", "post_read_state", "output_ratio_stats"):
        for metric_name, value in result[stats_name].items():
            if isinstance(value, (int, float)) and not math.isfinite(float(value)):
                raise RuntimeError(
                    f"Evaluation produced non-finite {stats_name}.{metric_name}: {value}"
                )
    del outputs, token_nll
    return result


def summarize_condition(
    rows: list[dict[str, Any]],
    condition: str,
    *,
    prefix_token_count: int | None = None,
) -> dict[str, Any]:
    def condition_metrics(row: dict[str, Any]) -> dict[str, Any]:
        metrics = row["conditions"][condition]
        if prefix_token_count is not None:
            metrics = metrics["prefixes"][str(prefix_token_count)]
        return metrics

    metrics = [condition_metrics(row) for row in rows]
    values = [float(item["ce"]) for item in metrics]
    nll_sum = sum(float(item["nll_sum"]) for item in metrics)
    token_count = sum(int(item["token_count"]) for item in metrics)
    token_weighted_ce = nll_sum / token_count
    summary = {
        "condition": condition,
        "rows": len(rows),
        "token_count": token_count,
        "nll_sum": nll_sum,
        "token_weighted_ce": token_weighted_ce,
        "perplexity": math.exp(token_weighted_ce),
        "mean_row_ce": statistics.fmean(values),
        "median_row_ce": statistics.median(values),
        "population_std_row_ce": statistics.pstdev(values),
        "min_row_ce": min(values),
        "max_row_ce": max(values),
    }
    if prefix_token_count is not None:
        summary["prefix_token_count"] = prefix_token_count
    return summary


def summarize_gap(
    rows: list[dict[str, Any]],
    key: str,
    *,
    prefix_token_count: int | None = None,
) -> dict[str, float]:
    def row_gap(row: dict[str, Any]) -> float:
        gaps = row["gaps"]
        if prefix_token_count is not None:
            gaps = gaps["prefixes"][str(prefix_token_count)]
        return float(gaps[key])

    values = [row_gap(row) for row in rows]
    return {
        "mean": statistics.fmean(values),
        "median": statistics.median(values),
        "population_std": statistics.pstdev(values),
        "min": min(values),
        "max": max(values),
        "fraction_positive": sum(value > 0.0 for value in values) / len(values),
    }


def build_paired_summary(
    rows: list[dict[str, Any]],
    summaries: dict[str, dict[str, Any]],
    *,
    prefix_token_count: int | None = None,
) -> dict[str, Any]:
    return {
        "memory_advantage_vs_no_write_token_weighted_ce": (
            summaries[CONDITION_NO_WRITE]["token_weighted_ce"]
            - summaries[CONDITION_CORRECT]["token_weighted_ce"]
        ),
        "memory_specificity_vs_shuffled_token_weighted_ce": (
            summaries[CONDITION_SHUFFLED]["token_weighted_ce"]
            - summaries[CONDITION_CORRECT]["token_weighted_ce"]
        ),
        "shuffled_vs_no_write_token_weighted_ce": (
            summaries[CONDITION_SHUFFLED]["token_weighted_ce"]
            - summaries[CONDITION_NO_WRITE]["token_weighted_ce"]
        ),
        "per_row_no_write_minus_correct_ce": summarize_gap(
            rows,
            "no_write_minus_correct_ce",
            prefix_token_count=prefix_token_count,
        ),
        "per_row_shuffled_minus_correct_ce": summarize_gap(
            rows,
            "shuffled_minus_correct_ce",
            prefix_token_count=prefix_token_count,
        ),
        "per_row_shuffled_minus_no_write_ce": summarize_gap(
            rows,
            "shuffled_minus_no_write_ce",
            prefix_token_count=prefix_token_count,
        ),
    }


def source_identity(source_row: dict[str, Any], row_index: int) -> dict[str, Any]:
    identity = {"row_index": row_index}
    if "loss_probe_source_index" in source_row:
        identity["loss_probe_source_index"] = source_row["loss_probe_source_index"]
    preprocessing = source_row.get("memory_preprocessing")
    if isinstance(preprocessing, dict):
        for key in ("source_repo", "source_split", "source_index"):
            if key in preprocessing:
                identity[key] = preprocessing[key]
    return identity


def main() -> None:
    args = parse_args()
    checkpoint = args.checkpoint.expanduser().resolve()
    tokenized_path = args.tokenized_dataset.expanduser().resolve()
    source_path = args.source_jsonl.expanduser().resolve()
    output_path = (
        default_output_path(
            checkpoint,
            memory_fusion_placement=args.memory_fusion_placement,
            memory_fusion_residual_scale=args.memory_fusion_residual_scale,
        )
        if args.output is None
        else args.output.expanduser().resolve()
    )

    protocol = load_protocol(checkpoint)
    tokenized = load_from_disk(str(tokenized_path))
    source_rows = read_jsonl(source_path)
    ready_metadata = validate_artifacts(
        checkpoint=checkpoint,
        tokenized=tokenized,
        tokenized_path=tokenized_path,
        source_path=source_path,
        source_rows=source_rows,
        protocol=protocol,
    )
    donors = make_mismatched_donors(tokenized, args.shuffle_seed)

    model, _ = load_model_and_tokenizer(
        base_model=str(args.base_model.expanduser().resolve()),
        device=args.device,
        dtype=args.dtype,
        attn_implementation=args.attn_implementation,
        delta_mem_root=args.delta_mem_root,
        memory_dir=checkpoint,
        memory_fusion_placement=args.memory_fusion_placement,
        memory_fusion_residual_scale=args.memory_fusion_residual_scale,
    )
    model.eval()
    effective_configs = {
        name: module.delta_config.to_dict()
        for name, module in iter_delta_mem_modules(model)
    }
    if not effective_configs:
        raise RuntimeError("Loaded model has no attached Delta-Mem modules")
    unique_effective_configs = {
        json.dumps(config, sort_keys=True) for config in effective_configs.values()
    }
    if len(unique_effective_configs) != 1:
        raise RuntimeError("Attached Delta-Mem modules have inconsistent effective configs")
    effective_adapter_config = next(iter(effective_configs.values()))
    started_at = time.time()
    rows: list[dict[str, Any]] = []
    with torch.inference_mode():
        for row_index in range(len(tokenized)):
            target_row = tokenized[row_index]
            donor_index = donors[row_index]
            donor_row = tokenized[donor_index]
            condition_results = {
                CONDITION_CORRECT: evaluate_condition(
                    model=model,
                    target_row=target_row,
                    write_row=target_row,
                    device=args.device,
                    read_write_enabled=bool(protocol["episode_read_write_enabled"]),
                ),
                CONDITION_NO_WRITE: evaluate_condition(
                    model=model,
                    target_row=target_row,
                    write_row=None,
                    device=args.device,
                    read_write_enabled=False,
                ),
                CONDITION_SHUFFLED: evaluate_condition(
                    model=model,
                    target_row=target_row,
                    write_row=donor_row,
                    device=args.device,
                    read_write_enabled=bool(protocol["episode_read_write_enabled"]),
                ),
            }
            correct_ce = condition_results[CONDITION_CORRECT]["ce"]
            no_write_ce = condition_results[CONDITION_NO_WRITE]["ce"]
            shuffled_ce = condition_results[CONDITION_SHUFFLED]["ce"]
            prefix_gaps = {}
            for prefix_token_count in PREFIX_TOKEN_COUNTS:
                prefix_key = str(prefix_token_count)
                prefix_correct_ce = condition_results[CONDITION_CORRECT]["prefixes"][
                    prefix_key
                ]["ce"]
                prefix_no_write_ce = condition_results[CONDITION_NO_WRITE]["prefixes"][
                    prefix_key
                ]["ce"]
                prefix_shuffled_ce = condition_results[CONDITION_SHUFFLED]["prefixes"][
                    prefix_key
                ]["ce"]
                prefix_gaps[prefix_key] = {
                    "no_write_minus_correct_ce": prefix_no_write_ce - prefix_correct_ce,
                    "shuffled_minus_correct_ce": prefix_shuffled_ce - prefix_correct_ce,
                    "shuffled_minus_no_write_ce": prefix_shuffled_ce - prefix_no_write_ce,
                }
            row_result = {
                **source_identity(source_rows[row_index], row_index),
                "shuffled_write_donor": source_identity(source_rows[donor_index], donor_index),
                "write_tokens": len(target_row["write_input_ids"]),
                "read_tokens": len(target_row["input_ids"]),
                "conditions": condition_results,
                "gaps": {
                    "no_write_minus_correct_ce": no_write_ce - correct_ce,
                    "shuffled_minus_correct_ce": shuffled_ce - correct_ce,
                    "shuffled_minus_no_write_ce": shuffled_ce - no_write_ce,
                    "prefixes": prefix_gaps,
                },
            }
            rows.append(row_result)
            print(
                f"row {row_index + 1:02d}/{len(tokenized)} "
                f"correct={correct_ce:.6f} no_write={no_write_ce:.6f} "
                f"shuffled={shuffled_ce:.6f} donor={donor_index}",
                flush=True,
            )
    reset_runtime(model, write_enabled=True)

    summaries = {
        condition: summarize_condition(rows, condition)
        for condition in CONDITIONS
    }
    paired = build_paired_summary(rows, summaries)
    prefix_summaries = {}
    for prefix_token_count in PREFIX_TOKEN_COUNTS:
        condition_summaries = {
            condition: summarize_condition(
                rows,
                condition,
                prefix_token_count=prefix_token_count,
            )
            for condition in CONDITIONS
        }
        prefix_summaries[str(prefix_token_count)] = {
            "conditions": condition_summaries,
            "paired_gaps": build_paired_summary(
                rows,
                condition_summaries,
                prefix_token_count=prefix_token_count,
            ),
        }
    adapter_config = json.loads(
        (checkpoint / "delta_mem_config.json").read_text(encoding="utf-8")
    )
    result = {
        "schema": "rwkv_ms_episode_answer_token_ce_ablation.v2",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "elapsed_seconds": time.time() - started_at,
        "provenance": {
            "base_model": str(args.base_model.expanduser().resolve()),
            "checkpoint": str(checkpoint),
            "tokenized_dataset": str(tokenized_path),
            "tokenized_fingerprint": getattr(tokenized, "_fingerprint", None),
            "tokenized_ready_metadata": ready_metadata,
            "source_jsonl": str(source_path),
            "source_jsonl_sha256": sha256_file(source_path),
            "training_protocol": protocol,
            "target_layers": adapter_config.get("target_layers"),
            "saved_adapter_config": adapter_config,
            "effective_adapter_config": effective_adapter_config,
            "memory_fusion_overrides": {
                "memory_fusion_placement": args.memory_fusion_placement,
                "memory_fusion_residual_scale": args.memory_fusion_residual_scale,
            },
            "shuffle_seed": args.shuffle_seed,
            "shuffled_donor_indices": donors,
            "device": args.device,
            "dtype": args.dtype,
            "attn_implementation": args.attn_implementation,
            "torch_version": torch.__version__,
        },
        "condition_definitions": {
            CONDITION_CORRECT: (
                "Reset state, write the target row's saved write tensors, then read the target "
                "with the checkpoint protocol's read-write setting."
            ),
            CONDITION_NO_WRITE: (
                "Reset to zero/absent state, skip the write phase, and keep writes disabled "
                "throughout the target read. Delta heads remain attached."
            ),
            CONDITION_SHUFFLED: (
                "Reset state, write a deterministic different row's saved write tensors, then "
                "read the unchanged target with the checkpoint protocol's read-write setting."
            ),
        },
        "metric_definition": (
            "Float32 causal cross-entropy over labels[:, 1:] where labels != -100 and "
            "attention_mask != 0. Aggregate CE is total NLL divided by total supervised tokens; "
            "prefix metrics use the first 1, 8, 16, or 32 supervised targets in sequence order."
        ),
        "summaries": summaries,
        "paired_gaps": paired,
        "prefix_summaries": prefix_summaries,
        "rows": rows,
    }
    write_json(output_path, result)
    print(f"wrote {output_path}", flush=True)
    for condition in CONDITIONS:
        summary = summaries[condition]
        print(
            f"{condition}: ce={summary['token_weighted_ce']:.6f} "
            f"tokens={summary['token_count']} ppl={summary['perplexity']:.3f}",
            flush=True,
        )
    print(
        "gaps: "
        f"no_write-correct={paired['memory_advantage_vs_no_write_token_weighted_ce']:.6f} "
        f"shuffled-correct={paired['memory_specificity_vs_shuffled_token_weighted_ce']:.6f}",
        flush=True,
    )


if __name__ == "__main__":
    main()
