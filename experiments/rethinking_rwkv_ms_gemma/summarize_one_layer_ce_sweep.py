#!/usr/bin/env python3
"""Validate and rank the six independent CE-only one-layer probes."""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import statistics
from typing import Any, Iterable

try:
    from .validate_one_layer_ce_source import validate_source
except ImportError:
    from validate_one_layer_ce_source import validate_source


LAYERS = (4, 5, 10, 11, 22, 23)
ATTENTION_TYPES = {
    4: "sliding_attention",
    5: "full_attention",
    10: "sliding_attention",
    11: "full_attention",
    22: "sliding_attention",
    23: "full_attention",
}
DEFAULT_RUN_ROOT = Path(
    "/run/media/xiaol/B214449214445C0B/delta_mem_outputs/novel_rwkv_ms_memory"
)
DEFAULT_SWEEP_ROOT = (
    DEFAULT_RUN_ROOT / "v4_one_layer_ce_gate128_content_control_seed20260724_n32"
)
DEFAULT_TRAIN_FILE = Path(
    "/run/media/xiaol/B214449214445C0B/delta_mem_data/novel_agent_memory/"
    "novel_memory_content_control_probe_seed20260724_n32.jsonl"
)
EXPECTED_TRAIN_SHA256 = "0aa7472d3c7fe3b5501801fc380f570b82a048c6e535e800263c6e1c2ee08a2d"
EXPECTED_TRAIN_ROWS = 32
STEPS_PER_EPOCH = 32


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sweep-root", type=Path, default=DEFAULT_SWEEP_ROOT)
    parser.add_argument("--train-file", type=Path, default=DEFAULT_TRAIN_FILE)
    parser.add_argument("--expected-sha256", default=EXPECTED_TRAIN_SHA256)
    parser.add_argument("--expected-steps", type=int, default=128)
    parser.add_argument("--output-json", type=Path)
    parser.add_argument(
        "--allow-incomplete",
        action="store_true",
        help="Exit successfully while reporting missing, invalid, or partial runs.",
    )
    parser.add_argument(
        "--stdout-only",
        action="store_true",
        help="Print the table and JSON destination status without writing JSON.",
    )
    return parser.parse_args()


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


def layer_directory(sweep_root: Path, layer: int) -> Path:
    return sweep_root / f"layer_{layer:02d}"


def checkpoint_step(path: Path) -> int:
    try:
        return int(path.name.removeprefix("checkpoint-"))
    except ValueError:
        return -1


def select_checkpoint(run_dir: Path, expected_steps: int) -> Path | None:
    trainer_dir = run_dir / "trainer"
    gate_checkpoint = trainer_dir / f"checkpoint-{expected_steps}"
    if gate_checkpoint.is_dir():
        return gate_checkpoint
    checkpoints = sorted(
        (
            path
            for path in trainer_dir.glob("checkpoint-*")
            if path.is_dir() and checkpoint_step(path) >= 0
        ),
        key=checkpoint_step,
    )
    return checkpoints[-1] if checkpoints else None


def values_match(actual: Any, expected: Any) -> bool:
    if isinstance(expected, float):
        try:
            return math.isclose(float(actual), expected, rel_tol=0.0, abs_tol=1e-12)
        except (TypeError, ValueError):
            return False
    return actual == expected


def validate_fields(
    payload: dict[str, Any], expected: dict[str, Any], *, source: str
) -> list[str]:
    errors: list[str] = []
    for key, expected_value in expected.items():
        actual_value = payload.get(key)
        if not values_match(actual_value, expected_value):
            errors.append(
                f"{source}.{key}: expected {expected_value!r}, got {actual_value!r}"
            )
    return errors


def summarize_values(values: Iterable[float]) -> dict[str, float | int]:
    materialized = [float(value) for value in values]
    if not materialized:
        raise ValueError("cannot summarize an empty sequence")
    return {
        "count": len(materialized),
        "first": materialized[0],
        "last": materialized[-1],
        "mean": math.fsum(materialized) / len(materialized),
        "median": statistics.median(materialized),
        "min": min(materialized),
        "max": max(materialized),
    }


def training_rows(trainer_state: dict[str, Any]) -> list[dict[str, Any]]:
    required = (
        "step",
        "loss",
        "delta/memory_keep_loss",
        "delta/memory_kl_loss",
        "delta/memory_teacher_loss",
        "delta/memory_wmem",
        "delta/max_state_norm",
        "grad_norm",
    )
    rows = [
        row
        for row in trainer_state.get("log_history", [])
        if all(key in row for key in required)
    ]
    rows.sort(key=lambda row: int(row["step"]))
    steps = [int(row["step"]) for row in rows]
    if not steps:
        raise ValueError("trainer_state.json has no complete CE training rows")
    if steps != list(range(1, steps[-1] + 1)):
        raise ValueError("logged training steps are not contiguous from step 1")
    return rows


def analyze_run(
    *,
    run_dir: Path,
    layer: int,
    expected_steps: int,
    train_file: Path,
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "layer": layer,
        "attention_type": ATTENTION_TYPES[layer],
        "run_dir": str(run_dir),
        "status": "missing",
        "errors": [],
    }
    checkpoint = select_checkpoint(run_dir, expected_steps)
    if checkpoint is None:
        result["errors"].append("no checkpoint found")
        return result

    result["checkpoint"] = str(checkpoint)
    result["checkpoint_step"] = checkpoint_step(checkpoint)
    required_files = (
        "delta_mem_config.json",
        "trainer_state.json",
        "training_protocol.json",
    )
    missing = [name for name in required_files if not (checkpoint / name).is_file()]
    if missing:
        result["status"] = "invalid"
        result["errors"].append(f"checkpoint missing files: {', '.join(missing)}")
        return result

    config = read_json(checkpoint / "delta_mem_config.json")
    protocol = read_json(checkpoint / "training_protocol.json")
    trainer_state = read_json(checkpoint / "trainer_state.json")
    expected_protocol = {
        "train_file": str(train_file.resolve()),
        "tokenized_samples": EXPECTED_TRAIN_ROWS,
        "train_samples": EXPECTED_TRAIN_ROWS,
        "eval_samples": 0,
        "training_mode": "episode",
        "assistant_loss_mode": "final_assistant_only",
        "max_length": 256,
        "max_write_length": 512,
        "teacher_max_length": 768,
        "episode_recent_messages": 1,
        "episode_read_write_enabled": False,
        "memory_loss_mode": "context_dropout_ce",
        "memory_dropout_no_memory_prob": 0.0,
        "memory_dropout_state_only_prob": 0.0,
        "memory_base_kl_weight": 0.0,
        "validation_split_ratio": 0.0,
        "seed": 42,
        "data_seed": 42,
        "per_device_train_batch_size": 1,
        "gradient_accumulation_steps": 1,
        "learning_rate": 0.001,
        "lr_scheduler_type": "constant_with_warmup",
        "warmup_steps": 8,
        "weight_decay": 0.0,
        "optim": "adamw_torch_fused",
        "num_train_epochs": 4.0,
        "max_steps": expected_steps,
        "save_steps": 32,
    }
    expected_config = {
        "rank": 8,
        "alpha": 8.0,
        "memory_backend": "rwkv_ms",
        "beta_bias_init": 0.0,
        "output_init": "base_slice_fixed",
        "online_gain": 0.2,
        "target_layers": [layer],
        "delta_heads": ["q", "o"],
        "trainable_delta_scale": True,
        "rwkv_ms_num_states": 4,
        "rwkv_ms_chunk_size": 128,
        "memory_readout_mode": "delta",
        "memory_write_source": "learned_hidden",
        "memory_write_granularity": "token",
    }
    errors = validate_fields(protocol, expected_protocol, source="protocol")
    errors.extend(validate_fields(config, expected_config, source="config"))

    try:
        rows = training_rows(trainer_state)
    except ValueError as error:
        result["status"] = "invalid"
        result["errors"] = errors + [str(error)]
        result["tokenized_fingerprint"] = protocol.get("tokenized_fingerprint")
        return result

    global_step = int(trainer_state.get("global_step", 0))
    logged_step = int(rows[-1]["step"])
    if global_step != checkpoint_step(checkpoint):
        errors.append(
            f"checkpoint/global-step mismatch: directory={checkpoint_step(checkpoint)} state={global_step}"
        )
    if logged_step != global_step:
        errors.append(f"last logged step {logged_step} does not equal global step {global_step}")
    if global_step > expected_steps:
        errors.append(f"global step {global_step} exceeds ranking gate {expected_steps}")

    for row in rows:
        step = int(row["step"])
        task_ce = float(row["delta/memory_keep_loss"])
        total_loss = float(row["loss"])
        if not math.isclose(total_loss, task_ce, rel_tol=0.0, abs_tol=1e-6):
            errors.append(f"step {step}: total loss is not CE-only task loss")
            break
        if abs(float(row["delta/memory_kl_loss"])) > 1e-12:
            errors.append(f"step {step}: logged KL is nonzero")
            break
        if abs(float(row["delta/memory_teacher_loss"])) > 1e-12:
            errors.append(f"step {step}: teacher CE was unexpectedly computed")
            break
        if not math.isclose(
            float(row["delta/memory_wmem"]), 1.0, rel_tol=0.0, abs_tol=1e-12
        ):
            errors.append(f"step {step}: memory was dropped")
            break

    epoch_summaries: list[dict[str, Any]] = []
    for epoch_index in range(1, math.ceil(logged_step / STEPS_PER_EPOCH) + 1):
        start = (epoch_index - 1) * STEPS_PER_EPOCH
        epoch_rows = rows[start : start + STEPS_PER_EPOCH]
        epoch_summaries.append(
            {
                "epoch": epoch_index,
                "step_start": int(epoch_rows[0]["step"]),
                "step_end": int(epoch_rows[-1]["step"]),
                "complete": len(epoch_rows) == STEPS_PER_EPOCH,
                "task_ce": summarize_values(
                    float(row["delta/memory_keep_loss"]) for row in epoch_rows
                ),
                "max_state_norm": max(
                    float(row["delta/max_state_norm"]) for row in epoch_rows
                ),
                "max_grad_norm": max(float(row["grad_norm"]) for row in epoch_rows),
            }
        )

    complete = global_step == expected_steps and logged_step == expected_steps
    result.update(
        {
            "status": "complete" if complete and not errors else "invalid" if errors else "partial",
            "errors": errors,
            "global_step": global_step,
            "expected_steps": expected_steps,
            "tokenized_fingerprint": protocol.get("tokenized_fingerprint"),
            "epoch_summaries": epoch_summaries,
            "overall_task_ce": summarize_values(
                float(row["delta/memory_keep_loss"]) for row in rows
            ),
            "final_epoch_mean_ce": float(epoch_summaries[-1]["task_ce"]["mean"]),
            "loss_auc_mean_ce": math.fsum(
                float(row["delta/memory_keep_loss"]) for row in rows
            )
            / len(rows),
            "max_state_norm": max(float(row["delta/max_state_norm"]) for row in rows),
            "last_state_norm": float(rows[-1]["delta/max_state_norm"]),
            "max_grad_norm": max(float(row["grad_norm"]) for row in rows),
        }
    )
    if len(epoch_summaries) >= 2:
        first_mean = float(epoch_summaries[0]["task_ce"]["mean"])
        final_mean = float(epoch_summaries[-1]["task_ce"]["mean"])
        result["first_to_final_ce_decrease"] = first_mean - final_mean
        result["first_to_final_relative_decrease"] = (
            (first_mean - final_mean) / first_mean if first_mean else None
        )
    return result


def summarize_sweep(
    *,
    sweep_root: Path,
    train_file: Path,
    expected_sha256: str,
    expected_steps: int,
) -> dict[str, Any]:
    if expected_steps <= 0 or expected_steps % STEPS_PER_EPOCH != 0:
        raise ValueError(
            f"expected steps must be a positive multiple of {STEPS_PER_EPOCH}"
        )
    if not train_file.is_file():
        raise FileNotFoundError(f"training file is missing: {train_file}")
    source_contract = validate_source(
        source=train_file,
        expected_sha256=expected_sha256,
    )
    actual_sha256 = str(source_contract["sha256"])
    train_rows = int(source_contract["rows"])

    runs = [
        analyze_run(
            run_dir=layer_directory(sweep_root, layer),
            layer=layer,
            expected_steps=expected_steps,
            train_file=train_file,
        )
        for layer in LAYERS
    ]
    cross_run_errors: list[str] = []
    fingerprints = {
        str(run["tokenized_fingerprint"])
        for run in runs
        if run.get("tokenized_fingerprint") is not None
    }
    if len(fingerprints) > 1:
        cross_run_errors.append(
            "tokenized fingerprints differ across layer runs: " + ", ".join(sorted(fingerprints))
        )

    rankable = [run for run in runs if run["status"] == "complete"]
    rankable.sort(
        key=lambda run: (
            float(run["final_epoch_mean_ce"]),
            float(run["loss_auc_mean_ce"]),
            float(run["max_state_norm"]),
            int(run["layer"]),
        )
    )
    ranking = [
        {
            "rank": rank,
            "layer": int(run["layer"]),
            "attention_type": run["attention_type"],
            "final_epoch_mean_ce": float(run["final_epoch_mean_ce"]),
            "loss_auc_mean_ce": float(run["loss_auc_mean_ce"]),
            "max_state_norm": float(run["max_state_norm"]),
        }
        for rank, run in enumerate(rankable, start=1)
    ]
    ranking_complete = len(rankable) == len(LAYERS) and not cross_run_errors
    return {
        "schema_version": 1,
        "sweep_root": str(sweep_root.resolve()),
        "layers": list(LAYERS),
        "ranking_rule": [
            "final_epoch_mean_ce ascending",
            "loss_auc_mean_ce ascending",
            "max_state_norm ascending",
            "layer ascending",
        ],
        "ranking_scope": "CE-only first gate; this does not establish memory dependence",
        "expected_steps": expected_steps,
        "expected_epochs": expected_steps // STEPS_PER_EPOCH,
        "dataset": {
            "path": str(train_file.resolve()),
            "sha256": actual_sha256,
            "rows": train_rows,
            "seed": 42,
            "data_seed": 42,
            "content_control": source_contract,
        },
        "cross_run_errors": cross_run_errors,
        "ranking_complete": ranking_complete,
        "ranking": ranking,
        "runs": runs,
    }


def print_summary(summary: dict[str, Any]) -> None:
    header = (
        "layer",
        "type",
        "step",
        "final_epoch_ce",
        "loss_auc_ce",
        "max_state",
        "status",
    )
    print("\t".join(header))
    for run in summary["runs"]:
        print(
            "\t".join(
                (
                    str(run["layer"]),
                    str(run["attention_type"]),
                    str(run.get("global_step", "-")),
                    (
                        f"{float(run['final_epoch_mean_ce']):.6f}"
                        if "final_epoch_mean_ce" in run
                        else "-"
                    ),
                    (
                        f"{float(run['loss_auc_mean_ce']):.6f}"
                        if "loss_auc_mean_ce" in run
                        else "-"
                    ),
                    (
                        f"{float(run['max_state_norm']):.6g}"
                        if "max_state_norm" in run
                        else "-"
                    ),
                    str(run["status"]),
                )
            )
        )
    if summary["ranking"]:
        print("ranking=" + ",".join(str(row["layer"]) for row in summary["ranking"]))
    if summary["cross_run_errors"]:
        for error in summary["cross_run_errors"]:
            print(f"cross_run_error={error}")


def main() -> int:
    args = parse_args()
    summary = summarize_sweep(
        sweep_root=args.sweep_root,
        train_file=args.train_file,
        expected_sha256=args.expected_sha256,
        expected_steps=args.expected_steps,
    )
    print_summary(summary)
    output_json = args.output_json or args.sweep_root / "one_layer_ce_summary.json"
    if not args.stdout_only:
        write_json_atomic(output_json, summary)
        print(f"summary_json={output_json}")
    if not summary["ranking_complete"] and not args.allow_incomplete:
        print("ranking is incomplete; rerun with --allow-incomplete for a non-failing partial report")
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
