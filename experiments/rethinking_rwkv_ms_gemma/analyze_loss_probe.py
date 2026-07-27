#!/usr/bin/env python3
"""Analyze a completed delta-Mem loss probe and render its loss curve."""

from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime, timezone
import hashlib
import html
import json
import math
import os
from pathlib import Path
import re
import shutil
import statistics
import subprocess
import tempfile
from typing import Any, Iterable, Sequence


METRIC_KEYS = {
    "task_ce": "delta/memory_keep_loss",
    "total_loss": "loss",
    "teacher_ce": "delta/memory_teacher_loss",
    "kl_loss": "delta/memory_kl_loss",
    "grad_norm": "grad_norm",
}
LAYER_PATTERN = re.compile(r"\.layers\.(\d+)\.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--checkpoint-start", type=int, default=16)
    parser.add_argument("--checkpoint-end", type=int)
    parser.add_argument("--window-size", type=int, default=16)
    parser.add_argument("--target-task-ce", type=float, default=1.7)
    parser.add_argument("--min-relative-improvement", type=float, default=0.01)
    parser.add_argument("--output-json", type=Path)
    parser.add_argument("--output-png", type=Path)
    return parser.parse_args()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json_atomic(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def quantile(values: Sequence[float], probability: float) -> float:
    ordered = sorted(values)
    if not ordered:
        raise ValueError("Cannot calculate a quantile of an empty sequence")
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * probability
    lower_index = math.floor(position)
    upper_index = math.ceil(position)
    if lower_index == upper_index:
        return ordered[lower_index]
    fraction = position - lower_index
    return ordered[lower_index] * (1.0 - fraction) + ordered[upper_index] * fraction


def summarize(values: Iterable[float]) -> dict[str, float | int]:
    materialized = [float(value) for value in values]
    if not materialized:
        raise ValueError("Cannot summarize an empty sequence")
    return {
        "count": len(materialized),
        "first": materialized[0],
        "last": materialized[-1],
        "mean": math.fsum(materialized) / len(materialized),
        "median": statistics.median(materialized),
        "population_std": statistics.pstdev(materialized),
        "min": min(materialized),
        "p05": quantile(materialized, 0.05),
        "p95": quantile(materialized, 0.95),
        "max": max(materialized),
    }


def summarize_rows(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    return {
        metric_name: summarize(float(row[source_key]) for row in rows)
        for metric_name, source_key in METRIC_KEYS.items()
    }


def contiguous_training_rows(trainer_state: dict[str, Any]) -> list[dict[str, Any]]:
    rows = [
        row
        for row in trainer_state.get("log_history", [])
        if all(key in row for key in ("step", *METRIC_KEYS.values()))
    ]
    rows.sort(key=lambda row: int(row["step"]))
    steps = [int(row["step"]) for row in rows]
    if not steps:
        raise ValueError("trainer_state.json contains no complete training records")
    expected = list(range(steps[0], steps[-1] + 1))
    if steps != expected:
        raise ValueError("Training log steps are not contiguous")
    if steps[0] != 1:
        raise ValueError(f"Training log begins at step {steps[0]}, not step 1")
    return rows


def grouped_metrics(
    rows: Sequence[dict[str, Any]],
    *,
    group_key,
    label_key: str,
) -> list[dict[str, Any]]:
    grouped: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[int(group_key(row))].append(row)
    return [
        {
            label_key: group_id,
            "step_start": int(group_rows[0]["step"]),
            "step_end": int(group_rows[-1]["step"]),
            "metrics": summarize_rows(group_rows),
        }
        for group_id, group_rows in sorted(grouped.items())
    ]


def comparison(before: dict[str, Any], after: dict[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for metric_name in METRIC_KEYS:
        before_mean = float(before[metric_name]["mean"])
        after_mean = float(after[metric_name]["mean"])
        decrease = before_mean - after_mean
        result[metric_name] = {
            "before_mean": before_mean,
            "after_mean": after_mean,
            "absolute_decrease": decrease,
            "relative_decrease": decrease / before_mean if before_mean else None,
        }
    return result


def objective_identity(
    rows: Sequence[dict[str, Any]], protocol: dict[str, Any]
) -> dict[str, Any]:
    weight = float(protocol["memory_base_kl_weight"])
    residuals = [
        float(row["loss"])
        - (
            float(row["delta/memory_keep_loss"])
            + weight * float(row["delta/memory_kl_loss"])
        )
        for row in rows
    ]
    tolerance = 1e-6
    max_abs_residual = max(abs(value) for value in residuals)
    return {
        "logged_task_ce_key": "delta/memory_keep_loss",
        "logged_teacher_ce_key": "delta/memory_teacher_loss",
        "logged_kl_key": "delta/memory_kl_loss",
        "logged_total_key": "loss",
        "formula": "total_loss = task_ce + memory_base_kl_weight * kl_loss",
        "memory_base_kl_weight": weight,
        "teacher_ce_in_total": False,
        "teacher_ce_role": "frozen full-context teacher diagnostic",
        "checked_steps": len(residuals),
        "residual": summarize(residuals),
        "max_abs_residual": max_abs_residual,
        "tolerance": tolerance,
        "identity_holds_within_tolerance": max_abs_residual <= tolerance,
    }


def load_adapter(path: Path) -> dict[str, Any]:
    try:
        import torch
    except ImportError as error:
        raise RuntimeError(
            "Adapter comparison requires a Python environment with PyTorch"
        ) from error
    adapter = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(adapter, dict) or not adapter:
        raise ValueError(f"Expected a non-empty tensor mapping in {path}")
    if not all(isinstance(name, str) and torch.is_tensor(value) for name, value in adapter.items()):
        raise ValueError(f"Expected only named tensors in {path}")
    return adapter


def adapter_movement(start_path: Path, end_path: Path) -> dict[str, Any]:
    import torch

    start = load_adapter(start_path)
    end = load_adapter(end_path)
    if start.keys() != end.keys():
        missing = sorted(set(start) - set(end))
        added = sorted(set(end) - set(start))
        raise ValueError(f"Adapter keys differ: missing={missing}, added={added}")

    per_tensor: list[dict[str, Any]] = []
    layer_accumulators: dict[int, dict[str, Any]] = defaultdict(
        lambda: {
            "tensor_count": 0,
            "changed_tensor_count": 0,
            "element_count": 0,
            "changed_element_count": 0,
            "start_l2_squared": 0.0,
            "end_l2_squared": 0.0,
            "delta_l2_squared": 0.0,
            "delta_max_abs": 0.0,
        }
    )
    global_accumulator = {
        "tensor_count": 0,
        "changed_tensor_count": 0,
        "element_count": 0,
        "changed_element_count": 0,
        "start_l2_squared": 0.0,
        "end_l2_squared": 0.0,
        "delta_l2_squared": 0.0,
        "delta_max_abs": 0.0,
    }

    for name in sorted(start):
        start_tensor = start[name]
        end_tensor = end[name]
        if start_tensor.shape != end_tensor.shape or start_tensor.dtype != end_tensor.dtype:
            raise ValueError(f"Tensor metadata differs for {name}")
        start_float = start_tensor.detach().to(dtype=torch.float64)
        end_float = end_tensor.detach().to(dtype=torch.float64)
        delta = end_float - start_float
        start_l2 = float(torch.linalg.vector_norm(start_float))
        end_l2 = float(torch.linalg.vector_norm(end_float))
        delta_l2 = float(torch.linalg.vector_norm(delta))
        delta_max_abs = float(delta.abs().max()) if delta.numel() else 0.0
        changed_elements = int(torch.count_nonzero(end_tensor != start_tensor).item())
        exact_equal = changed_elements == 0
        layer_match = LAYER_PATTERN.search(name)
        layer = int(layer_match.group(1)) if layer_match else None
        tensor_summary = {
            "name": name,
            "layer": layer,
            "shape": list(start_tensor.shape),
            "dtype": str(start_tensor.dtype).removeprefix("torch."),
            "element_count": start_tensor.numel(),
            "changed_element_count": changed_elements,
            "changed_element_fraction": (
                changed_elements / start_tensor.numel() if start_tensor.numel() else 0.0
            ),
            "exact_equal": exact_equal,
            "start_l2": start_l2,
            "end_l2": end_l2,
            "delta_l2": delta_l2,
            "relative_delta_l2_to_start": delta_l2 / start_l2 if start_l2 else None,
            "delta_max_abs": delta_max_abs,
        }
        per_tensor.append(tensor_summary)
        for accumulator in (
            global_accumulator,
            layer_accumulators[layer] if layer is not None else None,
        ):
            if accumulator is None:
                continue
            accumulator["tensor_count"] += 1
            accumulator["changed_tensor_count"] += int(not exact_equal)
            accumulator["element_count"] += start_tensor.numel()
            accumulator["changed_element_count"] += changed_elements
            accumulator["start_l2_squared"] += start_l2 * start_l2
            accumulator["end_l2_squared"] += end_l2 * end_l2
            accumulator["delta_l2_squared"] += delta_l2 * delta_l2
            accumulator["delta_max_abs"] = max(
                accumulator["delta_max_abs"], delta_max_abs
            )

    def finalize(accumulator: dict[str, Any]) -> dict[str, Any]:
        tensor_count = int(accumulator["tensor_count"])
        element_count = int(accumulator["element_count"])
        changed_tensor_count = int(accumulator["changed_tensor_count"])
        changed_element_count = int(accumulator["changed_element_count"])
        return {
            "tensor_count": tensor_count,
            "changed_tensor_count": changed_tensor_count,
            "unchanged_tensor_count": tensor_count - changed_tensor_count,
            "changed_tensor_fraction": (
                changed_tensor_count / tensor_count if tensor_count else 0.0
            ),
            "element_count": element_count,
            "changed_element_count": changed_element_count,
            "changed_element_fraction": (
                changed_element_count / element_count if element_count else 0.0
            ),
            "start_l2": math.sqrt(accumulator["start_l2_squared"]),
            "end_l2": math.sqrt(accumulator["end_l2_squared"]),
            "delta_l2": math.sqrt(accumulator["delta_l2_squared"]),
            "delta_max_abs": float(accumulator["delta_max_abs"]),
        }

    return {
        "start": {
            "path": str(start_path.resolve()),
            "size_bytes": start_path.stat().st_size,
            "sha256": sha256_file(start_path),
        },
        "end": {
            "path": str(end_path.resolve()),
            "size_bytes": end_path.stat().st_size,
            "sha256": sha256_file(end_path),
        },
        "global": finalize(global_accumulator),
        "by_layer": [
            {"layer": layer, **finalize(accumulator)}
            for layer, accumulator in sorted(layer_accumulators.items())
        ],
        "per_tensor": per_tensor,
    }


def verify_final_adapter(checkpoint_path: Path, final_path: Path) -> dict[str, Any]:
    import torch

    checkpoint = load_adapter(checkpoint_path)
    final = load_adapter(final_path)
    same_keys = checkpoint.keys() == final.keys()
    differing_tensors = (
        [name for name in checkpoint if not torch.equal(checkpoint[name], final[name])]
        if same_keys
        else []
    )
    checkpoint_sha256 = sha256_file(checkpoint_path)
    final_sha256 = sha256_file(final_path)
    return {
        "checkpoint_path": str(checkpoint_path.resolve()),
        "final_path": str(final_path.resolve()),
        "checkpoint_sha256": checkpoint_sha256,
        "final_sha256": final_sha256,
        "byte_identical": checkpoint_sha256 == final_sha256,
        "same_tensor_keys": same_keys,
        "tensor_exact_equal": same_keys and not differing_tensors,
        "differing_tensor_count": len(differing_tensors),
        "differing_tensors": differing_tensors,
    }


def moving_average(values: Sequence[float], width: int) -> list[float]:
    result: list[float] = []
    for index in range(len(values)):
        start = max(0, index - width + 1)
        window = values[start : index + 1]
        result.append(math.fsum(window) / len(window))
    return result


def svg_path(
    points: Sequence[tuple[float, float]], color: str, width: float, opacity: float
) -> str:
    coordinates = " ".join(f"{x:.3f},{y:.3f}" for x, y in points)
    return (
        f'<polyline points="{coordinates}" fill="none" stroke="{color}" '
        f'stroke-width="{width}" stroke-opacity="{opacity}" '
        'stroke-linejoin="round" stroke-linecap="round" />'
    )


def plot_svg(rows: Sequence[dict[str, Any]], run_name: str, target_task_ce: float) -> str:
    width, height = 1600, 1080
    left, right = 110, 1540
    plot_width = right - left
    panels = (
        {
            "top": 120,
            "bottom": 465,
            "title": "Task CE and frozen-teacher CE",
            "series": (
                ("Task CE", "delta/memory_keep_loss", "#2563eb"),
                ("Teacher CE", "delta/memory_teacher_loss", "#ea580c"),
            ),
            "target": target_task_ce,
        },
        {
            "top": 550,
            "bottom": 805,
            "title": "Total objective and KL term",
            "series": (
                ("Total loss", "loss", "#7c3aed"),
                ("KL loss", "delta/memory_kl_loss", "#dc2626"),
            ),
            "target": None,
        },
        {
            "top": 885,
            "bottom": 1020,
            "title": "Gradient norm",
            "series": (("Grad norm", "grad_norm", "#059669"),),
            "target": None,
        },
    )
    steps = [int(row["step"]) for row in rows]
    step_min, step_max = min(steps), max(steps)

    def x_position(step: int) -> float:
        return left + (step - step_min) / max(1, step_max - step_min) * plot_width

    fragments = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#f8fafc" />',
        f'<text x="{left}" y="52" font-family="sans-serif" font-size="28" font-weight="700" fill="#0f172a">Loss probe: {html.escape(run_name)}</text>',
        f'<text x="{left}" y="84" font-family="sans-serif" font-size="17" fill="#475569">Raw values are faint; solid lines are trailing 8-step means. Epoch boundaries are dashed.</text>',
    ]
    epoch_boundaries = [
        index
        for index in range(1, len(rows))
        if math.ceil(float(rows[index - 1]["epoch"]) - 1e-12)
        != math.ceil(float(rows[index]["epoch"]) - 1e-12)
    ]

    for panel in panels:
        top = int(panel["top"])
        bottom = int(panel["bottom"])
        panel_height = bottom - top
        all_values = [
            float(row[key])
            for _, key, _ in panel["series"]
            for row in rows
        ]
        target = panel["target"]
        if target is not None:
            all_values.append(float(target))
        value_min, value_max = min(all_values), max(all_values)
        padding = max((value_max - value_min) * 0.08, abs(value_max) * 0.01, 1e-12)
        value_min -= padding
        value_max += padding

        def y_position(value: float) -> float:
            return bottom - (value - value_min) / (value_max - value_min) * panel_height

        fragments.append(
            f'<text x="{left}" y="{top - 25}" font-family="sans-serif" font-size="20" font-weight="600" fill="#1e293b">{panel["title"]}</text>'
        )
        for tick_index in range(5):
            fraction = tick_index / 4
            y = bottom - fraction * panel_height
            value = value_min + fraction * (value_max - value_min)
            fragments.extend(
                [
                    f'<line x1="{left}" y1="{y:.3f}" x2="{right}" y2="{y:.3f}" stroke="#cbd5e1" stroke-width="1" />',
                    f'<text x="{left - 14}" y="{y + 5:.3f}" text-anchor="end" font-family="monospace" font-size="14" fill="#64748b">{value:.4g}</text>',
                ]
            )
        for boundary_index in epoch_boundaries:
            x = x_position(int(rows[boundary_index]["step"]) - 0.5)
            fragments.append(
                f'<line x1="{x:.3f}" y1="{top}" x2="{x:.3f}" y2="{bottom}" stroke="#94a3b8" stroke-width="1.5" stroke-dasharray="7 7" />'
            )
        if target is not None:
            y = y_position(float(target))
            fragments.extend(
                [
                    f'<line x1="{left}" y1="{y:.3f}" x2="{right}" y2="{y:.3f}" stroke="#0f766e" stroke-width="2" stroke-dasharray="10 6" />',
                    f'<text x="{right - 8}" y="{y - 8:.3f}" text-anchor="end" font-family="sans-serif" font-size="15" fill="#0f766e">target {target:g}</text>',
                ]
            )
        legend_x = left
        for label, key, color in panel["series"]:
            values = [float(row[key]) for row in rows]
            raw_points = [
                (x_position(step), y_position(value))
                for step, value in zip(steps, values, strict=True)
            ]
            smooth_points = [
                (x_position(step), y_position(value))
                for step, value in zip(steps, moving_average(values, 8), strict=True)
            ]
            fragments.extend(
                [
                    svg_path(raw_points, color, 1.3, 0.22),
                    svg_path(smooth_points, color, 3.2, 1.0),
                    f'<line x1="{legend_x}" y1="{top + 20}" x2="{legend_x + 35}" y2="{top + 20}" stroke="{color}" stroke-width="4" />',
                    f'<text x="{legend_x + 44}" y="{top + 26}" font-family="sans-serif" font-size="15" fill="#334155">{html.escape(label)}</text>',
                ]
            )
            legend_x += 190
        fragments.append(
            f'<rect x="{left}" y="{top}" width="{plot_width}" height="{panel_height}" fill="none" stroke="#64748b" stroke-width="1.5" />'
        )

    for tick_index in range(9):
        step = round(step_min + tick_index / 8 * (step_max - step_min))
        x = x_position(step)
        fragments.extend(
            [
                f'<line x1="{x:.3f}" y1="1020" x2="{x:.3f}" y2="1028" stroke="#475569" stroke-width="1.5" />',
                f'<text x="{x:.3f}" y="1052" text-anchor="middle" font-family="monospace" font-size="14" fill="#475569">{step}</text>',
            ]
        )
    fragments.append(
        f'<text x="{(left + right) / 2:.3f}" y="1074" text-anchor="middle" font-family="sans-serif" font-size="16" fill="#334155">optimizer step</text>'
    )
    fragments.append("</svg>")
    return "\n".join(fragments)


def render_png(
    rows: Sequence[dict[str, Any]], output_path: Path, run_name: str, target: float
) -> None:
    renderer = Path(__file__).with_name("render_loss_curve.py").resolve()
    candidates = [
        os.environ.get("PILLOW_PYTHON"),
        "/usr/bin/python3",
        shutil.which("python3"),
        shutil.which("python"),
    ]
    pillow_python = None
    for candidate in dict.fromkeys(item for item in candidates if item):
        check = subprocess.run(
            [candidate, "-c", "import PIL"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        if check.returncode == 0:
            pillow_python = candidate
            break
    if pillow_python is None:
        raise RuntimeError(
            "Rendering loss_curve.png requires a Pillow-enabled Python; set PILLOW_PYTHON"
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", suffix=".json", delete=False
    ) as handle:
        json.dump(
            {
                "run_name": run_name,
                "target_task_ce": target,
                "rows": rows,
            },
            handle,
        )
        temporary_spec = Path(handle.name)
    try:
        subprocess.run(
            [
                pillow_python,
                str(renderer),
                "--spec",
                str(temporary_spec),
                "--output",
                str(output_path),
            ],
            check=True,
        )
    finally:
        temporary_spec.unlink(missing_ok=True)


def main() -> None:
    args = parse_args()
    if args.window_size <= 0:
        raise ValueError("--window-size must be positive")
    if args.min_relative_improvement < 0:
        raise ValueError("--min-relative-improvement cannot be negative")

    run_dir = args.run_dir.expanduser().resolve()
    protocol_path = run_dir / "training_protocol.json"
    summary_path = run_dir / "training_summary.json"
    protocol = read_json(protocol_path)
    training_summary = read_json(summary_path)
    checkpoint_end = args.checkpoint_end or int(training_summary.get("global_step", 0))
    if checkpoint_end <= 0:
        checkpoint_dirs = sorted(
            int(path.name.removeprefix("checkpoint-"))
            for path in (run_dir / "trainer").glob("checkpoint-*")
            if path.name.removeprefix("checkpoint-").isdigit()
        )
        if not checkpoint_dirs:
            raise FileNotFoundError("No end checkpoint could be resolved")
        checkpoint_end = checkpoint_dirs[-1]
    trainer_state_path = (
        run_dir / "trainer" / f"checkpoint-{checkpoint_end}" / "trainer_state.json"
    )
    trainer_state = read_json(trainer_state_path)
    rows = contiguous_training_rows(trainer_state)
    if int(rows[-1]["step"]) != checkpoint_end:
        raise ValueError(
            f"End checkpoint is {checkpoint_end}, but final logged step is {rows[-1]['step']}"
        )

    per_epoch = grouped_metrics(
        rows,
        group_key=lambda row: math.ceil(float(row["epoch"]) - 1e-12),
        label_key="epoch",
    )
    per_window = grouped_metrics(
        rows,
        group_key=lambda row: (int(row["step"]) - 1) // args.window_size + 1,
        label_key="window",
    )
    first_epoch_metrics = per_epoch[0]["metrics"]
    final_epoch_metrics = per_epoch[-1]["metrics"]
    first_window_metrics = per_window[0]["metrics"]
    final_window_metrics = per_window[-1]["metrics"]
    first_epoch_task = float(first_epoch_metrics["task_ce"]["mean"])
    final_epoch_task = float(final_epoch_metrics["task_ce"]["mean"])
    task_decrease = first_epoch_task - final_epoch_task
    required_decrease = first_epoch_task * args.min_relative_improvement
    target_reached = final_epoch_task <= args.target_task_ce
    material_improvement = task_decrease >= required_decrease

    start_adapter_path = (
        run_dir
        / "trainer"
        / f"checkpoint-{args.checkpoint_start}"
        / "delta_mem_adapter.pt"
    )
    end_adapter_path = (
        run_dir
        / "trainer"
        / f"checkpoint-{checkpoint_end}"
        / "delta_mem_adapter.pt"
    )
    final_adapter_path = run_dir / "delta_mem_adapter.pt"
    movement = adapter_movement(start_adapter_path, end_adapter_path)
    final_verification = verify_final_adapter(end_adapter_path, final_adapter_path)
    objective = objective_identity(rows, protocol)
    gradient_values = [float(row["grad_norm"]) for row in rows]
    gradients = {
        "statistics": summarize(gradient_values),
        "finite_count": sum(math.isfinite(value) for value in gradient_values),
        "nonzero_count": sum(value != 0.0 for value in gradient_values),
        "all_finite": all(math.isfinite(value) for value in gradient_values),
        "all_nonzero": all(value != 0.0 for value in gradient_values),
    }
    objective_ok = bool(objective["identity_holds_within_tolerance"])
    final_adapter_ok = bool(final_verification["tensor_exact_equal"])
    decision = (
        "go"
        if target_reached and material_improvement and objective_ok and final_adapter_ok
        else "no_go"
    )
    reasons = []
    if not target_reached:
        reasons.append(
            f"final epoch mean task CE {final_epoch_task:.6f} remains above target {args.target_task_ce:.6f}"
        )
    if not material_improvement:
        reasons.append(
            f"task CE decreased {task_decrease:.6f}; required at least {required_decrease:.6f} ({args.min_relative_improvement:.2%})"
        )
    if not objective_ok:
        reasons.append("logged total loss does not match the configured objective")
    if not final_adapter_ok:
        reasons.append("final adapter differs from the end checkpoint")

    analysis = {
        "schema_version": 1,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "run": {
            "path": str(run_dir),
            "name": run_dir.name,
            "global_step": int(trainer_state["global_step"]),
            "max_steps": int(trainer_state["max_steps"]),
            "logged_training_steps": len(rows),
            "completed_epochs": float(trainer_state["epoch"]),
            "train_samples": int(training_summary["train_samples"]),
            "target_layers": training_summary["target_layers"],
        },
        "provenance": {
            "training_protocol": {
                "path": str(protocol_path),
                "sha256": sha256_file(protocol_path),
                "recorded_sha256": training_summary.get("training_protocol_sha256"),
            },
            "training_summary": {
                "path": str(summary_path),
                "sha256": sha256_file(summary_path),
            },
            "trainer_state": {
                "path": str(trainer_state_path),
                "sha256": sha256_file(trainer_state_path),
            },
            "train_file": protocol.get("train_file"),
            "tokenized_fingerprint": protocol.get("tokenized_fingerprint"),
        },
        "objective": {
            "memory_loss_mode": protocol.get("memory_loss_mode"),
            "memory_objective_version": protocol.get("memory_objective_version"),
            "episode_recent_messages": protocol.get("episode_recent_messages"),
            "episode_read_write_enabled": protocol.get("episode_read_write_enabled"),
            "max_length": protocol.get("max_length"),
            "max_write_length": protocol.get("max_write_length"),
            "teacher_max_length": protocol.get("teacher_max_length"),
            "identity": objective,
        },
        "metrics": {
            "overall": summarize_rows(rows),
            "by_epoch": per_epoch,
            "by_window": per_window,
            "first_to_final_epoch": comparison(
                first_epoch_metrics, final_epoch_metrics
            ),
            "first_to_final_window": comparison(
                first_window_metrics, final_window_metrics
            ),
        },
        "gradients": gradients,
        "adapter_movement": movement,
        "final_adapter_verification": final_verification,
        "verdict": {
            "decision": decision,
            "task_learning": "passed" if material_improvement else "failed",
            "target_task_ce": args.target_task_ce,
            "target_reached": target_reached,
            "final_epoch_mean_task_ce": final_epoch_task,
            "target_gap": final_epoch_task - args.target_task_ce,
            "first_epoch_mean_task_ce": first_epoch_task,
            "first_to_final_epoch_task_ce_decrease": task_decrease,
            "first_to_final_epoch_task_ce_relative_decrease": (
                task_decrease / first_epoch_task if first_epoch_task else None
            ),
            "minimum_relative_improvement": args.min_relative_improvement,
            "required_absolute_improvement": required_decrease,
            "objective_identity_ok": objective_ok,
            "gradients_all_finite_nonzero": gradients["all_finite"]
            and gradients["all_nonzero"],
            "adapter_moved": movement["global"]["changed_tensor_count"] > 0,
            "final_adapter_matches_end_checkpoint": final_adapter_ok,
            "reasons": reasons,
        },
    }

    output_json = args.output_json or run_dir / "loss_probe_analysis.json"
    output_png = args.output_png or run_dir / "loss_curve.png"
    write_json_atomic(output_json.expanduser().resolve(), analysis)
    render_png(rows, output_png.expanduser().resolve(), run_dir.name, args.target_task_ce)
    print(json.dumps(analysis["verdict"], indent=2, sort_keys=True))
    print(f"analysis_json={output_json.expanduser().resolve()}")
    print(f"loss_curve_png={output_png.expanduser().resolve()}")


if __name__ == "__main__":
    main()
