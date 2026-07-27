#!/usr/bin/env python3
"""Render a delta-Mem loss-probe plot specification with Pillow."""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
from typing import Any, Sequence

from PIL import Image, ImageDraw, ImageFont


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def load_font(size: int, *, bold: bool = False, mono: bool = False):
    if mono:
        candidates = (
            "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
            "/usr/share/fonts/TTF/DejaVuSansMono.ttf",
        )
    elif bold:
        candidates = (
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
            "/usr/share/fonts/TTF/DejaVuSans-Bold.ttf",
        )
    else:
        candidates = (
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
            "/usr/share/fonts/TTF/DejaVuSans.ttf",
        )
    for candidate in candidates:
        if Path(candidate).is_file():
            return ImageFont.truetype(candidate, size=size)
    return ImageFont.load_default(size=size)


def moving_average(values: Sequence[float], width: int) -> list[float]:
    result: list[float] = []
    for index in range(len(values)):
        start = max(0, index - width + 1)
        window = values[start : index + 1]
        result.append(math.fsum(window) / len(window))
    return result


def draw_dashed_line(
    draw: ImageDraw.ImageDraw,
    start: tuple[float, float],
    end: tuple[float, float],
    *,
    fill: str,
    width: int,
    dash: int = 8,
    gap: int = 7,
) -> None:
    start_x, start_y = start
    end_x, end_y = end
    distance = math.hypot(end_x - start_x, end_y - start_y)
    if distance == 0:
        return
    offset = 0.0
    while offset < distance:
        segment_end = min(offset + dash, distance)
        fraction_start = offset / distance
        fraction_end = segment_end / distance
        draw.line(
            (
                start_x + (end_x - start_x) * fraction_start,
                start_y + (end_y - start_y) * fraction_start,
                start_x + (end_x - start_x) * fraction_end,
                start_y + (end_y - start_y) * fraction_end,
            ),
            fill=fill,
            width=width,
        )
        offset += dash + gap


def draw_plot(spec: dict[str, Any], output_path: Path) -> None:
    rows = spec["rows"]
    run_name = str(spec["run_name"])
    target_task_ce = float(spec["target_task_ce"])
    width, height = 1600, 1080
    left, right = 110, 1540
    plot_width = right - left
    panels = (
        {
            "top": 120,
            "bottom": 465,
            "title": "Task CE and frozen-teacher CE",
            "series": (
                ("Task CE", "delta/memory_keep_loss", "#2563eb", "#bfdbfe"),
                ("Teacher CE", "delta/memory_teacher_loss", "#ea580c", "#fed7aa"),
            ),
            "target": target_task_ce,
        },
        {
            "top": 550,
            "bottom": 805,
            "title": "Total objective and KL term",
            "series": (
                ("Total loss", "loss", "#7c3aed", "#ddd6fe"),
                ("KL loss", "delta/memory_kl_loss", "#dc2626", "#fecaca"),
            ),
            "target": None,
        },
        {
            "top": 885,
            "bottom": 1020,
            "title": "Gradient norm",
            "series": (("Grad norm", "grad_norm", "#059669", "#a7f3d0"),),
            "target": None,
        },
    )
    steps = [int(row["step"]) for row in rows]
    step_min, step_max = min(steps), max(steps)
    image = Image.new("RGB", (width, height), "#f8fafc")
    draw = ImageDraw.Draw(image)
    title_font = load_font(28, bold=True)
    subtitle_font = load_font(17)
    panel_font = load_font(20, bold=True)
    label_font = load_font(15)
    tick_font = load_font(14, mono=True)

    def x_position(step: float) -> float:
        return left + (step - step_min) / max(1, step_max - step_min) * plot_width

    draw.text(
        (left, 30),
        f"Loss probe: {run_name}",
        font=title_font,
        fill="#0f172a",
    )
    draw.text(
        (left, 66),
        "Raw values are pale; solid lines are trailing 8-step means. Epoch boundaries are dashed.",
        font=subtitle_font,
        fill="#475569",
    )
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
            for _, key, _, _ in panel["series"]
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

        draw.text(
            (left, top - 32),
            panel["title"],
            font=panel_font,
            fill="#1e293b",
        )
        for tick_index in range(5):
            fraction = tick_index / 4
            y = bottom - fraction * panel_height
            value = value_min + fraction * (value_max - value_min)
            draw.line((left, y, right, y), fill="#cbd5e1", width=1)
            text = f"{value:.4g}"
            text_box = draw.textbbox((0, 0), text, font=tick_font)
            text_width = text_box[2] - text_box[0]
            draw.text(
                (left - 14 - text_width, y - 8), text, font=tick_font, fill="#64748b"
            )
        for boundary_index in epoch_boundaries:
            x = x_position(int(rows[boundary_index]["step"]) - 0.5)
            draw_dashed_line(
                draw,
                (x, top),
                (x, bottom),
                fill="#94a3b8",
                width=2,
            )
        if target is not None:
            y = y_position(float(target))
            draw_dashed_line(
                draw,
                (left, y),
                (right, y),
                fill="#0f766e",
                width=2,
                dash=11,
                gap=6,
            )
            label = f"target {target:g}"
            label_box = draw.textbbox((0, 0), label, font=label_font)
            label_width = label_box[2] - label_box[0]
            draw.text(
                (right - label_width - 8, y - 23),
                label,
                font=label_font,
                fill="#0f766e",
            )
        legend_x = left
        for label, key, color, pale_color in panel["series"]:
            values = [float(row[key]) for row in rows]
            raw_points = [
                (x_position(step), y_position(value))
                for step, value in zip(steps, values, strict=True)
            ]
            smooth_points = [
                (x_position(step), y_position(value))
                for step, value in zip(steps, moving_average(values, 8), strict=True)
            ]
            draw.line(raw_points, fill=pale_color, width=2, joint="curve")
            draw.line(smooth_points, fill=color, width=4, joint="curve")
            draw.line(
                (legend_x, top + 20, legend_x + 35, top + 20), fill=color, width=4
            )
            draw.text(
                (legend_x + 44, top + 11), label, font=label_font, fill="#334155"
            )
            legend_x += 190
        draw.rectangle((left, top, right, bottom), outline="#64748b", width=2)

    for tick_index in range(9):
        step = round(step_min + tick_index / 8 * (step_max - step_min))
        x = x_position(step)
        draw.line((x, 1020, x, 1028), fill="#475569", width=2)
        text = str(step)
        text_box = draw.textbbox((0, 0), text, font=tick_font)
        text_width = text_box[2] - text_box[0]
        draw.text((x - text_width / 2, 1034), text, font=tick_font, fill="#475569")
    axis_label = "optimizer step"
    axis_box = draw.textbbox((0, 0), axis_label, font=label_font)
    axis_width = axis_box[2] - axis_box[0]
    draw.text(
        ((left + right - axis_width) / 2, 1060),
        axis_label,
        font=label_font,
        fill="#334155",
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_name(f".{output_path.stem}.{os.getpid()}.tmp.png")
    image.save(temporary, format="PNG", optimize=True)
    temporary.replace(output_path)


def main() -> None:
    args = parse_args()
    spec = json.loads(args.spec.read_text(encoding="utf-8"))
    draw_plot(spec, args.output.expanduser().resolve())


if __name__ == "__main__":
    main()
