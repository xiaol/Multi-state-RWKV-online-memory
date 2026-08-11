#!/usr/bin/env python3
"""Materialize a leakage-safe native development split from publisher TRAIN rows."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from experiments.rethinking_rwkv_ms_gemma import prepare_natural_memory_gate as source


SCHEMA = "rwkv_ms_natural_memory_native_development.v1"
SPLIT_SEED = 20260812
FIT_FRACTION = 0.8
DEVELOPMENT_FRACTION = 0.2
EVOLUTION_PROTOCOL = Path(__file__).with_name(
    "natural_memory_native_evolution_protocol_v1.json"
)
EVOLUTION_PROTOCOL_PAYLOAD_SHA256 = (
    "219ea71766d3859569f1f49cc428eab5411e4ce8fa2bcae0a333ddfe78bbc749"
)
TASK_DIRECTORIES = {
    "attribution": "v3.2-attribution-best-candidate",
    "narrative": "v3.2-narrative-type-classification",
    "scene": "v4-scene-boundary-detection",
}
OUTPUT_SPLITS = {
    "fit": "train_derived_fit",
    "development": "train_derived_development",
}


def canonical_sha256(value: Any) -> str:
    return source.sha256_text(source.canonical_json(value))


def load_evolution_protocol(path: Path = EVOLUTION_PROTOCOL) -> Mapping[str, Any]:
    resolved = path.expanduser().resolve(strict=True)
    protocol = json.loads(resolved.read_text(encoding="utf-8"))
    if not isinstance(protocol, Mapping):
        raise ValueError("Evolution protocol must be an object")
    receipt = protocol.get("receipt")
    if not isinstance(receipt, Mapping):
        raise ValueError("Evolution protocol receipt is missing")
    unsigned = dict(protocol)
    unsigned.pop("receipt")
    payload_sha256 = canonical_sha256(unsigned)
    if (
        receipt.get("payload_sha256") != payload_sha256
        or payload_sha256 != EVOLUTION_PROTOCOL_PAYLOAD_SHA256
    ):
        raise ValueError("Evolution protocol payload hash differs")
    return protocol


def assign_native_component_splits(
    component_rows: Mapping[str, Sequence[str]],
    row_task: Mapping[str, str],
    *,
    seed: int = SPLIT_SEED,
) -> dict[str, str]:
    """Assign atomic components while balancing 80/20 row counts per task."""

    if not component_rows:
        raise ValueError("Native development split requires passage components")
    if set(row_task) != {
        row_id for row_ids in component_rows.values() for row_id in row_ids
    }:
        raise ValueError("Component rows and row-task bindings differ")
    ranked = sorted(
        component_rows,
        key=lambda component: source.sha256_text(f"{seed}:{component}"),
    )
    tasks = tuple(sorted(set(row_task.values())))
    dimensions = ("__all__", *tasks)
    weights: dict[str, dict[str, int]] = {}
    for component, row_ids in component_rows.items():
        counts = Counter(row_task[row_id] for row_id in row_ids)
        weights[component] = {
            "__all__": len(row_ids),
            **{task: counts[task] for task in tasks},
        }
    totals = {
        dimension: sum(weights[component][dimension] for component in ranked)
        for dimension in dimensions
    }
    fractions = {"fit": FIT_FRACTION, "development": DEVELOPMENT_FRACTION}
    targets = {
        split: {
            dimension: totals[dimension] * fraction
            for dimension in dimensions
        }
        for split, fraction in fractions.items()
    }
    loads = {
        split: {dimension: 0 for dimension in dimensions}
        for split in fractions
    }
    assignments: dict[str, str] = {}
    for component in ranked:
        present = [
            dimension
            for dimension in dimensions
            if weights[component][dimension] > 0
        ]
        choice = max(
            fractions,
            key=lambda split: (
                sum(
                    (targets[split][dimension] - loads[split][dimension])
                    / max(targets[split][dimension], 1.0)
                    for dimension in present
                )
                / len(present),
                -tuple(fractions).index(split),
            ),
        )
        assignments[component] = choice
        for dimension in dimensions:
            loads[choice][dimension] += weights[component][dimension]
    if set(assignments.values()) != set(fractions):
        raise ValueError("Native split assignment left one partition empty")
    return assignments


def _raw_line_maps(
    source_paths: Mapping[str, Path],
) -> dict[str, list[str]]:
    lines_by_task: dict[str, list[str]] = {}
    for task, requested in source_paths.items():
        path = requested.expanduser().resolve(strict=True)
        if source._forbidden_source_path(path):
            raise ValueError(f"Only publisher TRAIN rows may be materialized: {path}")
        rows: list[str] = []
        for raw_line in path.read_text(encoding="utf-8").splitlines():
            if not raw_line.strip():
                continue
            json.loads(raw_line)
            rows.append(raw_line)
        lines_by_task[task] = rows
    return lines_by_task


def build_native_development(
    output_dir: Path,
    *,
    source_paths: Mapping[str, Path] | None = None,
    split_seed: int = SPLIT_SEED,
) -> Mapping[str, Any]:
    source.require_hf_mirror()
    protocol = load_evolution_protocol()
    if split_seed != protocol["native_data_contract"]["split_seed"]:
        raise ValueError("Split seed differs from the evolution protocol")
    requested_output = output_dir.expanduser()
    if requested_output.is_symlink():
        raise ValueError("Native development output may not be a symbolic link")
    resolved_output = requested_output.resolve()
    if resolved_output.exists():
        raise ValueError(f"Native development output must be fresh: {resolved_output}")

    paths = dict(source_paths or source.default_source_paths())
    raw_rows, source_stats = source._load_raw_rows(
        paths,
        enforce_pinned_sources=True,
    )
    row_to_component, component_rows, signature_audit = source._component_assignments(
        raw_rows
    )
    row_task = {row.row_id: row.task for row in raw_rows}
    assignments = assign_native_component_splits(
        component_rows,
        row_task,
        seed=split_seed,
    )
    row_split = {
        row_id: assignments[component]
        for row_id, component in row_to_component.items()
    }
    fit_signatures = set().union(
        *(row.signatures for row in raw_rows if row_split[row.row_id] == "fit")
    )
    development_signatures = set().union(
        *(
            row.signatures
            for row in raw_rows
            if row_split[row.row_id] == "development"
        )
    )
    crossing_signatures = fit_signatures & development_signatures
    if crossing_signatures:
        raise RuntimeError("Normalized 32-character shingles cross native splits")

    raw_lines = _raw_line_maps(paths)
    resolved_output.mkdir(parents=True, exist_ok=False)
    output_files: dict[str, dict[str, Any]] = {}
    row_counts: Counter[tuple[str, str]] = Counter()
    ordered_row_ids: dict[str, dict[str, list[str]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for task in TASK_DIRECTORIES:
        task_rows = sorted(
            (row for row in raw_rows if row.task == task),
            key=lambda row: row.row_ordinal,
        )
        if len(raw_lines[task]) != len(task_rows) or any(
            source._canonical_row_id(task, json.loads(raw_line)) != row.row_id
            for raw_line, row in zip(raw_lines[task], task_rows, strict=True)
        ):
            raise RuntimeError(f"Raw publisher rows differ from validated {task} rows")
        task_dir = resolved_output / TASK_DIRECTORIES[task]
        task_dir.mkdir(parents=True, exist_ok=False)
        for split, output_name in OUTPUT_SPLITS.items():
            selected = sorted(
                (
                    row
                    for row in raw_rows
                    if row.task == task and row_split[row.row_id] == split
                ),
                key=lambda row: row.row_ordinal,
            )
            output_path = task_dir / f"{output_name}.jsonl"
            output_path.write_text(
                "".join(
                    raw_lines[task][row.row_ordinal] + "\n" for row in selected
                ),
                encoding="utf-8",
            )
            row_counts[(split, task)] = len(selected)
            ordered_row_ids[split][task] = [row.row_id for row in selected]
            output_files[f"{task}:{split}"] = {
                "path": str(output_path),
                "sha256": source.sha256_file(output_path),
                "rows": len(selected),
            }

    split_components = {
        split: sorted(
            component
            for component, assigned in assignments.items()
            if assigned == split
        )
        for split in OUTPUT_SPLITS
    }
    manifest: dict[str, Any] = {
        "schema": SCHEMA,
        "evolution_protocol": {
            "path": str(EVOLUTION_PROTOCOL.resolve(strict=True)),
            "file_sha256": source.sha256_file(EVOLUTION_PROTOCOL),
            "payload_sha256": EVOLUTION_PROTOCOL_PAYLOAD_SHA256,
        },
        "hf_endpoint": source.HF_MIRROR,
        "split_seed": split_seed,
        "split_unit": "connected_passage_component",
        "fractions": {
            "fit": FIT_FRACTION,
            "development": DEVELOPMENT_FRACTION,
        },
        "sources": source_stats,
        "source_semantic_rows": len(raw_rows),
        "component_count": len(component_rows),
        "component_counts": {
            split: len(components) for split, components in split_components.items()
        },
        "row_counts": {
            split: {
                task: row_counts[(split, task)] for task in TASK_DIRECTORIES
            }
            for split in OUTPUT_SPLITS
        },
        "ordered_row_ids_sha256": {
            split: {
                task: canonical_sha256(ordered_row_ids[split][task])
                for task in TASK_DIRECTORIES
            }
            for split in OUTPUT_SPLITS
        },
        "component_ids_sha256": {
            split: canonical_sha256(components)
            for split, components in split_components.items()
        },
        "leakage_audit": {
            **signature_audit,
            "cross_split_normalized_32_character_shingle_overlap": len(
                crossing_signatures
            ),
            "component_ids_crossing_splits": 0,
            "protected_splits_opened": [],
        },
        "outputs": output_files,
    }
    manifest["receipt"] = {
        "algorithm": "sha256",
        "payload_scope": "canonical_manifest_without_receipt",
        "payload_sha256": canonical_sha256(manifest),
    }
    manifest_path = resolved_output / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--split-seed", type=int, default=SPLIT_SEED)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    manifest = build_native_development(
        args.output_dir,
        split_seed=args.split_seed,
    )
    print(
        json.dumps(
            {
                "output_dir": str(args.output_dir.expanduser().resolve()),
                "row_counts": manifest["row_counts"],
                "receipt": manifest["receipt"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
