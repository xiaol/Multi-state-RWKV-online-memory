#!/usr/bin/env python3
"""Screen natural-memory split seeds for generator feasibility only."""

from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.rethinking_rwkv_ms_gemma import prepare_natural_memory_gate as source


SCHEMA = "rwkv_ms_natural_memory_split_seed_feasibility.v1"
BALANCE_LIMIT = 0.03


def canonical_json(payload: Any) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"))


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def source_paths_from_args(args: argparse.Namespace) -> dict[str, Path]:
    return {
        "attribution": args.attribution_train,
        "narrative": args.narrative_train,
        "scene": args.scene_train,
    }


def component_inputs(
    paths: Mapping[str, Path],
) -> tuple[
    list[source.Item],
    dict[str, list[str]],
    dict[str, dict[str, int]],
    dict[str, Any],
]:
    items, source_audit = source.load_items(paths, enforce_pinned_sources=True)
    component_rows: dict[str, set[str]] = defaultdict(set)
    component_task_weights: dict[str, dict[str, int]] = defaultdict(
        lambda: defaultdict(int)
    )
    for item in items:
        component_rows[item.component_id].add(item.row_id)
        component_task_weights[item.component_id][item.task] += 1
    return (
        items,
        {
            component: sorted(row_ids)
            for component, row_ids in component_rows.items()
        },
        {
            component: dict(task_counts)
            for component, task_counts in component_task_weights.items()
        },
        source_audit,
    )


def screen_seeds(
    *,
    paths: Mapping[str, Path],
    start_seed: int,
    count: int,
    selected_count: int,
) -> dict[str, Any]:
    if count <= 0:
        raise ValueError("Seed count must be positive")
    if selected_count <= 0 or selected_count > count:
        raise ValueError("Selected count must be within the screened seed count")
    items, component_rows, component_task_weights, source_audit = component_inputs(
        paths
    )
    screened: list[dict[str, Any]] = []
    for seed in range(start_seed, start_seed + count):
        assignments = source.assign_component_splits(
            component_rows,
            component_task_weights,
            seed=seed,
        )
        audit = source._split_audit(items, assignments)
        maximum_error = float(audit["maximum_item_fraction_abs_error"])
        screened.append(
            {
                "seed": seed,
                "feasible": maximum_error <= BALANCE_LIMIT,
                "maximum_item_fraction_abs_error": maximum_error,
                "component_assignment_sha256": sha256_text(
                    canonical_json(assignments)
                ),
            }
        )
    selected = [entry for entry in screened if entry["feasible"]][:selected_count]
    if len(selected) != selected_count:
        raise ValueError(
            f"Only {len(selected)} feasible seeds found; required {selected_count}"
        )
    payload: dict[str, Any] = {
        "schema": SCHEMA,
        "created_at": utc_now(),
        "screening_scope": {
            "classification": "generator_feasibility_only",
            "allowed_inputs": "publisher_training_splits_only",
            "balance_limit": BALANCE_LIMIT,
            "model_binding_opened": False,
            "episodes_generated": 0,
            "optimizer_updates": 0,
            "development_rows_generated": 0,
            "sealed_rows_generated": 0,
            "native_validation_opened": False,
            "test_opened": False,
            "hard32_opened": False,
        },
        "generator": {
            "path": str(Path(source.__file__).resolve()),
            "sha256": source.sha256_file(Path(source.__file__).resolve()),
            "assignment": "hash_ranked_task_stratified_weighted_component_v1",
        },
        "sources": source_audit["sources"],
        "source_row_count": source_audit["row_count"],
        "component_count": source_audit["component_count"],
        "start_seed": start_seed,
        "count": count,
        "selection_rule": (
            f"first_{selected_count}_feasible_seeds_in_ascending_screen_order"
        ),
        "selected_seeds": [entry["seed"] for entry in selected],
        "screened": screened,
    }
    payload["receipt"] = {
        "algorithm": "sha256",
        "payload_scope": "canonical_screen_without_receipt",
        "payload_sha256": sha256_text(canonical_json(payload)),
    }
    return payload


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    defaults = source._default_cli_sources()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--start-seed", type=int, required=True)
    parser.add_argument("--count", type=int, default=64)
    parser.add_argument("--selected-count", type=int, default=2)
    parser.add_argument("--attribution-train", type=Path, default=defaults["attribution"])
    parser.add_argument("--narrative-train", type=Path, default=defaults["narrative"])
    parser.add_argument("--scene-train", type=Path, default=defaults["scene"])
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    output = args.output.expanduser().resolve()
    if output.exists():
        raise ValueError(f"Output must be fresh: {output}")
    payload = screen_seeds(
        paths=source_paths_from_args(args),
        start_seed=args.start_seed,
        count=args.count,
        selected_count=args.selected_count,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "output": str(output),
                "selected_seeds": payload["selected_seeds"],
                "receipt": payload["receipt"]["payload_sha256"],
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
