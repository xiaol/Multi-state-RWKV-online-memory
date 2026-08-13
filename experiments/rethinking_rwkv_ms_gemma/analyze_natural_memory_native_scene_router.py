#!/usr/bin/env python3
"""Select and validate a conservative scene-boundary router."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = Path(__file__).resolve().parents[2]
for import_root in (SCRIPT_DIR, PROJECT_ROOT):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

from experiments.rethinking_rwkv_ms_gemma import (  # noqa: E402
    analyze_novel_agent_eval as recovery,
)
from experiments.rethinking_rwkv_ms_gemma import (  # noqa: E402
    run_natural_memory_native_routed_benchmark as runner,
)


SCHEMA = "rwkv_ms_natural_memory_native_scene_router.v1"
PROTOCOL = SCRIPT_DIR / "natural_memory_native_scene_router_protocol_v1.json"


def canonical_sha256(value: Any) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_protocol() -> Mapping[str, Any]:
    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    receipt = protocol.pop("receipt")
    digest = canonical_sha256(protocol)
    if (
        digest != runner.SCENE_ROUTER_PROTOCOL_PAYLOAD_SHA256
        or receipt.get("payload_sha256") != digest
    ):
        raise ValueError("Scene router protocol receipt differs")
    return protocol


def read_condition(
    root: Path,
    condition: str,
) -> tuple[dict[int, set[int] | None], list[dict[str, Any]]]:
    records: dict[int, set[int] | None] = {}
    artifacts: list[dict[str, Any]] = []
    for shard_index in range(4):
        path = root / f"shard-{shard_index}" / f"scene.{condition}.jsonl"
        rows = [
            json.loads(raw_line)
            for raw_line in path.read_text(encoding="utf-8").splitlines()
            if raw_line.strip()
        ]
        artifacts.append(
            {
                "path": str(path.resolve()),
                "rows": len(rows),
                "sha256": sha256_file(path),
            }
        )
        for record in rows:
            index = int(record["line_index"])
            if (
                record.get("schema") != runner.SCHEMA
                or record.get("task") != "scene"
                or record.get("condition") != condition
                or record.get("protocol_payload_sha256")
                != runner.PROTOCOL_PAYLOAD_SHA256
                or record.get("shard_index") != shard_index
                or index % 4 != shard_index
                or index < runner.SELECTION_ROWS
            ):
                raise ValueError(f"Invalid scene router record: {path}:{index}")
            prediction = record.get("prediction")
            records[index] = (
                set(int(value) for value in prediction)
                if isinstance(prediction, list)
                else None
            )
    expected = set(range(4, 361))
    if set(records) != expected:
        raise ValueError(f"Incomplete scene {condition} records")
    return records, artifacts


def read_gold(dataset_root: Path) -> tuple[dict[int, set[int]], dict[int, str]]:
    path = dataset_root / str(runner.TASKS["scene"]["path"])
    gold: dict[int, set[int]] = {}
    row_hashes: dict[int, str] = {}
    with path.open("r", encoding="utf-8") as handle:
        for index, raw_line in enumerate(line for line in handle if line.strip()):
            if index < 4:
                continue
            row = json.loads(raw_line)
            parsed = recovery.extract_json(str(row["messages"][-1]["content"]))
            gold[index] = recovery.strict_gold_boundaries(parsed)
            row_hashes[index] = hashlib.sha256(
                raw_line.rstrip("\n").encode("utf-8")
            ).hexdigest()
    return gold, row_hashes


def route(name: str, base: set[int], memory: set[int] | None) -> set[int]:
    memory_set = memory or set()
    if name == "base":
        return set(base)
    if name == "memory":
        return set(memory_set)
    if name == "intersection":
        return set(base & memory_set)
    if name == "union":
        return set(base | memory_set)
    if name == "conditional_intersection":
        return set(base & memory_set) if memory_set else set(base)
    raise ValueError(f"Unknown scene router: {name}")


def metrics(
    indices: Sequence[int],
    *,
    candidate: str,
    base: Mapping[int, set[int] | None],
    memory: Mapping[int, set[int] | None],
    gold: Mapping[int, set[int]],
) -> Mapping[str, Any]:
    tp = fp = fn = covered = 0
    for index in indices:
        base_set = base[index]
        if base_set is None:
            predicted = set()
        else:
            covered += 1
            predicted = route(candidate, base_set, memory[index])
        expected = gold[index]
        tp += len(predicted & expected)
        fp += len(predicted - expected)
        fn += len(expected - predicted)
    precision = 0.0 if tp + fp == 0 else tp / (tp + fp)
    recall = 0.0 if tp + fn == 0 else tp / (tp + fn)
    denominator = 2 * tp + fp + fn
    return {
        "rows": len(indices),
        "coverage": covered / len(indices),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": precision,
        "recall": recall,
        "micro_f1": 0.0 if denominator == 0 else 2 * tp / denominator,
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    protocol = validate_protocol()
    root = args.input_root.expanduser().resolve(strict=True)
    dataset_root = args.dataset_root.expanduser().resolve(strict=True)
    output = args.output.expanduser().resolve()
    if output.exists():
        raise ValueError(f"Scene router output must be fresh: {output}")
    base, base_artifacts = read_condition(root, "base")
    memory, memory_artifacts = read_condition(root, "memory")
    gold, _ = read_gold(dataset_root)
    fit_indices = tuple(index for index in sorted(gold) if index % 2 == 0)
    holdout_indices = tuple(index for index in sorted(gold) if index % 2 == 1)
    candidate_names = tuple(protocol["candidate_routers"])
    fit_metrics = {
        candidate: metrics(
            fit_indices,
            candidate=candidate,
            base=base,
            memory=memory,
            gold=gold,
        )
        for candidate in candidate_names
    }
    selected = min(
        candidate_names,
        key=lambda candidate: (
            -float(fit_metrics[candidate]["micro_f1"]),
            -float(fit_metrics[candidate]["precision"]),
            -float(fit_metrics[candidate]["recall"]),
            candidate,
        ),
    )
    holdout_metrics = {
        "base": metrics(
            holdout_indices,
            candidate="base",
            base=base,
            memory=memory,
            gold=gold,
        ),
        "selected": metrics(
            holdout_indices,
            candidate=selected,
            base=base,
            memory=memory,
            gold=gold,
        ),
    }
    fit_gain = (
        float(fit_metrics[selected]["micro_f1"])
        - float(fit_metrics["base"]["micro_f1"])
    )
    holdout_gain = (
        float(holdout_metrics["selected"]["micro_f1"])
        - float(holdout_metrics["base"]["micro_f1"])
    )
    gates = {
        "fit_gain_at_least_0.01": fit_gain >= 0.01,
        "holdout_gain_at_least_0.005": holdout_gain >= 0.005,
        "holdout_no_regression": holdout_gain >= 0.0,
        "holdout_coverage_at_least_0.95": (
            float(holdout_metrics["selected"]["coverage"]) >= 0.95
        ),
    }
    gates["passed"] = all(gates.values())
    result: dict[str, Any] = {
        "schema": SCHEMA,
        "protocol_payload_sha256": runner.SCENE_ROUTER_PROTOCOL_PAYLOAD_SHA256,
        "scope": {
            "rows": len(gold),
            "fit_rows": len(fit_indices),
            "holdout_rows": len(holdout_indices),
            "fit_partition": "even source index",
            "holdout_partition": "odd source index",
            "protected_splits_opened": [],
        },
        "fit": {
            "candidates": fit_metrics,
            "selected": selected,
            "selected_minus_base": fit_gain,
        },
        "holdout": {
            **holdout_metrics,
            "selected_minus_base": holdout_gain,
        },
        "gates": gates,
        "publisher_validation_authorized": gates["passed"],
        "provenance": {
            "base": base_artifacts,
            "memory": memory_artifacts,
        },
    }
    result["receipt"] = {
        "algorithm": "sha256",
        "payload_scope": "canonical_scene_router_without_receipt",
        "payload_sha256": canonical_sha256(result),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, ensure_ascii=True, sort_keys=True))
    return 0 if gates["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
