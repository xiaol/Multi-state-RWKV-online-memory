#!/usr/bin/env python3
"""Sign the frozen aligned-vector state-scalar materiality screen."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from experiments.rethinking_rwkv_ms_gemma import (
    run_natural_memory_native_projected_rwkv_hybrid_benchmark_eval as base,
)


PROTOCOL_PAYLOAD_SHA256 = "8edab2dfe3d0995d0e53b313297d008b7176a4415ca8f0ed6480362d9c894816"
SCHEMA = "rwkv_ms_natural_memory_native_aligned_vector_gate_state_scalar_screen_result.v1"
SHARD_HASHES = {
    "shard-0.jsonl": "2577b0060a65843590c5d2599653bf43e9d47026935df973e23c7268a167c976",
    "shard-1.jsonl": "ae40199368d63f592b53f3bf9feb14dd83f5bb4f28e053ad9a8bd84f5dca68e6",
    "shard-2.jsonl": "dde156c85cc4c12c3eb9790addd98db5c6e6dad6d7cf399c0d02d97741a1fccf",
    "shard-3.jsonl": "7c24e653fcf8952acdfcb55be563988bc22ab2576386e71f0142820f88ab2e7b",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_sha256(value: Any) -> str:
    payload = json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def load_rows(root: Path) -> tuple[list[Mapping[str, Any]], list[Mapping[str, Any]]]:
    rows: list[Mapping[str, Any]] = []
    provenance: list[Mapping[str, Any]] = []
    for filename, expected in sorted(SHARD_HASHES.items()):
        path = root / filename
        if sha256_file(path) != expected:
            raise ValueError(f"probe shard hash differs: {path}")
        shard_rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
        if len(shard_rows) != len({int(row["source_index"]) for row in shard_rows}):
            raise ValueError(f"duplicate source index in {path}")
        rows.extend(shard_rows)
        provenance.append({"path": str(path), "rows": len(shard_rows), "sha256": expected})
    rows.sort(key=lambda row: int(row["source_index"]))
    if len(rows) != 220 or [int(row["source_index"]) for row in rows] != sorted({int(row["source_index"]) for row in rows}):
        raise ValueError("probe row coverage differs")
    payload = [
        {"source_index": int(row["source_index"]), "row_sha256": str(row["row_sha256"])}
        for row in rows
    ]
    if canonical_sha256(payload) != base.AUTHORIZED_ROWS_PAYLOAD_SHA256:
        raise ValueError("probe rows are not the locked native endpoint")
    return rows, provenance


def ridge_alpha(rows: Sequence[Mapping[str, Any]], lam: float) -> np.ndarray:
    correct = np.asarray([row["correct"] for row in rows], dtype=np.float64)
    negatives = [np.asarray([row[name] for row in rows], dtype=np.float64) for name in ("matched_donor", "layer_permuted", "zero")]
    matrix = np.concatenate([correct, *negatives], axis=0)
    targets = np.concatenate([
        np.ones(len(rows), dtype=np.float64),
        -np.ones(len(rows) * len(negatives), dtype=np.float64),
    ])
    return np.linalg.solve(matrix.T @ matrix + lam * np.eye(matrix.shape[1]), matrix.T @ targets)


def metrics(rows: Sequence[Mapping[str, Any]], alpha: np.ndarray) -> Mapping[str, Any]:
    correct = np.asarray([row["correct"] for row in rows], dtype=np.float64)
    result: dict[str, Any] = {}
    for name in ("zero", "matched_donor", "layer_permuted"):
        negative = np.asarray([row[name] for row in rows], dtype=np.float64)
        gap = (correct @ alpha) - (negative @ alpha)
        result[name] = {
            "mean_gap": float(gap.mean()),
            "median_gap": float(np.median(gap)),
            "pairwise_positive_fraction": float(np.mean(gap > 0.0)),
            "p05_gap": float(np.quantile(gap, 0.05)),
            "p95_gap": float(np.quantile(gap, 0.95)),
        }
    return result


def analyze(root: Path) -> Mapping[str, Any]:
    rows, provenance = load_rows(root)
    candidates: list[Mapping[str, Any]] = []
    for lam in (1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0):
        alpha = ridge_alpha(rows, lam)
        values = metrics(rows, alpha)
        candidates.append({
            "ridge_lambda": lam,
            "alpha_l2": float(np.linalg.norm(alpha)),
            "metrics": values,
        })
    selected = max(
        candidates,
        key=lambda item: (
            item["metrics"]["matched_donor"]["pairwise_positive_fraction"],
            -item["ridge_lambda"],
        ),
    )
    donor_fraction = float(selected["metrics"]["matched_donor"]["pairwise_positive_fraction"])
    passed = donor_fraction >= 0.95
    result: dict[str, Any] = {
        "schema": SCHEMA,
        "status": "state_scalar_screen_passed_generation_calibration_authorized" if passed else "state_scalar_screen_failed_donor_separation_blocked",
        "passed": passed,
        "generation_calibration_authorized": passed,
        "protocol_payload_sha256": PROTOCOL_PAYLOAD_SHA256,
        "feature": "per-layer recurrent-state L2 norms",
        "rows": len(rows),
        "provenance": provenance,
        "candidate_sweeps": candidates,
        "selected_candidate": selected,
        "decision": {
            "minimum_matched_donor_pairwise_separation": 0.95,
            "observed_matched_donor_pairwise_separation": donor_fraction,
            "layer_permutation_separated": selected["metrics"]["layer_permuted"]["pairwise_positive_fraction"] >= 0.95,
            "zero_separated": selected["metrics"]["zero"]["pairwise_positive_fraction"] >= 0.95,
            "on_failure": "retire_aligned_vector_branch_and_return_to_learned_state_identity",
        },
        "protected_splits_opened": [],
    }
    result["receipt"] = {
        "algorithm": "sha256",
        "payload_scope": "canonical_result_without_receipt",
        "payload_sha256": canonical_sha256(result),
    }
    return result


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--probe-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    result = analyze(args.probe_root.expanduser().resolve())
    args.output.expanduser().resolve().write_text(
        json.dumps(result, ensure_ascii=True, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"status": result["status"], "passed": result["passed"], "decision": result["decision"], "receipt": result["receipt"]["payload_sha256"]}, sort_keys=True))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
