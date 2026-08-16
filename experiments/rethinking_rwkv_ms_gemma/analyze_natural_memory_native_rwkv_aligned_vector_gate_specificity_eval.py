#!/usr/bin/env python3
"""Aggregate and sign the aligned-gate native generation benchmark."""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import json
from pathlib import Path
import sys
from typing import Any, Iterator, Mapping, Sequence


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = Path(__file__).resolve().parents[2]
for import_root in (SCRIPT_DIR, PROJECT_ROOT):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

from experiments.rethinking_rwkv_ms_gemma import (  # noqa: E402
    analyze_natural_memory_native_rwkv_vector_gate_eval as base,
)
from experiments.rethinking_rwkv_ms_gemma import (  # noqa: E402
    run_natural_memory_native_rwkv_aligned_vector_gate_specificity_eval as evaluation,
)


SCHEMA = "rwkv_ms_natural_memory_native_aligned_vector_gate_specificity_result.v1"
MARGIN_MINIMUM = 0.005
COVERAGE_MINIMUM = 0.95
PARTITIONS_PER_SHARD = 1
PASS_STATUS = "aligned_vector_gate_native_recurrent_causal_gain_established"
PARTIAL_STATUS = "aligned_vector_gate_native_gain_without_full_causal_pass"
FAIL_STATUS = "aligned_vector_gate_native_gain_not_established"
ANALYZER_BINDING_PATH = Path(__file__)


def canonical_sha256(value: Any) -> str:
    return evaluation.canonical_sha256(value)


def sha256_file(path: Path) -> str:
    return evaluation.sha256_file(path)


@contextmanager
def analysis_bindings() -> Iterator[None]:
    bindings = {
        "evaluation": evaluation,
        "SCHEMA": SCHEMA,
        "MARGIN_MINIMUM": MARGIN_MINIMUM,
        "COVERAGE_MINIMUM": COVERAGE_MINIMUM,
        "PARTITIONS_PER_SHARD": PARTITIONS_PER_SHARD,
        "PASS_STATUS": PASS_STATUS,
        "PARTIAL_STATUS": PARTIAL_STATUS,
        "FAIL_STATUS": FAIL_STATUS,
        "INCLUDE_PROTOCOL_ERRATA": False,
        "ANALYZER_BINDING_PATH": ANALYZER_BINDING_PATH,
    }
    previous = {name: getattr(base, name) for name in bindings}
    try:
        for name, value in bindings.items():
            setattr(base, name, value)
        yield
    finally:
        for name, value in previous.items():
            setattr(base, name, value)


def read_records(
    root: Path,
) -> tuple[dict[str, dict[int, Mapping[str, Any]]], list[Mapping[str, Any]]]:
    with analysis_bindings():
        return base.read_records(root)


def aggregate_condition(records: Mapping[int, Mapping[str, Any]]) -> Mapping[str, Any]:
    return base.aggregate_condition(records)


def analyze(root: Path) -> Mapping[str, Any]:
    with analysis_bindings():
        return base.analyze(root)


def write_result(path: Path, result: Mapping[str, Any]) -> None:
    if path.exists():
        raise ValueError(f"Aligned-gate result output must be fresh: {path}")
    path.write_text(
        json.dumps(result, ensure_ascii=True, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evaluation-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    result = analyze(args.evaluation_root)
    write_result(args.output.expanduser().resolve(), result)
    print(
        json.dumps(
            {
                "status": result["status"],
                "passed": result["passed"],
                "condition_micro_f1": {
                    condition: metrics["micro_f1"]
                    for condition, metrics in result["condition_metrics"].items()
                },
                "causal_margins": result["causal_margins"],
                "result_receipt": result["receipt"]["payload_sha256"],
            },
            ensure_ascii=True,
            sort_keys=True,
        ),
        flush=True,
    )
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
