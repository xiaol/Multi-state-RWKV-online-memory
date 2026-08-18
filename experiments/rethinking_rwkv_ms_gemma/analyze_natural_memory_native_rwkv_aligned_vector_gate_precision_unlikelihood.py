#!/usr/bin/env python3
"""Aggregate and sign the precision-trained aligned-vector native benchmark."""

from __future__ import annotations

import argparse
from contextlib import contextmanager
from pathlib import Path
import sys
from typing import Any, Iterator, Sequence

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = Path(__file__).resolve().parents[2]
for root in (SCRIPT_DIR, PROJECT_ROOT):
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

from experiments.rethinking_rwkv_ms_gemma import (  # noqa: E402
    analyze_natural_memory_native_rwkv_vector_gate_eval as shared,
)
from experiments.rethinking_rwkv_ms_gemma import (  # noqa: E402
    run_natural_memory_native_rwkv_aligned_vector_gate_precision_unlikelihood_eval as evaluation,
)


SCHEMA = "rwkv_ms_natural_memory_native_aligned_vector_gate_precision_unlikelihood_result.v1"
PASS_STATUS = "aligned_vector_gate_precision_unlikelihood_native_gain_established"
PARTIAL_STATUS = "aligned_vector_gate_precision_unlikelihood_native_partial_gain"
FAIL_STATUS = "aligned_vector_gate_precision_unlikelihood_native_gain_not_established"
MARGIN_MINIMUM = 0.005
ANALYZER_BINDING_PATH = Path(__file__)


@contextmanager
def bindings() -> Iterator[None]:
    names = {
        "SCHEMA": SCHEMA,
        "PASS_STATUS": PASS_STATUS,
        "PARTIAL_STATUS": PARTIAL_STATUS,
        "FAIL_STATUS": FAIL_STATUS,
        "MARGIN_MINIMUM": MARGIN_MINIMUM,
        "ANALYZER_BINDING_PATH": ANALYZER_BINDING_PATH,
        "evaluation": evaluation,
        "INCLUDE_PROTOCOL_ERRATA": False,
    }
    previous = {name: getattr(shared, name) for name in names}
    try:
        for name, value in names.items():
            setattr(shared, name, value)
        yield
    finally:
        for name, value in previous.items():
            setattr(shared, name, value)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evaluation-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    with bindings():
        result = shared.analyze(args.evaluation_root)
        shared.write_result(args.output.expanduser().resolve(), result)
    print({"status": result["status"], "passed": result["passed"], "margins": result["causal_margins"], "receipt": result["receipt"]["payload_sha256"]})
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
