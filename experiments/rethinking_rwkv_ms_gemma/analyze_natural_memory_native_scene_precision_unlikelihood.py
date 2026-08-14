#!/usr/bin/env python3
"""Analyze and sign the locked precision-unlikelihood candidate."""

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
    analyze_natural_memory_native_scene_c16_residual as shared,
)
from experiments.rethinking_rwkv_ms_gemma import (  # noqa: E402
    materialize_natural_memory_native_scene_precision_unlikelihood as materializer,
)
from experiments.rethinking_rwkv_ms_gemma import (  # noqa: E402
    run_natural_memory_native_scene_precision_unlikelihood as training,
)
from experiments.rethinking_rwkv_ms_gemma import (  # noqa: E402
    run_natural_memory_native_scene_precision_unlikelihood_eval as runner,
)


SCHEMA = "rwkv_ms_natural_memory_native_scene_precision_unlikelihood_result.v1"
GATE_THRESHOLDS = dict(shared.GATE_THRESHOLDS)


def canonical_sha256(value: Any) -> str:
    return training.canonical_sha256(value)


@contextmanager
def _configured_analysis() -> Iterator[None]:
    replacements = {
        "training": training,
        "materializer": materializer,
        "runner": runner,
        "SCHEMA": SCHEMA,
        "GATE_THRESHOLDS": GATE_THRESHOLDS,
    }
    original = {name: getattr(shared, name) for name in replacements}
    try:
        for name, value in replacements.items():
            setattr(shared, name, value)
        yield
    finally:
        for name, value in original.items():
            setattr(shared, name, value)


def analyze(**kwargs: Any) -> Mapping[str, Any]:
    output = Path(kwargs["output"])
    with _configured_analysis():
        result = dict(shared.analyze(**kwargs))
    result["decision"] = (
        "TRAIN-only precision-unlikelihood endpoint cleared every preregistered gate; external replication still requires a separate protocol."
        if result["status"] == "passed"
        else "TRAIN-only precision-unlikelihood endpoint failed at least one preregistered gate; archive without external replication."
    )
    result["receipt"] = {
        "algorithm": "sha256",
        "payload_scope": "canonical_result_without_receipt",
        "payload_sha256": canonical_sha256(
            {key: value for key, value in result.items() if key != "receipt"}
        ),
    }
    output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return result


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--materialization-root", type=Path, required=True)
    parser.add_argument("--progression-root", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--reference-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    result = analyze(
        input_root=args.input_root.expanduser().resolve(strict=True),
        materialization_root=args.materialization_root.expanduser().resolve(strict=True),
        progression_root=args.progression_root.expanduser().resolve(strict=True),
        dataset_root=args.dataset_root.expanduser().resolve(strict=True),
        reference_root=args.reference_root.expanduser().resolve(strict=True),
        output=args.output.expanduser().resolve(),
    )
    print(
        json.dumps(
            {
                "status": result["status"],
                "metrics": result["metrics"],
                "deltas": result["deltas"],
                "receipt": result["receipt"]["payload_sha256"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
