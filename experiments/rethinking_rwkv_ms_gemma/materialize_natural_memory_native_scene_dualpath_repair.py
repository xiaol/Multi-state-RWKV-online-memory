#!/usr/bin/env python3
"""Materialize the locked dual-path repair endpoint unchanged."""

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
    materialize_natural_memory_native_scene_onpolicy_repair as shared,
)
from experiments.rethinking_rwkv_ms_gemma import (  # noqa: E402
    run_natural_memory_native_scene_dualpath_repair as training,
)


SCHEMA = "rwkv_ms_natural_memory_native_scene_dualpath_repair_materialization.v1"
PATCH_SCHEMA = "rwkv_ms_natural_memory_native_scene_dualpath_repair_patch.v1"
CANDIDATE_ID = "dualpath_repair_endpoint"


class _TrainingBinding:
    STARTING_GATE_STATE_SHA256 = training.shared.STARTING_GATE_STATE_SHA256

    def __getattr__(self, name: str) -> Any:
        return getattr(training, name)


TRAINING_BINDING = _TrainingBinding()


@contextmanager
def _configured_materializer() -> Iterator[None]:
    replacements = {
        "training": TRAINING_BINDING,
        "SCHEMA": SCHEMA,
        "PATCH_SCHEMA": PATCH_SCHEMA,
        "CANDIDATE_ID": CANDIDATE_ID,
    }
    original = {name: getattr(shared, name) for name in replacements}
    try:
        for name, value in replacements.items():
            setattr(shared, name, value)
        yield
    finally:
        for name, value in original.items():
            setattr(shared, name, value)


def canonical_sha256(value: Any) -> str:
    return training.canonical_sha256(value)


def sha256_file(path: Path) -> str:
    return training.sha256_file(path)


def load_training_endpoint(root: Path) -> tuple[Mapping[str, Any], Mapping[str, Any]]:
    with _configured_materializer():
        return shared.load_training_endpoint(root)


def materialize(*, training_root: Path, output_root: Path) -> Mapping[str, Any]:
    with _configured_materializer():
        return shared.materialize(training_root=training_root, output_root=output_root)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    result = materialize(
        training_root=args.training_root.expanduser().resolve(strict=True),
        output_root=args.output_root.expanduser().resolve(),
    )
    print(
        json.dumps(
            {
                "candidate_id": result["candidate_id"],
                "receipt": result["receipt"]["payload_sha256"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
