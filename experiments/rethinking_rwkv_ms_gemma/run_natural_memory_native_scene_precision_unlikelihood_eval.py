#!/usr/bin/env python3
"""Generate the locked precision-unlikelihood candidate on open TRAIN rows."""

from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
import sys
from typing import Any, Iterator, Mapping, Sequence


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = Path(__file__).resolve().parents[2]
for import_root in (SCRIPT_DIR, PROJECT_ROOT):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

from experiments.rethinking_rwkv_ms_gemma import (  # noqa: E402
    materialize_natural_memory_native_scene_precision_unlikelihood as materializer,
)
from experiments.rethinking_rwkv_ms_gemma import (  # noqa: E402
    run_natural_memory_native_scene_precision_unlikelihood as training,
)
from experiments.rethinking_rwkv_ms_gemma import (  # noqa: E402
    run_natural_memory_native_scene_seed_ensemble_eval as shared,
)


SCHEMA = "rwkv_ms_natural_memory_native_scene_precision_unlikelihood_eval_shard.v1"
INPUT_SCHEMA = "rwkv_ms_natural_memory_native_scene_precision_unlikelihood_eval_input.v1"
WORLD_SIZE = shared.WORLD_SIZE
ROWS = shared.ROWS
ROW_PAYLOAD_SHA256 = shared.ROW_PAYLOAD_SHA256
CONDITION = shared.CONDITION
_SHARED_INPUT_BINDING = shared.input_binding


def canonical_sha256(value: Any) -> str:
    return training.canonical_sha256(value)


def sha256_file(path: Path) -> str:
    return training.sha256_file(path)


def _bound_input_binding(**kwargs: Any) -> Mapping[str, Any]:
    value = dict(_SHARED_INPUT_BINDING(**kwargs))
    value["runner_sha256"] = sha256_file(Path(__file__))
    return value


@contextmanager
def _configured_engine() -> Iterator[None]:
    replacements = {
        "training": training,
        "materializer": materializer,
        "SCHEMA": SCHEMA,
        "INPUT_SCHEMA": INPUT_SCHEMA,
        "input_binding": _bound_input_binding,
    }
    original = {name: getattr(shared, name) for name in replacements}
    try:
        for name, value in replacements.items():
            setattr(shared, name, value)
        yield
    finally:
        for name, value in original.items():
            setattr(shared, name, value)


def validate_materialization(root: Path) -> tuple[Mapping[str, Any], Mapping[str, Any]]:
    with _configured_engine():
        return shared.validate_materialization(root)


def load_candidate_patch(*args: Any, **kwargs: Any) -> Mapping[str, Any]:
    with _configured_engine():
        return shared.load_candidate_patch(*args, **kwargs)


def input_binding(**kwargs: Any) -> Mapping[str, Any]:
    with _configured_engine():
        return _bound_input_binding(**kwargs)


def output_path(output_dir: Path) -> Path:
    with _configured_engine():
        return shared.output_path(output_dir)


def read_completed(path: Path) -> dict[int, Mapping[str, Any]]:
    with _configured_engine():
        return shared.read_completed(path)


def validate_resume(*args: Any, **kwargs: Any) -> None:
    with _configured_engine():
        shared.validate_resume(*args, **kwargs)


def generate(*args: Any, **kwargs: Any) -> Mapping[str, Any]:
    with _configured_engine():
        return shared.generate(*args, **kwargs)


def parse_args(argv: Sequence[str] | None = None) -> Any:
    with _configured_engine():
        return shared.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    with _configured_engine():
        return shared.main(argv)


if __name__ == "__main__":
    raise SystemExit(main())
