#!/usr/bin/env python3
"""Calibrate the straight-through top-1 internal RWKV router."""

from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence

from experiments.rethinking_rwkv_ms_gemma import (
    run_natural_memory_native_rwkv_recurrent_value_calibration as shared,
)
from experiments.rethinking_rwkv_ms_gemma import (
    run_natural_memory_native_rwkv_sharp_router_screen as screen,
)


SCRIPT_DIR = Path(__file__).resolve().parent
SCHEMA = "rwkv_ms_natural_memory_native_sharp_router_calibration.v1"
PROTOCOL = (
    SCRIPT_DIR / "natural_memory_native_rwkv_sharp_router_calibration_protocol_v1.json"
)
PROTOCOL_PAYLOAD_SHA256 = (
    "6282e6d3be1fab45c6cf69b6667dd462fa19db5ef4c31cd6814b2e0aa2dc3ce5"
)
SCREEN_RESULT = (
    SCRIPT_DIR
    / "local_artifacts/natural_memory_native_rwkv_sharp_router_screen_v2/result.json"
)
SCREEN_RESULT_BINDING = (
    "local_artifacts/natural_memory_native_rwkv_sharp_router_screen_v2/result.json"
)
SCREEN_RESULT_FILE_SHA256 = (
    "5ba5ea8141a3c5c89a8fd935a5837cf3f41eec5aa46b82c0d0e74688a36dbca5"
)
SCREEN_RESULT_RECEIPT = (
    "f65e65b427940799206769d385e933169d30b79796064cd6ac54f906cbaa37ef"
)
SELECTED_CANDIDATE = {
    "candidate_id": "recurrent_value_t1_k1",
    "hybrid_mode": "recurrent_value",
    "hybrid_gain": 0.03125,
    "read_temperature": 1.0,
    "read_top_k": 1,
}
SEED = 65


@contextmanager
def calibration_bindings() -> Iterator[None]:
    bindings = {
        "SCHEMA": SCHEMA,
        "PROTOCOL": PROTOCOL,
        "PROTOCOL_PAYLOAD_SHA256": PROTOCOL_PAYLOAD_SHA256,
        "SCREEN_RESULT": SCREEN_RESULT,
        "SCREEN_RESULT_BINDING": SCREEN_RESULT_BINDING,
        "SCREEN_RESULT_FILE_SHA256": SCREEN_RESULT_FILE_SHA256,
        "SCREEN_RESULT_RECEIPT": SCREEN_RESULT_RECEIPT,
        "SELECTED_CANDIDATE": SELECTED_CANDIDATE,
        "SEED": SEED,
        "RUNNER_BINDING_PATH": Path(__file__),
        "screen": screen,
    }
    previous = {name: getattr(shared, name) for name in bindings}
    try:
        for name, value in bindings.items():
            setattr(shared, name, value)
        yield
    finally:
        for name, value in previous.items():
            setattr(shared, name, value)


def validate_protocol() -> Mapping[str, Any]:
    with calibration_bindings():
        return shared.validate_protocol()


def validate_screen_result() -> Mapping[str, Any]:
    with calibration_bindings():
        return shared.validate_screen_result()


def run(**kwargs: Any) -> Mapping[str, Any]:
    with calibration_bindings():
        return shared.run(**kwargs)


def main(argv: Sequence[str] | None = None) -> int:
    with calibration_bindings():
        return shared.main(argv)


if __name__ == "__main__":
    raise SystemExit(main())
