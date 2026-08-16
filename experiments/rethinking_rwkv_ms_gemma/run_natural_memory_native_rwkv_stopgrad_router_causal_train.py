#!/usr/bin/env python3
"""Train stable RWKV readouts with stop-gradient route scores."""

from __future__ import annotations

from contextlib import contextmanager
import json
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence

from experiments.rethinking_rwkv_ms_gemma import (
    run_natural_memory_native_rwkv_stable_readout_causal_train as stable,
)
from experiments.rethinking_rwkv_ms_gemma import (
    run_natural_memory_native_rwkv_top2_abstention_causal_train as shared,
)


SCRIPT_DIR = Path(__file__).resolve().parent
SCHEMA = "rwkv_ms_natural_memory_native_stopgrad_router_causal_train.v1"
STEP_SCHEMA = "rwkv_ms_natural_memory_native_stopgrad_router_causal_train_step.v1"
INPUT_SCHEMA = "rwkv_ms_natural_memory_native_stopgrad_router_causal_train_input.v1"
PROTOCOL = (
    SCRIPT_DIR / "natural_memory_native_rwkv_stopgrad_router_causal_train_protocol_v1.json"
)
PROTOCOL_PAYLOAD_SHA256 = (
    "dd12008b1b9416e0f17d4ee7566d58dcbea4ecef97b6c63358ea7c13abab3b19"
)
SEED = 70
FAILED_INPUT_BINDING = (
    SCRIPT_DIR
    / "local_artifacts/"
    "natural_memory_native_rwkv_stable_readout_causal_train_failed_step2_v1/"
    "input_binding.json"
)
FAILED_INPUT_BINDING_SHA256 = (
    "b6b7c24819f8bcbea9b137aa574de13921d55d572cba57ac6d78d395ba2c6c72"
)
FAILED_PROGRESS = FAILED_INPUT_BINDING.parent / "training_progress.jsonl"
FAILED_PROGRESS_SHA256 = (
    "6da476fadbe39b01b7acc186fb943b7550dc2ed6fb9cf5311500d8378b9e9b02"
)
SELECTED_CANDIDATE = {
    **shared.SELECTED_CANDIDATE,
    "detach_read_scores": True,
}


def validate_protocol() -> Mapping[str, Any]:
    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    receipt = protocol.get("receipt")
    if not isinstance(receipt, Mapping):
        raise ValueError("Stop-gradient router protocol receipt is missing")
    unsigned = dict(protocol)
    unsigned.pop("receipt")
    digest = shared.canonical_sha256(unsigned)
    if digest != PROTOCOL_PAYLOAD_SHA256 or receipt.get("payload_sha256") != digest:
        raise ValueError("Stop-gradient router protocol payload differs")
    if shared.sha256_file(FAILED_INPUT_BINDING) != FAILED_INPUT_BINDING_SHA256:
        raise ValueError("Failed stable-readout input binding differs")
    if shared.sha256_file(FAILED_PROGRESS) != FAILED_PROGRESS_SHA256:
        raise ValueError("Failed stable-readout progress binding differs")
    if protocol.get("protected_splits_opened_by_this_protocol") != []:
        raise ValueError("Stop-gradient router training may not open protected data")
    return protocol


@contextmanager
def training_bindings() -> Iterator[None]:
    with stable.training_bindings():
        bindings = {
            "SCHEMA": SCHEMA,
            "STEP_SCHEMA": STEP_SCHEMA,
            "INPUT_SCHEMA": INPUT_SCHEMA,
            "PROTOCOL": PROTOCOL,
            "PROTOCOL_PAYLOAD_SHA256": PROTOCOL_PAYLOAD_SHA256,
            "SEED": SEED,
            "SELECTED_CANDIDATE": SELECTED_CANDIDATE,
            "RUNNER_BINDING_PATH": Path(__file__),
            "PASS_STATUS": "stopgrad_router_heldout_gate_passed_generation_authorized",
            "FAIL_STATUS": "stopgrad_router_heldout_gate_failed_generation_blocked",
            "validate_protocol": validate_protocol,
        }
        previous = {name: getattr(shared, name) for name in bindings}
        try:
            for name, value in bindings.items():
                setattr(shared, name, value)
            yield
        finally:
            for name, value in previous.items():
                setattr(shared, name, value)


def validate_calibration_result() -> Mapping[str, Any]:
    return shared.validate_calibration_result()


def run(**kwargs: Any) -> Mapping[str, Any]:
    with training_bindings():
        return shared.run(**kwargs)


def main(argv: Sequence[str] | None = None) -> int:
    with training_bindings():
        return shared.main(argv)


if __name__ == "__main__":
    raise SystemExit(main())
