#!/usr/bin/env python3
"""Run the BF16-safe addressed-MoE outer-FFN causal gate."""

from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
import sys
from typing import Any, Iterator, Mapping, Sequence

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[1]
for import_root in (SCRIPT_DIR, PROJECT_ROOT):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

from experiments.rethinking_rwkv_ms_gemma import (
    run_natural_memory_native_rwkv_addressed_moe_outer_ffn_causal_train as base,
)
from experiments.rethinking_rwkv_ms_gemma import (
    run_natural_memory_native_rwkv_addressed_moe_outer_ffn_scaled_screen as screen,
)

PROTOCOL = SCRIPT_DIR / "natural_memory_native_rwkv_addressed_moe_outer_ffn_causal_train_protocol_v2.json"
PROTOCOL_PAYLOAD_SHA256 = (
    "a124dca10f395ebbbdb4a8e962372ee8e664fd50b12e0adc28e61b13a39a006b"
)
SCREEN_RESULT = SCRIPT_DIR / "local_artifacts/natural_memory_native_rwkv_addressed_moe_outer_ffn_scaled_screen_v2/result.json"
SCREEN_RESULT_FILE_SHA256 = "6917c25c0fc3d8353780a873c2f50c19b13bf9ecb7c95e6550ac61f9ddf5b3d5"
SCREEN_RESULT_RECEIPT = "0fe44645f1830670193bae3910b9e90ac8502ade8b354717a47c968063d837b0"
SCHEMA = "rwkv_ms_natural_memory_native_addressed_moe_outer_ffn_causal_train.v2"
STEP_SCHEMA = "rwkv_ms_natural_memory_native_addressed_moe_outer_ffn_causal_train_step.v2"
INPUT_SCHEMA = "rwkv_ms_natural_memory_native_addressed_moe_outer_ffn_causal_train_input.v2"
SEED = 98
UPDATES = 16
CONTRAST_WEIGHT = 1.0
MIN_ACCEPTED_ROWS_PER_UPDATE = 6
MAX_TOTAL_REJECTED_ROWS = 8
ENDPOINT_CANDIDATE_ROWS = 19
TRAINING_PREFIX_SHA256 = "121c2eea4904545b2f8c44ef2e4b16e905ca76809af9c34de5b7f848825872fd"
HELDOUT_ORDINALS = (545, 765, 582, 718, 863, 1432, 1349, 1436, 29, 586, 456, 596)
HELDOUT_PAYLOAD_SHA256 = "97b043c06f2981446705b7c8aba3574d11d52a25ff99fa3ce242343f475cd417"
SELECTED_CANDIDATE = screen.SELECTED_CANDIDATE
PASS_STATUS = "addressed_moe_outer_ffn_heldout_passed_generation_authorized"
FAIL_STATUS = "addressed_moe_outer_ffn_heldout_failed_generation_blocked"


def validate_protocol() -> Mapping[str, Any]:
    return base.validate_protocol()


@contextmanager
def bindings() -> Iterator[None]:
    overrides = {
        "SCHEMA": SCHEMA,
        "STEP_SCHEMA": STEP_SCHEMA,
        "INPUT_SCHEMA": INPUT_SCHEMA,
        "PROTOCOL": PROTOCOL,
        "PROTOCOL_PAYLOAD_SHA256": PROTOCOL_PAYLOAD_SHA256,
        "SCREEN_RESULT": SCREEN_RESULT,
        "SCREEN_RESULT_FILE_SHA256": SCREEN_RESULT_FILE_SHA256,
        "SCREEN_RESULT_RECEIPT": SCREEN_RESULT_RECEIPT,
        "SEED": SEED,
        "UPDATES": UPDATES,
        "CONTRAST_WEIGHT": CONTRAST_WEIGHT,
        "MIN_ACCEPTED_ROWS_PER_UPDATE": MIN_ACCEPTED_ROWS_PER_UPDATE,
        "MAX_TOTAL_REJECTED_ROWS": MAX_TOTAL_REJECTED_ROWS,
        "ENDPOINT_CANDIDATE_ROWS": ENDPOINT_CANDIDATE_ROWS,
        "TRAINING_PREFIX_SHA256": TRAINING_PREFIX_SHA256,
        "HELDOUT_ORDINALS": HELDOUT_ORDINALS,
        "HELDOUT_PAYLOAD_SHA256": HELDOUT_PAYLOAD_SHA256,
        "SELECTED_CANDIDATE": SELECTED_CANDIDATE,
        "PASS_STATUS": PASS_STATUS,
        "FAIL_STATUS": FAIL_STATUS,
        "screen": screen,
        "load_model": base.load_model,
        "validate_screen_result": base.validate_screen_result,
        "validate_protocol": base.validate_protocol,
    }
    previous = {name: getattr(base, name) for name in overrides}
    try:
        for name, value in overrides.items():
            setattr(base, name, value)
        with base.bindings():
            previous_shared_screen = base.base.affine_train.shared.screen
            previous_configurer = base.base.affine_train.shared.TRAINABLE_CONFIGURER
            base.base.affine_train.shared.screen = screen
            base.base.affine_train.shared.TRAINABLE_CONFIGURER = (
                base.configure_outer_ffn_parameters
            )
            try:
                yield
            finally:
                base.base.affine_train.shared.screen = previous_shared_screen
                base.base.affine_train.shared.TRAINABLE_CONFIGURER = previous_configurer
    finally:
        for name, value in previous.items():
            setattr(base, name, value)


def run(**kwargs: Any) -> Mapping[str, Any]:
    with bindings():
        return base.base.affine_train.shared.run(**kwargs)


def main(argv: Sequence[str] | None = None) -> int:
    with bindings():
        return base.base.affine_train.shared.main(argv)


if __name__ == "__main__":
    raise SystemExit(main())
