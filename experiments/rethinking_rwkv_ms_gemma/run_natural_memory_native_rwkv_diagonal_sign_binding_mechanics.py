#!/usr/bin/env python3
"""Four-A100 mechanics gate for the diagonal-sign RWKV identity binding."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Sequence

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[1]
for root in (SCRIPT_DIR, PROJECT_ROOT):
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

from experiments.rethinking_rwkv_ms_gemma import rwkv_diagonal_sign_integration as sign
from experiments.rethinking_rwkv_ms_gemma import run_natural_memory_native_rwkv_headwise_rotary_binding_mechanics as mechanics


PROTOCOL = SCRIPT_DIR / "natural_memory_native_rwkv_diagonal_sign_binding_fullkey_mechanics_protocol_v1.json"
PROTOCOL_PAYLOAD_SHA256 = "bfd48c8c5b0cf3e1fb3036cc9ed5ca8cfb17c62dd5bef48339cbde5b37b11897"

mechanics.rotary = sign
mechanics.PROTOCOL = PROTOCOL
mechanics.PROTOCOL_PAYLOAD_SHA256 = PROTOCOL_PAYLOAD_SHA256
mechanics.SCHEMA = "rwkv_ms_natural_memory_native_diagonal_sign_binding_mechanics.v1"
mechanics.PASS_STATUS = "diagonal_sign_binding_mechanics_passed_causal_endpoint_authorized"
mechanics.FAIL_STATUS = "diagonal_sign_binding_mechanics_failed_causal_endpoint_blocked"
mechanics.CORRECT_MAX_ABS_TOLERANCE = 0.001


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-model", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    mechanics.run(
        base_model=args.base_model.expanduser().resolve(strict=True),
        dataset_root=args.dataset_root.expanduser().resolve(strict=True),
        output_dir=args.output_dir.expanduser().resolve(),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
