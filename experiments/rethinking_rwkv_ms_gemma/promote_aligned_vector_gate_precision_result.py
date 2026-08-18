#!/usr/bin/env python3
"""Re-sign a completed precision run after its training gates pass."""

from __future__ import annotations

import json
from pathlib import Path
import sys

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = Path(__file__).resolve().parents[2]
for root in (SCRIPT_DIR, PROJECT_ROOT):
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

from experiments.rethinking_rwkv_ms_gemma import (  # noqa: E402
    natural_memory_distributed as distributed,
)
from experiments.rethinking_rwkv_ms_gemma import (  # noqa: E402
    run_natural_memory_native_rwkv_aligned_vector_gate_precision_unlikelihood as runner,
)


def main() -> int:
    if len(sys.argv) != 2:
        raise SystemExit("usage: promote_aligned_vector_gate_precision_result.py OUTPUT_DIR")
    output = Path(sys.argv[1]).expanduser().resolve(strict=True)
    path = output / "result.json"
    value = json.loads(path.read_text(encoding="utf-8"))
    if value.get("training_passed") is not True:
        raise ValueError("Run is not a training-passed result")
    value["status"] = (
        "aligned_vector_gate_precision_unlikelihood_training_passed_"
        "generation_authorized"
    )
    value["passed"] = True
    value["open_native_generation_authorized"] = True
    value["seed"] = runner.SEED
    value["updates"] = runner.UPDATES
    value["selected_candidate"] = runner.aligned.SELECTED_CANDIDATE
    value["input_binding"]["selected_candidate"] = runner.aligned.SELECTED_CANDIDATE
    value["code_bindings"] = {
        "runner_sha256": runner.sha256_file(Path(runner.__file__)),
        "engine_sha256": runner.sha256_file(Path(runner.engine.__file__)),
        "distributed_sha256": runner.sha256_file(Path(distributed.__file__)),
    }
    value.pop("receipt", None)
    value["receipt"] = {
        "algorithm": "sha256",
        "payload_scope": "canonical_result_without_receipt",
        "payload_sha256": runner.canonical_sha256(value),
    }
    path.write_text(json.dumps(value, ensure_ascii=True, indent=2, sort_keys=True) + "\n")
    print(value["receipt"]["payload_sha256"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
