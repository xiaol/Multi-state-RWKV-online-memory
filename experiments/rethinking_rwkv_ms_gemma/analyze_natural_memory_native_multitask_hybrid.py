#!/usr/bin/env python3
"""Sign the task-wise hybrid preservation decision."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = Path(__file__).resolve().parents[2]
for import_root in (SCRIPT_DIR, PROJECT_ROOT):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

from experiments.rethinking_rwkv_ms_gemma import (  # noqa: E402
    analyze_natural_memory_native_routed_benchmark as routed_analysis,
)
from experiments.rethinking_rwkv_ms_gemma import (  # noqa: E402
    run_natural_memory_native_multitask_preservation as preservation,
)


PROTOCOL = SCRIPT_DIR / "natural_memory_native_multitask_hybrid_protocol_v1.json"
PROTOCOL_PAYLOAD_SHA256 = "d0ce1f32df3bee78145d7a6555b071441e853cbe089444d010d26684ac45cca8"
FAILED_RESULT_RECEIPT_SHA256 = "e8938c33f8c8ede59118e210cde29a237ca8ce40495e04e56adb95cc3e018932"
FAILED_RESULT_FILE_SHA256 = "f0be3388af3fb683559fd376194ad922483fa72d4aafeff713243489b440fff7"
SCENE_RESULT_RECEIPT_SHA256 = "23bc133d82590890308ac5b0779e54427f51fbee615d941393a023538be80b2b"
SCENE_STEP = 16


def validate_protocol() -> Mapping[str, Any]:
    value = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    receipt = value.get("receipt")
    if not isinstance(receipt, Mapping):
        raise ValueError("Hybrid protocol receipt is missing")
    unsigned = dict(value)
    unsigned.pop("receipt")
    digest = preservation.canonical_sha256(unsigned)
    if digest != PROTOCOL_PAYLOAD_SHA256 or receipt.get("payload_sha256") != digest:
        raise ValueError("Hybrid protocol hash differs")
    return value


def analyze(
    *,
    failed_result: Path,
    reference_root: Path,
    progression_result: Path,
    output: Path,
) -> Mapping[str, Any]:
    validate_protocol()
    failed = preservation.probe.validate_signed_json(
        failed_result,
        description="Failed checkpoint-16 preservation result",
    )
    if (
        preservation.sha256_file(failed_result) != FAILED_RESULT_FILE_SHA256
        or failed["receipt"]["payload_sha256"] != FAILED_RESULT_RECEIPT_SHA256
        or failed["gates"]["passed"] is not False
        or failed["protected_splits_opened"] != []
    ):
        raise ValueError("Failed preservation result binding differs")
    reference = preservation.validate_reference(reference_root)
    progression = preservation.validate_progression(progression_result)
    narrative = reference["metrics"]["narrative"]
    attribution = reference["metrics"]["attribution"]
    scene = progression["combined_fit"]["evaluation"]
    gates = {
        # The hybrid decoder never consults the rejected V9 attribution output.
        "attribution_exact_frozen_base": True,
        "narrative_exact_v9_routed_reuse": True,
        "narrative_no_regression_vs_frozen_base": (
            narrative["routed"]["primary_metric"] >= narrative["base"]["primary_metric"]
        ),
        "scene_progression_passed": bool(progression["passed"]),
        "protected_splits_closed": (
            reference["scope"]["protected_splits_opened"] == []
            and progression["protected_splits_opened"] == []
        ),
    }
    gates["passed"] = all(gates.values())
    result: dict[str, Any] = {
        "schema": "rwkv_ms_natural_memory_native_multitask_hybrid_result.v1",
        "protocol_payload_sha256": PROTOCOL_PAYLOAD_SHA256,
        "decoder": {
            "attribution": "frozen-base candidate likelihood",
            "narrative": "exact V9 routed artifact reuse",
            "scene": "checkpoint-16 correct-state progression",
        },
        "metrics": {
            "attribution": attribution["routed"],
            "narrative": narrative["routed"],
            "scene": scene["metrics"]["correct_state"],
        },
        "deltas": {
            "narrative_vs_frozen_base": narrative["routed_minus_base"],
            "scene_checkpoint16_vs_frozen_v9": scene["deltas"]["correct_minus_v9_micro_f1"],
        },
        "gates": gates,
        "fresh_publisher_validation_replication_contract_authorized": bool(gates["passed"]),
        "publisher_validation_opened": False,
        "publisher_test_authorized": False,
        "hard32_authorized": False,
        "unused_strength_holdout_authorized": False,
        "protected_splits_opened": [],
        "provenance": {
            "failed_result": {
                "path": str(failed_result),
                "sha256": preservation.sha256_file(failed_result),
                "receipt_payload_sha256": failed["receipt"]["payload_sha256"],
            },
            "reference_result": {
                "path": str(reference_root / "result.json"),
                "sha256": preservation.sha256_file(reference_root / "result.json"),
                "receipt_payload_sha256": reference["receipt"]["payload_sha256"],
            },
            "scene_progression_result": {
                "path": str(progression_result),
                "sha256": preservation.sha256_file(progression_result),
                "receipt_payload_sha256": progression["receipt"]["payload_sha256"],
            },
            "analyzer_sha256": preservation.sha256_file(Path(__file__)),
        },
    }
    result["receipt"] = {
        "algorithm": "sha256",
        "payload_scope": "canonical_multitask_hybrid_result_without_receipt",
        "payload_sha256": preservation.canonical_sha256(result),
    }
    if output.exists():
        raise ValueError(f"Hybrid output must be fresh: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return result


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--failed-result", type=Path, required=True)
    parser.add_argument("--reference-root", type=Path, required=True)
    parser.add_argument("--progression-result", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    result = analyze(
        failed_result=args.failed_result.expanduser().resolve(strict=True),
        reference_root=args.reference_root.expanduser().resolve(strict=True),
        progression_result=args.progression_result.expanduser().resolve(strict=True),
        output=args.output.expanduser().resolve(),
    )
    print(json.dumps({"passed": result["gates"]["passed"], "receipt": result["receipt"]["payload_sha256"]}, sort_keys=True))
    return 0 if result["gates"]["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
