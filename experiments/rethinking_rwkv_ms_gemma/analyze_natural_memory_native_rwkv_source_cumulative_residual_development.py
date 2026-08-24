#!/usr/bin/env python3
"""Recompute the cumulative-residual development gate with BF16 tolerance."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from experiments.rethinking_rwkv_ms_gemma import (
    run_natural_memory_native_rwkv_source_cumulative_residual_development_screen as screen,
)


SCRIPT_DIR = Path(__file__).resolve().parent
RESULT_SCHEMA = (
    "rwkv_ms_natural_memory_native_rwkv_source_cumulative_residual_mechanics.v1."
    "development_screen"
)
ANALYSIS_SCHEMA = (
    "rwkv_ms_natural_memory_native_rwkv_source_cumulative_residual_"
    "development_analysis.v1"
)
EXPECTED_RESULT_SHA256 = (
    "5eba745db6b245d5df5e8a2f16d058f41a162b2f551f958041e412ab1cd1abf9"
)
EXPECTED_RESULT_RECEIPT = (
    "768c88849c97017469d72af928eaef1dc01b934ec0004b69f89a983a9992e831"
)
DEFAULT_RESULT = SCRIPT_DIR / (
    "local_artifacts/"
    "natural_memory_native_rwkv_source_cumulative_residual_development_screen_v2/"
    "result.json"
)
DEFAULT_OUTPUT = DEFAULT_RESULT.with_name("analysis.json")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_result(path: Path) -> dict[str, Any]:
    if path.name != "result.json" or sha256_file(path) != EXPECTED_RESULT_SHA256:
        raise ValueError("Cumulative-residual development result file hash differs")
    result = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(result, Mapping) or result.get("schema") != RESULT_SCHEMA:
        raise ValueError("Cumulative-residual development result schema differs")
    screen.validate_receipt(
        result,
        scope="canonical_result_without_receipt",
        description="Cumulative-residual development result",
    )
    if (
        result["receipt"]["payload_sha256"] != EXPECTED_RESULT_RECEIPT
        or result.get("development_rows_opened") != 64
        or result.get("protected_mechanics_rows_opened") != 0
        or result.get("protected_causal_rows_opened") != 0
        or result.get("model_or_adapter_parameters_updated") is not False
        or result.get("native_benchmark_opened") is not False
    ):
        raise ValueError("Cumulative-residual development result binding differs")
    return dict(result)


def analyze(result: Mapping[str, Any]) -> dict[str, Any]:
    rows = result.get("rows")
    architecture = result.get("architecture")
    variants = architecture.get("variants") if isinstance(architecture, Mapping) else None
    if not isinstance(rows, list) or len(rows) != 64 or not isinstance(variants, Mapping):
        raise ValueError("Cumulative-residual development result coverage differs")
    analysis = {}
    for variant, config in variants.items():
        variant_rows = []
        for row in rows:
            value = copy.deepcopy(row["variants"][variant])
            invariants = value["invariants"]
            invariants.pop("selected_gate_equation_exact", None)
            invariants["selected_gate_equation_bf16_close"] = (
                value["selected_gate_max_absolute_error"]
                <= screen.BF16_GATE_MASS_ATOL
            )
            variant_rows.append(value)
        analysis[variant] = screen.aggregate_development_variant(
            variant_rows,
            config["score_anchor_layers"],
        )
    passing = [
        variant for variant, value in analysis.items() if value["development_pass"]
    ]
    ranking = sorted(
        analysis,
        key=lambda variant: (
            analysis[variant]["target_ce_margins"]["donor_both_minus_target"][
                "positive_fraction"
            ],
            analysis[variant]["target_ce_margins"]["donor_both_minus_target"][
                "mean"
            ],
            analysis[variant]["target_ce_margins"]["gain_vs_provider_off"][
                "mean"
            ],
        ),
        reverse=True,
    )
    payload: dict[str, Any] = {
        "schema": ANALYSIS_SCHEMA,
        "status": (
            "development_variant_selected"
            if passing
            else "development_failed_donor_causality_family_not_promoted"
        ),
        "passed": bool(passing),
        "selected_variant": passing[0] if len(passing) == 1 else None,
        "best_diagnostic_variant": ranking[0],
        "ranking": ranking,
        "source_result": {
            "path": str(DEFAULT_RESULT.relative_to(SCRIPT_DIR)),
            "sha256": EXPECTED_RESULT_SHA256,
            "receipt": EXPECTED_RESULT_RECEIPT,
        },
        "audit_correction": {
            "field": "selected memory mass equation",
            "original_check": "float32 byte equality after BF16 model execution",
            "corrected_tolerance": screen.BF16_GATE_MASS_ATOL,
            "maximum_observed_error": max(
                row["variants"][variant]["selected_gate_max_absolute_error"]
                for row in rows
                for variant in variants
            ),
            "all_rows_all_variants_within_tolerance": all(
                row["variants"][variant]["selected_gate_max_absolute_error"]
                <= screen.BF16_GATE_MASS_ATOL
                for row in rows
                for variant in variants
            ),
        },
        "analysis": analysis,
        "decision": {
            "promote_to_protected_mechanics": False,
            "reason": (
                "No variant reached the predeclared 0.75 matched-donor-positive "
                "row fraction despite passing source selection and mechanics at "
                "unsaturated scales."
            ),
            "next_architecture": (
                "freeze the scale-1 layer-17 source router and train a small "
                "state-valued outer FFN on open development data with correct, "
                "matched-donor, layer-roll, and zero-state contrast"
            ),
        },
        "protected_mechanics_rows_opened": 0,
        "protected_causal_rows_opened": 0,
        "native_benchmark_opened": False,
    }
    payload["receipt"] = {
        "algorithm": "sha256",
        "payload_scope": "canonical_analysis_without_receipt",
        "payload_sha256": screen.canonical_sha256(payload),
    }
    return payload


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result", type=Path, default=DEFAULT_RESULT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    output = args.output.expanduser().resolve()
    if output.exists():
        raise ValueError(f"Cumulative-residual analysis output must be fresh: {output}")
    result = load_result(args.result.expanduser().resolve(strict=True))
    analysis = analyze(result)
    screen.signed_json(output, analysis)
    print(json.dumps(analysis, ensure_ascii=True, sort_keys=True))


if __name__ == "__main__":
    main()
