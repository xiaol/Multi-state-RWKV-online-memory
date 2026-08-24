from __future__ import annotations

import json
from pathlib import Path

from experiments.rethinking_rwkv_ms_gemma import (
    analyze_natural_memory_native_rwkv_source_cumulative_residual_development as analyzer,
)
from experiments.rethinking_rwkv_ms_gemma import (
    run_natural_memory_native_rwkv_source_cumulative_residual_development_screen as screen,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
RESULT_ROOT = (
    PROJECT_ROOT
    / "experiments/rethinking_rwkv_ms_gemma/local_artifacts/"
    "natural_memory_native_rwkv_source_cumulative_residual_development_screen_v2"
)


def test_signed_development_analysis_recomputes_from_pinned_result() -> None:
    result = analyzer.load_result(RESULT_ROOT / "result.json")
    expected = json.loads((RESULT_ROOT / "analysis.json").read_text(encoding="utf-8"))

    assert analyzer.analyze(result) == expected
    screen.validate_receipt(
        expected,
        scope="canonical_analysis_without_receipt",
        description="Cumulative-residual development analysis",
    )
    assert expected["status"] == (
        "development_failed_donor_causality_family_not_promoted"
    )
    assert expected["passed"] is False
    assert expected["selected_variant"] is None
    assert expected["best_diagnostic_variant"] == "renew_at_17_scale_1"


def test_scale_one_passes_mechanics_but_fails_donor_causality() -> None:
    analysis = json.loads(
        (RESULT_ROOT / "analysis.json").read_text(encoding="utf-8")
    )
    scale_one = analysis["analysis"]["renew_at_17_scale_1"]

    assert scale_one["mechanics_pass"] is True
    assert scale_one["donor_causal_pass"] is False
    assert scale_one["terminal_target_selected_fraction"] == 0.953125
    assert scale_one["material_predictor_change_fraction"] == {
        "correct_vs_provider_off": 1.0,
        "donor_address_only_vs_single_target": 1.0,
        "donor_both_vs_single_target": 1.0,
        "donor_state_only_vs_single_target": 1.0,
        "layer_address_only_vs_single_target": 1.0,
        "layer_both_vs_single_target": 1.0,
        "layer_state_only_vs_single_target": 1.0,
    }
    assert scale_one["target_ce_margins"]["gain_vs_provider_off"] == {
        "mean": 0.012149795889854431,
        "positive_fraction": 0.5,
    }
    assert scale_one["target_ce_margins"]["donor_both_minus_target"] == {
        "mean": 0.017373159527778625,
        "positive_fraction": 0.609375,
    }


def test_bf16_gate_audit_and_data_firewall_are_explicit() -> None:
    analysis = json.loads(
        (RESULT_ROOT / "analysis.json").read_text(encoding="utf-8")
    )
    result = analyzer.load_result(RESULT_ROOT / "result.json")

    assert analysis["audit_correction"] == {
        "field": "selected memory mass equation",
        "original_check": "float32 byte equality after BF16 model execution",
        "corrected_tolerance": 1.0 / 512.0,
        "maximum_observed_error": 0.0019413232803344727,
        "all_rows_all_variants_within_tolerance": True,
    }
    assert result["hardware"]["world_size"] == 4
    assert result["hardware"]["four_distinct_a100s"] is True
    assert result["hardware"]["hf_endpoint"] == "https://hf-mirror.com"
    assert result["development_rows_opened"] == 64
    assert result["protected_mechanics_rows_opened"] == 0
    assert result["protected_causal_rows_opened"] == 0
    assert result["native_benchmark_opened"] is False


def test_development_runner_contains_no_protected_bundle_open_call() -> None:
    source = Path(screen.__file__).read_text(encoding="utf-8")

    assert "read_authorized_bundle(" not in source
    assert "allow_mechanics=True" not in source
    assert "allow_causal=True" not in source
    assert set(screen.DEVELOPMENT_VARIANTS) == {
        "renew_at_17_scale_0_5",
        "renew_at_17_scale_1",
        "renew_at_17_scale_2",
        "renew_at_17_scale_4",
        "renew_at_17_scale_8",
        "renew_at_17_scale_16",
        "renew_at_23_scale_4",
    }
