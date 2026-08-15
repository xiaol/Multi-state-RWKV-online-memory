from __future__ import annotations

from experiments.rethinking_rwkv_ms_gemma import (
    analyze_natural_memory_native_projected_rwkv_hybrid_benchmark as analysis,
)


def _record(value, *, carrier: str = "carrier"):
    return {
        "prediction": value,
        "projected_carrier_sha256": carrier,
        "projected_carrier_byte_identical": True,
    }


def _outputs(*, hybrid_matches_projected: bool = False):
    gold = {index: {index} for index in range(4)}
    projected_values = {0: [0], 1: [1], 2: [99], 3: [99]}
    hybrid_values = projected_values if hybrid_matches_projected else {
        index: [index] for index in gold
    }
    bad_values = {index: [99] for index in gold}
    outputs = {"projected_control": {}, "hybrid_candidate": {}}
    for seed in analysis.evaluator.SEEDS:
        outputs["projected_control"][seed] = {
            "correct_state": {
                index: _record(value) for index, value in projected_values.items()
            }
        }
        outputs["hybrid_candidate"][seed] = {
            "correct_recurrent_state": {
                index: _record(value) for index, value in hybrid_values.items()
            },
            "zero_recurrent_state": {
                index: _record(value) for index, value in bad_values.items()
            },
            "matched_donor_recurrent_state": {
                index: _record(value) for index, value in bad_values.items()
            },
            "layer_permuted_recurrent_state": {
                index: _record(value) for index, value in bad_values.items()
            },
            "projected_only_bypass": {
                index: _record(value) for index, value in bad_values.items()
            },
        }
    return outputs, gold


def test_signed_analysis_pass_requires_benchmark_and_causal_gates() -> None:
    outputs, gold = _outputs()

    result = analysis.build_analysis(outputs, gold)

    assert result["benchmark_gain_established"] is True
    assert result["recurrent_rwkv_causal_attribution_established"] is True
    assert result["gates"]["passed"] is True
    assert result["status"] == "native_benchmark_and_recurrent_causal_pass"
    assert result["aggregates"]["mean_hybrid_minus_projected_micro_f1"] == 0.5
    assert result["aggregates"][
        "zero_recurrent_exactly_matches_projected_bypass_predictions"
    ] is True


def test_analysis_rejects_no_hybrid_gain() -> None:
    outputs, gold = _outputs(hybrid_matches_projected=True)

    result = analysis.build_analysis(outputs, gold)

    assert result["benchmark_gain_established"] is False
    assert result["recurrent_rwkv_causal_attribution_established"] is False
    assert result["gates"]["mean_hybrid_minus_projected_micro_f1_minimum"] is False
    assert result["gates"][
        "paired_output_change_fraction_hybrid_vs_projected_minimum"
    ] is False


def test_analysis_detects_projected_carrier_drift() -> None:
    outputs, gold = _outputs()
    outputs["hybrid_candidate"][57]["zero_recurrent_state"][0][
        "projected_carrier_sha256"
    ] = "changed"

    result = analysis.build_analysis(outputs, gold)

    assert result["gates"][
        "projected_carrier_hash_fixed_for_every_hybrid_intervention"
    ] is False
    assert result["gates"]["passed"] is False
