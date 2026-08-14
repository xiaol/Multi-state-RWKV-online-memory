from __future__ import annotations

import pytest

from experiments.rethinking_rwkv_ms_gemma import (
    analyze_natural_memory_native_scene_consistency_router as analyzer,
)


def test_consistency_router_protocol_is_hash_bound_and_adaptive() -> None:
    protocol = analyzer.validate_protocol()

    assert analyzer.PROTOCOL_PAYLOAD_SHA256 == (
        "9601af58d9bcebf908fcbcae6bb79e65b207fec238f45ee553377f5f18a52056"
    )
    assert protocol["study_scope"]["classification"].startswith("post-hoc")
    assert protocol["study_scope"]["independent_evidence"] is False
    assert protocol["router"]["gold_available_at_inference"] is False
    assert protocol["router"]["candidate_order"] == list(analyzer.POLICY_ORDER)
    assert protocol["protected_splits_opened_by_this_study"] == []


def test_consistency_router_uses_only_bounded_proposal_changes() -> None:
    assert analyzer.proposal_reason({1, 4, 6}, {4, 6}) == "strict_subset"
    assert analyzer.proposal_reason(set(), {9}) == "abstention_singleton"
    assert analyzer.proposal_reason({1}, {1, 2}) is None
    assert analyzer.proposal_reason(set(), {1, 2}) is None
    assert analyzer.proposal_reason(None, {1}) is None

    assert analyzer.route_prediction({1, 4, 6}, {4, 6}, policy_id="combined") == {
        4,
        6,
    }
    assert analyzer.route_prediction(set(), {9}, policy_id="combined") == {9}
    assert analyzer.route_prediction({1}, {1, 2}, policy_id="combined") == {1}
    assert analyzer.route_prediction({1}, None, policy_id="combined") == {1}
    with pytest.raises(ValueError, match="Unknown consistency-router policy"):
        analyzer.route_prediction(set(), {1}, policy_id="oracle")


def test_consistency_router_policy_family_is_complete() -> None:
    checkpoint = {0: {1, 4}, 1: set(), 2: {2}}
    proposal = {0: {4}, 1: {7}, 2: {2, 3}}

    predictions = analyzer.policy_predictions(checkpoint, proposal)

    assert list(predictions) == list(analyzer.POLICY_ORDER)
    assert predictions["checkpoint"] == checkpoint
    assert predictions["strict_subset"] == {0: {4}, 1: set(), 2: {2}}
    assert predictions["abstention_singleton"] == {0: {1, 4}, 1: {7}, 2: {2}}
    assert predictions["combined"] == {0: {4}, 1: {7}, 2: {2}}


def test_consistency_router_thresholds_match_protocol() -> None:
    protocol = analyzer.validate_protocol()
    thresholds = protocol["evaluation"]["development_gates"]

    assert protocol["evaluation"]["fold_assignment_payload_sha256"] == (
        analyzer.FOLD_ASSIGNMENT_PAYLOAD_SHA256
    )
    assert {
        int(key): value for key, value in protocol["evaluation"]["fold_counts"].items()
    } == analyzer.FOLD_COUNTS
    assert analyzer.GATE_THRESHOLDS == {
        "coverage": thresholds["coverage_minimum"],
        "oof_minus_checkpoint_16_micro_f1": thresholds[
            "oof_minus_checkpoint_16_micro_f1_minimum"
        ],
        "oof_minus_v9_micro_f1": thresholds["oof_minus_v9_micro_f1_minimum"],
        "oof_output_change_fraction_vs_checkpoint_16": thresholds[
            "oof_output_change_fraction_vs_checkpoint_16_minimum"
        ],
        "combined_selected_folds": thresholds["combined_selected_folds_minimum"],
        "combined_tp_gain": thresholds["combined_tp_gain_minimum"],
        "combined_fp_delta_maximum": thresholds["combined_fp_delta_maximum"],
    }
