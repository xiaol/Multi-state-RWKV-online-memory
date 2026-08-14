from __future__ import annotations

import torch

from experiments.rethinking_rwkv_ms_gemma import (
    analyze_natural_memory_native_scene_repair_bridge as analyzer,
)
from experiments.rethinking_rwkv_ms_gemma import (
    materialize_natural_memory_native_scene_repair_bridges as materializer,
)


def test_repair_bridge_protocol_is_hash_bound_and_convex() -> None:
    protocol = materializer.validate_protocol()
    recipes = protocol["candidate_materialization"]["candidates"]

    assert materializer.PROTOCOL_PAYLOAD_SHA256 == (
        "44a97af8f0249f9c0cf129d6566bf0882f8ca1fc8f868c334cd2c7ae98fb0d7f"
    )
    assert len(recipes) == 3
    assert len({recipe["candidate_id"] for recipe in recipes}) == len(recipes)
    for recipe in recipes:
        assert set(recipe["weights"]) == set(materializer.SOURCE_ORDER)
        assert all(weight >= 0 for weight in recipe["weights"].values())
        assert sum(recipe["weights"].values()) == recipe["denominator"]


def test_repair_bridge_mixing_uses_locked_source_order() -> None:
    sources = {
        name: {"gate": torch.tensor([index], dtype=torch.float32)}
        for index, name in enumerate(materializer.SOURCE_ORDER)
    }
    recipe = {
        "candidate_id": "test",
        "denominator": 4,
        "weights": {
            "onpolicy": 1,
            "dualpath": 3,
        },
    }

    mixed = materializer.mix_state(sources, recipe)

    assert mixed["gate"].item() == 0.75
    assert mixed["gate"].dtype == torch.float32


def test_repair_bridge_fold_contract_matches_protocol() -> None:
    protocol = materializer.validate_protocol()
    evaluation = protocol["evaluation"]
    protocol_thresholds = evaluation["development_gates"]

    assert evaluation["fold_assignment_payload_sha256"] == analyzer.FOLD_ASSIGNMENT_PAYLOAD_SHA256
    assert {int(key): value for key, value in evaluation["fold_counts"].items()} == analyzer.FOLD_COUNTS
    assert analyzer.GATE_THRESHOLDS == {
        "coverage": protocol_thresholds["coverage_minimum"],
        "oof_minus_checkpoint_16_micro_f1": protocol_thresholds[
            "oof_minus_checkpoint_16_micro_f1_minimum"
        ],
        "oof_minus_v9_micro_f1": protocol_thresholds[
            "oof_minus_v9_micro_f1_minimum"
        ],
        "oof_output_change_fraction_vs_checkpoint_16": protocol_thresholds[
            "oof_output_change_fraction_vs_checkpoint_16_minimum"
        ],
        "learned_bridge_selected_folds": protocol_thresholds[
            "learned_bridge_selected_folds_minimum"
        ],
        "modal_learned_bridge_selected_folds": protocol_thresholds[
            "modal_learned_bridge_selected_folds_minimum"
        ],
        "modal_bridge_minus_checkpoint_16_micro_f1": protocol_thresholds[
            "modal_bridge_minus_checkpoint_16_micro_f1_minimum"
        ],
    }
    assert protocol["study_scope"]["independent_evidence"] is False
    assert protocol["protected_splits_opened_by_this_study"] == []
