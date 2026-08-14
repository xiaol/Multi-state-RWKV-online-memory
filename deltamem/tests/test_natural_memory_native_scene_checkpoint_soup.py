from __future__ import annotations

import json

import torch

from experiments.rethinking_rwkv_ms_gemma import (
    analyze_natural_memory_native_scene_checkpoint_soup as analyzer,
)
from experiments.rethinking_rwkv_ms_gemma import (
    materialize_natural_memory_native_scene_checkpoint_soups as materializer,
)


def test_checkpoint_soup_protocol_is_hash_bound_and_convex() -> None:
    protocol = materializer.validate_protocol()
    recipes = protocol["candidate_materialization"]["candidates"]

    assert len(recipes) == 7
    assert len({recipe["candidate_id"] for recipe in recipes}) == len(recipes)
    for recipe in recipes:
        assert set(recipe["weights"]) == set(materializer.SOURCE_ORDER)
        assert all(weight >= 0 for weight in recipe["weights"].values())
        assert sum(recipe["weights"].values()) == recipe["denominator"]


def test_checkpoint_soup_mixing_uses_locked_source_order() -> None:
    sources = {
        name: {"gate": torch.tensor([index], dtype=torch.float32)}
        for index, name in enumerate(materializer.SOURCE_ORDER)
    }
    recipe = {
        "candidate_id": "test",
        "denominator": 4,
        "weights": {
            "v9": 1,
            "checkpoint_8": 0,
            "checkpoint_16": 3,
            "checkpoint_32": 0,
        },
    }

    mixed = materializer.mix_state(sources, recipe)

    assert mixed["gate"].item() == 1.5
    assert mixed["gate"].dtype == torch.float32


def test_checkpoint_soup_fold_contract_matches_protocol() -> None:
    protocol = materializer.validate_protocol()
    evaluation = protocol["evaluation"]

    assert evaluation["fold_assignment_payload_sha256"] == analyzer.FOLD_ASSIGNMENT_PAYLOAD_SHA256
    assert {int(key): value for key, value in evaluation["fold_counts"].items()} == analyzer.FOLD_COUNTS
    assert json.dumps(analyzer.GATE_THRESHOLDS, sort_keys=True)
