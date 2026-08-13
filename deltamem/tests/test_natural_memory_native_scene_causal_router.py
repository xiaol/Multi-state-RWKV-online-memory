from __future__ import annotations

import torch

from experiments.rethinking_rwkv_ms_gemma import (
    analyze_natural_memory_native_scene_causal_router as analysis,
)
from experiments.rethinking_rwkv_ms_gemma import (
    run_natural_memory_native_scene_causal as runner,
)


def test_scene_causal_protocol_receipt_is_bound() -> None:
    protocol = runner.validate_protocol()

    assert protocol["data_scope"]["reported_rows"] == 357
    assert tuple(protocol["router_candidates"]) == analysis.CANDIDATE_NAMES
    assert protocol["authorization"]["publisher_validation_predictions_allowed_as_input"] is False
    assert protocol["authorization"]["publisher_test_authorized"] is False
    assert protocol["authorization"]["hard32_authorized"] is False


def test_scene_causal_runner_code_hash_is_bound() -> None:
    assert runner.sha256_file(runner.Path(runner.__file__)) == (
        analysis.EXPECTED_RUNNER_SHA256
    )


def test_donor_mapping_is_length_matched_gold_distinct_and_deterministic() -> None:
    rows = [
        {"source_index": 0, "gold": {"boundaries": [0]}},
        {"source_index": 1, "gold": {"boundaries": [1]}},
        {"source_index": 2, "gold": {"boundaries": [2]}},
        {"source_index": 3, "gold": {"boundaries": [3]}},
        {"source_index": 4, "gold": {"boundaries": [1]}},
        {"source_index": 5, "gold": {"boundaries": [2]}},
        {"source_index": 6, "gold": {"boundaries": [3]}},
    ]
    counts = [1, 1, 1, 1, 100, 103, 97]

    mapping = runner.build_donor_mapping(rows, counts)

    assert mapping == {4: 5, 5: 4, 6: 4}


def test_layer_permutation_moves_complete_bundles() -> None:
    module_names = tuple(f"model.layers.{index}.self_attn" for index in range(42))
    state = {}
    for index, module_name in enumerate(module_names):
        for suffix in runner.STATE_SUFFIXES:
            dtype = torch.bool if suffix == ".__projected_kv_occupied" else torch.float32
            state[f"{module_name}{suffix}"] = torch.tensor([index], dtype=dtype)

    permuted = runner.permute_layer_state(state, module_names)

    for suffix in runner.STATE_SUFFIXES:
        assert permuted[f"{module_names[0]}{suffix}"].item() == 1
        assert permuted[f"{module_names[-1]}{suffix}"].item() == 0


def test_router_methods_cover_recall_recovery_options() -> None:
    base = {2, 5, 9}
    memory = {3, 9}

    assert analysis.route("intersection", base, memory) == {9}
    assert analysis.route("memory_plus_small_base_3", base, memory) == {2, 3, 5, 9}
    assert analysis.route("memory_union_near_base_1", base, memory) == {2, 3, 9}
    assert analysis.route("snap_memory_to_base_1", base, memory) == {2, 9}
    assert analysis.route("intersection_else_memory", base, memory) == {9}


def test_fit_and_holdout_partition_are_disjoint() -> None:
    indices = tuple(range(4, 361))
    fit = {index for index in indices if index % 5 != 4}
    holdout = {index for index in indices if index % 5 == 4}

    assert not (fit & holdout)
    assert fit | holdout == set(indices)
    assert len(fit) == 285
    assert len(holdout) == 72
