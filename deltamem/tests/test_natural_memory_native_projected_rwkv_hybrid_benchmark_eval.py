from __future__ import annotations

import torch

from experiments.rethinking_rwkv_ms_gemma import (
    run_natural_memory_native_projected_rwkv_hybrid_benchmark_eval as evaluator,
)


def _development_file():
    return evaluator.training.DATASET_ROOT / evaluator.DATASET_RELATIVE_PATH


def test_authorized_evaluation_selection_is_locked_before_parsing() -> None:
    metadata = evaluator.raw_line_metadata(_development_file())
    rows = evaluator.parse_authorized_rows(metadata)

    assert len(metadata) == 361
    assert len(rows) == evaluator.EVALUATION_ROWS
    assert all(int(row["source_index"]) >= evaluator.EXCLUDED_INITIAL_ROWS for row in rows)
    assert {int(row["source_index"]) for row in rows}.isdisjoint(
        {0, 1, 2, 3}
    )


def test_donor_mapping_is_different_gold_and_deterministic() -> None:
    metadata = evaluator.raw_line_metadata(_development_file())
    rows = evaluator.parse_authorized_rows(metadata)
    counts = {int(row["source_index"]): index % 17 for index, row in enumerate(rows)}

    first = evaluator.build_donor_mapping(rows, counts)
    second = evaluator.build_donor_mapping(rows, counts)

    assert first == second
    assert set(first) == {int(row["source_index"]) for row in rows}
    gold = {
        int(row["source_index"]): evaluator.recovery.strict_gold_boundaries(row["gold"])
        for row in rows
    }
    assert all(gold[source] != gold[donor] for source, donor in first.items())


def test_layer_permutation_preserves_keys_and_changes_digest() -> None:
    module_names = tuple(f"model.layers.{index}.self_attn" for index in range(42))
    state = {}
    for index, name in enumerate(module_names):
        for suffix in evaluator.RECURRENT_SUFFIXES:
            state[f"{name}{suffix}"] = torch.full((1, 2), float(index))

    permuted = evaluator.permute_recurrent_state(state, module_names)

    assert set(permuted) == set(state)
    assert evaluator.tensor_digest(permuted) != evaluator.tensor_digest(state)
    assert torch.equal(
        permuted[f"{module_names[0]}{evaluator.RECURRENT_SUFFIXES[0]}"],
        state[f"{module_names[1]}{evaluator.RECURRENT_SUFFIXES[0]}"],
    )


def test_zero_recurrent_state_preserves_shapes_and_values_contract() -> None:
    recurrent = {
        "layer": torch.tensor([[1.0, -2.0]]),
        "layer.__rwkv_ms_positions": torch.tensor([3], dtype=torch.long),
        "layer.__rwkv_ms_previous_source": torch.tensor([7], dtype=torch.long),
    }

    zeroed = evaluator.zero_recurrent_state(recurrent)

    assert set(zeroed) == set(recurrent)
    assert zeroed["layer"].dtype == recurrent["layer"].dtype
    assert zeroed["layer.__rwkv_ms_positions"].dtype == torch.long
    assert torch.count_nonzero(zeroed["layer"]).item() == 0
