from __future__ import annotations

import torch

from experiments.rethinking_rwkv_ms_gemma import (
    run_natural_memory_native_rwkv_query_state_bilinear_crossfit as screen,
)


def test_component_crossfit_is_exact_and_donor_disjoint() -> None:
    mapping = {
        0: 1,
        1: 0,
        **{index: index + 1 if index % 2 == 0 else index - 1 for index in range(2, 220)},
    }
    split, payload = screen.crossfit_split(mapping)
    assert sum(value == "train" for value in split.values()) == 176
    assert sum(value == "heldout" for value in split.values()) == 44
    assert all(split[source] == split[donor] for source, donor in mapping.items())
    assert payload["component_count"] == 110


def test_layerwise_bilinear_shapes() -> None:
    head = screen.LayerwiseBilinear()
    query = torch.randn(3, screen.LAYERS, screen.STATE_DIM)
    state = torch.randn_like(query)
    assert head.score(query, state).shape == (3, screen.LAYERS)


def test_tiny_synthetic_train_eval_contract(monkeypatch) -> None:
    monkeypatch.setattr(screen, "TRAIN_ROWS", 8)
    monkeypatch.setattr(screen, "HELDOUT_ROWS", 4)
    monkeypatch.setattr(screen, "TRAIN_STEPS", 8)
    generator = torch.Generator().manual_seed(11)
    records = []
    for source in range(12):
        query = torch.randn(
            screen.LAYERS,
            screen.STATE_DIM,
            generator=generator,
        )
        correct = query + 0.01 * torch.randn(
            query.shape,
            generator=generator,
        )
        donor = -query + 0.01 * torch.randn(
            query.shape,
            generator=generator,
        )
        permuted = correct.roll(1, dims=0)
        records.append(
            {
                "source_index": source,
                "split": "train" if source < 8 else "heldout",
                "query": query.tolist(),
                "correct": correct.tolist(),
                "matched_donor": donor.tolist(),
                "layer_permuted": permuted.tolist(),
            }
        )
    result = screen.train_and_evaluate(records)
    assert result["loss"]["final"] <= result["loss"]["initial"]
    assert result["metrics"]["heldout"]["donor"]["finite"] is True
    assert result["weights_saved"] is False
