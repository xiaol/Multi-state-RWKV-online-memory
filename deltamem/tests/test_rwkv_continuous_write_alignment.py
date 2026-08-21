from __future__ import annotations

import inspect

import torch

from experiments.rethinking_rwkv_ms_gemma import rwkv_continuous_write_alignment as alignment
from experiments.rethinking_rwkv_ms_gemma import rwkv_continuous_write_integration as integration
from experiments.rethinking_rwkv_ms_gemma.rwkv_continuous_write_integration import (
    ContinuousWriteConditioner,
)


def test_reduced_rank_ridge_recovers_disjoint_synthetic_identity() -> None:
    generator = torch.Generator().manual_seed(31)
    module_names = ("layer.0", "layer.1", "layer.2", "layer.3")
    rows = 64
    addresses = torch.randn(
        rows,
        len(module_names),
        alignment.ADDRESS_DIM,
        generator=generator,
    )
    true_maps = torch.randn(
        len(module_names),
        alignment.ADDRESS_DIM,
        alignment.STATE_DIM,
        generator=generator,
    )
    receptance = torch.einsum("nla,lad->nld", addresses, true_maps)
    maps = alignment.fit_layer_maps(
        addresses[:48],
        receptance[:48],
        module_names,
        rank=alignment.MAP_RANK,
        ridge=alignment.RIDGE,
    )
    heldout_addresses = addresses[48:]
    heldout_receptance = receptance[48:]
    donor_indices = torch.arange(16, dtype=torch.long).roll(1)
    metrics = alignment.alignment_metrics(
        heldout_addresses,
        heldout_receptance,
        donor_indices,
        module_names,
        maps,
    )

    assert metrics["finite"] is True
    assert metrics["donor_positive_row_fraction"] >= 0.95
    assert metrics["donor_mean_gap"] > 0.05
    assert metrics["layer_permuted_positive_row_fraction"] >= 0.95


def test_zero_address_maps_to_exact_zero_direction() -> None:
    weights = alignment.FrozenMapWeights(
        down=torch.randn(alignment.MAP_RANK, alignment.ADDRESS_DIM),
        up=torch.randn(alignment.STATE_DIM, alignment.MAP_RANK),
    )
    address = torch.zeros(3, alignment.ADDRESS_DIM)
    direction = alignment.mapped_direction(address, weights)

    assert torch.equal(direction, torch.zeros_like(direction))


def test_fitted_weights_load_frozen_into_runtime_conditioner() -> None:
    generator = torch.Generator().manual_seed(43)
    addresses = torch.randn(48, alignment.ADDRESS_DIM, generator=generator)
    target_map = torch.randn(
        alignment.ADDRESS_DIM,
        alignment.STATE_DIM,
        generator=generator,
    )
    receptance = addresses @ target_map
    weights = alignment.fit_reduced_rank_ridge(addresses, receptance)
    conditioner = ContinuousWriteConditioner(
        alignment.ADDRESS_DIM,
        alignment.STATE_DIM,
        rank=alignment.MAP_RANK,
        seed=47,
        k_gain=0.25,
        a_gain=0.25,
        b_gain=0.25,
        trainable_map=True,
    )

    conditioner.load_frozen_map(weights.down, weights.up)

    assert torch.equal(conditioner.down, weights.down)
    assert torch.equal(conditioner.up, weights.up)
    assert conditioner.down.requires_grad is False
    assert conditioner.up.requires_grad is False


def test_runtime_default_rank_matches_precommitted_alignment_rank() -> None:
    default_rank = inspect.signature(integration.install).parameters["rank"].default

    assert default_rank == alignment.MAP_RANK == 16
