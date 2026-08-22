from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

from experiments.rethinking_rwkv_ms_gemma import (
    rwkv_address_decoded_token_replacement as ad_rtr,
)
from experiments.rethinking_rwkv_ms_gemma import rwkv_continuous_write_alignment as alignment


def _frozen_map() -> alignment.FrozenMapWeights:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(17)
    down = torch.randn(
        ad_rtr.MAP_RANK,
        ad_rtr.ADDRESS_DIM,
        generator=generator,
    )
    up = torch.randn(
        ad_rtr.STATE_DIM,
        ad_rtr.MAP_RANK,
        generator=generator,
    )
    return alignment.FrozenMapWeights(down=down, up=up)


def _slot_tensors(batch_size: int = 3) -> tuple[torch.Tensor, ...]:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(29)
    state = torch.randn(
        batch_size,
        ad_rtr.STATE_HEADS,
        ad_rtr.SLOTS,
        ad_rtr.STATE_DIM,
        ad_rtr.STATE_DIM,
        generator=generator,
    )
    keys = torch.randn(
        batch_size,
        ad_rtr.SLOTS,
        ad_rtr.ADDRESS_DIM,
        generator=generator,
    )
    values = torch.randn(
        batch_size,
        ad_rtr.SLOTS,
        ad_rtr.STATE_DIM,
        generator=generator,
    )
    occupied = torch.ones(batch_size, ad_rtr.SLOTS, dtype=torch.bool)
    return state, keys, values, occupied


def test_address_decoded_slots_match_direct_state_contraction() -> None:
    state, keys, values, occupied = _slot_tensors()
    weights = _frozen_map()

    slots = ad_rtr.address_decoded_slots(state, keys, values, occupied, weights)

    expected_direction = alignment.mapped_direction(keys, weights)
    expected = torch.einsum(
        "bhsij,bsj->bhsi",
        state.float(),
        expected_direction,
    )[:, 0]
    assert slots.directions.shape == (3, 4, 32)
    assert slots.contracted.shape == (3, 4, 32)
    assert torch.allclose(slots.directions, expected_direction)
    assert torch.allclose(slots.contracted, expected)


def test_slot_validation_fails_closed_on_shape_dtype_and_nonfinite_values() -> None:
    state, keys, values, occupied = _slot_tensors()
    weights = _frozen_map()

    with pytest.raises(ValueError, match="state shape differs"):
        ad_rtr.address_decoded_slots(state[:, :, :3], keys, values, occupied, weights)
    with pytest.raises(ValueError, match="occupied mask must be boolean"):
        ad_rtr.address_decoded_slots(state, keys, values, occupied.float(), weights)
    bad_keys = keys.clone()
    bad_keys[0, 0, 0] = torch.nan
    with pytest.raises(ValueError, match="slot tensors are nonfinite"):
        ad_rtr.address_decoded_slots(state, bad_keys, values, occupied, weights)


def test_full_rank_ridge_decoder_is_bias_free_and_preserves_exact_zero() -> None:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(41)
    contracted = torch.randn(16, 4, 32, generator=generator)
    true_weight = torch.randn(32, 32, generator=generator)
    values = F.linear(contracted, true_weight)
    occupied = torch.ones(16, 4, dtype=torch.bool)

    decoder = ad_rtr.fit_full_rank_ridge_decoder(
        contracted,
        values,
        occupied,
        ridge=1e-8,
    )
    decoded = decoder.decode(contracted)

    assert decoder.payload() == {
        "input_dim": 32,
        "output_dim": 32,
        "bias": False,
        "rank_reduction": False,
        "ridge": 1e-8,
    }
    assert torch.allclose(decoded, values, rtol=2e-4, atol=2e-4)
    assert torch.equal(decoder.decode(torch.zeros_like(contracted)), torch.zeros_like(contracted))


def test_reconstruction_metrics_cover_all_controls_and_exact_zero_paths() -> None:
    state, keys, _, occupied = _slot_tensors(batch_size=12)
    weights = _frozen_map()
    contracted = ad_rtr.address_decoded_slots(
        state,
        keys,
        torch.zeros(12, 4, 32),
        occupied,
        weights,
    ).contracted
    values = contracted.detach().clone()
    decoder = ad_rtr.fit_full_rank_ridge_decoder(
        contracted,
        values,
        occupied,
        ridge=1e-6,
    )

    metrics = ad_rtr.reconstruction_control_metrics(
        correct_state=state,
        matched_donor_state=torch.zeros_like(state),
        layer_roll_state=-state,
        keys=keys,
        wrong_address_keys=torch.zeros_like(keys),
        values=values,
        occupied=occupied,
        weights=weights,
        decoder=decoder,
    )

    assert metrics["finite"] is True
    assert metrics["active_slots"] == 48
    assert metrics["active_rows"] == 12
    assert metrics["cosine"]["correct"] > 0.999
    assert metrics["cosine"]["matched_donor_state"] == 0.0
    assert metrics["cosine"]["wrong_address"] == 0.0
    assert metrics["cosine"]["layer_roll"] < -0.999
    assert all(value > 0.999 for value in metrics["positive_slot_fractions"].values())
    assert all(value > 0.999 for value in metrics["positive_row_fractions"].values())
    assert all(metrics["zero_audit"].values())


def test_ridge_and_metric_masks_require_occupied_slots() -> None:
    state, keys, values, occupied = _slot_tensors(batch_size=2)
    empty = torch.zeros_like(occupied)
    weights = _frozen_map()
    contracted = ad_rtr.address_decoded_slots(
        state,
        keys,
        values,
        occupied,
        weights,
    ).contracted

    with pytest.raises(ValueError, match="no occupied slots"):
        ad_rtr.fit_full_rank_ridge_decoder(
            contracted,
            values,
            empty,
            ridge=1.0,
        )


def _layered_fixture() -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    tuple[str, ...],
    dict[str, alignment.FrozenMapWeights],
    dict[str, ad_rtr.FullRankRidgeDecoder],
]:
    rows = 4
    modules = 4
    generator = torch.Generator(device="cpu")
    generator.manual_seed(71)
    base_state = torch.randn(2, 1, 32, 32, generator=generator)
    base_keys = torch.randn(2, 64, generator=generator)
    row_sign = torch.tensor([1.0, -1.0, 1.0, -1.0])
    module_sign = torch.tensor([1.0, -1.0, 1.0, -1.0])
    pair_group = torch.tensor([0, 0, 1, 1])
    state = torch.zeros(rows, modules, 1, 4, 32, 32)
    keys = torch.zeros(rows, modules, 4, 64)
    occupied = torch.zeros(rows, modules, 4, dtype=torch.bool)
    for row in range(rows):
        for module in range(modules):
            slot = (row + module) % 4
            occupied[row, module, slot] = True
            state[row, module, :, slot] = (
                row_sign[row]
                * module_sign[module]
                * base_state[pair_group[row]]
            )
            keys[row, module, slot] = (
                row_sign[row]
                * module_sign[module]
                * base_keys[pair_group[row]]
            )
    donor_indices = torch.tensor([1, 0, 3, 2], dtype=torch.long)
    module_names = tuple(f"layer.{index}" for index in range(modules))
    maps = {name: _frozen_map() for name in module_names}
    values = torch.empty(rows, modules, 4, 32)
    decoders: dict[str, ad_rtr.FullRankRidgeDecoder] = {}
    for module, name in enumerate(module_names):
        dummy_values = torch.zeros(rows, 4, 32)
        contracted = ad_rtr.address_decoded_slots(
            state[:, module],
            keys[:, module],
            dummy_values,
            occupied[:, module],
            maps[name],
        ).contracted
        values[:, module] = contracted
        decoders[name] = ad_rtr.fit_full_rank_ridge_decoder(
            contracted,
            values[:, module],
            occupied[:, module],
            ridge=1e-8,
        )
    return (
        state,
        keys,
        values,
        occupied,
        donor_indices,
        module_names,
        maps,
        decoders,
    )


def test_layered_metrics_construct_donor_address_and_layer_roll_controls() -> None:
    (
        state,
        keys,
        values,
        occupied,
        donor_indices,
        module_names,
        maps,
        decoders,
    ) = _layered_fixture()

    metrics = ad_rtr.layered_reconstruction_control_metrics(
        state=state,
        keys=keys,
        values=values,
        occupied=occupied,
        donor_indices=donor_indices,
        module_names=module_names,
        maps=maps,
        decoders=decoders,
        cosine_threshold=0.99,
    )

    assert metrics["finite"] is True
    assert metrics["rows"] == 4
    assert metrics["modules"] == 4
    assert metrics["correct_mean_cosine"] > 0.999
    assert metrics["correct_at_least_threshold_fraction"] == 1.0
    assert all(value > 1.999 for value in metrics["mean_gaps"].values())
    assert all(value == 1.0 for value in metrics["positive_row_fractions"].values())
    assert len(metrics["per_row"]["correct_cosine"]) == 4
    assert set(metrics["per_module"]) == set(module_names)
    assert all(
        module["correct_at_least_threshold_fraction"] == 1.0
        for module in metrics["per_module"].values()
    )
    assert metrics["zero_audit"]["all_modules_exact"] is True


def test_layered_metrics_fail_closed_on_donor_and_module_inventory() -> None:
    (
        state,
        keys,
        values,
        occupied,
        donor_indices,
        module_names,
        maps,
        decoders,
    ) = _layered_fixture()
    bad_donors = donor_indices.clone()
    bad_donors[0] = state.shape[0]

    with pytest.raises(ValueError, match="donor index is out of range"):
        ad_rtr.layered_reconstruction_control_metrics(
            state=state,
            keys=keys,
            values=values,
            occupied=occupied,
            donor_indices=bad_donors,
            module_names=module_names,
            maps=maps,
            decoders=decoders,
            cosine_threshold=0.0,
        )
    with pytest.raises(ValueError, match="map or decoder inventory differs"):
        ad_rtr.layered_reconstruction_control_metrics(
            state=state,
            keys=keys,
            values=values,
            occupied=occupied,
            donor_indices=donor_indices,
            module_names=module_names,
            maps={module_names[0]: maps[module_names[0]]},
            decoders=decoders,
            cosine_threshold=0.0,
        )
