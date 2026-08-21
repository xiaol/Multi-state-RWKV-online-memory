"""Frozen reduced-rank alignment from full slot addresses to RWKV receptance."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import torch


ADDRESS_DIM = 64
STATE_DIM = 32
MAP_RANK = 16
RIDGE = 1.0


@dataclass(frozen=True)
class FrozenMapWeights:
    down: torch.Tensor
    up: torch.Tensor

    def payload(self) -> dict[str, Any]:
        return {
            "address_dim": int(self.down.shape[1]),
            "state_dim": int(self.up.shape[0]),
            "rank": int(self.down.shape[0]),
        }


def _rms_normalize(value: torch.Tensor) -> torch.Tensor:
    value = value.float()
    square_mean = value.square().mean(dim=-1, keepdim=True)
    active = square_mean.gt(0.0)
    normalized = value / square_mean.clamp_min(1e-12).sqrt()
    return torch.where(active, normalized, torch.zeros_like(normalized))


def fit_reduced_rank_ridge(
    addresses: torch.Tensor,
    receptance: torch.Tensor,
    *,
    rank: int = MAP_RANK,
    ridge: float = RIDGE,
) -> FrozenMapWeights:
    if addresses.ndim != 2 or receptance.ndim != 2:
        raise ValueError("Continuous-write alignment inputs must be matrices")
    if addresses.shape[0] != receptance.shape[0] or addresses.shape[0] < 2:
        raise ValueError("Continuous-write alignment row counts differ")
    if addresses.shape[1] != ADDRESS_DIM or receptance.shape[1] != STATE_DIM:
        raise ValueError("Continuous-write alignment feature widths differ")
    if not 1 <= int(rank) <= min(ADDRESS_DIM, STATE_DIM):
        raise ValueError("Continuous-write alignment rank is invalid")
    if not float(ridge) > 0.0:
        raise ValueError("Continuous-write ridge must be positive")
    if not bool(torch.isfinite(addresses).all().item()) or not bool(
        torch.isfinite(receptance).all().item()
    ):
        raise ValueError("Continuous-write alignment inputs are nonfinite")

    source = _rms_normalize(addresses).double().cpu()
    target = _rms_normalize(receptance).double().cpu()
    covariance = source.T @ source
    right_hand_side = source.T @ target
    full = torch.linalg.solve(
        covariance + float(ridge) * torch.eye(ADDRESS_DIM, dtype=torch.float64),
        right_hand_side,
    )
    left, singular, right = torch.linalg.svd(full, full_matrices=False)
    retained = int(rank)
    root = singular[:retained].clamp_min(0.0).sqrt()
    down = (left[:, :retained] * root).T.contiguous().float()
    up = (right[:retained].T * root).contiguous().float()
    if not bool(torch.isfinite(down).all().item()) or not bool(
        torch.isfinite(up).all().item()
    ):
        raise RuntimeError("Continuous-write fitted map is nonfinite")
    return FrozenMapWeights(down=down, up=up)


def mapped_direction(
    addresses: torch.Tensor,
    weights: FrozenMapWeights,
) -> torch.Tensor:
    if addresses.shape[-1] != weights.down.shape[1]:
        raise ValueError("Continuous-write mapped address width differs")
    latent = torch.nn.functional.linear(
        _rms_normalize(addresses),
        weights.down.float(),
    )
    mapped = torch.nn.functional.linear(latent, weights.up.float())
    return _rms_normalize(mapped)


def fit_layer_maps(
    addresses: torch.Tensor,
    receptance: torch.Tensor,
    module_names: Sequence[str],
    *,
    rank: int = MAP_RANK,
    ridge: float = RIDGE,
) -> dict[str, FrozenMapWeights]:
    if addresses.ndim != 3 or receptance.ndim != 3:
        raise ValueError("Continuous-write layered alignment inputs must be rank three")
    if addresses.shape[:2] != receptance.shape[:2]:
        raise ValueError("Continuous-write layered row/module axes differ")
    if addresses.shape[1] != len(module_names) or len(set(module_names)) != len(
        module_names
    ):
        raise ValueError("Continuous-write alignment module inventory differs")
    return {
        name: fit_reduced_rank_ridge(
            addresses[:, index],
            receptance[:, index],
            rank=rank,
            ridge=ridge,
        )
        for index, name in enumerate(module_names)
    }


def apply_layer_maps(
    addresses: torch.Tensor,
    module_names: Sequence[str],
    maps: Mapping[str, FrozenMapWeights],
) -> torch.Tensor:
    if addresses.ndim != 3 or addresses.shape[1] != len(module_names):
        raise ValueError("Continuous-write map application axes differ")
    if set(maps) != set(module_names):
        raise ValueError("Continuous-write fitted map inventory differs")
    return torch.stack(
        [
            mapped_direction(addresses[:, index], maps[name])
            for index, name in enumerate(module_names)
        ],
        dim=1,
    )


def alignment_metrics(
    addresses: torch.Tensor,
    receptance: torch.Tensor,
    donor_indices: torch.Tensor,
    module_names: Sequence[str],
    maps: Mapping[str, FrozenMapWeights],
) -> Mapping[str, float | bool]:
    if donor_indices.ndim != 1 or donor_indices.shape[0] != addresses.shape[0]:
        raise ValueError("Continuous-write donor indices differ")
    if donor_indices.dtype != torch.long:
        raise ValueError("Continuous-write donor indices must be int64")
    if bool(donor_indices.lt(0).any().item()) or bool(
        donor_indices.ge(addresses.shape[0]).any().item()
    ):
        raise ValueError("Continuous-write donor index is out of range")
    query = _rms_normalize(receptance)
    correct_direction = apply_layer_maps(addresses, module_names, maps)
    donor_direction = correct_direction.index_select(0, donor_indices)
    permuted_direction = apply_layer_maps(
        addresses.roll(1, dims=1),
        module_names,
        maps,
    )
    denominator = float(receptance.shape[-1])
    correct = (query * correct_direction).sum(dim=-1) / denominator
    donor = (query * donor_direction).sum(dim=-1) / denominator
    permuted = (query * permuted_direction).sum(dim=-1) / denominator
    donor_gap = correct - donor
    permuted_gap = correct - permuted
    finite = all(
        bool(torch.isfinite(value).all().item())
        for value in (correct, donor, permuted, donor_gap, permuted_gap)
    )
    return {
        "finite": finite,
        "donor_positive_module_fraction": float(donor_gap.gt(0.0).float().mean()),
        "donor_positive_row_fraction": float(
            donor_gap.mean(dim=1).gt(0.0).float().mean()
        ),
        "donor_mean_gap": float(donor_gap.mean()),
        "layer_permuted_positive_module_fraction": float(
            permuted_gap.gt(0.0).float().mean()
        ),
        "layer_permuted_positive_row_fraction": float(
            permuted_gap.mean(dim=1).gt(0.0).float().mean()
        ),
        "layer_permuted_mean_gap": float(permuted_gap.mean()),
    }
