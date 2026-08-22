"""Pure stage-0 reconstruction utilities for address-decoded RWKV slots."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Mapping, Sequence

import torch
import torch.nn.functional as F

from experiments.rethinking_rwkv_ms_gemma import rwkv_continuous_write_alignment as alignment


STATE_HEADS = 1
SLOTS = 4
STATE_DIM = 32
ADDRESS_DIM = 64
MAP_RANK = 16


@dataclass(frozen=True)
class AddressDecodedSlots:
    directions: torch.Tensor
    contracted: torch.Tensor


@dataclass(frozen=True)
class FullRankRidgeDecoder:
    weight: torch.Tensor
    ridge: float

    def __post_init__(self) -> None:
        if tuple(self.weight.shape) != (STATE_DIM, STATE_DIM):
            raise ValueError("AD-RTR ridge decoder weight shape differs")
        if not bool(torch.isfinite(self.weight).all().item()):
            raise ValueError("AD-RTR ridge decoder weight is nonfinite")
        if not math.isfinite(float(self.ridge)) or float(self.ridge) <= 0.0:
            raise ValueError("AD-RTR ridge must be finite and positive")
        if self.weight.requires_grad:
            raise ValueError("AD-RTR stage-0 ridge decoder must be frozen")

    def decode(self, contracted: torch.Tensor) -> torch.Tensor:
        if contracted.shape[-1] != STATE_DIM:
            raise ValueError("AD-RTR contracted value width differs")
        if not bool(torch.isfinite(contracted).all().item()):
            raise ValueError("AD-RTR contracted values are nonfinite")
        decoded = F.linear(
            contracted.float(),
            self.weight.to(device=contracted.device, dtype=torch.float32),
        )
        if not bool(torch.isfinite(decoded).all().item()):
            raise RuntimeError("AD-RTR decoded values are nonfinite")
        return decoded

    def payload(self) -> dict[str, Any]:
        return {
            "input_dim": STATE_DIM,
            "output_dim": STATE_DIM,
            "bias": False,
            "rank_reduction": False,
            "ridge": float(self.ridge),
        }


def validate_slot_tensors(
    state: torch.Tensor,
    keys: torch.Tensor,
    values: torch.Tensor,
    occupied: torch.Tensor,
) -> None:
    batch_size = state.shape[0] if state.ndim > 0 else 0
    expected = {
        "state": (batch_size, STATE_HEADS, SLOTS, STATE_DIM, STATE_DIM),
        "keys": (batch_size, SLOTS, ADDRESS_DIM),
        "values": (batch_size, SLOTS, STATE_DIM),
        "occupied": (batch_size, SLOTS),
    }
    actual = {
        "state": tuple(state.shape),
        "keys": tuple(keys.shape),
        "values": tuple(values.shape),
        "occupied": tuple(occupied.shape),
    }
    for name in expected:
        if actual[name] != expected[name]:
            raise ValueError(
                f"AD-RTR {name} shape differs: expected={expected[name]} actual={actual[name]}"
            )
    if batch_size < 1:
        raise ValueError("AD-RTR batch must be nonempty")
    if not state.is_floating_point() or not keys.is_floating_point() or not values.is_floating_point():
        raise ValueError("AD-RTR state, keys, and values must be floating point")
    if occupied.dtype != torch.bool:
        raise ValueError("AD-RTR occupied mask must be boolean")
    if not all(
        bool(torch.isfinite(tensor).all().item()) for tensor in (state, keys, values)
    ):
        raise ValueError("AD-RTR slot tensors are nonfinite")


def _validate_frozen_map(weights: alignment.FrozenMapWeights) -> None:
    if tuple(weights.down.shape) != (MAP_RANK, ADDRESS_DIM):
        raise ValueError("AD-RTR frozen address down-map shape differs")
    if tuple(weights.up.shape) != (STATE_DIM, MAP_RANK):
        raise ValueError("AD-RTR frozen address up-map shape differs")
    if weights.down.requires_grad or weights.up.requires_grad:
        raise ValueError("AD-RTR stage-0 address map must be frozen")
    if not bool(torch.isfinite(weights.down).all().item()) or not bool(
        torch.isfinite(weights.up).all().item()
    ):
        raise ValueError("AD-RTR frozen address map is nonfinite")


def address_decoded_slots(
    state: torch.Tensor,
    keys: torch.Tensor,
    values: torch.Tensor,
    occupied: torch.Tensor,
    weights: alignment.FrozenMapWeights,
) -> AddressDecodedSlots:
    validate_slot_tensors(state, keys, values, occupied)
    _validate_frozen_map(weights)
    device_weights = alignment.FrozenMapWeights(
        down=weights.down.detach().to(device=keys.device, dtype=torch.float32),
        up=weights.up.detach().to(device=keys.device, dtype=torch.float32),
    )
    directions = alignment.mapped_direction(keys.float(), device_weights)
    contracted_heads = torch.einsum(
        "bhsij,bsj->bhsi",
        state.float(),
        directions,
    )
    contracted = contracted_heads[:, 0]
    if tuple(directions.shape) != (state.shape[0], SLOTS, STATE_DIM):
        raise RuntimeError("AD-RTR decoded address shape differs")
    if tuple(contracted.shape) != (state.shape[0], SLOTS, STATE_DIM):
        raise RuntimeError("AD-RTR state contraction shape differs")
    if not bool(torch.isfinite(directions).all().item()) or not bool(
        torch.isfinite(contracted).all().item()
    ):
        raise RuntimeError("AD-RTR address decoding or contraction is nonfinite")
    return AddressDecodedSlots(directions=directions, contracted=contracted)


def fit_full_rank_ridge_decoder(
    contracted: torch.Tensor,
    projected_values: torch.Tensor,
    occupied: torch.Tensor,
    *,
    ridge: float,
) -> FullRankRidgeDecoder:
    if tuple(contracted.shape) != tuple(projected_values.shape):
        raise ValueError("AD-RTR contracted and projected value shapes differ")
    if contracted.ndim != 3 or contracted.shape[1:] != (SLOTS, STATE_DIM):
        raise ValueError("AD-RTR ridge inputs must have shape [batch, 4, 32]")
    if tuple(occupied.shape) != tuple(contracted.shape[:2]) or occupied.dtype != torch.bool:
        raise ValueError("AD-RTR ridge occupied mask differs")
    if not math.isfinite(float(ridge)) or float(ridge) <= 0.0:
        raise ValueError("AD-RTR ridge must be finite and positive")
    if not bool(torch.isfinite(contracted).all().item()) or not bool(
        torch.isfinite(projected_values).all().item()
    ):
        raise ValueError("AD-RTR ridge inputs are nonfinite")
    if not bool(occupied.any().item()):
        raise ValueError("AD-RTR ridge fit has no occupied slots")

    source = contracted[occupied].detach().double().cpu()
    target = projected_values[occupied].detach().double().cpu()
    covariance = source.T @ source
    right_hand_side = source.T @ target
    transform = torch.linalg.solve(
        covariance + float(ridge) * torch.eye(STATE_DIM, dtype=torch.float64),
        right_hand_side,
    )
    weight = transform.T.contiguous().float().detach()
    return FullRankRidgeDecoder(weight=weight, ridge=float(ridge))


def fit_address_decoded_ridge_decoder(
    state: torch.Tensor,
    keys: torch.Tensor,
    values: torch.Tensor,
    occupied: torch.Tensor,
    weights: alignment.FrozenMapWeights,
    *,
    ridge: float,
) -> tuple[FullRankRidgeDecoder, AddressDecodedSlots]:
    slots = address_decoded_slots(state, keys, values, occupied, weights)
    decoder = fit_full_rank_ridge_decoder(
        slots.contracted,
        values,
        occupied,
        ridge=ridge,
    )
    return decoder, slots


def _masked_row_means(values: torch.Tensor, occupied: torch.Tensor) -> torch.Tensor:
    counts = occupied.sum(dim=-1)
    active_rows = counts.gt(0)
    if not bool(active_rows.any().item()):
        raise ValueError("AD-RTR metrics have no occupied rows")
    sums = (values * occupied.to(dtype=values.dtype)).sum(dim=-1)
    return (sums / counts.clamp_min(1).to(dtype=values.dtype))[active_rows]


def canonicalize_active_slots(
    value: torch.Tensor,
    source_occupied: torch.Tensor,
    target_occupied: torch.Tensor,
    *,
    slot_dim: int,
) -> torch.Tensor:
    if source_occupied.shape != target_occupied.shape:
        raise ValueError("AD-RTR control occupancy shapes differ")
    if source_occupied.dtype != torch.bool or target_occupied.dtype != torch.bool:
        raise ValueError("AD-RTR control occupancy must be boolean")
    prefix = source_occupied.shape[:-1]
    if (
        tuple(value.shape[: len(prefix)]) != tuple(prefix)
        or value.shape[slot_dim] != SLOTS
    ):
        raise ValueError("AD-RTR control value and occupancy shapes differ")
    if not bool(source_occupied.sum(dim=-1).eq(1).all().item()) or not bool(
        target_occupied.sum(dim=-1).eq(1).all().item()
    ):
        raise ValueError("AD-RTR control occupancy must contain exactly one slot")
    source_index = source_occupied.to(dtype=torch.long).argmax(dim=-1)
    target_index = target_occupied.to(dtype=torch.long).argmax(dim=-1)
    index_shape = [*prefix, *([1] * (value.ndim - len(prefix)))]
    expanded_shape = list(value.shape)
    expanded_shape[slot_dim] = 1
    source_gather = source_index.reshape(index_shape).expand(expanded_shape)
    target_scatter = target_index.reshape(index_shape).expand(expanded_shape)
    active = value.gather(slot_dim, source_gather)
    return torch.zeros_like(value).scatter(slot_dim, target_scatter, active)


def reconstruction_control_metrics(
    *,
    correct_state: torch.Tensor,
    matched_donor_state: torch.Tensor,
    layer_roll_state: torch.Tensor,
    keys: torch.Tensor,
    wrong_address_keys: torch.Tensor,
    values: torch.Tensor,
    occupied: torch.Tensor,
    weights: alignment.FrozenMapWeights,
    decoder: FullRankRidgeDecoder,
) -> Mapping[str, Any]:
    controls = {
        "correct": (correct_state, keys),
        "matched_donor_state": (matched_donor_state, keys),
        "wrong_address": (correct_state, wrong_address_keys),
        "layer_roll": (layer_roll_state, keys),
    }
    decoded: dict[str, torch.Tensor] = {}
    cosine: dict[str, torch.Tensor] = {}
    for name, (state, address) in controls.items():
        slots = address_decoded_slots(state, address, values, occupied, weights)
        decoded[name] = decoder.decode(slots.contracted)
        cosine[name] = F.cosine_similarity(
            decoded[name].float(),
            values.float(),
            dim=-1,
            eps=1e-6,
        )

    active_cosine = {
        name: score.masked_select(occupied) for name, score in cosine.items()
    }
    gaps = {
        name: active_cosine["correct"] - active_cosine[name]
        for name in ("matched_donor_state", "wrong_address", "layer_roll")
    }
    row_gaps = {
        name: _masked_row_means(cosine["correct"] - cosine[name], occupied)
        for name in gaps
    }

    zero_state_slots = address_decoded_slots(
        torch.zeros_like(correct_state),
        keys,
        values,
        occupied,
        weights,
    )
    zero_address_slots = address_decoded_slots(
        correct_state,
        torch.zeros_like(keys),
        values,
        occupied,
        weights,
    )
    zero_state_decoded = decoder.decode(zero_state_slots.contracted)
    zero_address_decoded = decoder.decode(zero_address_slots.contracted)
    zero = torch.zeros_like(zero_state_slots.contracted)
    finite = all(
        bool(torch.isfinite(tensor).all().item())
        for tensor in (
            *decoded.values(),
            *cosine.values(),
            *gaps.values(),
            *row_gaps.values(),
            zero_state_decoded,
            zero_address_decoded,
        )
    )
    return {
        "finite": finite,
        "active_slots": int(occupied.sum().item()),
        "active_rows": int(occupied.any(dim=-1).sum().item()),
        "cosine": {
            name: float(score.mean().item()) for name, score in active_cosine.items()
        },
        "mean_gaps": {name: float(gap.mean().item()) for name, gap in gaps.items()},
        "positive_slot_fractions": {
            name: float(gap.gt(0.0).float().mean().item()) for name, gap in gaps.items()
        },
        "positive_row_fractions": {
            name: float(gap.gt(0.0).float().mean().item())
            for name, gap in row_gaps.items()
        },
        "zero_audit": {
            "zero_state_direction_finite": bool(
                torch.isfinite(zero_state_slots.directions).all().item()
            ),
            "zero_state_contraction_exact_zero": torch.equal(
                zero_state_slots.contracted, zero
            ),
            "zero_state_decoded_exact_zero": torch.equal(zero_state_decoded, zero),
            "zero_address_direction_exact_zero": torch.equal(
                zero_address_slots.directions,
                torch.zeros_like(zero_address_slots.directions),
            ),
            "zero_address_contraction_exact_zero": torch.equal(
                zero_address_slots.contracted, zero
            ),
            "zero_address_decoded_exact_zero": torch.equal(zero_address_decoded, zero),
        },
    }


def layered_reconstruction_control_metrics(
    *,
    state: torch.Tensor,
    keys: torch.Tensor,
    values: torch.Tensor,
    occupied: torch.Tensor,
    donor_indices: torch.Tensor,
    module_names: Sequence[str],
    maps: Mapping[str, alignment.FrozenMapWeights],
    decoders: Mapping[str, FullRankRidgeDecoder],
    cosine_threshold: float,
) -> Mapping[str, Any]:
    if state.ndim != 6:
        raise ValueError("Layered AD-RTR state must have rank six")
    rows, modules = state.shape[:2]
    expected = {
        "state": (rows, modules, STATE_HEADS, SLOTS, STATE_DIM, STATE_DIM),
        "keys": (rows, modules, SLOTS, ADDRESS_DIM),
        "values": (rows, modules, SLOTS, STATE_DIM),
        "occupied": (rows, modules, SLOTS),
    }
    actual = {
        "state": tuple(state.shape),
        "keys": tuple(keys.shape),
        "values": tuple(values.shape),
        "occupied": tuple(occupied.shape),
    }
    for name in expected:
        if actual[name] != expected[name]:
            raise ValueError(
                f"Layered AD-RTR {name} shape differs: "
                f"expected={expected[name]} actual={actual[name]}"
            )
    if rows < 1 or modules < 1:
        raise ValueError("Layered AD-RTR rows and modules must be nonempty")
    if occupied.dtype != torch.bool:
        raise ValueError("Layered AD-RTR occupied mask must be boolean")
    if not all(
        bool(torch.isfinite(tensor).all().item()) for tensor in (state, keys, values)
    ):
        raise ValueError("Layered AD-RTR tensors are nonfinite")
    if donor_indices.shape != (rows,) or donor_indices.dtype != torch.long:
        raise ValueError("Layered AD-RTR donor indices must have shape [rows] and int64 dtype")
    if bool(donor_indices.lt(0).any().item()) or bool(
        donor_indices.ge(rows).any().item()
    ):
        raise ValueError("Layered AD-RTR donor index is out of range")
    names = tuple(module_names)
    if len(names) != modules or len(set(names)) != modules:
        raise ValueError("Layered AD-RTR module inventory differs")
    if set(maps) != set(names) or set(decoders) != set(names):
        raise ValueError("Layered AD-RTR map or decoder inventory differs")
    if not math.isfinite(float(cosine_threshold)) or not -1.0 <= float(
        cosine_threshold
    ) <= 1.0:
        raise ValueError("Layered AD-RTR cosine threshold must be finite and in [-1, 1]")

    donor_occupied = occupied.index_select(0, donor_indices)
    donor_state = canonicalize_active_slots(
        state.index_select(0, donor_indices),
        donor_occupied,
        occupied,
        slot_dim=3,
    )
    donor_keys = canonicalize_active_slots(
        keys.index_select(0, donor_indices),
        donor_occupied,
        occupied,
        slot_dim=2,
    )
    layer_roll_occupied = torch.roll(occupied, shifts=1, dims=1)
    layer_roll_state = canonicalize_active_slots(
        torch.roll(state, shifts=1, dims=1),
        layer_roll_occupied,
        occupied,
        slot_dim=3,
    )
    control_names = ("matched_donor_state", "wrong_address", "layer_roll")
    module_active = occupied.any(dim=-1)
    if bool((~module_active.any(dim=-1)).any().item()):
        raise ValueError("Layered AD-RTR row has no occupied modules")

    module_cosines: dict[str, list[torch.Tensor]] = {
        name: [] for name in ("correct", *control_names)
    }
    per_module: dict[str, Any] = {}
    zero_by_module: dict[str, Mapping[str, bool]] = {}
    for module_index, module_name in enumerate(names):
        module_metrics = reconstruction_control_metrics(
            correct_state=state[:, module_index],
            matched_donor_state=donor_state[:, module_index],
            layer_roll_state=layer_roll_state[:, module_index],
            keys=keys[:, module_index],
            wrong_address_keys=donor_keys[:, module_index],
            values=values[:, module_index],
            occupied=occupied[:, module_index],
            weights=maps[module_name],
            decoder=decoders[module_name],
        )
        zero_by_module[module_name] = module_metrics["zero_audit"]

        decoded_controls = {
            "correct": (state[:, module_index], keys[:, module_index]),
            "matched_donor_state": (
                donor_state[:, module_index],
                keys[:, module_index],
            ),
            "wrong_address": (state[:, module_index], donor_keys[:, module_index]),
            "layer_roll": (
                layer_roll_state[:, module_index],
                keys[:, module_index],
            ),
        }
        slot_cosines: dict[str, torch.Tensor] = {}
        for control_name, (control_state, control_keys) in decoded_controls.items():
            slots = address_decoded_slots(
                control_state,
                control_keys,
                values[:, module_index],
                occupied[:, module_index],
                maps[module_name],
            )
            decoded = decoders[module_name].decode(slots.contracted)
            slot_cosines[control_name] = F.cosine_similarity(
                decoded.float(),
                values[:, module_index].float(),
                dim=-1,
                eps=1e-6,
            )
            counts = occupied[:, module_index].sum(dim=-1)
            row_means = (
                slot_cosines[control_name]
                * occupied[:, module_index].to(dtype=slot_cosines[control_name].dtype)
            ).sum(dim=-1) / counts.clamp_min(1).to(dtype=slot_cosines[control_name].dtype)
            module_cosines[control_name].append(row_means)

        active_rows = module_active[:, module_index]
        correct_rows = module_cosines["correct"][-1][active_rows]
        per_module[module_name] = {
            "finite": bool(module_metrics["finite"]),
            "active_rows": int(active_rows.sum().item()),
            "active_slots": int(occupied[:, module_index].sum().item()),
            "correct_mean_cosine": float(correct_rows.mean().item()),
            "correct_at_least_threshold_fraction": float(
                correct_rows.ge(float(cosine_threshold)).float().mean().item()
            ),
            "mean_gaps": {
                name: float(
                    (
                        module_cosines["correct"][-1]
                        - module_cosines[name][-1]
                    )[active_rows]
                    .mean()
                    .item()
                )
                for name in control_names
            },
            "positive_row_fractions": {
                name: float(
                    (
                        module_cosines["correct"][-1]
                        - module_cosines[name][-1]
                    )[active_rows]
                    .gt(0.0)
                    .float()
                    .mean()
                    .item()
                )
                for name in control_names
            },
            "zero_audit": module_metrics["zero_audit"],
        }

    stacked = {
        name: torch.stack(per_control, dim=1)
        for name, per_control in module_cosines.items()
    }
    module_counts = module_active.sum(dim=-1)
    per_row_cosine = {
        name: (
            score * module_active.to(dtype=score.dtype)
        ).sum(dim=-1) / module_counts.to(dtype=score.dtype)
        for name, score in stacked.items()
    }
    per_row_gaps = {
        name: per_row_cosine["correct"] - per_row_cosine[name]
        for name in control_names
    }
    finite = all(
        bool(torch.isfinite(tensor).all().item())
        for tensor in (*per_row_cosine.values(), *per_row_gaps.values())
    ) and all(item["finite"] for item in per_module.values())
    zero_exact = all(
        all(audit.values()) for audit in zero_by_module.values()
    )
    return {
        "finite": finite,
        "rows": rows,
        "modules": modules,
        "cosine_threshold": float(cosine_threshold),
        "correct_mean_cosine": float(per_row_cosine["correct"].mean().item()),
        "correct_at_least_threshold_fraction": float(
            per_row_cosine["correct"]
            .ge(float(cosine_threshold))
            .float()
            .mean()
            .item()
        ),
        "mean_gaps": {
            name: float(gap.mean().item()) for name, gap in per_row_gaps.items()
        },
        "positive_row_fractions": {
            name: float(gap.gt(0.0).float().mean().item())
            for name, gap in per_row_gaps.items()
        },
        "per_row": {
            "correct_cosine": per_row_cosine["correct"].detach().cpu().tolist(),
            "gaps": {
                name: gap.detach().cpu().tolist() for name, gap in per_row_gaps.items()
            },
        },
        "per_module": per_module,
        "zero_audit": {
            "all_modules_exact": zero_exact,
            "per_module": zero_by_module,
        },
    }
