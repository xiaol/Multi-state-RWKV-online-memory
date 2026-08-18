"""Capture same-space projected values and addressed RWKV reads for identity loss."""

from __future__ import annotations

from typing import Any, Sequence

import torch
import torch.nn.functional as F

from deltamem.core.delta import iter_delta_mem_modules
from experiments.rethinking_rwkv_ms_gemma import rwkv_query_state_identity as query_identity


CapturedProjectedValueRead = query_identity.CapturedQueryStateRead


def install(model: torch.nn.Module) -> dict[str, Any]:
    audit = query_identity.install(model)
    return {
        **audit,
        "query_target": "detached_projected_slot_value",
        "query_address_gradient": "detached_frozen_projected_route_and_value",
    }


def clear(model: torch.nn.Module, *, clear_fixed: bool = True) -> None:
    query_identity.clear(model, clear_fixed=clear_fixed)


def capture_write_values(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    """Capture one detached row-level projected value per module after the write."""
    values_by_module: dict[str, torch.Tensor] = {}
    for module_name, module in iter_delta_mem_modules(model):
        routes = module.last_write_routes
        values = module.projected_kv_values
        if routes is None or values is None:
            raise RuntimeError(
                f"Projected-value identity write carrier is missing for {module_name}"
            )
        if routes.ndim != 3 or values.ndim != 3:
            raise ValueError(
                "Projected-value identity routes and values must be batched sequences"
            )
        if routes.shape[0] != values.shape[0] or routes.shape[-1] != values.shape[1]:
            raise ValueError("Projected-value identity route/value shapes differ")
        if routes.shape[1] != 1:
            route_mass = routes.float().abs().sum(dim=-1, keepdim=True).sum(
                dim=1, keepdim=True
            )
            if bool(route_mass.eq(0).any().item()):
                raise RuntimeError(
                    f"Projected-value identity write route is empty for {module_name}"
                )
            routes = routes.float().sum(dim=1, keepdim=True) / route_mass.clamp_min(1e-6)
        selected = torch.einsum("bts,bsd->btd", routes.float(), values.float()).detach()
        expected = (values.shape[0], 1, values.shape[-1])
        if tuple(selected.shape) != expected:
            raise RuntimeError(
                "Projected-value identity target shape differs: "
                f"module={module_name} expected={expected} actual={tuple(selected.shape)}"
            )
        if not bool(torch.isfinite(selected).all().item()):
            raise RuntimeError(f"Projected-value identity target is non-finite for {module_name}")
        if bool((selected.float().norm(dim=-1) <= 1e-6).any().item()):
            raise RuntimeError(f"Projected-value identity target is zero for {module_name}")
        values_by_module[module_name] = selected
    if not values_by_module:
        raise RuntimeError("Projected-value identity target capture is empty")
    return values_by_module


def set_fixed_target_values(
    model: torch.nn.Module,
    values_by_module: dict[str, torch.Tensor],
) -> None:
    query_identity.set_fixed_query_addresses(model, values_by_module)


def capture(model: torch.nn.Module) -> tuple[CapturedProjectedValueRead, ...]:
    return query_identity.capture(model)


def score_tensor(
    captured: Sequence[CapturedProjectedValueRead],
    labels: torch.Tensor,
) -> torch.Tensor:
    """Return [layer, answer-token] cosine scores before any layer reduction."""
    if not captured:
        raise ValueError("Projected-value identity capture is empty")
    valid = labels.ne(-100)
    if valid.ndim != 2 or not bool(valid.any().item()):
        raise ValueError("Projected-value identity requires answer target positions")
    scores: list[torch.Tensor] = []
    for read in captured:
        if tuple(read.query_address.shape[:2]) != tuple(valid.shape):
            raise ValueError("Projected-value identity labels do not match reads")
        target = F.normalize(read.query_address.float(), dim=-1, eps=1e-6)
        state = F.normalize(read.recurrent_read.float(), dim=-1, eps=1e-6)
        scores.append((target * state).sum(dim=-1).masked_select(valid))
    result = torch.stack(scores)
    if not bool(torch.isfinite(result).all().item()):
        raise RuntimeError("Projected-value identity scores are non-finite")
    return result


def active_hinge(
    positive: torch.Tensor,
    donor: torch.Tensor,
    *,
    margin: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    if positive.shape != donor.shape or positive.ndim != 2:
        raise ValueError("Projected-value identity score shapes differ")
    detached_margin = positive.new_tensor(margin) - positive.detach() + donor.detach()
    active = detached_margin.gt(0.0)
    hinge = F.relu(detached_margin)
    return active, hinge
