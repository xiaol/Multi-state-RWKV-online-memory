from __future__ import annotations

from dataclasses import dataclass
import math
from types import MethodType
from typing import Any, Sequence

import torch
import torch.nn.functional as F

from deltamem.core.delta import iter_delta_mem_modules


@dataclass(frozen=True)
class CapturedQueryStateRead:
    module_name: str
    query_address: torch.Tensor
    recurrent_read: torch.Tensor


def _fold_query_address(
    routes: torch.Tensor,
    keys: torch.Tensor,
    *,
    state_dim: int,
) -> torch.Tensor:
    if routes.ndim != 3 or keys.ndim != 3:
        raise ValueError("Query-state identity capture requires batched projected routes")
    if routes.shape[0] != keys.shape[0] or routes.shape[-1] != keys.shape[1]:
        raise ValueError("Query-state identity projected route/key shapes differ")
    key_dim = int(keys.shape[-1])
    if key_dim % state_dim:
        raise ValueError("Query-state identity key dimension must fold into RWKV state")
    selected = torch.einsum("bts,bsk->btk", routes.float(), keys.float())
    fold = key_dim // state_dim
    return selected.reshape(
        *selected.shape[:-1],
        fold,
        state_dim,
    ).sum(dim=-2) / math.sqrt(float(fold))


def _projected_reads_and_capture(
    module: Any,
    hidden_states: torch.Tensor,
) -> torch.Tensor:
    projected_reads = module.rwkv_query_state_identity_original_projected_read(
        hidden_states
    )
    fixed_address = module.rwkv_query_state_identity_fixed_address
    if fixed_address is not None:
        expected = (
            projected_reads.shape[0],
            1,
            projected_reads.shape[-1],
        )
        if tuple(fixed_address.shape) != expected:
            raise ValueError(
                "Fixed query-state identity address shape differs: "
                f"expected={expected} actual={tuple(fixed_address.shape)}"
            )
        module.rwkv_query_state_identity_query_address = fixed_address.expand(
            -1,
            projected_reads.shape[1],
            -1,
        )
        return projected_reads
    routes = module.last_read_routes
    keys = module.projected_kv_keys
    if routes is None or keys is None:
        module.rwkv_query_state_identity_query_address = None
        return projected_reads
    module.rwkv_query_state_identity_query_address = _fold_query_address(
        routes,
        keys,
        state_dim=int(projected_reads.shape[-1]),
    ).detach()
    return projected_reads


def _fuse_and_capture(
    module: Any,
    projected_reads: torch.Tensor,
    recurrent_reads: torch.Tensor,
    route_agreement: torch.Tensor | None = None,
    query_state_gate: torch.Tensor | None = None,
    global_recurrent_reads: torch.Tensor | None = None,
    hidden_states: torch.Tensor | None = None,
) -> torch.Tensor:
    fused = module.rwkv_query_state_identity_original_fuse(
        projected_reads,
        recurrent_reads,
        route_agreement,
        query_state_gate,
        global_recurrent_reads,
        hidden_states,
    )
    if module.rwkv_query_state_identity_query_address is None:
        module.rwkv_query_state_identity_recurrent_read = None
        return fused
    module.rwkv_query_state_identity_recurrent_read = recurrent_reads
    return fused


def install(model: torch.nn.Module) -> dict[str, Any]:
    modules = tuple(iter_delta_mem_modules(model))
    if not modules:
        raise ValueError("Query-state identity capture requires Delta-Mem modules")
    installed: list[str] = []
    for module_name, module in modules:
        if hasattr(module, "rwkv_query_state_identity_original_fuse"):
            raise ValueError(f"Query-state identity capture already installed on {module_name}")
        module.rwkv_query_state_identity_original_projected_read = (
            module._projected_kv_slot_token_reads
        )
        module.rwkv_query_state_identity_original_fuse = (
            module._fuse_projected_rwkv_reads
        )
        module.rwkv_query_state_identity_query_address = None
        module.rwkv_query_state_identity_recurrent_read = None
        module.rwkv_query_state_identity_fixed_address = None
        module._projected_kv_slot_token_reads = MethodType(
            _projected_reads_and_capture,
            module,
        )
        module._fuse_projected_rwkv_reads = MethodType(_fuse_and_capture, module)
        installed.append(module_name)
    return {
        "modules": len(installed),
        "module_names": tuple(installed),
        "forward_output_changed": False,
        "trainable_parameters_added": 0,
        "query_address_gradient": "detached_frozen_projected_route_and_key",
        "state_gradient": "live_addressed_rwkv_read",
    }


def clear(model: torch.nn.Module, *, clear_fixed: bool = True) -> None:
    for _, module in iter_delta_mem_modules(model):
        if hasattr(module, "rwkv_query_state_identity_query_address"):
            module.rwkv_query_state_identity_query_address = None
            module.rwkv_query_state_identity_recurrent_read = None
            if clear_fixed:
                module.rwkv_query_state_identity_fixed_address = None


def capture_write_addresses(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    addresses: dict[str, torch.Tensor] = {}
    for module_name, module in iter_delta_mem_modules(model):
        routes = module.last_write_routes
        keys = module.projected_kv_keys
        if routes is None or keys is None:
            raise RuntimeError(
                f"Query-state identity write address is missing for {module_name}"
            )
        if routes.ndim != 3 or keys.ndim != 3:
            raise ValueError(
                "Query-state identity write routes and keys must be batched sequences"
            )
        if routes.shape[0] != keys.shape[0] or routes.shape[-1] != keys.shape[1]:
            raise ValueError(
                "Query-state identity write route/key shapes differ"
            )
        if routes.shape[1] != 1:
            valid_route_mass = routes.float().abs().sum(dim=-1, keepdim=True)
            route_mass = valid_route_mass.sum(dim=1, keepdim=True)
            if bool(route_mass.eq(0).any().item()):
                raise RuntimeError(
                    f"Query-state identity write route is empty for {module_name}"
                )
            routes = routes.float().sum(dim=1, keepdim=True) / route_mass.clamp_min(1e-6)
        address = _fold_query_address(
            routes,
            keys,
            state_dim=int(module.state_read_dim),
        ).detach()
        expected = (keys.shape[0], 1, module.state_read_dim)
        if tuple(address.shape) != expected:
            raise RuntimeError(
                "Query-state identity write address shape differs: "
                f"module={module_name} expected={expected} actual={tuple(address.shape)}"
            )
        addresses[module_name] = address
    if not addresses:
        raise RuntimeError("Query-state identity write address capture is empty")
    return addresses


def set_fixed_query_addresses(
    model: torch.nn.Module,
    addresses: dict[str, torch.Tensor],
) -> None:
    observed: set[str] = set()
    for module_name, module in iter_delta_mem_modules(model):
        try:
            address = addresses[module_name]
        except KeyError as error:
            raise ValueError(
                f"Fixed query-state identity address is missing for {module_name}"
            ) from error
        module.rwkv_query_state_identity_fixed_address = address
        observed.add(module_name)
    if observed != set(addresses):
        raise ValueError("Fixed query-state identity address modules differ")


def capture(model: torch.nn.Module) -> tuple[CapturedQueryStateRead, ...]:
    captured: list[CapturedQueryStateRead] = []
    for module_name, module in iter_delta_mem_modules(model):
        query_address = getattr(
            module,
            "rwkv_query_state_identity_query_address",
            None,
        )
        recurrent_read = getattr(
            module,
            "rwkv_query_state_identity_recurrent_read",
            None,
        )
        if query_address is None or recurrent_read is None:
            raise RuntimeError(
                f"Query-state identity read was not captured for {module_name}"
            )
        if query_address.shape != recurrent_read.shape:
            raise RuntimeError(
                "Query-state identity capture shapes differ: "
                f"module={module_name} query={tuple(query_address.shape)} "
                f"state={tuple(recurrent_read.shape)}"
            )
        captured.append(
            CapturedQueryStateRead(
                module_name=module_name,
                query_address=query_address,
                recurrent_read=recurrent_read,
            )
        )
    if not captured:
        raise RuntimeError("Query-state identity capture is empty")
    return tuple(captured)


def mean_scores(
    positive: Sequence[CapturedQueryStateRead],
    donor: Sequence[CapturedQueryStateRead],
    labels: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    if len(positive) != len(donor) or not positive:
        raise ValueError("Query-state identity branches must contain the same layers")
    valid = labels.ne(-100)
    if valid.ndim != 2 or not bool(valid.any().item()):
        raise ValueError("Query-state identity requires answer target positions")
    positive_scores: list[torch.Tensor] = []
    donor_scores: list[torch.Tensor] = []
    for correct_read, donor_read in zip(positive, donor):
        if correct_read.module_name != donor_read.module_name:
            raise ValueError("Query-state identity layer ordering differs")
        if tuple(correct_read.query_address.shape[:2]) != tuple(valid.shape):
            raise ValueError("Query-state identity labels do not match read tokens")
        query = F.normalize(
            correct_read.query_address.float(),
            dim=-1,
            eps=1e-6,
        )
        correct_state = F.normalize(
            correct_read.recurrent_read.float(),
            dim=-1,
            eps=1e-6,
        )
        donor_state = F.normalize(
            donor_read.recurrent_read.float(),
            dim=-1,
            eps=1e-6,
        )
        positive_scores.append(
            (query * correct_state).sum(dim=-1).masked_select(valid).mean()
        )
        donor_scores.append(
            (query * donor_state).sum(dim=-1).masked_select(valid).mean()
        )
    return torch.stack(positive_scores).mean(), torch.stack(donor_scores).mean()


def mean_score(
    captured: Sequence[CapturedQueryStateRead],
    labels: torch.Tensor,
) -> torch.Tensor:
    if not captured:
        raise ValueError("Query-state identity branch is empty")
    valid = labels.ne(-100)
    if valid.ndim != 2 or not bool(valid.any().item()):
        raise ValueError("Query-state identity requires answer target positions")
    scores: list[torch.Tensor] = []
    for read in captured:
        if tuple(read.query_address.shape[:2]) != tuple(valid.shape):
            raise ValueError("Query-state identity labels do not match read tokens")
        query = F.normalize(read.query_address.float(), dim=-1, eps=1e-6)
        state = F.normalize(read.recurrent_read.float(), dim=-1, eps=1e-6)
        scores.append((query * state).sum(dim=-1).masked_select(valid).mean())
    score = torch.stack(scores).mean()
    if not bool(torch.isfinite(score).item()):
        raise RuntimeError("Query-state identity score is non-finite")
    return score


def donor_hinge(
    positive: Sequence[CapturedQueryStateRead],
    donor: Sequence[CapturedQueryStateRead],
    labels: torch.Tensor,
    *,
    margin: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if not math.isfinite(margin) or margin <= 0.0:
        raise ValueError("Query-state identity margin must be finite and positive")
    positive_score, donor_score = mean_scores(positive, donor, labels)
    loss = F.relu(positive_score.new_tensor(margin) - positive_score + donor_score)
    if not bool(torch.isfinite(torch.stack((positive_score, donor_score, loss))).all().item()):
        raise RuntimeError("Query-state identity scores are non-finite")
    return positive_score, donor_score, loss
