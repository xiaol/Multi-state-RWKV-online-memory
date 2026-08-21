"""Immutable projected-address conditioning for exact-v5 RWKV writes."""

from __future__ import annotations

from dataclasses import dataclass
import math
from types import MethodType
from typing import Any, Mapping

import torch
import torch.nn.functional as F
from torch import nn

from deltamem.core.delta import iter_delta_mem_modules


@dataclass(frozen=True)
class WriteAddressLatch:
    keys: torch.Tensor
    routes: torch.Tensor
    selected_keys: torch.Tensor
    address_seq: torch.Tensor
    address_version: int


class ContinuousWriteConditioner(nn.Module):
    def __init__(
        self,
        address_dim: int,
        feature_dim: int,
        *,
        rank: int,
        seed: int,
        k_gain: float,
        a_gain: float,
        b_gain: float,
        trainable_map: bool,
    ) -> None:
        super().__init__()
        if address_dim < 1 or feature_dim < 1 or rank < 1:
            raise ValueError("Continuous-write dimensions and rank must be positive")
        for name, gain in (("k_gain", k_gain), ("a_gain", a_gain), ("b_gain", b_gain)):
            if not math.isfinite(gain) or gain < 0.0:
                raise ValueError(f"Continuous-write {name} must be finite and nonnegative")

        generator = torch.Generator(device="cpu")
        generator.manual_seed(int(seed))
        down = torch.randn(rank, address_dim, generator=generator) / math.sqrt(
            float(address_dim)
        )
        up = torch.randn(feature_dim, rank, generator=generator) / math.sqrt(float(rank))
        self.down = nn.Parameter(down.float(), requires_grad=trainable_map)
        self.up = nn.Parameter(up.float(), requires_grad=trainable_map)
        self.address_dim = int(address_dim)
        self.feature_dim = int(feature_dim)
        self.rank = int(rank)
        self.k_gain = float(k_gain)
        self.a_gain = float(a_gain)
        self.b_gain = float(b_gain)

    def direction(self, address_seq: torch.Tensor) -> torch.Tensor:
        if address_seq.shape[-1] != self.address_dim:
            raise ValueError(
                "Continuous-write address width differs: "
                f"expected={self.address_dim} actual={address_seq.shape[-1]}"
            )
        address = address_seq.to(device=self.down.device, dtype=torch.float32)
        mapped = F.linear(F.linear(address, self.down.float()), self.up.float())
        square_mean = mapped.square().mean(dim=-1, keepdim=True)
        active = address.square().sum(dim=-1, keepdim=True).gt(0.0) & square_mean.gt(0.0)
        normalized = mapped / square_mean.clamp_min(1e-12).sqrt()
        return torch.where(active, normalized, torch.zeros_like(normalized))

    def forward(
        self,
        k: torch.Tensor,
        v: torch.Tensor,
        a: torch.Tensor,
        b: torch.Tensor,
        address_seq: torch.Tensor,
        token_mask: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        expected_prefix = tuple(k.shape[:-1])
        if tuple(address_seq.shape[:-1]) != expected_prefix:
            raise ValueError(
                "Continuous-write address prefix differs from RWKV features: "
                f"expected={expected_prefix} actual={tuple(address_seq.shape[:-1])}"
            )
        if not (k.shape == v.shape == a.shape == b.shape):
            raise ValueError("Continuous-write RWKV feature shapes differ")
        if k.shape[-1] != self.feature_dim:
            raise ValueError(
                "Continuous-write feature width differs: "
                f"expected={self.feature_dim} actual={k.shape[-1]}"
            )
        if token_mask is not None and tuple(token_mask.shape) != tuple(k.shape[:2]):
            raise ValueError(
                "Continuous-write token mask differs: "
                f"expected={tuple(k.shape[:2])} actual={tuple(token_mask.shape)}"
            )

        active = (
            address_seq.detach()
            .to(device=k.device)
            .square()
            .sum(dim=-1, keepdim=True)
            .gt(0.0)
        )
        if token_mask is not None:
            active = active & token_mask.to(device=active.device, dtype=torch.bool).unsqueeze(-1)
        if not bool(active.any().item()):
            return k, v, a, b

        direction = self.direction(address_seq).to(device=k.device)

        k_float = k.float()
        k_rms = k_float.square().mean(dim=-1, keepdim=True).clamp_min(1e-12).sqrt()
        candidate_k = k_float + self.k_gain * k_rms * direction
        candidate_a = a.float() * (1.0 + self.a_gain * torch.tanh(direction))
        candidate_b = b.float() * (1.0 + self.b_gain * torch.tanh(direction))

        conditioned_k = torch.where(active, candidate_k, k_float).to(dtype=k.dtype)
        conditioned_a = torch.where(active, candidate_a, a.float()).to(dtype=a.dtype)
        conditioned_b = torch.where(active, candidate_b, b.float()).to(dtype=b.dtype)
        return conditioned_k, v, conditioned_a, conditioned_b


def _assert_latch_intact(latch: WriteAddressLatch) -> None:
    if latch.address_seq._version != latch.address_version:
        raise RuntimeError("Continuous-write immutable address was mutated in place")


def _materialize_latch(
    module: Any,
    hidden_states: torch.Tensor,
    token_mask: torch.Tensor | None,
) -> WriteAddressLatch:
    batch_size, sequence_length, _ = hidden_states.shape
    keys = module.projected_kv_keys
    routes = module.last_write_routes
    if routes is None:
        routes = hidden_states.new_zeros(batch_size, 1, module.rwkv_ms_num_states)
    if keys is None:
        if bool(routes.detach().ne(0).any().item()):
            raise RuntimeError("Continuous-write routes exist without projected keys")
        keys = hidden_states.new_zeros(
            batch_size,
            module.rwkv_ms_num_states,
            module.projected_kv_key_dim,
        )

    expected_routes = (batch_size, 1, module.rwkv_ms_num_states)
    expected_keys = (
        batch_size,
        module.rwkv_ms_num_states,
        module.projected_kv_key_dim,
    )
    if tuple(routes.shape) != expected_routes:
        raise ValueError(
            "Continuous-write projected routes differ: "
            f"expected={expected_routes} actual={tuple(routes.shape)}"
        )
    if tuple(keys.shape) != expected_keys:
        raise ValueError(
            "Continuous-write projected keys differ: "
            f"expected={expected_keys} actual={tuple(keys.shape)}"
        )

    routes_snapshot = routes.detach().float().clone()
    keys_snapshot = keys.detach().float().clone()
    selected_keys = torch.einsum(
        "bps,bsd->bpd",
        routes_snapshot,
        keys_snapshot,
    ).detach().clone()
    address_seq = selected_keys.expand(-1, sequence_length, -1).clone()
    if token_mask is not None:
        expected_mask = (batch_size, sequence_length)
        if tuple(token_mask.shape) != expected_mask:
            raise ValueError(
                "Continuous-write token mask differs from projected write: "
                f"expected={expected_mask} actual={tuple(token_mask.shape)}"
            )
        address_seq = address_seq * token_mask.to(
            device=address_seq.device,
            dtype=address_seq.dtype,
        ).unsqueeze(-1)
    address_seq = address_seq.detach().clone()
    return WriteAddressLatch(
        keys=keys_snapshot,
        routes=routes_snapshot,
        selected_keys=selected_keys,
        address_seq=address_seq,
        address_version=address_seq._version,
    )


def _projected_slot_write(
    module: Any,
    hidden_states: torch.Tensor,
    token_mask: torch.Tensor | None,
) -> None:
    module.rwkv_continuous_write_latch = None
    module.rwkv_continuous_write_audit = None
    module.rwkv_continuous_write_original_projected_slot_write(hidden_states, token_mask)
    module.rwkv_continuous_write_latch = _materialize_latch(
        module,
        hidden_states,
        token_mask,
    )


def _reset_state(module: Any) -> None:
    module.rwkv_continuous_write_original_reset_state()
    module.rwkv_continuous_write_latch = None
    module.rwkv_continuous_write_audit = None


def _write_address_sequence(
    module: Any,
    hidden_states: torch.Tensor,
    token_mask: torch.Tensor | None,
) -> torch.Tensor:
    latch = module.rwkv_continuous_write_latch
    if latch is None:
        raise RuntimeError("Continuous-write address requested before projected slot write")
    _assert_latch_intact(latch)
    expected_shape = (
        hidden_states.shape[0],
        hidden_states.shape[1],
        module.projected_kv_key_dim,
    )
    if tuple(latch.address_seq.shape) != expected_shape:
        raise RuntimeError(
            "Continuous-write latched address shape differs: "
            f"expected={expected_shape} actual={tuple(latch.address_seq.shape)}"
        )
    return latch.address_seq


def _conditioned_features(
    module: Any,
    k: torch.Tensor,
    v: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    address_seq: torch.Tensor,
    token_mask: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    latch = module.rwkv_continuous_write_latch
    if latch is None:
        raise RuntimeError("Continuous-write conditioner has no latched address")
    _assert_latch_intact(latch)
    if address_seq is not latch.address_seq:
        raise RuntimeError("Continuous-write conditioner did not receive the latched tensor")

    if not module.rwkv_continuous_write_enabled:
        outputs = (k, v, a, b)
    else:
        outputs = module.rwkv_continuous_write_conditioner(
            k,
            v,
            a,
            b,
            address_seq,
            token_mask,
        )
    if outputs[1] is not v:
        raise RuntimeError("Continuous-write conditioner changed the RWKV value object")
    if module.rwkv_continuous_write_capture_enabled:
        module.rwkv_continuous_write_audit = {
            "conditioner_address": address_seq,
            "conditioner_address_value": address_seq.detach().clone(),
            "conditioner_address_object_id": id(address_seq),
            "latched_address_object_id": id(latch.address_seq),
            "value_object_id": id(v),
            "returned_value_object_id": id(outputs[1]),
        }
    _assert_latch_intact(latch)
    return outputs


def install(
    model: nn.Module,
    *,
    rank: int = 4,
    seed: int = 149,
    k_gain: float = 0.25,
    a_gain: float = 0.25,
    b_gain: float = 0.25,
    trainable_map: bool = False,
) -> Mapping[str, Any]:
    modules = tuple(iter_delta_mem_modules(model))
    if not modules:
        raise ValueError("Continuous-write integration requires Delta-Mem modules")
    installed: list[str] = []
    parameter_elements = 0
    for index, (name, module) in enumerate(modules):
        if hasattr(module, "rwkv_continuous_write_conditioner"):
            raise ValueError(f"Continuous-write integration is already installed on {name}")
        if module.rwkv_ms_hybrid_mode != "address_keyed_moe_deepembed_ffn":
            raise ValueError("Continuous-write integration requires exact-v5 hybrid mode")
        conditioner = ContinuousWriteConditioner(
            int(module.projected_kv_key_dim),
            int(module.state_read_dim),
            rank=int(rank),
            seed=int(seed) + index,
            k_gain=k_gain,
            a_gain=a_gain,
            b_gain=b_gain,
            trainable_map=trainable_map,
        ).to(device=module.memory_v_proj.device)
        module.add_module("rwkv_continuous_write_conditioner", conditioner)
        module.rwkv_continuous_write_original_projected_slot_write = (
            module._write_projected_kv_slots
        )
        module.rwkv_continuous_write_original_address_sequence = (
            module._projected_rwkv_write_address_sequence
        )
        module.rwkv_continuous_write_original_conditioner = (
            module._rwkv_ms_address_conditioned_write_features
        )
        module.rwkv_continuous_write_original_reset_state = module.reset_state
        module._write_projected_kv_slots = MethodType(_projected_slot_write, module)
        module._projected_rwkv_write_address_sequence = MethodType(
            _write_address_sequence,
            module,
        )
        module._rwkv_ms_address_conditioned_write_features = MethodType(
            _conditioned_features,
            module,
        )
        module.reset_state = MethodType(_reset_state, module)
        module.rwkv_continuous_write_enabled = True
        module.rwkv_continuous_write_capture_enabled = False
        module.rwkv_continuous_write_latch = None
        module.rwkv_continuous_write_audit = None
        installed.append(name)
        parameter_elements += sum(parameter.numel() for parameter in conditioner.parameters())
    return {
        "modules": len(installed),
        "module_names": tuple(installed),
        "rank": int(rank),
        "address_dim": int(modules[0][1].projected_kv_key_dim),
        "feature_dim": int(modules[0][1].state_read_dim),
        "parameter_tensors": 2 * len(installed),
        "parameter_elements": parameter_elements,
        "map_trainable": bool(trainable_map),
        "conditioned_features": ("k", "a", "b"),
        "value_identity": "same_object_and_bytes",
        "address_lifecycle": "single_post_projected_write_detached_full_address_latch",
        "live_key_or_route_recomputation": False,
    }


def set_enabled(model: nn.Module, enabled: bool) -> None:
    for _, module in iter_delta_mem_modules(model):
        module.rwkv_continuous_write_enabled = bool(enabled)


def set_capture(model: nn.Module, enabled: bool) -> None:
    for _, module in iter_delta_mem_modules(model):
        module.rwkv_continuous_write_capture_enabled = bool(enabled)
        module.rwkv_continuous_write_audit = None


def clear_transient(model: nn.Module) -> None:
    for _, module in iter_delta_mem_modules(model):
        module.rwkv_continuous_write_latch = None
        module.rwkv_continuous_write_audit = None
