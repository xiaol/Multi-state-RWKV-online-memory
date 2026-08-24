"""Experiment-scoped explicit address-key/RWKV-state-value virtual KV."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any

import torch
from torch import nn
import torch.nn.functional as F


@dataclass(frozen=True)
class VirtualKVShape:
    key_dim: int
    state_heads: int
    rank: int
    slots: int
    kv_heads: int
    head_dim: int
    probe_rank: int = 8
    value_hidden: int = 128
    seed: int = 211
    key_radius: float = 1.0
    value_radius: float = 1.0
    co_rotate_keys: bool = False

    def __post_init__(self) -> None:
        for name in (
            "key_dim",
            "state_heads",
            "rank",
            "slots",
            "kv_heads",
            "head_dim",
            "probe_rank",
            "value_hidden",
        ):
            if int(getattr(self, name)) < 1:
                raise ValueError(f"{name} must be >= 1")
        if self.probe_rank > self.rank:
            raise ValueError("probe_rank must be <= rank")
        for name in ("key_radius", "value_radius"):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and > 0")


class ExplicitRWKVVirtualKV(nn.Module):
    """Builds ephemeral K/V positions without mutating cache or state."""

    def __init__(self, shape: VirtualKVShape) -> None:
        super().__init__()
        self.shape_spec = shape
        generator = torch.Generator(device="cpu")
        generator.manual_seed(int(shape.seed))
        right_probe = torch.randn(
            shape.rank,
            shape.probe_rank,
            generator=generator,
            dtype=torch.float32,
        )
        right_probe = torch.linalg.qr(right_probe, mode="reduced").Q
        self.register_buffer("right_probe", right_probe, persistent=True)
        self.key_proj = nn.Parameter(
            torch.empty(shape.kv_heads * shape.head_dim, shape.key_dim)
        )
        self.value_in = nn.Parameter(
            torch.empty(
                shape.value_hidden,
                shape.state_heads * shape.rank * shape.probe_rank,
            )
        )
        self.value_out = nn.Parameter(
            torch.empty(shape.kv_heads * shape.head_dim, shape.value_hidden)
        )
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(int(shape.seed) + 7919)
            self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.key_proj, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.value_in, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.value_out, a=math.sqrt(5))

    @staticmethod
    def _rms_sphere(value: torch.Tensor, radius: float) -> torch.Tensor:
        return value / value.square().mean(dim=-1, keepdim=True).add(1e-12).sqrt() * radius

    @staticmethod
    def _rotate_half(value: torch.Tensor) -> torch.Tensor:
        first, second = value.chunk(2, dim=-1)
        return torch.cat((-second, first), dim=-1)

    def _co_rotate_keys(
        self,
        keys: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor] | None,
    ) -> torch.Tensor:
        if not self.shape_spec.co_rotate_keys:
            return keys
        if position_embeddings is None or len(position_embeddings) != 2:
            raise ValueError("co-rotated virtual keys require query position embeddings")
        cos, sin = position_embeddings
        expected = (keys.size(0), 1, self.shape_spec.head_dim)
        if tuple(cos.shape) != expected or tuple(sin.shape) != expected:
            raise ValueError(
                "co-rotated virtual key position geometry differs: "
                f"expected={expected} cos={tuple(cos.shape)} sin={tuple(sin.shape)}"
            )
        if not bool(torch.isfinite(cos.float()).all().item()) or not bool(
            torch.isfinite(sin.float()).all().item()
        ):
            raise ValueError("co-rotated virtual key position embeddings must be finite")
        cos = cos[:, None].to(device=keys.device, dtype=keys.dtype)
        sin = sin[:, None].to(device=keys.device, dtype=keys.dtype)
        return keys * cos + self._rotate_half(keys) * sin

    def _validate_inputs(
        self,
        state: torch.Tensor,
        address_keys: torch.Tensor,
        occupied: torch.Tensor,
        query_states: torch.Tensor,
        real_keys: torch.Tensor,
        real_values: torch.Tensor,
    ) -> None:
        shape = self.shape_spec
        expected_state = (state.size(0), shape.state_heads, shape.slots, shape.rank, shape.rank)
        if tuple(state.shape) != expected_state:
            raise ValueError(f"state shape differs: expected={expected_state} actual={tuple(state.shape)}")
        expected_keys = (state.size(0), shape.slots, shape.key_dim)
        if tuple(address_keys.shape) != expected_keys:
            raise ValueError(f"address key shape differs: expected={expected_keys} actual={tuple(address_keys.shape)}")
        if tuple(occupied.shape) != expected_keys[:2] or occupied.dtype != torch.bool:
            raise ValueError("occupied must be boolean [batch, slots]")
        if query_states.ndim != 4 or query_states.size(2) != 1:
            raise ValueError("virtual KV currently requires exactly one query token")
        if real_keys.ndim != 4 or real_values.ndim != 4:
            raise ValueError("real K/V must be rank 4")
        expected_real = (state.size(0), shape.kv_heads, real_keys.size(2), shape.head_dim)
        if tuple(real_keys.shape) != expected_real or tuple(real_values.shape) != expected_real:
            raise ValueError("real K/V geometry differs from virtual KV geometry")

    def _extended_mask(
        self,
        attention_mask: torch.Tensor | None,
        *,
        batch_size: int,
        query_length: int,
        real_length: int,
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        virtual_length = self.shape_spec.slots
        if attention_mask is None:
            mask_dtype = torch.float32
            result = torch.full(
                (batch_size, 1, query_length, real_length + virtual_length),
                torch.finfo(mask_dtype).min,
                device=device,
                dtype=mask_dtype,
            )
            result[:, :, :, :real_length] = 0.0
        else:
            if attention_mask.ndim != 4 or attention_mask.shape[0] != batch_size:
                raise ValueError("attention_mask must be [batch, heads, query, key]")
            if attention_mask.shape[-2] != query_length or attention_mask.shape[-1] != real_length:
                raise ValueError("attention_mask dimensions differ from real K/V")
            if attention_mask.dtype == torch.bool:
                result = attention_mask.to(device=device).clone()
            else:
                if not attention_mask.dtype.is_floating_point:
                    raise ValueError("attention_mask must be bool or floating point")
                result = attention_mask.to(device=device).clone()
            mask_dtype = result.dtype
            result = torch.cat(
                (
                    result,
                    torch.full(
                        (batch_size, result.shape[1], query_length, virtual_length),
                        False if mask_dtype == torch.bool else torch.finfo(mask_dtype).min,
                        device=device,
                        dtype=mask_dtype,
                    ),
                ),
                dim=-1,
            )
        active = torch.zeros(
            (batch_size, virtual_length), device=device, dtype=torch.bool
        )
        return result, active

    def forward(
        self,
        *,
        state: torch.Tensor,
        address_keys: torch.Tensor,
        occupied: torch.Tensor,
        query_states: torch.Tensor,
        real_keys: torch.Tensor,
        real_values: torch.Tensor,
        attention_mask: torch.Tensor | None,
        position_embeddings: tuple[torch.Tensor, torch.Tensor] | None = None,
        module: Any | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None:
        del module
        self._validate_inputs(
            state,
            address_keys,
            occupied,
            query_states,
            real_keys,
            real_values,
        )
        if not bool(torch.isfinite(state.float()).all().item()) or not bool(
            torch.isfinite(address_keys.float()).all().item()
        ):
            raise ValueError("virtual KV inputs must be finite")
        state_float = state.float()
        active = occupied & state_float.square().sum(dim=(-1, -2)).sum(dim=1).gt(0.0)
        if not bool(active.any().item()):
            return None
        if bool(address_keys.square().sum(dim=-1).eq(0.0).logical_and(active).any().item()):
            raise ValueError("active virtual KV address is exactly zero")
        probed = torch.einsum("bhsij,jp->bhsip", state_float, self.right_probe.float())
        probed = probed.permute(0, 2, 1, 3, 4).reshape(
            state.size(0),
            self.shape_spec.slots,
            self.shape_spec.state_heads * self.shape_spec.rank * self.shape_spec.probe_rank,
        )
        probe_rms = probed.square().mean(dim=-1, keepdim=True).add(1e-6).sqrt()
        value_hidden = F.silu(F.linear(probed / probe_rms, self.value_in.float()))
        values = F.linear(value_hidden, self.value_out.float()).reshape(
            state.size(0),
            self.shape_spec.slots,
            self.shape_spec.kv_heads,
            self.shape_spec.head_dim,
        )
        values = self._rms_sphere(values, self.shape_spec.value_radius)
        values = torch.where(active[:, :, None, None], values, torch.zeros_like(values))
        if bool(values[active].square().sum(dim=(-1, -2)).eq(0.0).any().item()):
            raise RuntimeError("active virtual KV value collapsed to zero")
        values = values.permute(0, 2, 1, 3).contiguous()

        addresses = address_keys.float()
        addresses = addresses / addresses.square().mean(dim=-1, keepdim=True).add(1e-6).sqrt()
        keys = F.linear(addresses, self.key_proj.float()).reshape(
            state.size(0),
            self.shape_spec.slots,
            self.shape_spec.kv_heads,
            self.shape_spec.head_dim,
        )
        keys = self._rms_sphere(keys, self.shape_spec.key_radius)
        if bool(keys[active].square().sum(dim=(-1, -2)).eq(0.0).any().item()):
            raise RuntimeError("active virtual KV key collapsed to zero")
        keys = keys.permute(0, 2, 1, 3).contiguous()
        keys = self._co_rotate_keys(keys, position_embeddings)
        mask, _ = self._extended_mask(
            attention_mask,
            batch_size=state.size(0),
            query_length=query_states.size(2),
            real_length=real_keys.size(2),
            device=query_states.device,
        )
        if mask.dtype == torch.bool:
            virtual_allow = active[:, None, None, :]
        else:
            virtual_allow = torch.where(
                active[:, None, None, :],
                torch.zeros((), device=mask.device, dtype=mask.dtype),
                torch.full(
                    (), torch.finfo(mask.dtype).min, device=mask.device, dtype=mask.dtype
                ),
            )
        mask[:, :, :, real_keys.size(2):] = virtual_allow
        return keys.to(real_keys), values.to(real_values), mask
