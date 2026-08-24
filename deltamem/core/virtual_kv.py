"""Experiment-scoped explicit address-key/RWKV-state-value virtual KV."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Mapping, Sequence

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
        attention_bias: torch.Tensor | None = None,
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
        active = (
            occupied
            & state_float.square().sum(dim=(-1, -2)).sum(dim=1).gt(0.0)
            & address_keys.float().square().sum(dim=-1).gt(0.0)
        )
        if not bool(active.any().item()):
            return None
        if attention_bias is not None:
            expected_bias = (state.size(0), self.shape_spec.slots)
            if tuple(attention_bias.shape) != expected_bias:
                raise ValueError(
                    "virtual attention bias shape differs: "
                    f"expected={expected_bias} actual={tuple(attention_bias.shape)}"
                )
            if not attention_bias.dtype.is_floating_point or not bool(
                torch.isfinite(attention_bias.float()).all().item()
            ):
                raise ValueError("virtual attention bias must be finite floating point")
            if self.shape_spec.co_rotate_keys:
                raise ValueError("bias-routed virtual KV does not use co-rotated keys")
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

        if attention_bias is None:
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
        else:
            keys = state.new_zeros(
                state.size(0),
                self.shape_spec.kv_heads,
                self.shape_spec.slots,
                self.shape_spec.head_dim,
            )
        mask, _ = self._extended_mask(
            attention_mask,
            batch_size=state.size(0),
            query_length=query_states.size(2),
            real_length=real_keys.size(2),
            device=query_states.device,
        )
        if attention_bias is not None and mask.dtype == torch.bool:
            raise ValueError("virtual attention bias requires an additive floating mask")
        if mask.dtype == torch.bool:
            virtual_allow = active[:, None, None, :]
        elif attention_bias is not None:
            virtual_allow = torch.where(
                active[:, None, None, :],
                attention_bias[:, None, None, :].to(
                    device=mask.device,
                    dtype=mask.dtype,
                ),
                torch.full(
                    (), torch.finfo(mask.dtype).min, device=mask.device, dtype=mask.dtype
                ),
            )
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


class FrozenCompatibilityMap(nn.Module):
    """Frozen reduced-rank map from address space to RWKV receptance space."""

    def __init__(self, down: torch.Tensor, up: torch.Tensor) -> None:
        super().__init__()
        if down.ndim != 2 or up.ndim != 2 or down.size(0) != up.size(1):
            raise ValueError("compatibility map must be down[rank,address] and up[state,rank]")
        if not bool(torch.isfinite(down.float()).all().item()) or not bool(
            torch.isfinite(up.float()).all().item()
        ):
            raise ValueError("compatibility map weights must be finite")
        self.register_buffer("down", down.detach().float().clone(), persistent=True)
        self.register_buffer("up", up.detach().float().clone(), persistent=True)

    @property
    def address_dim(self) -> int:
        return int(self.down.size(1))

    @property
    def state_dim(self) -> int:
        return int(self.up.size(0))

    @staticmethod
    def _rms_normalize(value: torch.Tensor) -> torch.Tensor:
        value = value.float()
        square_mean = value.square().mean(dim=-1, keepdim=True)
        normalized = value / square_mean.clamp_min(1e-12).sqrt()
        return torch.where(square_mean.gt(0.0), normalized, torch.zeros_like(normalized))

    def forward(self, addresses: torch.Tensor) -> torch.Tensor:
        if addresses.ndim != 3 or addresses.size(-1) != self.address_dim:
            raise ValueError("compatibility addresses must be [batch, slots, address_dim]")
        normalized = self._rms_normalize(addresses)
        latent = F.linear(normalized, self.down.to(device=addresses.device))
        mapped = F.linear(latent, self.up.to(device=addresses.device))
        return self._rms_normalize(mapped)


class CumulativeRWKVCompatibilityRouter(nn.Module):
    """Strict causal-depth compatibility router for ephemeral RWKV virtual K/V."""

    def __init__(
        self,
        *,
        builders: Mapping[int, ExplicitRWKVVirtualKV],
        maps: Mapping[int, Any],
        anchor_layers: Sequence[int] = (5, 11, 17, 23),
        compatibility_scale: float | None = None,
        required_receptance_calls: int = 2,
    ) -> None:
        super().__init__()
        anchors = tuple(int(layer) for layer in anchor_layers)
        if not anchors or tuple(sorted(set(anchors))) != anchors:
            raise ValueError("anchor_layers must be unique and strictly increasing")
        if set(builders) != set(anchors) or set(maps) != set(anchors):
            raise ValueError("builders and maps must exactly cover anchor_layers")
        if int(required_receptance_calls) < 1:
            raise ValueError("required_receptance_calls must be >= 1")
        self.anchor_layers = anchors
        self.required_receptance_calls = int(required_receptance_calls)
        self.builders = nn.ModuleDict({str(layer): builders[layer] for layer in anchors})
        frozen_maps: dict[str, FrozenCompatibilityMap] = {}
        state_dim = None
        for layer in anchors:
            weights = maps[layer]
            down = getattr(weights, "down", None)
            up = getattr(weights, "up", None)
            if down is None or up is None:
                try:
                    down, up = weights
                except (TypeError, ValueError) as error:
                    raise ValueError(
                        "each compatibility map must expose down/up tensors"
                    ) from error
            if not isinstance(down, torch.Tensor) or not isinstance(up, torch.Tensor):
                raise ValueError("compatibility map down/up values must be tensors")
            frozen = FrozenCompatibilityMap(down, up)
            builder_shape = builders[layer].shape_spec
            expected_state_dim = builder_shape.state_heads * builder_shape.rank
            if (
                frozen.address_dim != builder_shape.key_dim
                or frozen.state_dim != expected_state_dim
            ):
                raise ValueError(
                    f"compatibility map geometry differs from builder at layer {layer}"
                )
            if state_dim is None:
                state_dim = frozen.state_dim
            elif frozen.state_dim != state_dim:
                raise ValueError("compatibility maps must share one state dimension")
            frozen_maps[str(layer)] = frozen
        assert state_dim is not None
        scale = float(state_dim if compatibility_scale is None else compatibility_scale)
        if not math.isfinite(scale) or scale <= 0.0:
            raise ValueError("compatibility_scale must be finite and > 0")
        self.compatibility_scale = scale
        self.maps = nn.ModuleDict(frozen_maps)
        self._states: dict[int, torch.Tensor] | None = None
        self._addresses: dict[int, torch.Tensor] | None = None
        self._occupied: dict[int, torch.Tensor] | None = None
        self._source_ids: torch.Tensor | None = None
        self._running_score_sum: torch.Tensor | None = None
        self._running_active: torch.Tensor | None = None
        self._next_anchor_index = 0
        self._completed = False
        self._diagnostics: list[dict[str, Any]] = []

    @property
    def active(self) -> bool:
        return self._states is not None

    @property
    def completed(self) -> bool:
        return self._completed

    @property
    def diagnostics(self) -> tuple[Mapping[str, Any], ...]:
        return tuple(self._diagnostics)

    @staticmethod
    def _clone_bank(bank: Mapping[int, torch.Tensor]) -> dict[int, torch.Tensor]:
        return {int(layer): value.detach().clone() for layer, value in bank.items()}

    def _clear_context(self, *, keep_completion: bool = False) -> None:
        self._states = None
        self._addresses = None
        self._occupied = None
        self._source_ids = None
        self._running_score_sum = None
        self._running_active = None
        self._next_anchor_index = 0
        if not keep_completion:
            self._completed = False
            self._diagnostics.clear()

    def abort_forward(self) -> None:
        self._clear_context()

    def begin_forward(
        self,
        *,
        states: Mapping[int, torch.Tensor],
        address_keys: Mapping[int, torch.Tensor],
        occupied: Mapping[int, torch.Tensor],
        source_ids: Mapping[int, torch.Tensor],
    ) -> None:
        if self.active or self.completed:
            self.abort_forward()
            raise RuntimeError("previous cumulative virtual-KV forward was not ended")
        expected_layers = set(self.anchor_layers)
        if (
            set(states) != expected_layers
            or set(address_keys) != expected_layers
            or set(occupied) != expected_layers
            or set(source_ids) != expected_layers
        ):
            raise ValueError("cumulative virtual-KV banks must exactly cover all anchors")
        reference_sources = source_ids[self.anchor_layers[0]]
        reference_occupied = occupied[self.anchor_layers[0]]
        if reference_sources.ndim != 2 or reference_sources.dtype.is_floating_point:
            raise ValueError("source_ids must be an integer [batch, slots] tensor")
        if reference_occupied.ndim != 2 or reference_occupied.dtype != torch.bool:
            raise ValueError("occupied must be boolean [batch, slots]")
        batch_slots = tuple(reference_sources.shape)
        if tuple(reference_occupied.shape) != batch_slots:
            raise ValueError("source_ids and occupied shapes differ")
        for layer in self.anchor_layers:
            builder_shape = self.builders[str(layer)].shape_spec
            expected_state = (
                batch_slots[0],
                builder_shape.state_heads,
                batch_slots[1],
                builder_shape.rank,
                builder_shape.rank,
            )
            expected_address = (*batch_slots, builder_shape.key_dim)
            if tuple(states[layer].shape) != expected_state:
                raise ValueError(f"shadow RWKV state geometry differs at layer {layer}")
            if tuple(address_keys[layer].shape) != expected_address:
                raise ValueError(f"shadow address geometry differs at layer {layer}")
            if (
                tuple(occupied[layer].shape) != batch_slots
                or occupied[layer].dtype != torch.bool
            ):
                raise ValueError(f"shadow occupancy geometry differs at layer {layer}")
            if (
                tuple(source_ids[layer].shape) != batch_slots
                or source_ids[layer].dtype.is_floating_point
                or not torch.equal(source_ids[layer], reference_sources)
            ):
                raise ValueError(f"shadow source alignment differs at layer {layer}")
            if not torch.equal(occupied[layer], reference_occupied):
                raise ValueError(f"shadow occupancy alignment differs at layer {layer}")
            if not bool(torch.isfinite(states[layer].float()).all().item()) or not bool(
                torch.isfinite(address_keys[layer].float()).all().item()
            ):
                raise ValueError(f"shadow bank contains nonfinite values at layer {layer}")
        self._states = self._clone_bank(states)
        self._addresses = self._clone_bank(address_keys)
        self._occupied = self._clone_bank(occupied)
        self._source_ids = reference_sources.detach().clone()
        self._running_score_sum = torch.zeros(
            batch_slots,
            device=address_keys[self.anchor_layers[0]].device,
            dtype=torch.float32,
        )
        self._running_active = reference_occupied.detach().clone().to(
            device=self._running_score_sum.device
        )
        self._next_anchor_index = 0
        self._completed = False
        self._diagnostics.clear()

    def provider_for(self, layer: int):
        layer = int(layer)
        if layer not in self.anchor_layers:
            raise ValueError(f"layer {layer} is not a cumulative router anchor")

        def provider(**kwargs):
            return self._provide(layer=layer, **kwargs)

        return provider

    def _provide(
        self,
        *,
        layer: int,
        query_states: torch.Tensor,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        attention_mask: torch.Tensor | None,
        position_embeddings: tuple[torch.Tensor, torch.Tensor] | None,
        module: Any,
        **kwargs,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None:
        del kwargs
        try:
            if not self.active or self._completed:
                raise RuntimeError("cumulative virtual-KV provider has no active forward")
            expected_layer = self.anchor_layers[self._next_anchor_index]
            if layer != expected_layer:
                raise RuntimeError(
                    f"cumulative virtual-KV anchor order differs: "
                    f"expected={expected_layer} actual={layer}"
                )
            module_layer = getattr(module, "layer_idx", layer)
            if int(module_layer) != layer:
                raise RuntimeError("cumulative virtual-KV provider/module layer mismatch")
            receptance = getattr(module, "rwkv_virtual_router_receptance", None)
            receptance_calls = int(
                getattr(module, "rwkv_virtual_router_receptance_calls", 0)
            )
            if receptance is None or receptance_calls != self.required_receptance_calls:
                raise RuntimeError(
                    "cumulative virtual-KV provider requires the audited current RWKV "
                    "receptance lifecycle"
                )
            if receptance.ndim != 4 or receptance.size(1) != 1:
                raise ValueError("current RWKV receptance must be [batch, 1, heads, rank]")
            assert self._states is not None
            assert self._addresses is not None
            assert self._occupied is not None
            assert self._source_ids is not None
            assert self._running_score_sum is not None
            assert self._running_active is not None
            state = self._states[layer]
            addresses = self._addresses[layer]
            occupied = self._occupied[layer]
            flattened_receptance = receptance[:, 0].float().reshape(receptance.size(0), -1)
            compatibility_map = self.maps[str(layer)]
            if (
                flattened_receptance.size(0) != state.size(0)
                or flattened_receptance.size(1) != compatibility_map.state_dim
            ):
                raise ValueError("current RWKV receptance geometry differs from router map")
            normalized_receptance = FrozenCompatibilityMap._rms_normalize(
                flattened_receptance
            )
            mapped_addresses = compatibility_map(addresses)
            local_scores = (
                mapped_addresses * normalized_receptance.unsqueeze(1)
            ).mean(dim=-1)
            local_active = (
                occupied
                & state.float().square().sum(dim=(-1, -2)).sum(dim=1).gt(0.0)
                & addresses.float().square().sum(dim=-1).gt(0.0)
            )
            self._running_score_sum = self._running_score_sum + local_scores
            self._running_active = self._running_active & local_active
            count = self._next_anchor_index + 1
            accumulated_scores = self._running_score_sum / float(count)
            attention_bias = self.compatibility_scale * accumulated_scores
            output = self.builders[str(layer)](
                state=state,
                address_keys=addresses,
                occupied=self._running_active,
                query_states=query_states,
                real_keys=key_states,
                real_values=value_states,
                attention_mask=attention_mask,
                position_embeddings=position_embeddings,
                attention_bias=attention_bias,
                module=module,
            )
            self._diagnostics.append(
                {
                    "layer": layer,
                    "count": count,
                    "source_ids": self._source_ids.detach().clone(),
                    "local_scores": local_scores.detach().clone(),
                    "accumulated_scores": accumulated_scores.detach().clone(),
                    "attention_bias": attention_bias.detach().clone(),
                    "active": self._running_active.detach().clone(),
                }
            )
            self._next_anchor_index += 1
            if self._next_anchor_index == len(self.anchor_layers):
                self._completed = True
                self._clear_context(keep_completion=True)
            return output
        except Exception:
            self.abort_forward()
            raise

    def end_forward(self) -> tuple[Mapping[str, Any], ...]:
        if not self.completed:
            self.abort_forward()
            raise RuntimeError("cumulative virtual-KV forward ended before all anchors")
        diagnostics = tuple(self._diagnostics)
        self._clear_context()
        return diagnostics
