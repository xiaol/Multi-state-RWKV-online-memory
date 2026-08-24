from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F


class FrozenCompatibilityMap(nn.Module):
    """Frozen reduced-rank map from source addresses to RWKV receptance space."""

    def __init__(self, down: torch.Tensor, up: torch.Tensor) -> None:
        super().__init__()
        if down.ndim != 2 or up.ndim != 2 or down.size(0) != up.size(1):
            raise ValueError(
                "compatibility map must be down[rank,address] and up[state,rank]"
            )
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
    def rms_normalize(value: torch.Tensor) -> torch.Tensor:
        value = value.float()
        square_mean = value.square().mean(dim=-1, keepdim=True)
        normalized = value / square_mean.clamp_min(1e-12).sqrt()
        return torch.where(square_mean.gt(0.0), normalized, torch.zeros_like(normalized))

    def forward(self, addresses: torch.Tensor) -> torch.Tensor:
        if addresses.ndim != 3 or addresses.size(-1) != self.address_dim:
            raise ValueError("compatibility addresses must be [batch, slots, address_dim]")
        normalized = self.rms_normalize(addresses)
        latent = F.linear(normalized, self.down.to(device=addresses.device))
        mapped = F.linear(latent, self.up.to(device=addresses.device))
        return self.rms_normalize(mapped)


class SourceBoundOuterFFN(nn.Module):
    """Zero-preserving query-gated correction for one selected RWKV read."""

    def __init__(
        self,
        *,
        state_dim: int,
        query_dim: int,
        bottleneck_dim: int,
    ) -> None:
        super().__init__()
        if int(state_dim) < 1 or int(query_dim) < 1 or int(bottleneck_dim) < 1:
            raise ValueError("outer FFN dimensions must be positive")
        self.state_dim = int(state_dim)
        self.query_dim = int(query_dim)
        self.bottleneck_dim = int(bottleneck_dim)
        self.state_down = nn.Linear(
            self.state_dim, self.bottleneck_dim, bias=False
        )
        self.query_gate = nn.Linear(
            self.query_dim, self.bottleneck_dim, bias=False
        )
        self.output_up = nn.Linear(
            self.bottleneck_dim, self.query_dim, bias=False
        )
        nn.init.kaiming_uniform_(self.state_down.weight, a=math.sqrt(5.0))
        nn.init.zeros_(self.query_gate.weight)
        nn.init.zeros_(self.output_up.weight)

    @staticmethod
    def rms_normalize(value: torch.Tensor) -> torch.Tensor:
        value = value.float()
        square_mean = value.square().mean(dim=-1, keepdim=True)
        normalized = value / square_mean.clamp_min(1e-12).sqrt()
        return torch.where(square_mean.gt(0.0), normalized, torch.zeros_like(value))

    def forward(
        self,
        *,
        native_read: torch.Tensor,
        hidden_query: torch.Tensor,
        base_hidden_read: torch.Tensor,
    ) -> tuple[torch.Tensor, Mapping[str, torch.Tensor]]:
        if native_read.ndim != 3 or native_read.size(-1) != self.state_dim:
            raise ValueError("outer FFN native read geometry differs")
        if hidden_query.ndim != 3 or hidden_query.size(-1) != self.query_dim:
            raise ValueError("outer FFN hidden query geometry differs")
        if tuple(hidden_query.shape) != tuple(base_hidden_read.shape):
            raise ValueError("outer FFN hidden read/query geometry differs")
        if tuple(native_read.shape[:2]) != tuple(hidden_query.shape[:2]):
            raise ValueError("outer FFN state/query sequence geometry differs")

        normalized_state = self.rms_normalize(native_read)
        normalized_query = self.rms_normalize(hidden_query)
        state_value = F.silu(self.state_down(normalized_state))
        query_gate = 2.0 * torch.sigmoid(self.query_gate(normalized_query))
        correction = self.output_up(state_value * query_gate)
        combined = base_hidden_read.float() + correction.float()
        state_active = native_read.float().square().sum(dim=-1, keepdim=True).gt(0.0)
        combined = torch.where(state_active, combined, torch.zeros_like(combined))
        square_mean = combined.square().mean(dim=-1, keepdim=True)
        direction = torch.tanh(
            combined / square_mean.clamp_min(1e-12).sqrt()
        )
        direction = torch.where(
            state_active & square_mean.gt(0.0),
            direction,
            torch.zeros_like(direction),
        )
        return direction, {
            "state_value": state_value,
            "query_gate": query_gate,
            "correction": correction,
            "combined_hidden_read": combined,
        }


class SourceCumulativeResidualRouter(nn.Module):
    """Source-canonical cumulative routing into a bounded terminal RWKV residual."""

    def __init__(
        self,
        *,
        maps: Mapping[int, Any],
        anchor_layers: Sequence[int] = (5, 11, 17, 23),
        compatibility_scale: float = 32.0,
        residual_gain: float = 1.0 / 32.0,
        required_receptance_calls: int = 2,
        outer_ffn: SourceBoundOuterFFN | None = None,
    ) -> None:
        super().__init__()
        anchors = tuple(int(layer) for layer in anchor_layers)
        if not anchors or tuple(sorted(set(anchors))) != anchors:
            raise ValueError("anchor_layers must be unique and strictly increasing")
        if set(maps) != set(anchors):
            raise ValueError("maps must exactly cover anchor_layers")
        if int(required_receptance_calls) < 1:
            raise ValueError("required_receptance_calls must be >= 1")
        scale = float(compatibility_scale)
        gain = float(residual_gain)
        if not math.isfinite(scale) or scale <= 0.0:
            raise ValueError("compatibility_scale must be finite and > 0")
        if not math.isfinite(gain) or not (0.0 < gain <= 1.0):
            raise ValueError("residual_gain must be finite and satisfy 0 < gain <= 1")

        frozen_maps: dict[str, FrozenCompatibilityMap] = {}
        state_dim = None
        address_dim = None
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
            if state_dim is None:
                state_dim = frozen.state_dim
                address_dim = frozen.address_dim
            elif frozen.state_dim != state_dim or frozen.address_dim != address_dim:
                raise ValueError("compatibility maps must share one geometry")
            frozen_maps[str(layer)] = frozen

        self.anchor_layers = anchors
        self.compatibility_scale = scale
        self.residual_gain = gain
        self.required_receptance_calls = int(required_receptance_calls)
        self.outer_ffn = outer_ffn
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

    @staticmethod
    def _canonical_gather(value: torch.Tensor, order: torch.Tensor, axis: int) -> torch.Tensor:
        shape = [order.size(0)] + [1] * (value.ndim - 1)
        shape[axis] = order.size(1)
        index = order.view(*shape).expand(
            *[
                order.size(0)
                if dimension == 0
                else order.size(1)
                if dimension == axis
                else value.size(dimension)
                for dimension in range(value.ndim)
            ]
        )
        return value.gather(axis, index)

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
            raise RuntimeError("previous cumulative residual forward was not ended")
        expected_layers = set(self.anchor_layers)
        if (
            set(states) != expected_layers
            or set(address_keys) != expected_layers
            or set(occupied) != expected_layers
            or set(source_ids) != expected_layers
        ):
            raise ValueError("cumulative residual banks must exactly cover all anchors")

        canonical_states: dict[int, torch.Tensor] = {}
        canonical_addresses: dict[int, torch.Tensor] = {}
        canonical_occupied: dict[int, torch.Tensor] = {}
        canonical_sources = None
        batch_slots = None
        for layer in self.anchor_layers:
            local_sources = source_ids[layer]
            local_occupied = occupied[layer]
            local_state = states[layer]
            local_addresses = address_keys[layer]
            compatibility_map = self.maps[str(layer)]
            if local_sources.ndim != 2 or local_sources.dtype.is_floating_point:
                raise ValueError("source_ids must be integer [batch, slots] tensors")
            if local_occupied.ndim != 2 or local_occupied.dtype != torch.bool:
                raise ValueError("occupied must be boolean [batch, slots] tensors")
            if tuple(local_occupied.shape) != tuple(local_sources.shape):
                raise ValueError("source_ids and occupied shapes differ")
            if local_state.ndim != 5 or local_addresses.ndim != 3:
                raise ValueError("state/address bank ranks differ from the RWKV contract")
            local_batch_slots = tuple(local_sources.shape)
            if batch_slots is None:
                batch_slots = local_batch_slots
            elif local_batch_slots != batch_slots:
                raise ValueError("source bank shape differs across anchors")
            if (
                local_state.size(0) != local_batch_slots[0]
                or local_state.size(2) != local_batch_slots[1]
                or local_state.size(1) * local_state.size(3) != compatibility_map.state_dim
                or local_state.size(3) != local_state.size(4)
            ):
                raise ValueError(f"shadow RWKV state geometry differs at layer {layer}")
            if tuple(local_addresses.shape[:2]) != local_batch_slots or (
                local_addresses.size(-1) != compatibility_map.address_dim
            ):
                raise ValueError(f"shadow address geometry differs at layer {layer}")
            if not bool(torch.isfinite(local_state.float()).all().item()) or not bool(
                torch.isfinite(local_addresses.float()).all().item()
            ):
                raise ValueError(f"shadow bank contains nonfinite values at layer {layer}")

            sorted_sources, order = torch.sort(local_sources, dim=1, stable=True)
            if bool(sorted_sources[:, 1:].eq(sorted_sources[:, :-1]).any().item()):
                raise ValueError("source_ids must be unique within every row")
            if canonical_sources is None:
                canonical_sources = sorted_sources.detach().clone()
            elif not torch.equal(sorted_sources, canonical_sources):
                raise ValueError("source identity set differs across anchors")
            canonical_states[layer] = self._canonical_gather(
                local_state.detach(), order, 2
            ).clone()
            canonical_addresses[layer] = self._canonical_gather(
                local_addresses.detach(), order, 1
            ).clone()
            canonical_occupied[layer] = self._canonical_gather(
                local_occupied.detach(), order, 1
            ).clone()

        assert canonical_sources is not None
        self._states = canonical_states
        self._addresses = canonical_addresses
        self._occupied = canonical_occupied
        self._source_ids = canonical_sources
        self._running_active = canonical_occupied[self.anchor_layers[0]].clone()
        self._running_score_sum = None
        self._next_anchor_index = 0
        self._completed = False
        self._diagnostics.clear()

    def provider_for(self, layer: int):
        layer = int(layer)
        if layer not in self.anchor_layers:
            raise ValueError(f"layer {layer} is not a cumulative residual anchor")

        def provider(**kwargs):
            return self._provide(layer=layer, **kwargs)

        return provider

    @staticmethod
    def _module_capture(module: Any) -> tuple[torch.Tensor, torch.Tensor, int]:
        receptance = getattr(module, "rwkv_residual_router_receptance", None)
        readout_gate = getattr(module, "rwkv_residual_router_gate", None)
        calls = int(getattr(module, "rwkv_residual_router_receptance_calls", 0))
        if receptance is None or readout_gate is None:
            raise RuntimeError(
                "cumulative residual provider requires live RWKV receptance and gate captures"
            )
        return receptance, readout_gate, calls

    def _terminal_residual(
        self,
        *,
        state: torch.Tensor,
        receptance: torch.Tensor,
        readout_gate: torch.Tensor,
        scores: torch.Tensor,
        active: torch.Tensor,
        hidden_states: torch.Tensor,
        token_mask: torch.Tensor | None,
        module: Any,
    ) -> tuple[torch.Tensor, Mapping[str, torch.Tensor]]:
        batch_size, seq_len = receptance.shape[:2]
        any_active = active.any(dim=1)
        if not bool(any_active.any().item()):
            zeros = hidden_states.new_zeros(hidden_states.shape)
            source_routes = scores.new_zeros(scores.shape)
            memory_mass = scores.new_zeros(batch_size, seq_len, 1)
            selected = torch.full(
                (batch_size, seq_len),
                -1,
                dtype=torch.long,
                device=scores.device,
            )
            return zeros, {
                "soft_source_routes": source_routes,
                "source_routes": source_routes,
                "memory_mass": memory_mass,
                "selected_slot": selected,
                "raw_read": receptance.new_zeros(batch_size, seq_len, receptance.size(2) * receptance.size(3)),
                "native_read": receptance.new_zeros(batch_size, seq_len, receptance.size(2) * receptance.size(3)),
                "hidden_read": zeros.float(),
            }

        scaled_scores = self.compatibility_scale * scores.float()
        masked_scores = scaled_scores.masked_fill(
            ~active[:, None, :], torch.finfo(scaled_scores.dtype).min
        )
        selected = masked_scores.argmax(dim=-1)
        selected_scores = masked_scores.gather(-1, selected.unsqueeze(-1))
        memory_mass = torch.sigmoid(selected_scores)
        soft_source_routes = torch.softmax(masked_scores, dim=-1)
        hard_source_routes = F.one_hot(
            selected, num_classes=masked_scores.size(-1)
        ).to(dtype=soft_source_routes.dtype)
        hard_source_routes = hard_source_routes * any_active[:, None, None].to(
            dtype=hard_source_routes.dtype
        )
        source_routes = (
            hard_source_routes
            + soft_source_routes
            - soft_source_routes.detach()
        )

        slot_reads = torch.einsum(
            "bhsij,bthj->bthsi", state.float(), receptance.float()
        )
        raw_read = torch.einsum(
            "bts,bthsi->bthi", source_routes, slot_reads
        ).reshape(batch_size, seq_len, -1)
        core = getattr(module, "hrm_rwkv7_core", None)
        if core is None:
            raise RuntimeError("cumulative residual provider requires an RWKV readout core")
        native_read = core.readout(raw_read.to(dtype=readout_gate.dtype), readout_gate)
        projector = getattr(module, "_project_delta_head", None)
        delta_o_proj = getattr(module, "delta_o_proj", None)
        if projector is None or delta_o_proj is None:
            raise RuntimeError("cumulative residual provider requires the native delta-O head")
        hidden_read = projector(native_read, delta_o_proj, "o")
        if hidden_read is None:
            raise RuntimeError("cumulative residual provider requires an active O delta head")
        outer_diagnostics: Mapping[str, torch.Tensor] = {}
        if self.outer_ffn is None:
            square_mean = hidden_read.float().square().mean(dim=-1, keepdim=True)
            direction = torch.tanh(
                hidden_read.float() / square_mean.clamp_min(1e-12).sqrt()
            )
            direction = torch.where(
                square_mean.gt(0.0), direction, torch.zeros_like(direction)
            )
        else:
            direction, outer_diagnostics = self.outer_ffn(
                native_read=native_read,
                hidden_query=hidden_states,
                base_hidden_read=hidden_read,
            )
        residual = self.residual_gain * memory_mass * direction
        if token_mask is not None:
            residual = residual * token_mask.to(
                device=residual.device, dtype=residual.dtype
            ).unsqueeze(-1)
        residual = torch.where(
            any_active[:, None, None], residual, torch.zeros_like(residual)
        ).to(dtype=hidden_states.dtype)
        return residual, {
            "soft_source_routes": soft_source_routes,
            "source_routes": hard_source_routes,
            "memory_mass": memory_mass,
            "selected_slot": torch.where(
                any_active[:, None], selected, torch.full_like(selected, -1)
            ),
            "raw_read": raw_read,
            "native_read": native_read,
            "hidden_read": hidden_read,
            **outer_diagnostics,
        }

    def _provide(
        self,
        *,
        layer: int,
        hidden_states: torch.Tensor,
        token_mask: torch.Tensor | None,
        module: Any,
        **kwargs,
    ) -> torch.Tensor | None:
        del kwargs
        try:
            if not self.active or self._completed:
                raise RuntimeError("cumulative residual provider has no active forward")
            expected_layer = self.anchor_layers[self._next_anchor_index]
            if layer != expected_layer:
                raise RuntimeError(
                    "cumulative residual anchor order differs: "
                    f"expected={expected_layer} actual={layer}"
                )
            if int(getattr(module, "layer_idx", layer)) != layer:
                raise RuntimeError("cumulative residual provider/module layer mismatch")
            receptance, readout_gate, calls = self._module_capture(module)
            if calls != self.required_receptance_calls:
                raise RuntimeError(
                    "cumulative residual provider requires the audited current RWKV "
                    "receptance lifecycle"
                )
            if receptance.ndim != 4 or readout_gate.ndim != 3:
                raise ValueError("live RWKV capture ranks differ from the router contract")
            if tuple(receptance.shape[:2]) != tuple(hidden_states.shape[:2]) or (
                tuple(readout_gate.shape[:2]) != tuple(hidden_states.shape[:2])
            ):
                raise ValueError("live RWKV capture sequence shape differs from hidden states")

            assert self._states is not None
            assert self._addresses is not None
            assert self._occupied is not None
            assert self._source_ids is not None
            assert self._running_active is not None
            state = self._states[layer].to(device=receptance.device)
            addresses = self._addresses[layer].to(device=receptance.device)
            occupied = self._occupied[layer].to(device=receptance.device)
            flattened_receptance = receptance.float().reshape(
                receptance.size(0), receptance.size(1), -1
            )
            compatibility_map = self.maps[str(layer)]
            if flattened_receptance.size(-1) != compatibility_map.state_dim:
                raise ValueError("current RWKV receptance geometry differs from router map")
            normalized_receptance = compatibility_map.rms_normalize(
                flattened_receptance
            )
            mapped_addresses = compatibility_map(addresses)
            local_scores = torch.einsum(
                "btd,bsd->bts", normalized_receptance, mapped_addresses
            ) / float(compatibility_map.state_dim)
            local_active = (
                occupied
                & state.float().square().sum(dim=(-1, -2)).sum(dim=1).gt(0.0)
                & addresses.float().square().sum(dim=-1).gt(0.0)
            )
            if self._running_score_sum is None:
                self._running_score_sum = torch.zeros_like(local_scores)
            elif tuple(self._running_score_sum.shape) != tuple(local_scores.shape):
                raise ValueError("query geometry differs across cumulative anchors")
            self._running_score_sum = self._running_score_sum + local_scores
            self._running_active = self._running_active.to(local_active.device) & local_active
            count = self._next_anchor_index + 1
            accumulated_scores = self._running_score_sum / float(count)
            residual = None
            terminal: Mapping[str, torch.Tensor] = {}
            if layer == self.anchor_layers[-1]:
                residual, terminal = self._terminal_residual(
                    state=state,
                    receptance=receptance,
                    readout_gate=readout_gate,
                    scores=accumulated_scores,
                    active=self._running_active,
                    hidden_states=hidden_states,
                    token_mask=token_mask,
                    module=module,
                )
            self._diagnostics.append(
                {
                    "layer": layer,
                    "count": count,
                    "source_ids": self._source_ids.detach().clone(),
                    "local_scores": local_scores.detach().clone(),
                    "accumulated_scores": accumulated_scores.detach().clone(),
                    "active": self._running_active.detach().clone(),
                    **{name: value.detach().clone() for name, value in terminal.items()},
                    **(
                        {"residual": residual.detach().clone()}
                        if residual is not None
                        else {}
                    ),
                }
            )
            self._next_anchor_index += 1
            if self._next_anchor_index == len(self.anchor_layers):
                self._completed = True
                self._clear_context(keep_completion=True)
            return residual
        except Exception:
            self.abort_forward()
            raise

    def end_forward(self) -> tuple[Mapping[str, Any], ...]:
        if not self.completed:
            self.abort_forward()
            raise RuntimeError("cumulative residual forward ended before all anchors")
        diagnostics = tuple(self._diagnostics)
        self._clear_context()
        return diagnostics
