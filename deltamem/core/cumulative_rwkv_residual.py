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


class SourceBoundJointIdentityFFN(nn.Module):
    """Zero-preserving RWKV value gated by address/receptance identity features."""

    def __init__(
        self,
        *,
        state_dim: int,
        hidden_dim: int,
        anchor_count: int,
        bottleneck_dim: int,
    ) -> None:
        super().__init__()
        dimensions = (state_dim, hidden_dim, anchor_count, bottleneck_dim)
        if any(int(value) < 1 for value in dimensions):
            raise ValueError("joint identity FFN dimensions must be positive")
        self.state_dim = int(state_dim)
        self.hidden_dim = int(hidden_dim)
        self.anchor_count = int(anchor_count)
        self.bottleneck_dim = int(bottleneck_dim)
        self.identity_dim = 2 * self.anchor_count * self.state_dim
        self.state_down = nn.Linear(
            self.state_dim, self.bottleneck_dim, bias=False
        )
        self.query_gate = nn.Linear(
            self.identity_dim, self.bottleneck_dim, bias=False
        )
        self.output_up = nn.Linear(
            self.bottleneck_dim, self.hidden_dim, bias=False
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
        identity_features: torch.Tensor,
        base_hidden_read: torch.Tensor,
    ) -> tuple[torch.Tensor, Mapping[str, torch.Tensor]]:
        if native_read.ndim != 3 or native_read.size(-1) != self.state_dim:
            raise ValueError("joint identity native read geometry differs")
        if (
            identity_features.ndim != 3
            or identity_features.size(-1) != self.identity_dim
        ):
            raise ValueError("joint identity feature geometry differs")
        if tuple(native_read.shape[:2]) != tuple(identity_features.shape[:2]):
            raise ValueError("joint identity state/feature sequence geometry differs")
        if (
            base_hidden_read.ndim != 3
            or base_hidden_read.size(-1) != self.hidden_dim
            or tuple(base_hidden_read.shape[:2]) != tuple(native_read.shape[:2])
        ):
            raise ValueError("joint identity hidden read geometry differs")

        normalized_state = self.rms_normalize(native_read)
        state_value = F.silu(self.state_down(normalized_state))
        identity_gate = 2.0 * torch.sigmoid(
            self.query_gate(identity_features.float())
        )
        correction = self.output_up(state_value * identity_gate)
        state_active = native_read.float().square().sum(dim=-1, keepdim=True).gt(0.0)
        direction = torch.tanh(correction.float())
        direction = torch.where(
            state_active,
            direction,
            torch.zeros_like(direction),
        )
        return direction, {
            "state_value": state_value,
            "query_gate": identity_gate,
            "correction": correction,
            "combined_hidden_read": correction,
        }


class SourceBoundMultiAnchorBundleFFN(nn.Module):
    """Zero-preserving CrossGLU over one source's native reads at every anchor."""

    def __init__(
        self,
        *,
        state_dim: int,
        hidden_dim: int,
        anchor_count: int,
        bottleneck_dim: int,
    ) -> None:
        super().__init__()
        dimensions = (state_dim, hidden_dim, anchor_count, bottleneck_dim)
        if any(int(value) < 1 for value in dimensions):
            raise ValueError("multi-anchor bundle FFN dimensions must be positive")
        self.state_dim = int(state_dim)
        self.hidden_dim = int(hidden_dim)
        self.anchor_count = int(anchor_count)
        self.bottleneck_dim = int(bottleneck_dim)
        self.bundle_dim = self.anchor_count * self.state_dim
        self.state_down = nn.Linear(
            self.bundle_dim, self.bottleneck_dim, bias=False
        )
        self.query_gate = nn.Linear(
            self.hidden_dim, self.bottleneck_dim, bias=False
        )
        self.output_up = nn.Linear(
            self.bottleneck_dim, self.hidden_dim, bias=False
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
        native_reads: torch.Tensor,
        hidden_query: torch.Tensor,
    ) -> tuple[torch.Tensor, Mapping[str, torch.Tensor]]:
        if (
            native_reads.ndim != 4
            or native_reads.size(-2) != self.anchor_count
            or native_reads.size(-1) != self.state_dim
        ):
            raise ValueError("multi-anchor native read geometry differs")
        if hidden_query.ndim != 3 or hidden_query.size(-1) != self.hidden_dim:
            raise ValueError("multi-anchor hidden query geometry differs")
        if tuple(native_reads.shape[:2]) != tuple(hidden_query.shape[:2]):
            raise ValueError("multi-anchor state/query sequence geometry differs")

        normalized_reads = self.rms_normalize(native_reads)
        bundle = normalized_reads.flatten(start_dim=-2)
        state_value = F.silu(self.state_down(bundle))
        normalized_query = self.rms_normalize(hidden_query)
        query_gate = 2.0 * torch.sigmoid(self.query_gate(normalized_query))
        correction = self.output_up(state_value * query_gate)
        bundle_active = native_reads.float().square().sum(
            dim=(-1, -2), keepdim=False
        ).unsqueeze(-1).gt(0.0)
        direction = torch.tanh(correction.float())
        direction = torch.where(
            bundle_active,
            direction,
            torch.zeros_like(direction),
        )
        return direction, {
            "anchor_native_reads": native_reads,
            "normalized_anchor_reads": normalized_reads,
            "native_read_bundle": bundle,
            "state_value": state_value,
            "query_gate": query_gate,
            "correction": correction,
            "combined_hidden_read": correction,
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
        route_weights: Sequence[float] | None = None,
        outer_ffn: SourceBoundOuterFFN
        | SourceBoundJointIdentityFFN
        | SourceBoundMultiAnchorBundleFFN
        | None = None,
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
        weighted_routing = route_weights is not None
        if route_weights is None:
            normalized_route_weights = tuple(1.0 / len(anchors) for _ in anchors)
        else:
            if len(route_weights) != len(anchors):
                raise ValueError("route_weights must cover every anchor")
            raw_route_weights = tuple(float(weight) for weight in route_weights)
            if (
                not all(math.isfinite(weight) and weight >= 0.0 for weight in raw_route_weights)
                or not any(weight > 0.0 for weight in raw_route_weights)
            ):
                raise ValueError("route_weights must be finite, nonnegative, and nonzero")
            total_route_weight = sum(raw_route_weights)
            normalized_route_weights = tuple(
                weight / total_route_weight for weight in raw_route_weights
            )

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
        self.route_weights = normalized_route_weights
        self.weighted_routing = weighted_routing
        self.outer_ffn = outer_ffn
        self.maps = nn.ModuleDict(frozen_maps)
        self._states: dict[int, torch.Tensor] | None = None
        self._addresses: dict[int, torch.Tensor] | None = None
        self._occupied: dict[int, torch.Tensor] | None = None
        self._source_ids: torch.Tensor | None = None
        self._running_score_sum: torch.Tensor | None = None
        self._local_scores: dict[int, torch.Tensor] = {}
        self._running_active: torch.Tensor | None = None
        self._normalized_receptance: dict[int, torch.Tensor] = {}
        self._mapped_addresses: dict[int, torch.Tensor] = {}
        self._slot_reads: dict[int, torch.Tensor] = {}
        self._readout_gates: dict[int, torch.Tensor] = {}
        self._readout_cores: dict[int, Any] = {}
        self._memory_mass_override: torch.Tensor | None = None
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
        self._local_scores.clear()
        self._running_active = None
        self._normalized_receptance.clear()
        self._mapped_addresses.clear()
        self._slot_reads.clear()
        self._readout_gates.clear()
        self._readout_cores.clear()
        self._memory_mass_override = None
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
        memory_mass_override: torch.Tensor | None = None,
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
        if memory_mass_override is not None:
            if (
                memory_mass_override.ndim != 3
                or tuple(memory_mass_override.shape)
                != (canonical_sources.size(0), 1, 1)
                or not bool(torch.isfinite(memory_mass_override.float()).all().item())
                or bool(memory_mass_override.lt(0.0).any().item())
                or bool(memory_mass_override.gt(1.0).any().item())
            ):
                raise ValueError(
                    "memory_mass_override must be finite [batch,1,1] values in [0,1]"
                )
        self._states = canonical_states
        self._addresses = canonical_addresses
        self._occupied = canonical_occupied
        self._source_ids = canonical_sources
        self._running_active = canonical_occupied[self.anchor_layers[0]].clone()
        self._running_score_sum = None
        self._local_scores.clear()
        self._normalized_receptance.clear()
        self._mapped_addresses.clear()
        self._slot_reads.clear()
        self._readout_gates.clear()
        self._readout_cores.clear()
        self._memory_mass_override = (
            memory_mass_override.detach().clone()
            if memory_mass_override is not None
            else None
        )
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
        computed_memory_mass = torch.sigmoid(selected_scores)
        if self._memory_mass_override is None:
            memory_mass = computed_memory_mass
        else:
            if tuple(self._memory_mass_override.shape) != (
                batch_size,
                1,
                1,
            ):
                raise RuntimeError("memory mass override lifecycle geometry differs")
            memory_mass = self._memory_mass_override.to(
                device=scores.device, dtype=computed_memory_mass.dtype
            ).expand(batch_size, seq_len, 1)
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

        outer_diagnostics: Mapping[str, torch.Tensor] = {}
        if isinstance(self.outer_ffn, SourceBoundMultiAnchorBundleFFN):
            if (
                set(self._slot_reads) != set(self.anchor_layers)
                or set(self._readout_gates) != set(self.anchor_layers)
                or set(self._readout_cores) != set(self.anchor_layers)
            ):
                raise RuntimeError("multi-anchor native read lifecycle is incomplete")
            raw_reads = []
            native_reads = []
            for anchor in self.anchor_layers:
                anchor_raw_read = torch.einsum(
                    "bts,bthsi->bthi", source_routes, self._slot_reads[anchor]
                ).reshape(batch_size, seq_len, -1)
                anchor_gate = self._readout_gates[anchor]
                anchor_native_read = self._readout_cores[anchor].readout(
                    anchor_raw_read.to(dtype=anchor_gate.dtype), anchor_gate
                )
                raw_reads.append(anchor_raw_read)
                native_reads.append(anchor_native_read)
            anchor_raw_reads = torch.stack(raw_reads, dim=2)
            anchor_native_reads = torch.stack(native_reads, dim=2)
            raw_read = anchor_raw_reads[:, :, -1]
            native_read = anchor_native_reads[:, :, -1]
            hidden_read = hidden_states.new_zeros(hidden_states.shape).float()
            direction, outer_diagnostics = self.outer_ffn(
                native_reads=anchor_native_reads,
                hidden_query=hidden_states,
            )
            outer_diagnostics = {
                **outer_diagnostics,
                "anchor_raw_reads": anchor_raw_reads,
            }
        else:
            slot_reads = torch.einsum(
                "bhsij,bthj->bthsi", state.float(), receptance.float()
            )
            raw_read = torch.einsum(
                "bts,bthsi->bthi", source_routes, slot_reads
            ).reshape(batch_size, seq_len, -1)
            core = getattr(module, "hrm_rwkv7_core", None)
            if core is None:
                raise RuntimeError(
                    "cumulative residual provider requires an RWKV readout core"
                )
            native_read = core.readout(
                raw_read.to(dtype=readout_gate.dtype), readout_gate
            )
            projector = getattr(module, "_project_delta_head", None)
            delta_o_proj = getattr(module, "delta_o_proj", None)
            if projector is None or delta_o_proj is None:
                raise RuntimeError(
                    "cumulative residual provider requires the native delta-O head"
                )
            hidden_read = projector(native_read, delta_o_proj, "o")
            if hidden_read is None:
                raise RuntimeError(
                    "cumulative residual provider requires an active O delta head"
                )
        if self.outer_ffn is None:
            square_mean = hidden_read.float().square().mean(dim=-1, keepdim=True)
            direction = torch.tanh(
                hidden_read.float() / square_mean.clamp_min(1e-12).sqrt()
            )
            direction = torch.where(
                square_mean.gt(0.0), direction, torch.zeros_like(direction)
            )
        elif isinstance(self.outer_ffn, SourceBoundJointIdentityFFN):
            identity_parts = []
            for layer in self.anchor_layers:
                normalized_receptance = self._normalized_receptance.get(layer)
                mapped_addresses = self._mapped_addresses.get(layer)
                if normalized_receptance is None or mapped_addresses is None:
                    raise RuntimeError("joint identity feature lifecycle is incomplete")
                selected_address = mapped_addresses.gather(
                    1,
                    selected.unsqueeze(-1).expand(
                        -1, -1, mapped_addresses.size(-1)
                    ),
                )
                identity_parts.extend(
                    (
                        normalized_receptance * selected_address,
                        (normalized_receptance - selected_address).abs(),
                    )
                )
            identity_features = torch.cat(identity_parts, dim=-1)
            direction, outer_diagnostics = self.outer_ffn(
                native_read=native_read,
                identity_features=identity_features,
                base_hidden_read=hidden_read,
            )
            outer_diagnostics = {
                **outer_diagnostics,
                "identity_features": identity_features,
            }
        elif isinstance(self.outer_ffn, SourceBoundOuterFFN):
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
            "computed_memory_mass": computed_memory_mass,
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
            self._normalized_receptance[layer] = normalized_receptance
            self._mapped_addresses[layer] = mapped_addresses
            local_scores = torch.einsum(
                "btd,bsd->bts", normalized_receptance, mapped_addresses
            ) / float(compatibility_map.state_dim)
            local_active = (
                occupied
                & state.float().square().sum(dim=(-1, -2)).sum(dim=1).gt(0.0)
                & addresses.float().square().sum(dim=-1).gt(0.0)
            )
            if isinstance(self.outer_ffn, SourceBoundMultiAnchorBundleFFN):
                core = getattr(module, "hrm_rwkv7_core", None)
                if core is None:
                    raise RuntimeError(
                        "multi-anchor bundle requires an RWKV readout core at every anchor"
                    )
                self._slot_reads[layer] = torch.einsum(
                    "bhsij,bthj->bthsi", state.float(), receptance.float()
                )
                self._readout_gates[layer] = readout_gate
                self._readout_cores[layer] = core
            if self._running_score_sum is None:
                self._running_score_sum = torch.zeros_like(local_scores)
            elif tuple(self._running_score_sum.shape) != tuple(local_scores.shape):
                raise ValueError("query geometry differs across cumulative anchors")
            self._running_score_sum = self._running_score_sum + local_scores
            self._running_active = self._running_active.to(local_active.device) & local_active
            count = self._next_anchor_index + 1
            self._local_scores[layer] = local_scores
            if layer == self.anchor_layers[-1] and self.weighted_routing:
                accumulated_scores = sum(
                    self._local_scores[anchor] * weight
                    for anchor, weight in zip(self.anchor_layers, self.route_weights)
                )
            else:
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
                    "route_weights": self.route_weights,
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
