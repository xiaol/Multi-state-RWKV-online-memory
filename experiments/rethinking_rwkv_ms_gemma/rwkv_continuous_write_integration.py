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
from experiments.rethinking_rwkv_ms_gemma import rwkv_continuous_write_alignment


@dataclass(frozen=True)
class WriteAddressLatch:
    keys: torch.Tensor
    routes: torch.Tensor
    selected_keys: torch.Tensor
    effective_selected_keys: torch.Tensor
    address_seq: torch.Tensor
    folded_address_seq: torch.Tensor
    selected_keys_version: int
    effective_selected_keys_version: int
    address_version: int
    folded_address_version: int
    address_override_applied: bool


@dataclass(frozen=True)
class QueuedWriteAddressOverride:
    selected_keys: torch.Tensor
    selected_keys_version: int


CONTINUOUS_MODE = "continuous"
INHERITED_EXACT_V5_MODE = "inherited_exact_v5"
RAW_UNCONDITIONED_MODE = "raw_unconditioned"
CONTROL_MODES = frozenset(
    (CONTINUOUS_MODE, INHERITED_EXACT_V5_MODE, RAW_UNCONDITIONED_MODE)
)


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

    def load_frozen_map(
        self,
        down: torch.Tensor,
        up: torch.Tensor,
    ) -> None:
        if tuple(down.shape) != tuple(self.down.shape):
            raise ValueError(
                "Continuous-write down-map shape differs: "
                f"expected={tuple(self.down.shape)} actual={tuple(down.shape)}"
            )
        if tuple(up.shape) != tuple(self.up.shape):
            raise ValueError(
                "Continuous-write up-map shape differs: "
                f"expected={tuple(self.up.shape)} actual={tuple(up.shape)}"
            )
        if not bool(torch.isfinite(down).all().item()) or not bool(
            torch.isfinite(up).all().item()
        ):
            raise ValueError("Continuous-write frozen map is nonfinite")
        with torch.no_grad():
            self.down.copy_(down.to(device=self.down.device, dtype=self.down.dtype))
            self.up.copy_(up.to(device=self.up.device, dtype=self.up.dtype))
        self.down.requires_grad_(False)
        self.up.requires_grad_(False)

    def direction(self, address_seq: torch.Tensor) -> torch.Tensor:
        if address_seq.shape[-1] != self.address_dim:
            raise ValueError(
                "Continuous-write address width differs: "
                f"expected={self.address_dim} actual={address_seq.shape[-1]}"
            )
        address = address_seq.to(device=self.down.device, dtype=torch.float32)
        if not bool(torch.isfinite(address).all().item()):
            raise ValueError("Continuous-write address is nonfinite")
        active = address.square().sum(dim=-1, keepdim=True).gt(0.0)
        normalized_address = rwkv_continuous_write_alignment._rms_normalize(address)
        mapped = F.linear(
            F.linear(normalized_address, self.down.float()),
            self.up.float(),
        )
        if not bool(torch.isfinite(mapped).all().item()):
            raise RuntimeError("Continuous-write active address mapped nonfinitely")
        square_mean = mapped.square().mean(dim=-1, keepdim=True)
        if bool((active & square_mean.le(0.0)).any().item()):
            raise RuntimeError("Continuous-write active address mapped to zero direction")
        normalized = rwkv_continuous_write_alignment._rms_normalize(mapped)
        if not bool(torch.isfinite(normalized).all().item()):
            raise RuntimeError("Continuous-write normalized direction is nonfinite")
        return normalized

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
    if latch.selected_keys._version != latch.selected_keys_version:
        raise RuntimeError("Continuous-write immutable natural address was mutated in place")
    if (
        latch.effective_selected_keys._version
        != latch.effective_selected_keys_version
    ):
        raise RuntimeError("Continuous-write immutable effective address was mutated in place")
    if latch.address_seq._version != latch.address_version:
        raise RuntimeError("Continuous-write immutable address was mutated in place")
    if latch.folded_address_seq._version != latch.folded_address_version:
        raise RuntimeError("Continuous-write immutable folded address was mutated in place")


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
    queued_override = module.rwkv_continuous_write_address_override_queue
    if queued_override is None:
        effective_selected_keys = selected_keys
        address_override_applied = False
    else:
        if not isinstance(queued_override, QueuedWriteAddressOverride):
            raise RuntimeError("Continuous-write queued address override contract differs")
        if (
            queued_override.selected_keys._version
            != queued_override.selected_keys_version
        ):
            raise RuntimeError("Continuous-write queued address override was mutated")
        expected_selected = (
            batch_size,
            1,
            module.projected_kv_key_dim,
        )
        if tuple(queued_override.selected_keys.shape) != expected_selected:
            raise ValueError(
                "Continuous-write queued address override shape differs: "
                f"expected={expected_selected} "
                f"actual={tuple(queued_override.selected_keys.shape)}"
            )
        if queued_override.selected_keys.dtype != torch.float32:
            raise ValueError("Continuous-write queued address override must be float32")
        if queued_override.selected_keys.device != selected_keys.device:
            raise ValueError("Continuous-write queued address override device differs")
        if not bool(torch.isfinite(queued_override.selected_keys).all().item()):
            raise ValueError("Continuous-write queued address override is nonfinite")
        effective_selected_keys = queued_override.selected_keys
        address_override_applied = True
    address_seq = effective_selected_keys.expand(-1, sequence_length, -1).clone()
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
    if module.projected_kv_key_dim % module.state_read_dim != 0:
        raise ValueError("Continuous-write address cannot be folded to exact-v5 width")
    fold = module.projected_kv_key_dim // module.state_read_dim
    folded_selected = selected_keys.reshape(
        batch_size,
        1,
        fold,
        module.state_read_dim,
    ).sum(dim=2) / math.sqrt(float(fold))
    folded_address_seq = folded_selected.to(dtype=hidden_states.dtype).expand(
        -1,
        sequence_length,
        -1,
    )
    if token_mask is not None:
        folded_address_seq = folded_address_seq * token_mask.to(
            device=folded_address_seq.device,
            dtype=folded_address_seq.dtype,
        ).unsqueeze(-1)
    folded_address_seq = folded_address_seq.detach().clone()
    return WriteAddressLatch(
        keys=keys_snapshot,
        routes=routes_snapshot,
        selected_keys=selected_keys,
        effective_selected_keys=effective_selected_keys,
        address_seq=address_seq,
        folded_address_seq=folded_address_seq,
        selected_keys_version=selected_keys._version,
        effective_selected_keys_version=effective_selected_keys._version,
        address_version=address_seq._version,
        folded_address_version=folded_address_seq._version,
        address_override_applied=address_override_applied,
    )


def _projected_slot_write(
    module: Any,
    hidden_states: torch.Tensor,
    token_mask: torch.Tensor | None,
) -> None:
    module.rwkv_continuous_write_latch = None
    module.rwkv_continuous_write_audit = None
    module.rwkv_continuous_write_original_projected_slot_write(hidden_states, token_mask)
    latch = _materialize_latch(
        module,
        hidden_states,
        token_mask,
    )
    module.rwkv_continuous_write_latch = latch
    if latch.address_override_applied:
        module.rwkv_continuous_write_address_override_queue = None


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

    mode = module.rwkv_continuous_write_mode
    if mode == RAW_UNCONDITIONED_MODE:
        outputs = (k, v, a, b)
    elif mode == INHERITED_EXACT_V5_MODE:
        outputs = module.rwkv_continuous_write_original_conditioner(
            k,
            v,
            a,
            b,
            latch.folded_address_seq,
            token_mask,
        )
    elif mode == CONTINUOUS_MODE:
        outputs = module.rwkv_continuous_write_conditioner(
            k,
            v,
            a,
            b,
            address_seq,
            token_mask,
        )
    else:
        raise RuntimeError(f"Unknown continuous-write control mode: {mode}")
    if mode != INHERITED_EXACT_V5_MODE and outputs[1] is not v:
        raise RuntimeError("Continuous-write conditioner changed the RWKV value object")
    if module.rwkv_continuous_write_capture_enabled:
        module.rwkv_continuous_write_audit = {
            "mode": mode,
            "latched_selected_keys": latch.selected_keys.detach().clone(),
            "effective_selected_keys": latch.effective_selected_keys.detach().clone(),
            "natural_selected_keys_object_id": id(latch.selected_keys),
            "effective_selected_keys_object_id": id(latch.effective_selected_keys),
            "address_override_applied": latch.address_override_applied,
            "effective_full64_consumed_by_mode": mode == CONTINUOUS_MODE,
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
    rank: int = rwkv_continuous_write_alignment.MAP_RANK,
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
        module.rwkv_continuous_write_mode = CONTINUOUS_MODE
        module.rwkv_continuous_write_capture_enabled = False
        module.rwkv_continuous_write_latch = None
        module.rwkv_continuous_write_audit = None
        module.rwkv_continuous_write_address_override_queue = None
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
        "default_mode": CONTINUOUS_MODE,
        "baseline_mode": INHERITED_EXACT_V5_MODE,
        "raw_control_mode": RAW_UNCONDITIONED_MODE,
        "effective_full64_override": "one_shot_selected_key_only",
        "pending_effective_full64_overrides": 0,
    }


def set_enabled(model: nn.Module, enabled: bool) -> None:
    set_mode(model, CONTINUOUS_MODE if enabled else INHERITED_EXACT_V5_MODE)


def set_mode(model: nn.Module, mode: str) -> None:
    if mode not in CONTROL_MODES:
        raise ValueError(f"Unknown continuous-write control mode: {mode}")
    for _, module in iter_delta_mem_modules(model):
        module.rwkv_continuous_write_mode = mode


def set_capture(model: nn.Module, enabled: bool) -> None:
    for _, module in iter_delta_mem_modules(model):
        module.rwkv_continuous_write_capture_enabled = bool(enabled)
        module.rwkv_continuous_write_audit = None


def queue_effective_full64_address_overrides(
    model: nn.Module,
    overrides: Mapping[str, torch.Tensor],
) -> None:
    modules = tuple(iter_delta_mem_modules(model))
    expected_names = {name for name, _ in modules}
    if set(overrides) != expected_names:
        raise ValueError("Continuous-write address override module inventory differs")
    prepared: list[tuple[Any, QueuedWriteAddressOverride]] = []
    for name, module in modules:
        if module.rwkv_continuous_write_address_override_queue is not None:
            raise RuntimeError(
                f"Continuous-write address override is already queued: {name}"
            )
        value = overrides[name]
        if not isinstance(value, torch.Tensor):
            raise TypeError("Continuous-write address override must be a tensor")
        if (
            value.ndim != 3
            or value.shape[1] != 1
            or value.shape[2] != module.projected_kv_key_dim
        ):
            raise ValueError(
                "Continuous-write address override shape differs: "
                f"expected=[batch,1,{module.projected_kv_key_dim}] "
                f"actual={tuple(value.shape)}"
            )
        if value.dtype != torch.float32:
            raise ValueError("Continuous-write address override must be float32")
        if not bool(torch.isfinite(value).all().item()):
            raise ValueError("Continuous-write address override is nonfinite")
        selected_keys = value.detach().to(
            device=module.memory_v_proj.device,
            dtype=torch.float32,
        ).contiguous().clone()
        prepared.append(
            (
                module,
                QueuedWriteAddressOverride(
                    selected_keys=selected_keys,
                    selected_keys_version=selected_keys._version,
                ),
            )
        )
    for module, queued in prepared:
        module.rwkv_continuous_write_address_override_queue = queued


def clear_effective_full64_address_overrides(model: nn.Module) -> None:
    for _, module in iter_delta_mem_modules(model):
        module.rwkv_continuous_write_address_override_queue = None


def pending_effective_full64_address_override_names(
    model: nn.Module,
) -> tuple[str, ...]:
    return tuple(
        name
        for name, module in iter_delta_mem_modules(model)
        if module.rwkv_continuous_write_address_override_queue is not None
    )


def clear_transient(model: nn.Module) -> None:
    for _, module in iter_delta_mem_modules(model):
        module.rwkv_continuous_write_latch = None
        module.rwkv_continuous_write_audit = None
        module.rwkv_continuous_write_address_override_queue = None
