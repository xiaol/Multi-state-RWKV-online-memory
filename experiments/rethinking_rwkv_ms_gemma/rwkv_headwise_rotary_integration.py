"""Runtime hooks for headwise rotary address-value binding in RWKV-MS."""

from __future__ import annotations

import math
from types import MethodType
from typing import Any, Mapping

import torch

from deltamem.core.delta import iter_delta_mem_modules

from .rwkv_headwise_rotary_binding import HeadwiseRotaryBinding


def _fold_projected_address(
    module: Any,
    routes: torch.Tensor,
    *,
    sequence_length: int,
) -> torch.Tensor:
    keys = module.projected_kv_keys
    if keys is None:
        raise RuntimeError("Rotary binding requires projected slot keys")
    if routes.ndim != 3 or keys.ndim != 3:
        raise ValueError("Rotary projected routes and keys must be batched")
    if routes.shape[0] != keys.shape[0] or routes.shape[-1] != keys.shape[1]:
        raise ValueError("Rotary projected routes and keys differ")
    key_dim = int(keys.shape[-1])
    state_dim = int(module.state_read_dim)
    if key_dim % state_dim:
        raise ValueError("Rotary projected key width must fold into RWKV state")
    selected = torch.einsum("bts,bsd->btd", routes.float(), keys.float())
    fold = key_dim // state_dim
    selected = selected.reshape(
        selected.shape[0], selected.shape[1], fold, state_dim
    ).sum(dim=-2) / math.sqrt(float(fold))
    if selected.shape[1] == 1 and sequence_length != 1:
        selected = selected.expand(-1, sequence_length, -1)
    if selected.shape[1] != sequence_length:
        raise ValueError("Rotary query address sequence has an unexpected length")
    return selected


def _projected_slot_addresses(module: Any, *, sequence_length: int) -> torch.Tensor:
    keys = module.projected_kv_keys
    if keys is None or keys.ndim != 3:
        raise RuntimeError("Rotary slot unbinding requires projected slot keys")
    state_dim = int(module.state_read_dim)
    key_dim = int(keys.shape[-1])
    if key_dim % state_dim:
        raise ValueError("Rotary projected key width must fold into RWKV state")
    fold = key_dim // state_dim
    addresses = keys.float().reshape(
        keys.shape[0], keys.shape[1], fold, state_dim
    ).sum(dim=-2) / math.sqrt(float(fold))
    return addresses.detach().unsqueeze(1).expand(-1, sequence_length, -1, -1)


def _unbind_slot_reads(
    module: Any,
    slot_reads: torch.Tensor,
    *,
    sequence_length: int,
) -> torch.Tensor:
    batch_size, _, num_heads, num_slots, head_size = slot_reads.shape
    slot_values = slot_reads.permute(0, 1, 3, 2, 4).reshape(
        batch_size, sequence_length, num_slots, module.state_read_dim
    )
    addresses = _projected_slot_addresses(
        module, sequence_length=sequence_length
    ).to(device=slot_values.device)
    decoded = module.rwkv_headwise_rotary_binding.unbind(
        addresses, slot_values.float()
    )
    return decoded.reshape(
        batch_size, sequence_length, num_slots, num_heads, head_size
    ).permute(0, 1, 3, 2, 4)


def _record_read(
    module: Any,
    kind: str,
    *,
    address: torch.Tensor,
    raw: torch.Tensor,
    decoded: torch.Tensor,
) -> None:
    if not getattr(module, "rwkv_rotary_capture_enabled", False):
        return
    captures = module.rwkv_rotary_read_captures
    captures[kind] = {
        "address": address.detach().float(),
        "raw": raw.detach().float(),
        "decoded": decoded.detach().float(),
    }


def _address_conditioned_write_features(
    module: Any,
    k: torch.Tensor,
    v: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    address_seq: torch.Tensor,
    token_mask: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    features = module.rwkv_rotary_original_address_conditioned_write_features(
        k, v, a, b, address_seq, token_mask
    )
    if getattr(module, "rwkv_rotary_capture_enabled", False):
        module.rwkv_rotary_write_address = address_seq.detach().float()
    if not module.rwkv_rotary_binding_enabled:
        return features
    feature_k, feature_v, feature_a, feature_b = features
    binder: HeadwiseRotaryBinding = module.rwkv_headwise_rotary_binding
    bound_v = binder.bind(address_seq.float(), feature_v.float())
    if not bool(torch.isfinite(bound_v).all().item()):
        raise RuntimeError("Rotary bound RWKV values are non-finite")
    if getattr(module, "rwkv_rotary_capture_enabled", False):
        module.rwkv_rotary_bound_values = bound_v.detach().float()
    return feature_k, bound_v, feature_a, feature_b


def _addressed_token_state_reads(
    module: Any,
    state: torch.Tensor,
    memory_source_seq: torch.Tensor,
    projected_routes: torch.Tensor,
    token_mask: torch.Tensor | None,
) -> torch.Tensor:
    batch_size, seq_len, _ = memory_source_seq.shape
    r_seq, slot_reads, readout_gate = module._rwkv_ms_token_state_read_basis(
        state, memory_source_seq, token_mask
    )
    routes = projected_routes.to(device=slot_reads.device, dtype=slot_reads.dtype)
    if token_mask is not None:
        routes = routes * token_mask.to(
            device=routes.device, dtype=routes.dtype
        ).unsqueeze(-1)
    raw = torch.einsum("bts,bthsi->bthi", routes, slot_reads).reshape(
        batch_size, seq_len, module.state_read_dim
    )
    address = _fold_projected_address(module, routes, sequence_length=seq_len)
    module.rwkv_rotary_query_address = address.detach()
    if module.rwkv_rotary_binding_enabled:
        decoded_slots = _unbind_slot_reads(
            module, slot_reads, sequence_length=seq_len
        )
        decoded = torch.einsum("bts,bthsi->bthi", routes, decoded_slots).reshape(
            batch_size, seq_len, module.state_read_dim
        )
    else:
        decoded = raw
    _record_read(
        module, "addressed", address=address, raw=raw, decoded=decoded
    )
    module.last_read_routes = routes
    return module.hrm_rwkv7_core.readout(
        decoded.to(dtype=readout_gate.dtype), readout_gate
    )


def _token_state_reads(
    module: Any,
    state: torch.Tensor,
    memory_source_seq: torch.Tensor,
    token_mask: torch.Tensor | None,
) -> torch.Tensor:
    batch_size, seq_len, _ = memory_source_seq.shape
    if seq_len == 0:
        module.last_read_routes = memory_source_seq.new_zeros(
            batch_size, 0, module.rwkv_ms_num_states
        )
        return memory_source_seq.new_zeros(
            batch_size, 0, module.state_read_dim
        )
    r_seq, slot_reads, readout_gate = module._rwkv_ms_token_state_read_basis(
        state, memory_source_seq, token_mask
    )
    routes = module._rwkv_ms_token_state_routes(
        state, r_seq, slot_reads, token_mask
    )
    raw = torch.einsum("bths,bthsi->bthi", routes, slot_reads).reshape(
        batch_size, seq_len, module.state_read_dim
    )
    address = getattr(module, "rwkv_rotary_query_address", None)
    if address is None or tuple(address.shape) != (batch_size, seq_len, module.state_read_dim):
        raise RuntimeError("Rotary global read has no matching projected query address")
    if module.rwkv_rotary_binding_enabled:
        decoded_slots = _unbind_slot_reads(
            module, slot_reads, sequence_length=seq_len
        )
        decoded = torch.einsum("bths,bthsi->bthi", routes, decoded_slots).reshape(
            batch_size, seq_len, module.state_read_dim
        )
    else:
        decoded = raw
    _record_read(module, "global", address=address, raw=raw, decoded=decoded)
    module.last_read_routes = routes.mean(dim=2)
    return module.hrm_rwkv7_core.readout(
        decoded.to(dtype=readout_gate.dtype), readout_gate
    )


def install(
    model: torch.nn.Module,
    *,
    state_dim: int = 32,
    head_size: int = 32,
    max_phase: float = math.pi,
    trainable_projection: bool = True,
) -> Mapping[str, Any]:
    modules = tuple(iter_delta_mem_modules(model))
    if not modules:
        raise ValueError("Rotary binding requires Delta-Mem modules")
    installed: list[str] = []
    parameter_tensors = 0
    parameter_elements = 0
    for module_name, module in modules:
        if hasattr(module, "rwkv_headwise_rotary_binding"):
            raise ValueError(f"Rotary binding already installed on {module_name}")
        if int(module.state_read_dim) != int(state_dim):
            raise ValueError("Rotary state width differs from protocol")
        if int(module.num_state_heads) != 1 or int(module.rank) != int(head_size):
            raise ValueError("Rotary protocol requires one RWKV head of width 32")
        if module.rwkv_ms_hybrid_mode != "address_keyed_moe_deepembed_ffn":
            raise ValueError(
                "Rotary integration requires address_keyed_moe_deepembed_ffn mode"
            )
        binder = HeadwiseRotaryBinding(
            state_dim,
            head_size=head_size,
            max_phase=max_phase,
            trainable_projection=trainable_projection,
        ).to(device=next(module.parameters()).device)
        module.add_module("rwkv_headwise_rotary_binding", binder)
        module.rwkv_rotary_original_address_conditioned_write_features = (
            module._rwkv_ms_address_conditioned_write_features
        )
        module.rwkv_rotary_original_addressed_token_state_reads = (
            module._rwkv_ms_addressed_token_state_reads
        )
        module.rwkv_rotary_original_token_state_reads = module._rwkv_ms_token_state_reads
        module._rwkv_ms_address_conditioned_write_features = MethodType(
            _address_conditioned_write_features, module
        )
        module._rwkv_ms_addressed_token_state_reads = MethodType(
            _addressed_token_state_reads, module
        )
        module._rwkv_ms_token_state_reads = MethodType(_token_state_reads, module)
        module.rwkv_rotary_binding_enabled = True
        module.rwkv_rotary_capture_enabled = False
        module.rwkv_rotary_query_address = None
        module.rwkv_rotary_write_address = None
        module.rwkv_rotary_bound_values = None
        module.rwkv_rotary_read_captures = {}
        installed.append(module_name)
        parameter_tensors += sum(1 for _ in binder.parameters())
        parameter_elements += sum(parameter.numel() for parameter in binder.parameters())
    return {
        "modules": len(installed),
        "module_names": tuple(installed),
        "state_dim": int(state_dim),
        "head_size": int(head_size),
        "parameters_per_layer": sum(
            parameter.numel() for parameter in next(iter_delta_mem_modules(model))[1].rwkv_headwise_rotary_binding.parameters()
        ),
        "parameter_tensors": parameter_tensors,
        "parameter_elements": parameter_elements,
        "binding_placement": "after_rwkv_v_projection_before_state_scan",
        "unbinding_placement": "after_rwkv_matrix_read_before_group_norm_output",
        "forward_output_changed": True,
        "projected_carrier_changed": False,
    }


def set_enabled(model: torch.nn.Module, enabled: bool) -> None:
    for _, module in iter_delta_mem_modules(model):
        module.rwkv_rotary_binding_enabled = bool(enabled)


def set_capture(model: torch.nn.Module, enabled: bool) -> None:
    for _, module in iter_delta_mem_modules(model):
        module.rwkv_rotary_capture_enabled = bool(enabled)
        if enabled:
            module.rwkv_rotary_read_captures = {}
        else:
            module.rwkv_rotary_read_captures = {}


def clear_transient(model: torch.nn.Module) -> None:
    for _, module in iter_delta_mem_modules(model):
        module.rwkv_rotary_query_address = None
        module.rwkv_rotary_write_address = None
        module.rwkv_rotary_bound_values = None
        module.rwkv_rotary_read_captures = {}


def named_binding_parameters(model: torch.nn.Module) -> tuple[tuple[str, torch.nn.Parameter], ...]:
    named = tuple(
        (name, parameter)
        for name, parameter in model.named_parameters()
        if "rwkv_headwise_rotary_binding" in name
    )
    return tuple(sorted(named, key=lambda item: item[0]))
