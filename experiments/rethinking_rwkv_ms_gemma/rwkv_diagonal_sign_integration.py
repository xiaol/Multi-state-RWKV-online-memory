"""Runtime hooks for exact diagonal-sign address binding in RWKV-MS."""

from __future__ import annotations

import math
from types import MethodType
from typing import Any, Mapping

import torch

from deltamem.core.delta import iter_delta_mem_modules

from .rwkv_diagonal_sign_binding import DiagonalSignBinding, deterministic_projection


def _fold_address(module: Any, routes: torch.Tensor, *, sequence_length: int) -> torch.Tensor:
    keys = module.projected_kv_keys
    if keys is None:
        raise RuntimeError("Diagonal-sign binding requires projected slot keys")
    selected = torch.einsum("bts,bsd->btd", routes.float(), keys.float())
    if selected.shape[1] == 1 and sequence_length != 1:
        selected = selected.expand(-1, sequence_length, -1)
    if selected.shape[1] != sequence_length:
        raise ValueError("Projected query address length differs from read sequence")
    return selected.detach()


def _slot_addresses(module: Any, *, sequence_length: int) -> torch.Tensor:
    keys = module.projected_kv_keys
    if keys is None:
        raise RuntimeError("Diagonal-sign slot unbinding requires projected keys")
    return keys.float().detach().unsqueeze(1).expand(-1, sequence_length, -1, -1)


def _unbind_slots(module: Any, slot_reads: torch.Tensor, *, sequence_length: int) -> torch.Tensor:
    batch_size, _, num_heads, num_slots, head_size = slot_reads.shape
    values = slot_reads.permute(0, 1, 3, 2, 4).reshape(
        batch_size, sequence_length, num_slots, module.state_read_dim
    )
    addresses = _slot_addresses(module, sequence_length=sequence_length)
    module.rwkv_diagonal_sign_slot_codes = module.rwkv_diagonal_sign_binding.codes(addresses).detach()
    decoded = module.rwkv_diagonal_sign_binding.unbind(addresses, values.float())
    return decoded.reshape(
        batch_size, sequence_length, num_slots, num_heads, head_size
    ).permute(0, 1, 3, 2, 4)


def _record(module: Any, kind: str, address: torch.Tensor, raw: torch.Tensor, decoded: torch.Tensor) -> None:
    if getattr(module, "rwkv_diagonal_sign_capture_enabled", False):
        module.rwkv_diagonal_sign_captures[kind] = {
            "address": address.detach().float(),
            "raw": raw.detach().float(),
            "decoded": decoded.detach().float(),
            "slot_codes": getattr(module, "rwkv_diagonal_sign_slot_codes", address.new_ones(*address.shape)),
            "write_codes": getattr(module, "rwkv_diagonal_sign_write_code", address.new_ones(*address.shape)),
        }


def _write_features(module: Any, k: torch.Tensor, v: torch.Tensor, a: torch.Tensor, b: torch.Tensor, address_seq: torch.Tensor, token_mask: torch.Tensor | None) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    features = module.rwkv_diagonal_sign_original_write_features(k, v, a, b, address_seq, token_mask)
    if getattr(module, "rwkv_diagonal_sign_capture_enabled", False):
        module.rwkv_diagonal_sign_write_address = address_seq.detach().float()
        module.rwkv_rotary_write_address = module.rwkv_diagonal_sign_write_address
    feature_k, feature_v, feature_a, feature_b = features
    binding_address = address_seq.float()
    routes = module.last_write_routes
    if routes is not None and routes.ndim == 3 and routes.shape[1] == 1:
        slot_addresses = _slot_addresses(
            module, sequence_length=address_seq.shape[1]
        )
        selected_slot = routes.argmax(dim=-1).view(
            routes.shape[0], 1, 1, 1
        ).expand(
            -1, address_seq.shape[1], 1, module.projected_kv_key_dim
        )
        binding_address = slot_addresses.gather(2, selected_slot).squeeze(2)
    module.rwkv_diagonal_sign_write_code = module.rwkv_diagonal_sign_binding.codes(binding_address).detach()
    if not module.rwkv_diagonal_sign_enabled:
        return features
    bound_v = module.rwkv_diagonal_sign_binding.bind(binding_address, feature_v)
    if not bool(torch.isfinite(bound_v).all().item()):
        raise RuntimeError("Diagonal-sign bound values are non-finite")
    return feature_k, bound_v, feature_a, feature_b


def _addressed_reads(module: Any, state: torch.Tensor, memory_source_seq: torch.Tensor, projected_routes: torch.Tensor, token_mask: torch.Tensor | None) -> torch.Tensor:
    batch_size, seq_len, _ = memory_source_seq.shape
    _, slot_reads, readout_gate = module._rwkv_ms_token_state_read_basis(state, memory_source_seq, token_mask)
    routes = projected_routes.to(device=slot_reads.device, dtype=slot_reads.dtype)
    if token_mask is not None:
        routes = routes * token_mask.to(device=routes.device, dtype=routes.dtype).unsqueeze(-1)
    raw = torch.einsum("bts,bthsi->bthi", routes, slot_reads).reshape(batch_size, seq_len, module.state_read_dim)
    address = _fold_address(module, routes, sequence_length=seq_len)
    module.rwkv_diagonal_sign_query_address = address
    module.rwkv_diagonal_sign_slot_codes = module.rwkv_diagonal_sign_binding.codes(
        _slot_addresses(module, sequence_length=seq_len)
    ).detach()
    if module.rwkv_diagonal_sign_enabled:
        decoded_slots = _unbind_slots(module, slot_reads, sequence_length=seq_len)
        decoded = torch.einsum("bts,bthsi->bthi", routes, decoded_slots).reshape(batch_size, seq_len, module.state_read_dim)
    else:
        decoded = raw
    _record(module, "addressed", address, raw, decoded)
    module.last_read_routes = routes
    return module.hrm_rwkv7_core.readout(decoded.to(dtype=readout_gate.dtype), readout_gate)


def _global_reads(module: Any, state: torch.Tensor, memory_source_seq: torch.Tensor, token_mask: torch.Tensor | None) -> torch.Tensor:
    batch_size, seq_len, _ = memory_source_seq.shape
    if seq_len == 0:
        return memory_source_seq.new_zeros(batch_size, 0, module.state_read_dim)
    r_seq, slot_reads, readout_gate = module._rwkv_ms_token_state_read_basis(state, memory_source_seq, token_mask)
    routes = module._rwkv_ms_token_state_routes(state, r_seq, slot_reads, token_mask)
    raw = torch.einsum("bths,bthsi->bthi", routes, slot_reads).reshape(batch_size, seq_len, module.state_read_dim)
    address = getattr(module, "rwkv_diagonal_sign_query_address", None)
    if address is None or tuple(address.shape) != (batch_size, seq_len, module.projected_kv_key_dim):
        raise RuntimeError("Diagonal-sign global read has no same-step query address")
    if module.rwkv_diagonal_sign_enabled:
        decoded_slots = _unbind_slots(module, slot_reads, sequence_length=seq_len)
        decoded = torch.einsum("bths,bthsi->bthi", routes, decoded_slots).reshape(batch_size, seq_len, module.state_read_dim)
    else:
        decoded = raw
    _record(module, "global", address, raw, decoded)
    module.last_read_routes = routes.mean(dim=2)
    return module.hrm_rwkv7_core.readout(decoded.to(dtype=readout_gate.dtype), readout_gate)


def install(model: torch.nn.Module, *, state_dim: int = 32, head_size: int = 32, seed: int = 115, frequency: float = 64.0, trainable_projection: bool = True) -> Mapping[str, Any]:
    modules = tuple(iter_delta_mem_modules(model))
    if not modules:
        raise ValueError("Diagonal-sign binding requires Delta-Mem modules")
    installed: list[str] = []
    for index, (name, module) in enumerate(modules):
        if int(module.state_read_dim) != int(state_dim) or int(module.num_state_heads) != 1 or int(module.rank) != int(head_size):
            raise ValueError("Diagonal-sign protocol requires one RWKV width-32 head")
        if module.rwkv_ms_hybrid_mode != "address_keyed_moe_deepembed_ffn":
            raise ValueError("Diagonal-sign binding requires address-keyed write mode")
        address_dim = int(module.projected_kv_key_dim)
        projection = deterministic_projection(address_dim, int(seed) + index, state_dim)
        binder = DiagonalSignBinding(state_dim, address_dim=address_dim, projection=projection, frequency=frequency, trainable_projection=trainable_projection).to(next(module.parameters()).device)
        module.add_module("rwkv_diagonal_sign_binding", binder)
        module.rwkv_diagonal_sign_original_write_features = module._rwkv_ms_address_conditioned_write_features
        module.rwkv_diagonal_sign_original_addressed_reads = module._rwkv_ms_addressed_token_state_reads
        module.rwkv_diagonal_sign_original_global_reads = module._rwkv_ms_token_state_reads
        module._rwkv_ms_address_conditioned_write_features = MethodType(_write_features, module)
        module._rwkv_ms_addressed_token_state_reads = MethodType(_addressed_reads, module)
        module._rwkv_ms_token_state_reads = MethodType(_global_reads, module)
        module.rwkv_diagonal_sign_enabled = True
        module.rwkv_diagonal_sign_capture_enabled = False
        module.rwkv_diagonal_sign_query_address = None
        module.rwkv_diagonal_sign_write_address = None
        module.rwkv_diagonal_sign_captures = {}
        module.rwkv_rotary_query_address = None
        module.rwkv_rotary_write_address = None
        module.rwkv_rotary_read_captures = module.rwkv_diagonal_sign_captures
        installed.append(name)
    return {
        "modules": len(installed),
        "module_names": tuple(installed),
        "state_dim": int(state_dim),
        "head_size": int(head_size),
        "frequency": float(frequency),
        "address_dim": int(next(iter_delta_mem_modules(model))[1].projected_kv_key_dim),
        "parameters_per_layer": int(next(iter_delta_mem_modules(model))[1].projected_kv_key_dim) * int(state_dim),
        "parameter_tensors": len(installed),
        "parameter_elements": len(installed) * int(next(iter_delta_mem_modules(model))[1].projected_kv_key_dim) * int(state_dim),
        "binding_placement": "after_rwkv_v_projection_before_state_scan",
        "unbinding_placement": "per_slot_after_rwkv_matrix_read_before_group_norm_output",
        "diagonal_involution": True,
        "projected_carrier_changed": False,
    }


def set_enabled(model: torch.nn.Module, enabled: bool) -> None:
    for _, module in iter_delta_mem_modules(model):
        module.rwkv_diagonal_sign_enabled = bool(enabled)


def set_capture(model: torch.nn.Module, enabled: bool) -> None:
    for _, module in iter_delta_mem_modules(model):
        module.rwkv_diagonal_sign_capture_enabled = bool(enabled)
        module.rwkv_diagonal_sign_captures = {}
        module.rwkv_rotary_read_captures = module.rwkv_diagonal_sign_captures


def clear_transient(model: torch.nn.Module) -> None:
    for _, module in iter_delta_mem_modules(model):
        module.rwkv_diagonal_sign_query_address = None
        module.rwkv_diagonal_sign_write_address = None
        module.rwkv_diagonal_sign_captures = {}
        module.rwkv_rotary_query_address = None
        module.rwkv_rotary_write_address = None
        module.rwkv_rotary_read_captures = module.rwkv_diagonal_sign_captures


def named_binding_parameters(model: torch.nn.Module) -> tuple[tuple[str, torch.nn.Parameter], ...]:
    return tuple(sorted(((name, parameter) for name, parameter in model.named_parameters() if "rwkv_diagonal_sign_binding" in name), key=lambda item: item[0]))
