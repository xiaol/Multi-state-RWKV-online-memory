"""Runtime hooks for exact two-axis address binding in RWKV-MS."""

from __future__ import annotations

import hashlib
from types import MethodType
from typing import Any, Mapping

import torch

from deltamem.core.delta import iter_delta_mem_modules

from .rwkv_bidirectional_sign_binding import BidirectionalDiagonalSignBinding
from .rwkv_diagonal_sign_binding import deterministic_projection


def _tensor_sha256(value: torch.Tensor) -> str:
    tensor = value.detach().cpu().contiguous()
    digest = hashlib.sha256()
    digest.update(str(tensor.dtype).encode("ascii"))
    digest.update(str(tuple(tensor.shape)).encode("ascii"))
    digest.update(tensor.view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def _byte_equal(left: torch.Tensor, right: torch.Tensor) -> bool:
    return bool(
        torch.equal(
            left.detach().contiguous().view(torch.uint8),
            right.detach().contiguous().view(torch.uint8),
        )
    )


def _fold_address(
    module: Any,
    routes: torch.Tensor,
    *,
    sequence_length: int,
) -> torch.Tensor:
    keys = module.projected_kv_keys
    if keys is None:
        raise RuntimeError("Bidirectional sign binding requires projected slot keys")
    selected = torch.einsum("bts,bsd->btd", routes.float(), keys.float())
    if selected.shape[1] == 1 and sequence_length != 1:
        selected = selected.expand(-1, sequence_length, -1)
    if selected.shape[1] != sequence_length:
        raise ValueError("Bidirectional projected address length differs")
    return selected.detach()


def _write_address(module: Any, sequence_length: int) -> torch.Tensor:
    routes = module.last_write_routes
    keys = module.projected_kv_keys
    if routes is None or keys is None:
        raise RuntimeError("Bidirectional write binding requires routes and keys")
    if routes.ndim != 3 or keys.ndim != 3:
        raise ValueError("Bidirectional write route/key ranks differ")
    if routes.shape[0] != keys.shape[0] or routes.shape[-1] != keys.shape[1]:
        raise ValueError("Bidirectional write route/key shapes differ")
    if routes.shape[1] not in {1, sequence_length}:
        raise ValueError("Bidirectional write route length differs")
    if not bool(routes.float().abs().sum(dim=-1).gt(0.0).all().item()):
        raise RuntimeError("Bidirectional write route is empty")
    address = _fold_address(module, routes, sequence_length=sequence_length)
    if tuple(address.shape) != (
        keys.shape[0],
        sequence_length,
        module.projected_kv_key_dim,
    ):
        raise RuntimeError("Bidirectional write address shape differs")
    return address


def _slot_codes(
    module: Any,
    *,
    sequence_length: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    keys = module.projected_kv_keys
    if keys is None:
        raise RuntimeError("Bidirectional slot decode requires projected keys")
    addresses = keys.float().detach()
    left, right = module.rwkv_bidirectional_sign_binding.codes(addresses)
    return (
        left.unsqueeze(1).expand(-1, sequence_length, -1, -1),
        right.unsqueeze(1).expand(-1, sequence_length, -1, -1),
    )


def _decoded_slot_reads(
    module: Any,
    state: torch.Tensor,
    r_seq: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    batch_size, sequence_length, num_heads, head_size = r_seq.shape
    left, right = _slot_codes(module, sequence_length=sequence_length)
    num_slots = state.shape[2]
    left_heads = left.reshape(
        batch_size,
        sequence_length,
        num_slots,
        num_heads,
        head_size,
    ).permute(0, 1, 3, 2, 4)
    right_heads = right.reshape(
        batch_size,
        sequence_length,
        num_slots,
        num_heads,
        head_size,
    ).permute(0, 1, 3, 2, 4)
    bound_receptance = r_seq.float().unsqueeze(3) * right_heads
    encoded_reads = torch.einsum(
        "bhsij,bthsj->bthsi",
        state.float(),
        bound_receptance,
    )
    decoded = encoded_reads * left_heads
    return decoded, left, right


def _record(
    module: Any,
    kind: str,
    slot_addresses: torch.Tensor,
    raw: torch.Tensor,
    decoded: torch.Tensor,
    left_slot_codes: torch.Tensor,
    right_slot_codes: torch.Tensor,
    native_receptance: torch.Tensor,
) -> None:
    if not getattr(module, "rwkv_bidirectional_sign_capture_enabled", False):
        return
    write_left = module.rwkv_bidirectional_sign_write_left_code
    write_right = module.rwkv_bidirectional_sign_write_right_code
    if write_left is None or write_right is None:
        raise RuntimeError("Bidirectional read capture has no write codes")
    module.rwkv_bidirectional_sign_captures[kind] = {
        "slot_addresses": slot_addresses.detach().clone(),
        "raw": raw.detach().clone(),
        "decoded": decoded.detach().clone(),
        "slot_codes": left_slot_codes.detach().clone(),
        "write_codes": write_left.detach().clone(),
        "right_slot_codes": right_slot_codes.detach().clone(),
        "right_write_codes": write_right.detach().clone(),
        "native_receptance": native_receptance.detach().clone(),
    }


def _write_features(
    module: Any,
    k: torch.Tensor,
    v: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    address_seq: torch.Tensor,
    token_mask: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    features = module.rwkv_bidirectional_sign_original_write_features(
        k,
        v,
        a,
        b,
        address_seq,
        token_mask,
    )
    full_address = _write_address(module, address_seq.shape[1])
    left, right = module.rwkv_bidirectional_sign_binding.codes(full_address)
    module.rwkv_bidirectional_sign_write_address = full_address
    module.rwkv_bidirectional_sign_write_left_code = left.detach()
    module.rwkv_bidirectional_sign_write_right_code = right.detach()
    module.rwkv_rotary_write_address = full_address
    if not module.rwkv_bidirectional_sign_enabled:
        return features
    bound = module.rwkv_bidirectional_sign_binding.bind_features(
        full_address,
        *features,
    )
    if not all(bool(torch.isfinite(value).all().item()) for value in bound):
        raise RuntimeError("Bidirectional bound write features are non-finite")
    return bound


def _projected_slot_write(
    module: Any,
    hidden_states: torch.Tensor,
    token_mask: torch.Tensor | None,
) -> None:
    if module.rwkv_bidirectional_sign_pending_rebase is not None:
        raise RuntimeError("Bidirectional slot-key rebase was delayed past its scan")
    old_keys = module.projected_kv_keys
    old_occupied = module.projected_kv_occupied
    if (old_keys is None) != (old_occupied is None):
        raise RuntimeError("Bidirectional projected slot sidecars are incomplete")
    old_keys = None if old_keys is None else old_keys.detach().clone()
    old_occupied = (
        None if old_occupied is None else old_occupied.detach().clone().to(torch.bool)
    )
    module.rwkv_bidirectional_sign_original_projected_slot_write(
        hidden_states,
        token_mask,
    )
    routes = module.last_write_routes
    new_keys = module.projected_kv_keys
    new_occupied = module.projected_kv_occupied
    if routes is None:
        return
    if new_keys is None or new_occupied is None:
        raise RuntimeError("Bidirectional projected slot write omitted sidecars")
    selected = routes.detach().to(torch.bool).any(dim=1)
    if tuple(selected.shape) != tuple(new_occupied.shape):
        raise RuntimeError("Bidirectional projected write routes differ from slot state")
    if old_keys is None:
        old_keys = torch.zeros_like(new_keys)
        old_occupied = torch.zeros_like(new_occupied, dtype=torch.bool)
    assert old_occupied is not None
    if tuple(old_keys.shape) != tuple(new_keys.shape):
        raise RuntimeError("Bidirectional old/new projected key shapes differ")
    unselected = ~selected
    if not _byte_equal(new_keys[unselected], old_keys[unselected]):
        raise RuntimeError("Bidirectional projected write changed an unselected key")
    key_changed = new_keys.detach().ne(old_keys).any(dim=-1)
    module.rwkv_bidirectional_sign_pending_rebase = {
        "old_keys": old_keys,
        "new_keys": new_keys.detach().clone(),
        "selected": selected,
        "rebase": selected & old_occupied & key_changed,
        "insert": selected & ~old_occupied,
    }


def _backend_scan(
    module: Any,
    state: torch.Tensor,
    memory_q_seq: torch.Tensor,
    memory_k_seq: torch.Tensor,
    memory_v_seq: torch.Tensor,
    beta_seq: torch.Tensor,
    lambda_seq: torch.Tensor,
    write_route_seq: torch.Tensor | None = None,
    rwkv_write_route_seq: torch.Tensor | None = None,
    rwkv_write_address_seq: torch.Tensor | None = None,
    read_route_seq: torch.Tensor | None = None,
    token_mask: torch.Tensor | None = None,
    write_only: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    pending = module.rwkv_bidirectional_sign_pending_rebase
    if pending is not None and not write_only:
        raise RuntimeError("Bidirectional slot-key rebase reached a non-write scan")
    scan_state = state
    if pending is not None:
        module.rwkv_bidirectional_sign_pending_rebase = None
        selected = pending["selected"]
        insert = pending["insert"]
        rebase = pending["rebase"]
        if state.ndim != 5 or tuple(state.shape[:1] + state.shape[2:3]) != tuple(
            selected.shape
        ):
            raise RuntimeError("Bidirectional recurrent state differs from projected slots")
        insert_state = state.masked_select(
            insert.unsqueeze(1).unsqueeze(-1).unsqueeze(-1)
        )
        if insert_state.numel() and not bool(insert_state.eq(0).all().item()):
            raise RuntimeError("Bidirectional inserted slot contains stale recurrent state")
        if module.rwkv_bidirectional_sign_enabled:
            heads = state.shape[1]
            old_addresses = pending["old_keys"].unsqueeze(1).expand(
                -1, heads, -1, -1
            )
            new_addresses = pending["new_keys"].unsqueeze(1).expand(
                -1, heads, -1, -1
            )
            candidate = module.rwkv_bidirectional_sign_binding.rebase_state(
                old_addresses,
                new_addresses,
                state,
            )
            mask = rebase.unsqueeze(1).unsqueeze(-1).unsqueeze(-1)
            scan_state = torch.where(mask, candidate, state)
        unchanged = ~rebase
        if not _byte_equal(
            scan_state.masked_select(unchanged.unsqueeze(1).unsqueeze(-1).unsqueeze(-1)),
            state.masked_select(unchanged.unsqueeze(1).unsqueeze(-1).unsqueeze(-1)),
        ):
            raise RuntimeError("Bidirectional rebase changed an unselected state slot")
        if module.rwkv_bidirectional_sign_enabled:
            module.rwkv_bidirectional_sign_rebase_events += int(rebase.sum().item())
        if module.rwkv_bidirectional_sign_capture_enabled:
            module.rwkv_bidirectional_sign_rebase_capture = {
                key: value.detach().clone()
                for key, value in pending.items()
            }
            module.rwkv_bidirectional_sign_rebase_capture.update(
                {
                    "state_before": state.detach().clone(),
                    "state_after": scan_state.detach().clone(),
                }
            )
    return module.rwkv_bidirectional_sign_original_backend_scan(
        scan_state,
        memory_q_seq,
        memory_k_seq,
        memory_v_seq,
        beta_seq,
        lambda_seq,
        write_route_seq=write_route_seq,
        rwkv_write_route_seq=rwkv_write_route_seq,
        rwkv_write_address_seq=rwkv_write_address_seq,
        read_route_seq=read_route_seq,
        token_mask=token_mask,
        write_only=write_only,
    )


def _read_basis(
    module: Any,
    state: torch.Tensor,
    memory_source_seq: torch.Tensor,
    token_mask: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    r_seq, raw_slots, readout_gate = module.rwkv_bidirectional_sign_original_read_basis(
        state,
        memory_source_seq,
        token_mask,
    )
    sequence_length = memory_source_seq.shape[1]
    left, right = _slot_codes(module, sequence_length=sequence_length)
    decoded_slots = raw_slots
    if module.rwkv_bidirectional_sign_enabled:
        decoded_slots, left, right = _decoded_slot_reads(module, state, r_seq)
    kind = module.rwkv_bidirectional_sign_read_kind
    if kind is None:
        raise RuntimeError("Bidirectional read basis has no explicit call-site tag")
    slot_addresses = module.projected_kv_keys
    if slot_addresses is None:
        raise RuntimeError("Bidirectional read basis has no projected slot addresses")
    _record(
        module,
        kind,
        slot_addresses,
        raw_slots,
        decoded_slots,
        left,
        right,
        r_seq,
    )
    return r_seq, decoded_slots, readout_gate


def _addressed_reads(
    module: Any,
    state: torch.Tensor,
    memory_source_seq: torch.Tensor,
    projected_routes: torch.Tensor,
    token_mask: torch.Tensor | None,
) -> torch.Tensor:
    sequence = getattr(module, "rwkv_bidirectional_sign_read_sequence", None)
    if sequence is None:
        sequence = []
        module.rwkv_bidirectional_sign_read_sequence = sequence
    if sequence:
        raise RuntimeError("Bidirectional addressed read call order differs")
    sequence.append("addressed")
    sequence_length = memory_source_seq.shape[1]
    module.rwkv_bidirectional_sign_query_address = _fold_address(
        module,
        projected_routes,
        sequence_length=sequence_length,
    )
    module.rwkv_bidirectional_sign_read_kind = "addressed"
    try:
        result = module.rwkv_bidirectional_sign_original_addressed_reads(
            state,
            memory_source_seq,
            projected_routes,
            token_mask,
        )
        if getattr(module, "rwkv_bidirectional_sign_capture_enabled", False):
            module.rwkv_bidirectional_sign_captures["addressed"]["routes"] = (
                module.last_read_routes.detach().clone()
            )
        return result
    finally:
        module.rwkv_bidirectional_sign_read_kind = None


def _global_reads(
    module: Any,
    state: torch.Tensor,
    memory_source_seq: torch.Tensor,
    token_mask: torch.Tensor | None,
) -> torch.Tensor:
    _, sequence_length, _ = memory_source_seq.shape
    if sequence_length == 0:
        return module.rwkv_bidirectional_sign_original_global_reads(
            state,
            memory_source_seq,
            token_mask,
        )
    sequence = getattr(module, "rwkv_bidirectional_sign_read_sequence", None)
    if sequence != ["addressed"]:
        raise RuntimeError("Bidirectional global read call order differs")
    sequence.append("global")
    module.rwkv_bidirectional_sign_read_kind = "global"
    try:
        result = module.rwkv_bidirectional_sign_original_global_reads(
            state,
            memory_source_seq,
            token_mask,
        )
        if getattr(module, "rwkv_bidirectional_sign_capture_enabled", False):
            module.rwkv_bidirectional_sign_captures["global"]["routes"] = (
                module.last_read_routes.detach().clone()
            )
        return result
    finally:
        module.rwkv_bidirectional_sign_read_kind = None
        module.rwkv_bidirectional_sign_read_sequence = []


def install(
    model: torch.nn.Module,
    *,
    state_dim: int = 32,
    head_size: int = 32,
    seed: int = 131,
    frequency: float = 64.0,
    trainable_projection: bool = False,
    expected_projection_sha256: Mapping[str, Mapping[str, str]] | None = None,
) -> Mapping[str, Any]:
    if trainable_projection:
        raise ValueError("Bidirectional sign projections are frozen by protocol")
    modules = tuple(iter_delta_mem_modules(model))
    if not modules:
        raise ValueError("Bidirectional sign binding requires Delta-Mem modules")
    installed: list[str] = []
    projection_sha256: dict[str, Mapping[str, str]] = {}
    for index, (name, module) in enumerate(modules):
        if (
            int(module.state_read_dim) != int(state_dim)
            or int(module.num_state_heads) != 1
            or int(module.rank) != int(head_size)
        ):
            raise ValueError("Bidirectional sign protocol requires one width-32 RWKV head")
        if module.rwkv_ms_hybrid_mode != "address_keyed_moe_deepembed_ffn":
            raise ValueError("Bidirectional sign binding requires exact-v5 hybrid mode")
        if int(getattr(module, "rwkv_ms_anchor_interval", 0)) != 0:
            raise ValueError("Bidirectional sign binding does not yet encode historical anchors")
        address_dim = int(module.projected_kv_key_dim)
        left_projection = deterministic_projection(
            address_dim,
            int(seed) + 2 * index,
            state_dim,
        )
        right_projection = deterministic_projection(
            address_dim,
            int(seed) + 2 * index + 1,
            state_dim,
        )
        hashes = {
            "left": _tensor_sha256(left_projection),
            "right": _tensor_sha256(right_projection),
        }
        if hashes["left"] == hashes["right"]:
            raise RuntimeError("Bidirectional left/right projections are tied")
        if expected_projection_sha256 is not None and dict(
            expected_projection_sha256.get(name, {})
        ) != hashes:
            raise ValueError(f"Bidirectional projection manifest differs for {name}")
        binder = BidirectionalDiagonalSignBinding(
            state_dim,
            address_dim=address_dim,
            left_projection=left_projection,
            right_projection=right_projection,
            frequency=frequency,
            trainable_projection=False,
        ).to(next(module.parameters()).device)
        module.add_module("rwkv_bidirectional_sign_binding", binder)
        module.rwkv_bidirectional_sign_original_write_features = (
            module._rwkv_ms_address_conditioned_write_features
        )
        module.rwkv_bidirectional_sign_original_read_basis = (
            module._rwkv_ms_token_state_read_basis
        )
        module.rwkv_bidirectional_sign_original_addressed_reads = (
            module._rwkv_ms_addressed_token_state_reads
        )
        module.rwkv_bidirectional_sign_original_global_reads = (
            module._rwkv_ms_token_state_reads
        )
        module.rwkv_bidirectional_sign_original_projected_slot_write = (
            module._write_projected_kv_slots
        )
        module.rwkv_bidirectional_sign_original_backend_scan = (
            module._memory_backend_scan
        )
        module._rwkv_ms_address_conditioned_write_features = MethodType(
            _write_features,
            module,
        )
        module._rwkv_ms_token_state_read_basis = MethodType(_read_basis, module)
        module._rwkv_ms_addressed_token_state_reads = MethodType(
            _addressed_reads,
            module,
        )
        module._rwkv_ms_token_state_reads = MethodType(_global_reads, module)
        module._write_projected_kv_slots = MethodType(_projected_slot_write, module)
        module._memory_backend_scan = MethodType(_backend_scan, module)
        module.rwkv_bidirectional_sign_enabled = True
        module.rwkv_bidirectional_sign_capture_enabled = False
        module.rwkv_bidirectional_sign_read_kind = None
        module.rwkv_bidirectional_sign_read_sequence = []
        module.rwkv_bidirectional_sign_query_address = None
        module.rwkv_bidirectional_sign_write_address = None
        module.rwkv_bidirectional_sign_write_left_code = None
        module.rwkv_bidirectional_sign_write_right_code = None
        module.rwkv_bidirectional_sign_captures = {}
        module.rwkv_bidirectional_sign_pending_rebase = None
        module.rwkv_bidirectional_sign_rebase_events = 0
        module.rwkv_bidirectional_sign_rebase_capture = None
        module.rwkv_rotary_write_address = None
        module.rwkv_rotary_read_captures = module.rwkv_bidirectional_sign_captures
        installed.append(name)
        projection_sha256[name] = hashes
    if expected_projection_sha256 is not None and set(expected_projection_sha256) != set(
        installed
    ):
        raise ValueError("Bidirectional projection manifest module names differ")
    return {
        "modules": len(installed),
        "module_names": tuple(installed),
        "state_dim": int(state_dim),
        "head_size": int(head_size),
        "frequency": float(frequency),
        "address_dim": int(modules[0][1].projected_kv_key_dim),
        "parameters_per_layer": 2
        * int(modules[0][1].projected_kv_key_dim)
        * int(state_dim),
        "parameter_tensors": 2 * len(installed),
        "parameter_elements": 2
        * len(installed)
        * int(modules[0][1].projected_kv_key_dim)
        * int(state_dim),
        "binding_placement": "v-left and k/a/b-right before exact RWKV scan",
        "unbinding_placement": "single read-basis hook right-codes r and left-decodes every per-slot read before native routing",
        "outer_read_hooks_change_math": False,
        "read_call_sites_explicitly_tagged": True,
        "left_and_right_codes_independent": True,
        "projections_trainable": False,
        "projection_sha256": projection_sha256,
        "projected_carrier_changed": False,
        "state_rebase_on_slot_address_change_implemented": True,
        "rebase_boundary": "selected occupied projected-key changes immediately before write scan",
    }


def set_enabled(model: torch.nn.Module, enabled: bool) -> None:
    for _, module in iter_delta_mem_modules(model):
        current = bool(module.rwkv_bidirectional_sign_enabled)
        state = module.delta_state
        if current != bool(enabled) and state is not None and bool(state.ne(0).any().item()):
            raise RuntimeError("Cannot toggle bidirectional binding with live recurrent state")
        module.rwkv_bidirectional_sign_enabled = bool(enabled)


def set_capture(model: torch.nn.Module, enabled: bool) -> None:
    for _, module in iter_delta_mem_modules(model):
        module.rwkv_bidirectional_sign_capture_enabled = bool(enabled)
        module.rwkv_bidirectional_sign_captures = {}
        module.rwkv_bidirectional_sign_read_sequence = []
        module.rwkv_bidirectional_sign_pending_rebase = None
        module.rwkv_bidirectional_sign_rebase_capture = None
        module.rwkv_rotary_read_captures = module.rwkv_bidirectional_sign_captures


def clear_read_capture(model: torch.nn.Module) -> None:
    for _, module in iter_delta_mem_modules(model):
        if module.rwkv_bidirectional_sign_pending_rebase is not None:
            raise RuntimeError("Cannot clear read capture with a pending slot-key rebase")
        module.rwkv_bidirectional_sign_read_kind = None
        module.rwkv_bidirectional_sign_read_sequence = []
        module.rwkv_bidirectional_sign_query_address = None
        module.rwkv_bidirectional_sign_captures = {}
        module.rwkv_rotary_read_captures = module.rwkv_bidirectional_sign_captures


def clear_transient(model: torch.nn.Module) -> None:
    for _, module in iter_delta_mem_modules(model):
        if module.rwkv_bidirectional_sign_pending_rebase is not None:
            raise RuntimeError("Cannot clear a pending bidirectional slot-key rebase")
        module.rwkv_bidirectional_sign_read_kind = None
        module.rwkv_bidirectional_sign_read_sequence = []
        module.rwkv_bidirectional_sign_query_address = None
        module.rwkv_bidirectional_sign_write_address = None
        module.rwkv_bidirectional_sign_write_left_code = None
        module.rwkv_bidirectional_sign_write_right_code = None
        module.rwkv_bidirectional_sign_captures = {}
        module.rwkv_bidirectional_sign_rebase_capture = None
        module.rwkv_rotary_write_address = None
        module.rwkv_rotary_read_captures = module.rwkv_bidirectional_sign_captures
