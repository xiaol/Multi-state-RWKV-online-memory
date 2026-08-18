from __future__ import annotations

from types import MethodType
from typing import Any, Mapping

import torch
import torch.nn.functional as F

from deltamem.core.delta import iter_delta_mem_modules


FEATURE_NAMES = ("k", "v", "a", "b")


def _conditioned_features(
    module: Any,
    k: torch.Tensor,
    v: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    address_seq: torch.Tensor,
    token_mask: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    expected_shape = (*k.shape[:-1], module.state_read_dim)
    if tuple(address_seq.shape) != expected_shape:
        raise ValueError(
            "Learned RWKV write addresses must match the write feature shape: "
            f"expected={expected_shape} actual={tuple(address_seq.shape)}"
        )
    address = address_seq.to(device=k.device, dtype=torch.float32)
    address_square_mean = address.square().mean(dim=-1, keepdim=True)
    active = address_square_mean.gt(0.0)
    address_rms = (address_square_mean + 1e-12).sqrt()
    if token_mask is not None:
        expected_mask_shape = k.shape[:2]
        if tuple(token_mask.shape) != expected_mask_shape:
            raise ValueError(
                "Learned RWKV token mask must match the write sequence: "
                f"expected={expected_mask_shape} actual={tuple(token_mask.shape)}"
            )
        active = active & token_mask.to(device=k.device, dtype=torch.bool).unsqueeze(-1)
    direction = torch.tanh(address / address_rms.clamp_min(1e-6))
    gain = float(module.rwkv_ms_write_address_gain)

    def learned_delta(feature_name: str) -> torch.Tensor:
        down = getattr(module, f"rwkv_learned_write_{feature_name}_down")
        up = getattr(module, f"rwkv_learned_write_{feature_name}_up")
        hidden = F.linear(direction, down.float())
        return torch.tanh(F.linear(hidden, up.float()))

    def additive(feature: torch.Tensor, feature_name: str) -> torch.Tensor:
        feature_float = feature.float()
        feature_rms = (feature_float.square().mean(dim=-1, keepdim=True) + 1e-12).sqrt()
        candidate = feature_float + gain * feature_rms * learned_delta(feature_name)
        return torch.where(active, candidate, feature_float).to(dtype=feature.dtype)

    def multiplicative(feature: torch.Tensor, feature_name: str) -> torch.Tensor:
        feature_float = feature.float()
        candidate = feature_float * (1.0 + gain * learned_delta(feature_name))
        return torch.where(active, candidate, feature_float).to(dtype=feature.dtype)

    return additive(k, "k"), additive(v, "v"), multiplicative(a, "a"), multiplicative(b, "b")


def install(model: torch.nn.Module, *, rank: int = 2) -> Mapping[str, Any]:
    if rank < 1:
        raise ValueError("Learned RWKV write rank must be positive")
    modules = tuple(iter_delta_mem_modules(model))
    if not modules:
        raise ValueError("Learned RWKV write installation requires Delta-Mem modules")
    installed_names: list[str] = []
    parameter_count = 0
    for module_name, module in modules:
        if hasattr(module, "rwkv_learned_write_k_down"):
            raise ValueError(f"Learned RWKV write is already installed on {module_name}")
        module.rwkv_learned_write_rank = int(rank)
        for feature_name in FEATURE_NAMES:
            down_name = f"rwkv_learned_write_{feature_name}_down"
            up_name = f"rwkv_learned_write_{feature_name}_up"
            down = torch.empty(rank, module.state_read_dim, device=module.memory_v_proj.device)
            up = torch.zeros(module.state_read_dim, rank, device=module.memory_v_proj.device)
            torch.nn.init.normal_(down, mean=0.0, std=1.0 / module.state_read_dim**0.5)
            module.register_parameter(down_name, torch.nn.Parameter(down.float()))
            module.register_parameter(up_name, torch.nn.Parameter(up.float()))
            installed_names.extend((f"{module_name}.{down_name}", f"{module_name}.{up_name}"))
            parameter_count += down.numel() + up.numel()
        module.rwkv_learned_write_original = module._rwkv_ms_address_conditioned_write_features
        module._rwkv_ms_address_conditioned_write_features = MethodType(
            _conditioned_features,
            module,
        )
    return {
        "modules": len(modules),
        "rank": int(rank),
        "parameter_tensors": len(installed_names),
        "parameter_elements": parameter_count,
        "parameter_names": tuple(installed_names),
        "initialized_as_exact_noop": True,
    }


def parameter_suffixes() -> tuple[str, ...]:
    return tuple(
        f".rwkv_learned_write_{feature_name}_{direction}"
        for feature_name in FEATURE_NAMES
        for direction in ("down", "up")
    )
