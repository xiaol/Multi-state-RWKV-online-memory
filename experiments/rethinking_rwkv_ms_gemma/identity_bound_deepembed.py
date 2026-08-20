"""Learned projected-value/RWKV identity binding for the DeepEmbed route."""

from __future__ import annotations

from types import MethodType
from typing import Any, Sequence

import torch
from torch import nn
from torch.nn import functional as F

from deltamem.core.delta import iter_delta_mem_modules
from experiments.rethinking_rwkv_ms_gemma import rwkv_projected_value_identity as value_identity


class StateIdentityBinder(nn.Module):
    """Score a projected value/state pair and gate a bounded RWKV correction."""

    def __init__(
        self,
        state_dim: int,
        *,
        hidden_dim: int = 8,
        temperature: float = 4.0,
        threshold: float = 0.0,
        bias: float = -2.0,
        max_gain: float = 0.125,
    ) -> None:
        super().__init__()
        if int(state_dim) < 1 or int(hidden_dim) < 1:
            raise ValueError("state and hidden dimensions must be positive")
        if not torch.isfinite(torch.tensor([temperature, threshold, bias, max_gain])).all():
            raise ValueError("binder hyperparameters must be finite")
        if float(temperature) <= 0.0 or float(max_gain) <= 0.0:
            raise ValueError("binder temperature and gain must be positive")
        self.state_dim = int(state_dim)
        self.hidden_dim = int(hidden_dim)
        self.temperature = float(temperature)
        self.threshold = float(threshold)
        self.bias = float(bias)
        self.max_gain = float(max_gain)
        self.query_map = nn.Linear(self.state_dim, self.state_dim, bias=False)
        self.state_map = nn.Linear(self.state_dim, self.state_dim, bias=False)
        self.pair_down = nn.Linear(2 * self.state_dim, self.hidden_dim, bias=False)
        self.pair_up = nn.Linear(self.hidden_dim, 1, bias=False)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        with torch.no_grad():
            self.query_map.weight.copy_(torch.eye(self.state_dim))
            self.state_map.weight.copy_(torch.eye(self.state_dim))
            self.pair_down.weight.zero_()
            self.pair_down.weight[:, : self.hidden_dim].copy_(
                torch.eye(self.hidden_dim)
            )
            self.pair_up.weight.fill_(0.01)

    def score(self, projected_value: torch.Tensor, recurrent_read: torch.Tensor) -> torch.Tensor:
        if projected_value.shape != recurrent_read.shape:
            raise ValueError("projected value and recurrent read shapes differ")
        if projected_value.shape[-1] != self.state_dim:
            raise ValueError("binder input width differs from state dimension")
        query = F.normalize(self.query_map(projected_value.float()), dim=-1, eps=1e-6)
        state = F.normalize(self.state_map(recurrent_read.float()), dim=-1, eps=1e-6)
        cosine = (query * state).sum(dim=-1)
        pair = torch.cat((query * state, (query - state).abs()), dim=-1)
        residual = self.pair_up(F.silu(self.pair_down(pair))).squeeze(-1)
        return cosine + residual

    def gate(self, score: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(
            self.temperature * (score - self.threshold) + self.bias
        )

    def correction(
        self,
        projected: torch.Tensor,
        recurrent_read: torch.Tensor,
        projected_value: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        score = self.score(projected_value, recurrent_read)
        gate = self.gate(score).unsqueeze(-1)
        recurrent_rms = recurrent_read.float().square().mean(dim=-1, keepdim=True).sqrt()
        direction = torch.tanh(recurrent_read.float() / recurrent_rms.clamp_min(1e-6))
        carrier_rms = projected.float().square().mean(dim=-1, keepdim=True).sqrt()
        correction = self.max_gain * carrier_rms * gate * direction
        return correction.to(dtype=projected.dtype), score, gate


def _module_key(module_name: str) -> str:
    return module_name.replace(".", "__")


def install(model: torch.nn.Module, *, device: torch.device | None = None) -> dict[str, Any]:
    modules = tuple(iter_delta_mem_modules(model))
    if not modules:
        raise ValueError("identity-bound DeepEmbed requires Delta-Mem modules")
    binders = nn.ModuleDict()
    names: list[str] = []
    for module_name, module in modules:
        if hasattr(module, "rwkv_identity_binder"):
            raise ValueError(f"identity binder already installed on {module_name}")
        binder = StateIdentityBinder(int(module.state_read_dim))
        if device is not None:
            binder = binder.to(device=device)
        binders[_module_key(module_name)] = binder
        module.rwkv_identity_binder_key = _module_key(module_name)
        module.rwkv_identity_last_score = None
        module.rwkv_identity_last_gate = None
        module.rwkv_identity_last_correction = None
        names.append(module_name)
    model.rwkv_identity_binder_bank = binders
    for module_name, module in modules:
        module.rwkv_identity_binder = binders[module.rwkv_identity_binder_key]
        original_fuse = module._fuse_projected_rwkv_reads

        def bound_fuse(
            current_module: Any,
            projected_reads: torch.Tensor,
            recurrent_reads: torch.Tensor,
            route_agreement: torch.Tensor | None = None,
            query_state_gate: torch.Tensor | None = None,
            global_recurrent_reads: torch.Tensor | None = None,
            hidden_states: torch.Tensor | None = None,
            *,
            _original_fuse: Any = original_fuse,
        ) -> torch.Tensor:
            fused = _original_fuse(
                projected_reads,
                recurrent_reads,
                route_agreement,
                query_state_gate,
                global_recurrent_reads,
                hidden_states,
            )
            query_value = getattr(
                current_module,
                "rwkv_query_state_identity_query_address",
                None,
            )
            if query_value is None:
                current_module.rwkv_identity_last_score = None
                current_module.rwkv_identity_last_gate = None
                current_module.rwkv_identity_last_correction = None
                return fused
            correction, score, gate = current_module.rwkv_identity_binder.correction(
                projected_reads,
                recurrent_reads,
                query_value,
            )
            current_module.rwkv_identity_last_score = score
            current_module.rwkv_identity_last_gate = gate
            current_module.rwkv_identity_last_correction = correction
            return fused + correction

        module._fuse_projected_rwkv_reads = MethodType(bound_fuse, module)
    return {
        "modules": len(names),
        "module_names": tuple(names),
        "forward_output_changed": True,
        "zero_state_exact_projected_only": True,
        "identity_target": "detached_projected_slot_value",
        "binder_parameters": sum(parameter.numel() for parameter in binders.parameters()),
        "parameters_per_layer": sum(
            parameter.numel() for parameter in next(iter(binders.values())).parameters()
        ),
    }


def score_tensor(
    captured: Sequence[value_identity.CapturedProjectedValueRead],
    labels: torch.Tensor,
    model: torch.nn.Module,
) -> torch.Tensor:
    if not captured:
        raise ValueError("identity binder capture is empty")
    valid = labels.ne(-100)
    if valid.ndim != 2 or not bool(valid.any().item()):
        raise ValueError("identity binder requires answer target positions")
    binders = getattr(model, "rwkv_identity_binder_bank", None)
    if binders is None:
        raise RuntimeError("identity binder bank is not installed")
    scores: list[torch.Tensor] = []
    for read in captured:
        binder = binders[_module_key(read.module_name)]
        score = binder.score(read.query_address, read.recurrent_read)
        if tuple(score.shape) != tuple(valid.shape):
            raise ValueError("identity binder score and labels differ")
        scores.append(score.masked_select(valid))
    result = torch.stack(scores)
    if not bool(torch.isfinite(result).all().item()):
        raise RuntimeError("identity binder score is non-finite")
    return result


def parameter_names(model: torch.nn.Module) -> tuple[str, ...]:
    return tuple(
        name
        for name, parameter in model.named_parameters()
        if "rwkv_identity_binder" in name and parameter.requires_grad
    )


def clear_runtime(model: torch.nn.Module) -> None:
    for _, module in iter_delta_mem_modules(model):
        module.rwkv_identity_last_score = None
        module.rwkv_identity_last_gate = None
        module.rwkv_identity_last_correction = None
