"""Shadow-certified live-value residual for the trained RWKV DeepEmbed path."""

from __future__ import annotations

import copy
from types import MethodType
from typing import Any, Mapping, Sequence

import torch
from torch import nn
from torch.nn import functional as F

from deltamem.core.delta import iter_delta_mem_modules
from experiments.rethinking_rwkv_ms_gemma.rwkv_query_state_bilinear import (
    ResidualBilinearIdentity,
)


class ShadowCertifiedLiveValue(nn.Module):
    """Use a detached shadow for identity and the live RWKV read for value."""

    def __init__(
        self,
        state_dim: int,
        *,
        identity: ResidualBilinearIdentity,
        threshold: float,
        temperature: float = 8.0,
        max_gain: float = 0.125,
    ) -> None:
        super().__init__()
        if int(state_dim) < 1:
            raise ValueError("state_dim must be positive")
        values = torch.tensor([threshold, temperature, max_gain], dtype=torch.float32)
        if not bool(torch.isfinite(values).all().item()):
            raise ValueError("shadow-certified binder constants must be finite")
        if float(temperature) <= 0.0 or not 0.0 < float(max_gain) <= 1.0:
            raise ValueError("temperature and max_gain are outside their bounds")
        if int(identity.state_dim) != int(state_dim):
            raise ValueError("identity head width differs from the live-value binder")
        self.state_dim = int(state_dim)
        self.temperature = float(temperature)
        self.max_gain = float(max_gain)
        self.identity = identity
        for parameter in self.identity.parameters():
            parameter.requires_grad_(False)
        self.register_buffer(
            "threshold",
            torch.tensor(float(threshold), dtype=torch.float32),
            persistent=True,
        )
        self.value_map = nn.Linear(self.state_dim, self.state_dim, bias=False)
        with torch.no_grad():
            self.value_map.weight.copy_(torch.eye(self.state_dim))

    @staticmethod
    def _exact_rms(value: torch.Tensor) -> torch.Tensor:
        square_mean = value.float().square().mean(dim=-1, keepdim=True)
        return torch.where(
            square_mean.gt(0.0),
            (square_mean + 1e-12).sqrt(),
            square_mean,
        )

    def score(self, query: torch.Tensor, shadow: torch.Tensor) -> torch.Tensor:
        if query.shape != shadow.shape or query.shape[-1] != self.state_dim:
            raise ValueError("shadow-certified query and shadow shapes differ")
        return self.identity.score(query.detach(), shadow.detach())

    def gate(self, score: torch.Tensor) -> torch.Tensor:
        return self.max_gain * torch.sigmoid(
            self.temperature * (score.float() - self.threshold)
        )

    def forward(
        self,
        projected: torch.Tensor,
        query: torch.Tensor,
        shadow: torch.Tensor,
        live_recurrent: torch.Tensor,
        *,
        gate_override: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        shapes = {
            tuple(projected.shape),
            tuple(query.shape),
            tuple(shadow.shape),
            tuple(live_recurrent.shape),
        }
        if len(shapes) != 1 or projected.shape[-1] != self.state_dim:
            raise ValueError("shadow-certified fusion inputs must share the state width")
        score = self.score(query, shadow)
        gate = self.gate(score).unsqueeze(-1)
        if gate_override is not None:
            if tuple(gate_override.shape) != tuple(gate.shape):
                raise ValueError("shadow-certified gate override shape differs")
            gate = gate_override.to(device=gate.device, dtype=gate.dtype)
        mapped_live = self.value_map(live_recurrent.float())
        live_rms = self._exact_rms(mapped_live)
        live_direction = torch.tanh(mapped_live / live_rms.clamp_min(1e-6))
        live_present = live_recurrent.float().square().sum(
            dim=-1,
            keepdim=True,
        ).gt(0.0)
        live_direction = torch.where(
            live_present,
            live_direction,
            torch.zeros_like(live_direction),
        )
        carrier_rms = self._exact_rms(projected)
        correction = carrier_rms * gate * live_direction
        if not bool(
            torch.isfinite(
                torch.cat(
                    (
                        correction.reshape(-1),
                        score.reshape(-1),
                        gate.reshape(-1),
                        mapped_live.reshape(-1),
                    )
                )
            ).all().item()
        ):
            raise RuntimeError("shadow-certified live-value output is non-finite")
        return correction.to(dtype=projected.dtype), score, gate, mapped_live

    def audit_payload(self) -> Mapping[str, Any]:
        return {
            "architecture": "shadow_certified_live_rwkv_value_residual",
            "identity_source": "detached_binder_disabled_v5_shadow",
            "value_source": "live_current_condition_rwkv_read",
            "base_fusion": "original_v5_fusion_preserved_then_additive_correction",
            "identity_parameters_frozen": True,
            "value_map": "bias_free_identity_initialized",
            "state_dim": self.state_dim,
            "temperature": self.temperature,
            "threshold": float(self.threshold.item()),
            "max_gain": self.max_gain,
            "zero_live_exact_no_correction": True,
        }


def _threshold_values(
    thresholds: Sequence[float] | torch.Tensor,
    *,
    layers: int,
) -> tuple[float, ...]:
    tensor = torch.as_tensor(thresholds, dtype=torch.float32).reshape(-1)
    if tensor.numel() != layers or not bool(torch.isfinite(tensor).all().item()):
        raise ValueError("shadow-certified thresholds must provide one finite value per layer")
    return tuple(float(value) for value in tensor.tolist())


def install(
    model: torch.nn.Module,
    head: Any,
    *,
    thresholds: Sequence[float] | torch.Tensor,
    temperature: float = 8.0,
    max_gain: float = 0.125,
) -> Mapping[str, Any]:
    modules = tuple(iter_delta_mem_modules(model))
    heads = tuple(head.heads)
    threshold_values = _threshold_values(thresholds, layers=len(modules))
    if not modules or len(heads) != len(modules):
        raise ValueError("shadow-certified modules and identity heads differ")
    names: list[str] = []
    for index, (module_name, module) in enumerate(modules):
        if hasattr(module, "rwkv_v5_shadow_certified_binder"):
            raise ValueError(f"shadow-certified binder already installed on {module_name}")
        identity = copy.deepcopy(heads[index])
        binder = ShadowCertifiedLiveValue(
            int(module.state_read_dim),
            identity=identity,
            threshold=threshold_values[index],
            temperature=temperature,
            max_gain=max_gain,
        ).to(device=module.memory_v_proj.device, dtype=torch.float32)
        module.rwkv_v5_shadow_certified_binder = binder
        module.rwkv_v5_shadow_certified_enabled = False
        module.rwkv_v5_shadow_query = None
        module.rwkv_v5_shadow_state = None
        module.rwkv_v5_shadow_gate_override = None
        module.rwkv_v5_shadow_correction_mask = None
        module.rwkv_v5_shadow_live_value_override = None
        module.rwkv_v5_shadow_last_score = None
        module.rwkv_v5_shadow_last_gate = None
        module.rwkv_v5_shadow_last_value = None
        module.rwkv_v5_shadow_last_correction = None
        original_fuse = module._fuse_projected_rwkv_reads

        def fused_with_shadow_certificate(
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
            if not current_module.rwkv_v5_shadow_certified_enabled:
                current_module.rwkv_v5_shadow_last_score = None
                current_module.rwkv_v5_shadow_last_gate = None
                current_module.rwkv_v5_shadow_last_value = None
                current_module.rwkv_v5_shadow_last_correction = None
                return fused
            query = current_module.rwkv_v5_shadow_query
            shadow = current_module.rwkv_v5_shadow_state
            if query is None or shadow is None:
                raise RuntimeError("enabled shadow-certified fusion is missing runtime inputs")
            live_value = current_module.rwkv_v5_shadow_live_value_override
            if live_value is None:
                live_value = recurrent_reads
            correction, score, gate, value = (
                current_module.rwkv_v5_shadow_certified_binder(
                    projected_reads,
                    query,
                    shadow,
                    live_value,
                    gate_override=current_module.rwkv_v5_shadow_gate_override,
                )
            )
            correction_mask = current_module.rwkv_v5_shadow_correction_mask
            if correction_mask is not None:
                if tuple(correction_mask.shape) != tuple(correction.shape[:-1]):
                    raise ValueError("shadow-certified correction mask shape differs")
                correction = correction * correction_mask.to(
                    device=correction.device,
                    dtype=correction.dtype,
                ).unsqueeze(-1)
            current_module.rwkv_v5_shadow_last_score = score
            current_module.rwkv_v5_shadow_last_gate = gate
            current_module.rwkv_v5_shadow_last_value = value
            current_module.rwkv_v5_shadow_last_correction = correction
            return fused + correction.to(dtype=fused.dtype)

        module._fuse_projected_rwkv_reads = MethodType(
            fused_with_shadow_certificate,
            module,
        )
        names.append(module_name)
    return {
        "modules": len(names),
        "module_names": tuple(names),
        "base_fusion_preserved": True,
        "identity_stream_detached": True,
        "live_rwkv_value_stream": True,
        "identity_parameters_frozen": True,
        "trainable_value_map_tensors": len(names),
        "trainable_value_map_elements": sum(
            module.rwkv_v5_shadow_certified_binder.value_map.weight.numel()
            for _, module in modules
        ),
        "token_correction_mask_supported": True,
        "detached_live_value_override_supported": True,
        "zero_live_exact_base_fusion": True,
    }


def set_runtime(
    modules: Sequence[tuple[str, Any]],
    *,
    queries: Mapping[str, torch.Tensor],
    shadows: Mapping[str, torch.Tensor],
    seq_len: int,
    gate_overrides: Mapping[str, torch.Tensor] | None = None,
    correction_masks: Mapping[str, torch.Tensor] | None = None,
    live_value_overrides: Mapping[str, torch.Tensor] | None = None,
) -> None:
    if int(seq_len) < 1:
        raise ValueError("shadow-certified runtime sequence length must be positive")
    for module_name, module in modules:
        query = queries[module_name]
        shadow = shadows[module_name]
        expected = (query.shape[0], int(seq_len), int(module.state_read_dim))
        if tuple(query.shape) == (query.shape[0], 1, int(module.state_read_dim)):
            query = query.expand(-1, int(seq_len), -1)
        if tuple(query.shape) != expected or tuple(shadow.shape) != expected:
            raise ValueError(f"shadow-certified runtime shape differs for {module_name}")
        module.rwkv_v5_shadow_query = query.detach()
        module.rwkv_v5_shadow_state = shadow.detach()
        module.rwkv_v5_shadow_gate_override = (
            None if gate_overrides is None else gate_overrides[module_name]
        )
        correction_mask = (
            None if correction_masks is None else correction_masks[module_name]
        )
        if correction_mask is not None and tuple(correction_mask.shape) != expected[:2]:
            raise ValueError(
                f"shadow-certified correction mask differs for {module_name}"
            )
        module.rwkv_v5_shadow_correction_mask = correction_mask
        live_value_override = (
            None
            if live_value_overrides is None
            else live_value_overrides[module_name]
        )
        if live_value_override is not None and tuple(live_value_override.shape) != expected:
            raise ValueError(
                f"shadow-certified live-value override differs for {module_name}"
            )
        module.rwkv_v5_shadow_live_value_override = (
            None if live_value_override is None else live_value_override.detach()
        )
        module.rwkv_v5_shadow_certified_enabled = True


def clear_runtime(
    modules: Sequence[tuple[str, Any]],
    *,
    disable: bool = True,
) -> None:
    for _, module in modules:
        if disable:
            module.rwkv_v5_shadow_certified_enabled = False
        module.rwkv_v5_shadow_query = None
        module.rwkv_v5_shadow_state = None
        module.rwkv_v5_shadow_gate_override = None
        module.rwkv_v5_shadow_correction_mask = None
        module.rwkv_v5_shadow_live_value_override = None
        module.rwkv_v5_shadow_last_score = None
        module.rwkv_v5_shadow_last_gate = None
        module.rwkv_v5_shadow_last_value = None
        module.rwkv_v5_shadow_last_correction = None


def configure_trainable_value_maps(
    model: torch.nn.Module,
) -> tuple[tuple[tuple[str, torch.nn.Parameter], ...], Mapping[str, Any]]:
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    selected: list[tuple[str, torch.nn.Parameter]] = []
    for name, parameter in model.named_parameters():
        if name.endswith("rwkv_v5_shadow_certified_binder.value_map.weight"):
            parameter.requires_grad_(True)
            selected.append((name, parameter))
    selected = sorted(selected)
    return tuple(selected), {
        "parameter_tensors": len(selected),
        "parameter_elements": sum(parameter.numel() for _, parameter in selected),
        "identity_parameters_frozen": True,
        "only_live_value_maps_trainable": True,
    }


def install_shifted_feedback(
    model: torch.nn.Module,
    *,
    gain: float = 0.03125,
) -> Mapping[str, Any]:
    if not torch.isfinite(torch.tensor(float(gain))) or not 0.0 < float(gain) <= 1.0:
        raise ValueError("shifted feedback gain must be finite and in (0, 1]")
    modules = tuple(iter_delta_mem_modules(model))
    if not modules:
        raise ValueError("shifted feedback requires Delta-Mem modules")
    names: list[str] = []
    for module_name, module in modules:
        if hasattr(module, "rwkv_v5_shifted_feedback_handle"):
            raise ValueError(f"shifted feedback already installed on {module_name}")
        module.rwkv_v5_shifted_feedback_enabled = False
        module.rwkv_v5_shifted_feedback_source = None
        module.rwkv_v5_shifted_feedback_last_applied = None

        def inject_shifted_feedback(
            current_module: Any,
            inputs: tuple[Any, ...],
        ) -> tuple[Any, ...] | None:
            current_module.rwkv_v5_shifted_feedback_last_applied = None
            if not current_module.rwkv_v5_shifted_feedback_enabled:
                return None
            if not inputs or not isinstance(inputs[0], torch.Tensor):
                raise RuntimeError("shifted feedback requires a tensor hidden-state input")
            hidden_states = inputs[0]
            source = current_module.rwkv_v5_shifted_feedback_source
            if source is None:
                raise RuntimeError("enabled shifted feedback is missing its previous-pass source")
            expected = (
                hidden_states.shape[0],
                hidden_states.shape[1],
                int(current_module.state_read_dim),
            )
            if tuple(source.shape) != expected:
                raise ValueError(
                    f"shifted feedback source shape differs: expected={expected} "
                    f"actual={tuple(source.shape)}"
                )
            shifted = torch.zeros_like(source, dtype=torch.float32)
            shifted[:, 1:] = source.detach().float()[:, :-1]
            mapped = F.linear(
                shifted,
                current_module.delta_o_proj.detach().float(),
            )
            mapped_rms = ShadowCertifiedLiveValue._exact_rms(mapped)
            direction = torch.tanh(mapped / mapped_rms.clamp_min(1e-6))
            present = shifted.square().sum(dim=-1, keepdim=True).gt(0.0)
            direction = torch.where(present, direction, torch.zeros_like(direction))
            carrier_rms = ShadowCertifiedLiveValue._exact_rms(hidden_states)
            correction = float(gain) * carrier_rms * direction
            if not bool(torch.isfinite(correction).all().item()):
                raise RuntimeError("shifted feedback correction is non-finite")
            correction = correction.to(device=hidden_states.device, dtype=hidden_states.dtype)
            current_module.rwkv_v5_shifted_feedback_last_applied = correction.detach()
            return (hidden_states + correction, *inputs[1:])

        module.rwkv_v5_shifted_feedback_handle = module.register_forward_pre_hook(
            inject_shifted_feedback
        )
        names.append(module_name)
    return {
        "modules": len(names),
        "module_names": tuple(names),
        "source": "detached_previous_pass_live_rwkv_value",
        "injection": "one_token_right_shift_before_attention_query_and_read_formation",
        "projection": "frozen_signed_v5_delta_o_proj",
        "gain": float(gain),
        "first_token_exact_zero": True,
        "zero_source_exact_no_correction": True,
    }


def set_shifted_feedback(
    modules: Sequence[tuple[str, Any]],
    feedback: Mapping[str, torch.Tensor],
) -> None:
    for module_name, module in modules:
        try:
            source = feedback[module_name]
        except KeyError as error:
            raise ValueError(
                f"shifted feedback source is missing for {module_name}"
            ) from error
        module.rwkv_v5_shifted_feedback_source = source.detach()
        module.rwkv_v5_shifted_feedback_enabled = True


def clear_shifted_feedback(
    modules: Sequence[tuple[str, Any]],
    *,
    disable: bool = True,
) -> None:
    for _, module in modules:
        if disable:
            module.rwkv_v5_shifted_feedback_enabled = False
        module.rwkv_v5_shifted_feedback_source = None
        module.rwkv_v5_shifted_feedback_last_applied = None
