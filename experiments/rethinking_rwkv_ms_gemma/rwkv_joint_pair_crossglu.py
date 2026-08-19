"""Joint query/state CrossGLU for RWKV memory identity mechanics."""

from __future__ import annotations

from typing import Any

import torch
from torch import nn
from torch.nn import functional as F

from .rwkv_query_state_bilinear import ResidualBilinearIdentity


class JointPairGatedCrossGLU(nn.Module):
    """Bounded pair-dependent state bridge with an exact zero-state contract."""

    def __init__(
        self,
        state_dim: int,
        *,
        identity: ResidualBilinearIdentity | None = None,
        max_gain: float = 0.25,
        temperature: float = 2.0,
    ) -> None:
        super().__init__()
        if int(state_dim) < 1:
            raise ValueError("state_dim must be positive")
        if not 0.0 < float(max_gain) <= 1.0:
            raise ValueError("max_gain must be in (0, 1]")
        if not torch.isfinite(torch.tensor(float(temperature))) or float(temperature) <= 0.0:
            raise ValueError("temperature must be finite and positive")
        self.state_dim = int(state_dim)
        self.max_gain = float(max_gain)
        self.temperature = float(temperature)
        self.identity = (
            identity
            if identity is not None
            else ResidualBilinearIdentity(self.state_dim, bottleneck=4)
        )
        if self.identity.state_dim != self.state_dim:
            raise ValueError("identity state dimension differs from bridge")
        self.gate_proj = nn.Linear(2 * self.state_dim, self.state_dim, bias=False)
        self.value_proj = nn.Linear(self.state_dim, self.state_dim, bias=False)
        self.output_proj = nn.Linear(self.state_dim, self.state_dim, bias=False)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        with torch.no_grad():
            self.gate_proj.weight.zero_()
            self.gate_proj.weight[:, : self.state_dim].copy_(
                torch.eye(self.state_dim, device=self.gate_proj.weight.device)
            )
            self.gate_proj.weight[:, self.state_dim :].copy_(
                0.5 * torch.eye(self.state_dim, device=self.gate_proj.weight.device)
            )
            self.value_proj.weight.copy_(
                torch.eye(self.state_dim, device=self.value_proj.weight.device)
            )
            self.output_proj.weight.copy_(
                torch.eye(self.state_dim, device=self.output_proj.weight.device)
            )

    def pair_features(
        self,
        query: torch.Tensor,
        state: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if query.shape != state.shape or query.shape[-1] != self.state_dim:
            raise ValueError("query/state shapes differ from bridge")
        mapped_query = self.identity.map_query(query)
        mapped_state = self.identity.map_state(state)
        query_norm = F.normalize(mapped_query, dim=-1, eps=1e-6)
        state_norm = F.normalize(mapped_state, dim=-1, eps=1e-6)
        joint = torch.cat((query_norm * state_norm, (query_norm - state_norm).abs()), dim=-1)
        return mapped_query, mapped_state, joint

    def forward(
        self,
        projected: torch.Tensor,
        query: torch.Tensor,
        recurrent_read: torch.Tensor,
        *,
        gate_override: torch.Tensor | None = None,
        gate_shuffle: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        shapes = {tuple(projected.shape), tuple(query.shape), tuple(recurrent_read.shape)}
        if len(shapes) != 1 or projected.shape[-1] != self.state_dim:
            raise ValueError("pair bridge inputs must share the state width")
        _, mapped_state, joint = self.pair_features(query, recurrent_read)
        gate_logits = self.temperature * self.gate_proj(joint.float())
        gate = self.max_gain * torch.sigmoid(gate_logits)
        if gate_override is not None:
            if tuple(gate_override.shape) != tuple(gate.shape):
                raise ValueError("gate override shape differs from computed gate")
            gate = gate_override
        elif gate_shuffle:
            gate = torch.roll(gate, shifts=1, dims=-1)
        value = self.value_proj(mapped_state.float())
        bridge = self.output_proj(value * gate.float())
        output = projected.float() + bridge
        return output, gate, value, bridge

    def audit_payload(self) -> dict[str, Any]:
        return {
            "fusion": "projected + output(value(state) * sigmoid(G[normalize(q)*normalize(s), abs(normalize(q)-normalize(s))]))",
            "identity_maps": "frozen source-and-donor-component-disjoint residual bilinear maps",
            "gate_projection": "bias-free deterministic [I, 0.5I]",
            "value_projection": "bias-free identity initialization",
            "output_projection": "bias-free identity initialization",
            "state_dim": self.state_dim,
            "max_gain": self.max_gain,
            "temperature": self.temperature,
            "zero_state_exact_projected_only": True,
        }
