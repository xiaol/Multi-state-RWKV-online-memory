"""Bounded output coupling for a learned RWKV state-identity head.

The identity score is used at the point where a hybrid read combines its
projected carrier and recurrent correction:

``projected + gate(query, recurrent_read) * recurrent_correction``.

The projected carrier is never edited.  In particular, an exactly zero
recurrent correction gives byte-identical projected-only output irrespective
of the learned compatibility score or gate value.
"""

from __future__ import annotations

from typing import Any

import torch
from torch import nn

from experiments.rethinking_rwkv_ms_gemma.rwkv_query_state_bilinear import (
    ResidualBilinearIdentity,
    bounded_recurrent_gate,
)


class BoundedOutputIdentityGate(nn.Module):
    """Gate a recurrent correction with learned low-rank compatibility.

    The compatibility maps are exact identity transforms at installation.
    The gate is bounded in ``[0, max_gate]`` and starts near closed through a
    finite negative bias, while retaining a nonzero derivative for its score.
    This is deliberately an output coupling; unlike probe-only heads, it can
    influence the answer logits once installed in the fusion path.
    """

    def __init__(
        self,
        state_dim: int,
        *,
        bottleneck: int = 4,
        max_gate: float = 0.25,
        temperature: float = 4.0,
        threshold: float = 0.0,
        bias: float = -6.0,
    ) -> None:
        super().__init__()
        if not 0.0 < float(max_gate) <= 1.0:
            raise ValueError("max_gate must be in (0, 1]")
        self.identity = ResidualBilinearIdentity(
            state_dim, bottleneck=bottleneck
        )
        self.max_gate = float(max_gate)
        self.temperature = float(temperature)
        self.threshold = float(threshold)
        self.bias = float(bias)

    def gate(self, query: torch.Tensor, recurrent_read: torch.Tensor) -> torch.Tensor:
        score = self.identity.score(query, recurrent_read)
        return self.max_gate * bounded_recurrent_gate(
            score,
            temperature=self.temperature,
            threshold=self.threshold,
            bias=self.bias,
        )

    def forward(
        self,
        projected: torch.Tensor,
        recurrent_correction: torch.Tensor,
        query: torch.Tensor,
        recurrent_read: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        shapes = {
            tuple(projected.shape),
            tuple(recurrent_correction.shape),
            tuple(query.shape),
            tuple(recurrent_read.shape),
        }
        if len(shapes) != 1:
            raise ValueError("output identity gate inputs must have identical shapes")
        gate = self.gate(query, recurrent_read)
        output = projected.float() + gate.unsqueeze(-1) * recurrent_correction.float()
        return output, gate

    def audit(self, reference: torch.Tensor) -> dict[str, Any]:
        if reference.shape[-1] != self.identity.state_dim:
            raise ValueError("reference width differs from identity head")
        with torch.no_grad():
            mapped_query = self.identity.map_query(reference)
            mapped_state = self.identity.map_state(reference)
            zero_output, gate = self(
                reference,
                torch.zeros_like(reference),
                reference,
                reference,
            )
        return {
            "state_dim": self.identity.state_dim,
            "bottleneck": self.identity.bottleneck,
            "parameters": sum(parameter.numel() for parameter in self.parameters()),
            "max_gate": self.max_gate,
            "initialized_query_map_exact_identity": bool(
                torch.equal(mapped_query, reference.float())
            ),
            "initialized_state_map_exact_identity": bool(
                torch.equal(mapped_state, reference.float())
            ),
            "zero_recurrent_exact_projected_only": bool(
                torch.equal(zero_output, reference.float())
            ),
            "initial_gate_min": float(gate.min().item()),
            "initial_gate_max": float(gate.max().item()),
        }


def architecture_payload() -> dict[str, Any]:
    """Stable, protocol-ready description of the conditional architecture."""
    return {
        "fusion": "projected + bounded_identity_gate(query,recurrent_read) * recurrent_correction",
        "projected_carrier": "preserved by reference; never edited by identity head",
        "zero_recurrent_identity": "zero recurrent correction yields exact projected-only output",
        "identity_head": "ResidualBilinearIdentity rank-4 query/state maps and cosine score",
        "gate": "0.25 * sigmoid(4 * (score - 0) - 6)",
        "output_coupled": True,
    }
