"""Small, state-space identity objective for the native RWKV experiments.

This module deliberately contains no model or dataset access.  It provides the
cheap next-candidate primitive: learn a bounded low-rank residual projection of
the frozen projected slot address, then identify the matching recurrent state
with in-batch InfoNCE.  The up projection is zero initialized, so installing
the projector is an exact no-op until optimization updates it.
"""

from __future__ import annotations

from typing import Tuple

import torch
from torch import nn
import torch.nn.functional as F


class LowRankQueryProjector(nn.Module):
    """Zero-noop rank-r residual map from projected address to RWKV state space."""

    def __init__(
        self,
        state_dim: int,
        rank: int = 4,
        *,
        residual_scale: float = 0.25,
    ) -> None:
        super().__init__()
        if state_dim < 1 or rank < 1:
            raise ValueError("state_dim and rank must be positive")
        if not 0.0 < residual_scale <= 1.0:
            raise ValueError("residual_scale must be in (0, 1]")
        self.state_dim = int(state_dim)
        self.rank = int(rank)
        self.residual_scale = float(residual_scale)
        self.down = nn.Parameter(torch.empty(rank, state_dim, dtype=torch.float32))
        self.up = nn.Parameter(torch.zeros(state_dim, rank, dtype=torch.float32))
        nn.init.normal_(self.down, mean=0.0, std=state_dim ** -0.5)

    def forward(self, address: torch.Tensor) -> torch.Tensor:
        if address.shape[-1] != self.state_dim:
            raise ValueError(
                f"address last dimension must be {self.state_dim}, "
                f"got {address.shape[-1]}"
            )
        x = address.float()
        hidden = torch.tanh(F.linear(x, self.down))
        return x + self.residual_scale * F.linear(hidden, self.up)

    @property
    def parameter_count(self) -> int:
        return int(self.down.numel() + self.up.numel())

    def audit(self, address: torch.Tensor) -> dict[str, float | bool | int]:
        with torch.no_grad():
            delta = self(address) - address.float()
        return {
            "state_dim": self.state_dim,
            "rank": self.rank,
            "parameter_count": self.parameter_count,
            "initialized_exact_noop": bool(torch.equal(delta, torch.zeros_like(delta))),
            "max_initial_abs_delta": float(delta.abs().max().item()),
        }


def info_nce_loss(
    projected_addresses: torch.Tensor,
    recurrent_states: torch.Tensor,
    *,
    temperature: float = 0.07,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return loss, logits, and positive-vs-hardest-negative margin.

    Inputs are ``[batch, state_dim]``.  The diagonal is the matched state and
    every other row is an in-batch donor negative.  No donor state is modified.
    """
    if projected_addresses.ndim != 2 or recurrent_states.ndim != 2:
        raise ValueError("InfoNCE inputs must be rank-2 tensors")
    if projected_addresses.shape != recurrent_states.shape:
        raise ValueError("InfoNCE address/state shapes must match")
    if projected_addresses.shape[0] < 2:
        raise ValueError("InfoNCE requires at least two rows for a negative")
    if not torch.isfinite(projected_addresses).all() or not torch.isfinite(recurrent_states).all():
        raise ValueError("InfoNCE inputs must be finite")
    if not 0.0 < temperature:
        raise ValueError("temperature must be positive")
    query = F.normalize(projected_addresses.float(), dim=-1, eps=1e-6)
    state = F.normalize(recurrent_states.float(), dim=-1, eps=1e-6)
    logits = query @ state.transpose(0, 1) / float(temperature)
    labels = torch.arange(logits.shape[0], device=logits.device)
    loss = F.cross_entropy(logits, labels)
    positive = logits.diagonal()
    negative = logits.masked_fill(torch.eye(logits.shape[0], device=logits.device, dtype=torch.bool), float("-inf"))
    margin = (positive - negative.max(dim=1).values).mean()
    return loss, logits, margin


def deterministic_self_test() -> dict[str, float | bool]:
    """CPU-only smoke proving gradients and a decreasing identity loss."""
    torch.manual_seed(20260819)
    addresses = F.normalize(torch.randn(8, 32), dim=-1)
    transform = torch.randn(32, 32)
    states = F.normalize(addresses @ transform, dim=-1)
    projector = LowRankQueryProjector(32, rank=4)
    audit = projector.audit(addresses)
    if audit["initialized_exact_noop"] is not True:
        raise AssertionError("projector is not an exact no-op at initialization")
    optimizer = torch.optim.AdamW(projector.parameters(), lr=0.05)
    initial, _, _ = info_nce_loss(projector(addresses), states)
    for _ in range(32):
        optimizer.zero_grad(set_to_none=True)
        loss, _, _ = info_nce_loss(projector(addresses), states)
        loss.backward()
        optimizer.step()
    final, _, margin = info_nce_loss(projector(addresses), states)
    if not bool(torch.isfinite(final).item() and final < initial):
        raise AssertionError(f"InfoNCE did not improve: initial={initial} final={final}")
    return {
        "initialized_exact_noop": True,
        "initial_loss": float(initial.item()),
        "final_loss": float(final.item()),
        "final_margin": float(margin.item()),
        "gradient_screen_passed": True,
    }


if __name__ == "__main__":
    print(deterministic_self_test())
