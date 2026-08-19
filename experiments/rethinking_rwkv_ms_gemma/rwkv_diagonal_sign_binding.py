"""Exact address binding for the diagonal RWKV state update.

RWKV-MS applies per-value-channel keep, erase, and write coefficients to the
left (value) axis of its recurrent matrix.  A dense orthogonal change of basis
does not commute with those diagonal updates.  This binder therefore uses an
address-conditioned diagonal involution instead: each value channel is
multiplied by ``+1`` or ``-1``.  The same code is its own inverse and commutes
with every diagonal value-axis update.

The projection can be fit on address-only pairs and frozen before causal CE
training.  ``codes`` is intentionally hard and detached; the causal endpoint
must not learn the identity map from answer tokens.
"""

from __future__ import annotations

import hashlib
import math
from typing import Any

import torch
from torch import nn


class DiagonalSignBinding(nn.Module):
    """Address-conditioned diagonal ``+/-1`` binding for RWKV value rows."""

    def __init__(
        self,
        state_dim: int,
        *,
        address_dim: int | None = None,
        projection: torch.Tensor | None = None,
        frequency: float = 64.0,
        trainable_projection: bool = True,
    ) -> None:
        super().__init__()
        if int(state_dim) < 1:
            raise ValueError("state_dim must be positive")
        self.state_dim = int(state_dim)
        self.address_dim = int(state_dim if address_dim is None else address_dim)
        if self.address_dim < 1:
            raise ValueError("address_dim must be positive")
        if not math.isfinite(float(frequency)) or float(frequency) <= 0.0:
            raise ValueError("frequency must be finite and positive")
        self.frequency = float(frequency)
        if projection is None:
            projection = torch.empty(self.address_dim, self.state_dim, dtype=torch.float32)
            nn.init.orthogonal_(projection)
        else:
            if tuple(projection.shape) != (self.address_dim, self.state_dim):
                raise ValueError("projection shape differs from state_dim")
            projection = projection.detach().float().clone()
        self.projection = nn.Parameter(
            projection,
            requires_grad=bool(trainable_projection),
        )

    def _normalize(self, address: torch.Tensor) -> torch.Tensor:
        if address.shape[-1] != self.address_dim:
            raise ValueError(
                f"address width differs from binder: expected={self.address_dim} "
                f"actual={address.shape[-1]}"
            )
        address = address.float()
        rms = address.square().mean(dim=-1, keepdim=True).add(1e-12).sqrt()
        return address / rms

    def logits(self, address: torch.Tensor) -> torch.Tensor:
        """Return address code logits before hard sign materialization."""

        normalized = self._normalize(address)
        return torch.einsum("...d,dc->...c", normalized, self.projection.float())

    def codes(self, address: torch.Tensor) -> torch.Tensor:
        """Return a detached diagonal code with exact ``+/-1`` entries.

        Zero addresses receive the all-ones code.  This keeps zero-state and
        zero-address paths exact while still allowing arbitrary nonzero
        addresses to produce a near-collision-free code.
        """

        normalized = self._normalize(address)
        logits = torch.einsum("...d,dc->...c", normalized, self.projection.float())
        signs = torch.where(
            torch.sin(logits * self.frequency).ge(0.0),
            torch.ones_like(logits),
            -torch.ones_like(logits),
        )
        active = address.float().abs().sum(dim=-1, keepdim=True).gt(0.0)
        return torch.where(active, signs, torch.ones_like(signs)).detach()

    def bind(self, write_address: torch.Tensor, value: torch.Tensor) -> torch.Tensor:
        if write_address.shape[:-1] != value.shape[:-1] or value.shape[-1] != self.state_dim:
            raise ValueError("write address/value leading shapes or value width differ")
        return (value.float() * self.codes(write_address)).to(dtype=value.dtype)

    def unbind(self, query_address: torch.Tensor, value: torch.Tensor) -> torch.Tensor:
        if query_address.shape[:-1] != value.shape[:-1] or value.shape[-1] != self.state_dim:
            raise ValueError("query address/value leading shapes or value width differ")
        # The code is an involution: D^{-1} = D.
        return (value.float() * self.codes(query_address)).to(dtype=value.dtype)

    def transfer(
        self,
        write_address: torch.Tensor,
        query_address: torch.Tensor,
        value: torch.Tensor,
    ) -> torch.Tensor:
        return self.unbind(query_address, self.bind(write_address, value))

    def hamming_distance(
        self,
        left_address: torch.Tensor,
        right_address: torch.Tensor,
    ) -> torch.Tensor:
        left = self.codes(left_address)
        right = self.codes(right_address)
        return left.ne(right).float().mean(dim=-1)

    def separation_hinge(
        self,
        left_address: torch.Tensor,
        right_address: torch.Tensor,
        *,
        margin: float,
        valid: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if not math.isfinite(float(margin)) or not 0.0 < margin <= 1.0:
            raise ValueError("separation margin must be in (0, 1]")
        left_soft = torch.tanh(self.logits(left_address))
        right_soft = torch.tanh(self.logits(right_address))
        distance = (left_soft - right_soft).abs().mean(dim=-1).mul(0.5)
        hinge = (distance.new_tensor(float(margin)) - distance).clamp_min(0.0)
        if valid is not None:
            if tuple(valid.shape) != tuple(hinge.shape):
                raise ValueError("separation mask shape differs from address rows")
            hinge = hinge.masked_select(valid)
            if hinge.numel() == 0:
                raise ValueError("separation mask selects no rows")
        return hinge.mean()

    def architecture_payload(self) -> dict[str, Any]:
        return {
            "mechanism": "address_conditioned_diagonal_sign_involution",
            "write": "v_bound=diag(sign(sin(frequency*P*rmsnorm(address))))@v",
            "read": "v_read=diag(sign(sin(frequency*P*rmsnorm(query_address))))@rwkv_state_read",
            "identity": "matching write/read codes cancel exactly",
            "state_dim": self.state_dim,
            "frequency": self.frequency,
            "address_dim": self.address_dim,
            "parameters_per_layer": self.address_dim * self.state_dim,
            "diagonal_involution": True,
            "commutes_with_rwkv_value_axis_diagonal_updates": True,
            "scalar_gate": False,
            "cosine_readout": False,
        }


def deterministic_projection(state_dim: int, seed: int, output_dim: int | None = None) -> torch.Tensor:
    """Return a reproducible orthogonal projection for mechanics-only runs."""

    generator = torch.Generator(device="cpu").manual_seed(int(seed))
    output_dim = int(state_dim if output_dim is None else output_dim)
    matrix = torch.randn(int(state_dim), output_dim, generator=generator)
    return torch.linalg.qr(matrix).Q


def projection_sha256(projection: torch.Tensor) -> str:
    payload = projection.detach().float().cpu().contiguous().numpy().tobytes()
    return hashlib.sha256(payload).hexdigest()


def fit_address_only_projection(
    left_address: torch.Tensor,
    right_address: torch.Tensor,
    *,
    steps: int = 1200,
    learning_rate: float = 0.03,
    margin: float = 0.8,
    weight_decay: float = 0.01,
    seed: int = 115,
) -> tuple[torch.Tensor, dict[str, float | int]]:
    """Fit a code projection using only matched address pairs.

    The returned projection is detached and intended to be frozen before
    causal answer training.  No values, logits, or answer labels enter this
    objective.
    """

    if left_address.shape != right_address.shape or left_address.ndim != 2:
        raise ValueError("address pairs must have identical [rows, state_dim] shapes")
    if int(steps) < 1 or not math.isfinite(float(learning_rate)) or learning_rate <= 0.0:
        raise ValueError("steps and learning_rate must be positive")
    if not 0.0 < float(margin) <= 2.0:
        raise ValueError("fit margin must be in (0, 2]")
    state_dim = int(left_address.shape[-1])
    generator = torch.Generator(device="cpu").manual_seed(int(seed))
    projection = torch.randn(state_dim, state_dim, generator=generator, dtype=torch.float32)
    projection = torch.linalg.qr(projection).Q.requires_grad_(True)
    left = left_address.detach().float()
    right = right_address.detach().float()
    left = left / left.square().mean(dim=-1, keepdim=True).add(1e-12).sqrt()
    right = right / right.square().mean(dim=-1, keepdim=True).add(1e-12).sqrt()
    optimizer = torch.optim.AdamW([projection], lr=float(learning_rate), weight_decay=float(weight_decay))
    for _ in range(int(steps)):
        left_soft = torch.tanh(left @ projection)
        right_soft = torch.tanh(right @ projection)
        distance = (left_soft - right_soft).abs().mean(dim=-1)
        separation = torch.relu(distance.new_tensor(float(margin)) - distance).mean()
        saturation = (1.0 - left_soft.abs()).mean() + (1.0 - right_soft.abs()).mean()
        balance = left_soft.mean(dim=0).square().mean() + right_soft.mean(dim=0).square().mean()
        loss = separation + 0.05 * saturation + 0.05 * balance
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        with torch.no_grad():
            projection.mul_(
                math.sqrt(float(state_dim))
                / projection.square().sum(dim=0, keepdim=True).sqrt().clamp_min(1e-8)
            )
    with torch.no_grad():
        left_code = torch.where(left @ projection >= 0.0, 1.0, -1.0)
        right_code = torch.where(right @ projection >= 0.0, 1.0, -1.0)
        hamming = left_code.ne(right_code).float().mean(dim=-1)
    metrics: dict[str, float | int] = {
        "steps": int(steps),
        "seed": int(seed),
        "rows": int(left.shape[0]),
        "mean_hamming": float(hamming.mean().item()),
        "positive_hamming_fraction": float(hamming.gt(0.0).float().mean().item()),
        "changed_row_fraction_at_0_05": float(hamming.ge(0.05).float().mean().item()),
    }
    return projection.detach(), metrics
