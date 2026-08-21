"""Exact two-axis address binding for RWKV recurrent matrices."""

from __future__ import annotations

from typing import Any

import torch
from torch import nn

from .rwkv_diagonal_sign_binding import DiagonalSignBinding


class BidirectionalDiagonalSignBinding(nn.Module):
    """Bind RWKV value rows and key columns with independent sign codes."""

    def __init__(
        self,
        state_dim: int,
        *,
        address_dim: int,
        left_projection: torch.Tensor,
        right_projection: torch.Tensor,
        frequency: float = 64.0,
        trainable_projection: bool = False,
    ) -> None:
        super().__init__()
        self.state_dim = int(state_dim)
        self.address_dim = int(address_dim)
        self.left = DiagonalSignBinding(
            self.state_dim,
            address_dim=self.address_dim,
            projection=left_projection,
            frequency=frequency,
            trainable_projection=trainable_projection,
        )
        self.right = DiagonalSignBinding(
            self.state_dim,
            address_dim=self.address_dim,
            projection=right_projection,
            frequency=frequency,
            trainable_projection=trainable_projection,
        )

    def codes(self, address: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.left.codes(address), self.right.codes(address)

    def bind_features(
        self,
        address: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        a: torch.Tensor,
        b: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        left_code, right_code = self.codes(address)
        return (
            (k.float() * right_code).to(dtype=k.dtype),
            (v.float() * left_code).to(dtype=v.dtype),
            (a.float() * right_code).to(dtype=a.dtype),
            (b.float() * right_code).to(dtype=b.dtype),
        )

    def encode_state(self, address: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
        left_code, right_code = self.codes(address)
        if state.shape[-2:] != (self.state_dim, self.state_dim):
            raise ValueError("RWKV state shape differs from bidirectional binder")
        if address.shape[:-1] != state.shape[:-2]:
            raise ValueError("RWKV state/address leading shapes differ")
        encoded = left_code.unsqueeze(-1) * state.float() * right_code.unsqueeze(-2)
        return encoded.to(dtype=state.dtype)

    def decoded_read(
        self,
        address: torch.Tensor,
        state: torch.Tensor,
        receptance: torch.Tensor,
    ) -> torch.Tensor:
        left_code, right_code = self.codes(address)
        if state.shape[-2:] != (self.state_dim, self.state_dim):
            raise ValueError("RWKV state shape differs from bidirectional binder")
        if receptance.shape[-1] != self.state_dim:
            raise ValueError("RWKV receptance width differs from bidirectional binder")
        bound_receptance = receptance.float() * right_code
        raw = torch.matmul(state.float(), bound_receptance.unsqueeze(-1)).squeeze(-1)
        return (raw * left_code).to(dtype=state.dtype)

    def rebase_state(
        self,
        old_address: torch.Tensor,
        new_address: torch.Tensor,
        state: torch.Tensor,
    ) -> torch.Tensor:
        old_left, old_right = self.codes(old_address)
        new_left, new_right = self.codes(new_address)
        rebased = (
            new_left.unsqueeze(-1)
            * old_left.unsqueeze(-1)
            * state.float()
            * old_right.unsqueeze(-2)
            * new_right.unsqueeze(-2)
        )
        return rebased.to(dtype=state.dtype)

    def architecture_payload(self) -> dict[str, Any]:
        return {
            "mechanism": "address_conditioned_bidirectional_diagonal_sign_involution",
            "state": "S_bound=D_value(address)@S@D_key(address)",
            "write": "v*=D_value; k/a/b*=D_key",
            "read": "r*=D_key; decoded_read=D_value@(S_bound@r*)",
            "identity": "matching left and right codes cancel exactly",
            "state_dim": self.state_dim,
            "address_dim": self.address_dim,
            "frequency": self.left.frequency,
            "parameters_per_layer": 2 * self.address_dim * self.state_dim,
            "left_and_right_codes_independent": True,
            "commutes_with_rwkv_diagonal_updates": True,
            "value_feature_preserved_after_correct_decode": True,
        }
