"""Invertible headwise address-value binding for RWKV online memory.

The current identity routes score a query against an already-written state.
This module instead changes the representation that enters the recurrent
state.  Each RWKV value head is interpreted as complex pairs and rotated by a
phase generated from the projected slot address.  Reading applies the inverse
rotation generated from the query address before RWKV normalization/readout.

For a matching address the two rotations cancel.  A matched-donor state with
a separated address code is decoded in the wrong basis.  The operation is
norm preserving and contains no scalar gate or cosine score.
"""

from __future__ import annotations

import math
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F


class HeadwiseRotaryBinding(nn.Module):
    """Bind values to addresses with invertible rotations inside RWKV heads."""

    def __init__(
        self,
        state_dim: int,
        *,
        head_size: int,
        max_phase: float = math.pi,
        trainable_projection: bool = True,
    ) -> None:
        super().__init__()
        if int(state_dim) < 1:
            raise ValueError("state_dim must be positive")
        if int(head_size) < 2 or int(head_size) % 2:
            raise ValueError("head_size must be positive and even")
        if int(state_dim) % int(head_size):
            raise ValueError("state_dim must be divisible by head_size")
        if not math.isfinite(float(max_phase)) or max_phase <= 0.0:
            raise ValueError("max_phase must be finite and positive")

        self.state_dim = int(state_dim)
        self.head_size = int(head_size)
        self.num_heads = self.state_dim // self.head_size
        self.pairs_per_head = self.head_size // 2
        self.max_phase = float(max_phase)
        projection = torch.empty(
            self.num_heads,
            self.pairs_per_head,
            self.head_size,
            dtype=torch.float32,
        )
        for head_projection in projection:
            nn.init.orthogonal_(head_projection)
        self.phase_projection = nn.Parameter(
            projection,
            requires_grad=bool(trainable_projection),
        )

    def _reshape_heads(self, value: torch.Tensor, *, name: str) -> torch.Tensor:
        if value.shape[-1] != self.state_dim:
            raise ValueError(
                f"{name} width differs from binder: "
                f"expected={self.state_dim} actual={value.shape[-1]}"
            )
        return value.float().reshape(*value.shape[:-1], self.num_heads, self.head_size)

    def phases(self, address: torch.Tensor) -> torch.Tensor:
        """Return bounded rotation phases for every head and complex pair."""

        heads = self._reshape_heads(address, name="address")
        square_mean = heads.square().mean(dim=-1, keepdim=True)
        rms = torch.where(
            square_mean.gt(0.0),
            (square_mean + 1e-12).sqrt(),
            torch.ones_like(square_mean),
        )
        normalized = heads / rms
        logits = torch.einsum(
            "...hi,hpi->...hp",
            normalized,
            self.phase_projection.float(),
        )
        return self.max_phase * torch.tanh(logits)

    def _rotate(
        self,
        address: torch.Tensor,
        value: torch.Tensor,
        *,
        inverse: bool,
    ) -> torch.Tensor:
        if address.shape[:-1] != value.shape[:-1]:
            raise ValueError(
                "address/value leading shapes differ: "
                f"address={tuple(address.shape)} value={tuple(value.shape)}"
            )
        value_heads = self._reshape_heads(value, name="value")
        value_pairs = value_heads.reshape(
            *value.shape[:-1],
            self.num_heads,
            self.pairs_per_head,
            2,
        )
        phase = self.phases(address)
        if inverse:
            phase = -phase
        cosine = torch.cos(phase)
        sine = torch.sin(phase)
        real = value_pairs[..., 0]
        imaginary = value_pairs[..., 1]
        rotated = torch.stack(
            (
                cosine * real - sine * imaginary,
                sine * real + cosine * imaginary,
            ),
            dim=-1,
        )
        return rotated.reshape(*value.shape).to(dtype=value.dtype)

    def bind(self, write_address: torch.Tensor, value: torch.Tensor) -> torch.Tensor:
        """Rotate a recurrent value into its address-specific write basis."""

        return self._rotate(write_address, value, inverse=False)

    def unbind(self, query_address: torch.Tensor, value: torch.Tensor) -> torch.Tensor:
        """Decode a recurrent read using the query address basis."""

        return self._rotate(query_address, value, inverse=True)

    def transfer(
        self,
        write_address: torch.Tensor,
        query_address: torch.Tensor,
        value: torch.Tensor,
    ) -> torch.Tensor:
        """Bind at write and unbind at read for an isolated value."""

        return self.unbind(query_address, self.bind(write_address, value))

    def phase_distance(
        self,
        left_address: torch.Tensor,
        right_address: torch.Tensor,
    ) -> torch.Tensor:
        """Return normalized circular RMS distance between two address codes."""

        if left_address.shape != right_address.shape:
            raise ValueError("phase-distance addresses must have identical shapes")
        delta = self.phases(left_address) - self.phases(right_address)
        wrapped = torch.atan2(torch.sin(delta), torch.cos(delta))
        return wrapped.square().mean(dim=(-2, -1)).sqrt() / math.pi

    def separation_hinge(
        self,
        left_address: torch.Tensor,
        right_address: torch.Tensor,
        *,
        margin: float,
        valid: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Penalize aliased address codes without observing answer values."""

        if not math.isfinite(float(margin)) or not 0.0 < margin <= 1.0:
            raise ValueError("separation margin must be in (0, 1]")
        distance = self.phase_distance(left_address, right_address)
        hinge = F.relu(distance.new_tensor(float(margin)) - distance)
        if valid is not None:
            if valid.shape != hinge.shape:
                raise ValueError("separation mask shape differs from address rows")
            hinge = hinge.masked_select(valid)
            if hinge.numel() == 0:
                raise ValueError("separation mask selects no rows")
        result = hinge.mean()
        if not bool(torch.isfinite(result).item()):
            raise RuntimeError("non-finite rotary binding separation loss")
        return result


def parameter_count(state_dim: int, head_size: int) -> int:
    if int(state_dim) < 1:
        raise ValueError("state_dim must be positive")
    if int(head_size) < 2 or int(head_size) % 2:
        raise ValueError("head_size must be positive and even")
    if int(state_dim) % int(head_size):
        raise ValueError("state_dim must be divisible by head_size")
    return (int(state_dim) // int(head_size)) * (int(head_size) // 2) * int(head_size)


def architecture_payload(state_dim: int = 32, head_size: int = 32) -> dict[str, Any]:
    """Return an auditable description for experiment protocols."""

    return {
        "mechanism": "headwise_complex_rotation_address_value_binding",
        "write": "v_bound=blockdiag(R(theta(write_address)))@v",
        "read": "v_read=blockdiag(R(-theta(query_address)))@rwkv_state_read",
        "identity": "matching write/read addresses cancel algebraically",
        "matched_donor": "a separated donor address is decoded in the target basis",
        "state_dim": int(state_dim),
        "head_size": int(head_size),
        "num_heads": int(state_dim) // int(head_size),
        "complex_pairs_per_head": int(head_size) // 2,
        "parameters_per_layer": parameter_count(state_dim, head_size),
        "norm_preserving": True,
        "scalar_gate": False,
        "cosine_readout": False,
        "placement": "bind RWKV v before state scan; unbind matrix read before group norm/output",
    }
