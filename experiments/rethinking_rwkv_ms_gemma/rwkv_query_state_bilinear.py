"""Low-rank query/state compatibility for a learned RWKV identity gate.

The existing query-state identity probe compares the folded projected-slot
address and the RWKV read in the same coordinate system.  That is a strong
assumption: the projected key is trained for slot routing while the recurrent
read is produced by the RWKV ``k/v/a/b`` dynamics.  This module provides a
small, bounded compatibility head that learns a residual coordinate map for
each side before taking a cosine score.

It is intentionally independent of Delta-Mem attachment code.  A runner can
install one instance per wrapped layer, train it with the internal donor hinge,
and only then use the score as a bounded recurrent-correction gate.  At
initialization both maps are exact identity transforms, so an identity screen
can be compared against the old cosine probe without a hidden coordinate
change.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F


@dataclass(frozen=True)
class PairwiseIdentityMetrics:
    """Aggregated metrics for a positive/negative state pair."""

    positive_score: torch.Tensor
    negative_score: torch.Tensor
    hinge: torch.Tensor


class ResidualBilinearIdentity(nn.Module):
    """A low-rank residual map followed by a cosine compatibility score.

    ``query`` and ``state`` are expected to have the same final dimension,
    normally ``rank * num_state_heads`` (32 in the current Gemma endpoint).
    The four matrices are tiny: for state dimension 32 and rank 4, one layer
    has 512 trainable values.  ``q_down``/``s_down`` are initialized with a
    small Gaussian and both up projections start at zero, making each map
    exactly identity at initialization while preserving a first-order update
    path through the up projection.
    """

    def __init__(
        self,
        state_dim: int,
        *,
        bottleneck: int = 4,
        down_init_std: float = 0.02,
    ) -> None:
        super().__init__()
        if int(state_dim) < 1:
            raise ValueError("state_dim must be positive")
        if int(bottleneck) < 1:
            raise ValueError("bottleneck must be positive")
        if not torch.isfinite(torch.tensor(float(down_init_std))) or down_init_std <= 0:
            raise ValueError("down_init_std must be finite and positive")
        self.state_dim = int(state_dim)
        self.bottleneck = int(bottleneck)
        self.q_down = nn.Parameter(torch.empty(self.bottleneck, self.state_dim))
        self.q_up = nn.Parameter(torch.zeros(self.state_dim, self.bottleneck))
        self.s_down = nn.Parameter(torch.empty(self.bottleneck, self.state_dim))
        self.s_up = nn.Parameter(torch.zeros(self.state_dim, self.bottleneck))
        nn.init.normal_(self.q_down, std=float(down_init_std))
        nn.init.normal_(self.s_down, std=float(down_init_std))

    def _map(
        self,
        value: torch.Tensor,
        down: torch.Tensor,
        up: torch.Tensor,
    ) -> torch.Tensor:
        if value.shape[-1] != self.state_dim:
            raise ValueError(
                "query/state feature width differs from identity head: "
                f"expected={self.state_dim} actual={value.shape[-1]}"
            )
        residual = F.linear(F.silu(F.linear(value.float(), down.float())), up.float())
        return value.float() + residual

    def map_query(self, query: torch.Tensor) -> torch.Tensor:
        return self._map(query, self.q_down, self.q_up)

    def map_state(self, state: torch.Tensor) -> torch.Tensor:
        return self._map(state, self.s_down, self.s_up)

    def score(self, query: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
        """Return a cosine compatibility score for each token/row."""

        if query.shape != state.shape:
            raise ValueError(
                "query/state features must have identical shapes: "
                f"query={tuple(query.shape)} state={tuple(state.shape)}"
            )
        mapped_query = F.normalize(self.map_query(query), dim=-1, eps=1e-6)
        mapped_state = F.normalize(self.map_state(state), dim=-1, eps=1e-6)
        return (mapped_query * mapped_state).sum(dim=-1)

    @staticmethod
    def masked_mean(score: torch.Tensor, valid: torch.Tensor | None = None) -> torch.Tensor:
        if valid is None:
            return score.mean()
        if score.shape != valid.shape:
            raise ValueError(
                "identity score and mask shapes differ: "
                f"score={tuple(score.shape)} mask={tuple(valid.shape)}"
            )
        selected = score.masked_select(valid)
        if selected.numel() == 0:
            raise ValueError("identity mask selects no answer positions")
        return selected.mean()

    def pairwise_hinge(
        self,
        query: torch.Tensor,
        positive_state: torch.Tensor,
        negative_state: torch.Tensor,
        *,
        valid: torch.Tensor | None = None,
        margin: float = 0.2,
    ) -> PairwiseIdentityMetrics:
        if not torch.isfinite(torch.tensor(float(margin))) or margin <= 0.0:
            raise ValueError("identity margin must be finite and positive")
        positive_score = self.masked_mean(self.score(query, positive_state), valid)
        negative_score = self.masked_mean(self.score(query, negative_state), valid)
        hinge = F.relu(positive_score.new_tensor(float(margin)) - positive_score + negative_score)
        if not bool(torch.isfinite(torch.stack((positive_score, negative_score, hinge))).all()):
            raise RuntimeError("non-finite bilinear identity metrics")
        return PairwiseIdentityMetrics(positive_score, negative_score, hinge)


def bounded_recurrent_gate(
    score: torch.Tensor,
    *,
    temperature: float = 4.0,
    threshold: float = 0.0,
    bias: float = -6.0,
) -> torch.Tensor:
    """Map compatibility scores to a bounded recurrent-correction gate.

    The negative bias keeps a newly installed head nearly inactive.  A runner
    should use the gate only after the cross-fit identity screen passes; this
    helper does not alter the projected carrier and returns values in ``[0,1]``.
    """

    if not torch.isfinite(torch.tensor([temperature, threshold, bias])).all() or temperature <= 0:
        raise ValueError("gate temperature must be finite and positive")
    return torch.sigmoid(float(temperature) * (score - float(threshold)) + float(bias))


def parameter_count(state_dim: int, bottleneck: int = 4) -> int:
    """Return trainable values per layer for audit/protocol generation."""

    return 4 * int(state_dim) * int(bottleneck)


def audit_payload(state_dim: int, bottleneck: int = 4) -> dict[str, Any]:
    """Return a deterministic architecture payload for signed protocols."""

    return {
        "feature_width": int(state_dim),
        "bottleneck": int(bottleneck),
        "maps": "identity_plus_low_rank_silu_residual_query_and_state",
        "score": "cosine(normalized(mapped_query),normalized(mapped_state))",
        "parameters_per_layer": parameter_count(state_dim, bottleneck),
        "initialization": "down_gaussian_std_0.02_up_zero_exact_identity",
        "query_gradient": "live_only_if_runner_uses_non_detached_query",
    }
