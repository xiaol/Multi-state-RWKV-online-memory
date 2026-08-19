from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

from experiments.rethinking_rwkv_ms_gemma.rwkv_query_state_infonce import (
    LowRankQueryProjector,
    deterministic_self_test,
    info_nce_loss,
)


def test_projector_starts_as_exact_noop() -> None:
    torch.manual_seed(3)
    projector = LowRankQueryProjector(8, rank=2)
    address = torch.randn(4, 8)
    assert torch.equal(projector(address), address)
    assert projector.audit(address)["initialized_exact_noop"] is True
    assert projector.parameter_count == 32


def test_infonce_rewards_matching_diagonal() -> None:
    states = F.normalize(torch.eye(4), dim=-1)
    matched_loss, _, matched_margin = info_nce_loss(states, states)
    shifted_loss, _, shifted_margin = info_nce_loss(states.roll(1, dims=0), states)
    assert matched_loss < shifted_loss
    assert matched_margin > 0
    assert shifted_margin < 0


def test_infonce_rejects_missing_negatives() -> None:
    with pytest.raises(ValueError, match="at least two"):
        info_nce_loss(torch.ones(1, 4), torch.ones(1, 4))


def test_deterministic_gradient_screen_improves_loss() -> None:
    result = deterministic_self_test()
    assert result["gradient_screen_passed"] is True
    assert result["initialized_exact_noop"] is True
    assert result["final_loss"] < result["initial_loss"]
    assert result["final_margin"] > 0
