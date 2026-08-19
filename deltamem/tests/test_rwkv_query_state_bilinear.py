from __future__ import annotations

import pytest
import torch

from experiments.rethinking_rwkv_ms_gemma.rwkv_query_state_bilinear import (
    ResidualBilinearIdentity,
    audit_payload,
    bounded_recurrent_gate,
    parameter_count,
)


def test_low_rank_identity_head_starts_as_exact_identity() -> None:
    torch.manual_seed(7)
    head = ResidualBilinearIdentity(8, bottleneck=2)
    query = torch.randn(2, 3, 8)
    state = torch.randn(2, 3, 8)
    assert torch.equal(head.map_query(query), query)
    assert torch.equal(head.map_state(state), state)
    assert parameter_count(8, 2) == 64


def test_pairwise_hinge_backpropagates_through_both_state_maps() -> None:
    torch.manual_seed(9)
    head = ResidualBilinearIdentity(8, bottleneck=2)
    query = torch.randn(2, 3, 8)
    positive = query.detach().clone().requires_grad_(True)
    negative = torch.randn(2, 3, 8, requires_grad=True)
    valid = torch.tensor([[True, True, False], [True, False, True]])
    metrics = head.pairwise_hinge(
        query,
        positive,
        negative,
        valid=valid,
        margin=0.2,
    )
    metrics.hinge.backward()
    assert torch.isfinite(metrics.hinge)
    assert positive.grad is not None
    assert negative.grad is not None
    assert all(parameter.grad is not None for parameter in head.parameters())


def test_shape_and_gate_contracts() -> None:
    head = ResidualBilinearIdentity(8, bottleneck=2)
    with pytest.raises(ValueError, match="identical shapes"):
        head.score(torch.randn(1, 8), torch.randn(1, 7))
    gate = bounded_recurrent_gate(torch.tensor([-1.0, 0.0, 1.0]))
    assert torch.all((gate >= 0.0) & (gate <= 1.0))
    payload = audit_payload(32, 4)
    assert payload["parameters_per_layer"] == 512
    assert payload["maps"].startswith("identity_plus_low_rank")
