from __future__ import annotations

from types import SimpleNamespace

import torch
from torch import nn

from experiments.rethinking_rwkv_ms_gemma import (
    rwkv_v5_shadow_certified_live_value as certified,
)
from experiments.rethinking_rwkv_ms_gemma.rwkv_query_state_bilinear import (
    ResidualBilinearIdentity,
)


def _binder(state_dim: int = 4) -> certified.ShadowCertifiedLiveValue:
    return certified.ShadowCertifiedLiveValue(
        state_dim,
        identity=ResidualBilinearIdentity(state_dim, bottleneck=2),
        threshold=0.0,
        temperature=8.0,
        max_gain=0.125,
    )


def test_shadow_certificate_uses_shadow_for_gate_and_live_state_for_value() -> None:
    binder = _binder()
    projected = torch.tensor([[[2.0, -1.0, 0.5, 1.0]]])
    query = torch.tensor([[[1.0, 0.0, 0.0, 0.0]]])
    correct_shadow = query.clone()
    donor_shadow = -query
    live = torch.tensor([[[0.25, 0.5, -0.75, 1.0]]])

    correct, _, correct_gate, correct_value = binder(
        projected, query, correct_shadow, live
    )
    donor, _, donor_gate, donor_value = binder(
        projected, query, donor_shadow, live
    )

    assert torch.equal(correct_value, donor_value)
    assert bool((correct_gate > donor_gate).all().item())
    assert not torch.equal(correct, donor)

    swapped_live = live.roll(1, dims=-1)
    _, _, swapped_gate, swapped_value = binder(
        projected, query, correct_shadow, swapped_live
    )
    assert torch.equal(correct_gate, swapped_gate)
    assert not torch.equal(correct_value, swapped_value)


def test_shadow_certificate_zero_live_state_has_exact_zero_correction() -> None:
    torch.manual_seed(23)
    binder = _binder()
    projected = torch.randn(2, 3, 4)
    query = torch.randn_like(projected)
    shadow = torch.randn_like(projected)

    correction, score, gate, value = binder(
        projected,
        query,
        shadow,
        torch.zeros_like(projected),
    )

    assert torch.equal(correction, torch.zeros_like(correction))
    assert torch.equal(value, torch.zeros_like(value))
    assert bool(torch.isfinite(score).all().item())
    assert bool(torch.isfinite(gate).all().item())


def test_shadow_identity_stream_is_detached_but_live_value_map_trains() -> None:
    torch.manual_seed(29)
    binder = _binder()
    projected = torch.randn(2, 3, 4, requires_grad=True)
    query = torch.randn(2, 3, 4, requires_grad=True)
    shadow = torch.randn(2, 3, 4, requires_grad=True)
    live = torch.randn(2, 3, 4, requires_grad=True)

    correction, _, _, _ = binder(projected, query, shadow, live)
    correction.square().mean().backward()

    assert query.grad is None
    assert shadow.grad is None
    assert live.grad is not None and bool(torch.isfinite(live.grad).all().item())
    assert binder.value_map.weight.grad is not None
    assert bool(torch.isfinite(binder.value_map.weight.grad).all().item())
    assert all(parameter.grad is None for parameter in binder.identity.parameters())


class _FakeV5Module(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.state_read_dim = 4
        self.memory_v_proj = nn.Parameter(torch.empty(4, 4))
        self.delta_o_proj = nn.Parameter(torch.eye(4))
        self.original_calls = 0

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return hidden_states

    def _fuse_projected_rwkv_reads(
        self,
        projected: torch.Tensor,
        recurrent: torch.Tensor,
        route_agreement=None,
        query_state_gate=None,
        global_recurrent_reads=None,
        hidden_states=None,
    ) -> torch.Tensor:
        self.original_calls += 1
        return projected + 0.25 * recurrent


def test_installed_wrapper_preserves_base_fusion_and_adds_afterward(monkeypatch) -> None:
    module = _FakeV5Module()
    monkeypatch.setattr(
        certified,
        "iter_delta_mem_modules",
        lambda model: (("layer", module),),
    )
    head = SimpleNamespace(
        heads=nn.ModuleList([ResidualBilinearIdentity(4, bottleneck=2)])
    )
    audit = certified.install(module, head, thresholds=[0.0])
    projected = torch.tensor([[[1.0, 2.0, -1.0, 0.5]]])
    recurrent = torch.tensor([[[0.25, -0.5, 0.75, 1.0]]])
    expected = projected + 0.25 * recurrent

    disabled = module._fuse_projected_rwkv_reads(projected, recurrent)

    assert torch.equal(disabled, expected)
    assert module.original_calls == 1
    assert audit["base_fusion_preserved"] is True

    certified.set_runtime(
        (("layer", module),),
        queries={"layer": projected},
        shadows={"layer": recurrent},
        seq_len=1,
    )
    enabled = module._fuse_projected_rwkv_reads(projected, recurrent)

    assert module.original_calls == 2
    assert not torch.equal(enabled, expected)
    assert module.rwkv_v5_shadow_last_correction is not None
    certified.clear_runtime((("layer", module),))
    restored = module._fuse_projected_rwkv_reads(projected, recurrent)
    assert torch.equal(restored, expected)


def test_installed_wrapper_masks_correction_to_causal_predictors(monkeypatch) -> None:
    module = _FakeV5Module()
    monkeypatch.setattr(
        certified,
        "iter_delta_mem_modules",
        lambda model: (("layer", module),),
    )
    head = SimpleNamespace(
        heads=nn.ModuleList([ResidualBilinearIdentity(4, bottleneck=2)])
    )
    certified.install(module, head, thresholds=[0.0])
    projected = torch.ones(1, 4, 4)
    recurrent = torch.arange(16, dtype=torch.float32).reshape(1, 4, 4) + 1
    mask = torch.tensor([[False, True, True, False]])
    expected = projected + 0.25 * recurrent

    certified.set_runtime(
        (("layer", module),),
        queries={"layer": projected},
        shadows={"layer": recurrent},
        seq_len=4,
        correction_masks={"layer": mask},
    )
    output = module._fuse_projected_rwkv_reads(projected, recurrent)

    assert torch.equal(output[:, ~mask[0]], expected[:, ~mask[0]])
    assert not torch.equal(output[:, mask[0]], expected[:, mask[0]])


def test_live_value_override_changes_only_certified_material_stream(monkeypatch) -> None:
    module = _FakeV5Module()
    monkeypatch.setattr(
        certified,
        "iter_delta_mem_modules",
        lambda model: (("layer", module),),
    )
    head = SimpleNamespace(
        heads=nn.ModuleList([ResidualBilinearIdentity(4, bottleneck=2)])
    )
    certified.install(module, head, thresholds=[0.0])
    projected = torch.ones(1, 2, 4)
    target_recurrent = torch.full((1, 2, 4), 2.0)
    donor_value = torch.tensor(
        [[[1.0, -1.0, 0.5, -0.5], [-0.25, 0.25, -0.75, 0.75]]]
    )
    expected_original_fusion = projected + 0.25 * target_recurrent

    certified.set_runtime(
        (("layer", module),),
        queries={"layer": projected},
        shadows={"layer": target_recurrent},
        seq_len=2,
        live_value_overrides={"layer": donor_value},
    )
    output = module._fuse_projected_rwkv_reads(projected, target_recurrent)

    assert module.original_calls == 1
    assert torch.equal(module.rwkv_v5_shadow_last_value, donor_value)
    assert not torch.equal(output, expected_original_fusion)


def test_trainable_isolation_selects_only_live_value_map(monkeypatch) -> None:
    module = _FakeV5Module()
    monkeypatch.setattr(
        certified,
        "iter_delta_mem_modules",
        lambda model: (("layer", module),),
    )
    head = SimpleNamespace(
        heads=nn.ModuleList([ResidualBilinearIdentity(4, bottleneck=2)])
    )
    certified.install(module, head, thresholds=[0.0])

    selected, audit = certified.configure_trainable_value_maps(module)

    assert len(selected) == 1
    assert selected[0][0].endswith(
        "rwkv_v5_shadow_certified_binder.value_map.weight"
    )
    assert audit["only_live_value_maps_trainable"] is True
    assert all(parameter.requires_grad is False for parameter in module.rwkv_v5_shadow_certified_binder.identity.parameters())


def test_shifted_feedback_is_causal_and_zero_source_is_exact(monkeypatch) -> None:
    module = _FakeV5Module()
    monkeypatch.setattr(
        certified,
        "iter_delta_mem_modules",
        lambda model: (("layer", module),),
    )
    audit = certified.install_shifted_feedback(module, gain=0.125)
    hidden = torch.ones(1, 4, 4)
    source = torch.tensor(
        [[[1.0, 0.0, 0.0, 0.0], [0.0, 2.0, 0.0, 0.0], [0.0, 0.0, 3.0, 0.0], [0.0, 0.0, 0.0, 4.0]]]
    )

    certified.set_shifted_feedback((("layer", module),), {"layer": source})
    output = module(hidden)

    assert audit["first_token_exact_zero"] is True
    assert torch.equal(output[:, 0], hidden[:, 0])
    assert not torch.equal(output[:, 1:], hidden[:, 1:])
    assert module.rwkv_v5_shifted_feedback_last_applied is not None

    certified.set_shifted_feedback(
        (("layer", module),),
        {"layer": torch.zeros_like(source)},
    )
    zero_output = module(hidden)
    assert torch.equal(zero_output, hidden)


def test_shifted_feedback_detaches_previous_pass(monkeypatch) -> None:
    module = _FakeV5Module()
    monkeypatch.setattr(
        certified,
        "iter_delta_mem_modules",
        lambda model: (("layer", module),),
    )
    certified.install_shifted_feedback(module)
    source = torch.randn(1, 3, 4, requires_grad=True)
    hidden = torch.randn(1, 3, 4, requires_grad=True)
    certified.set_shifted_feedback((("layer", module),), {"layer": source})

    module(hidden).square().mean().backward()

    assert source.grad is None
    assert hidden.grad is not None
