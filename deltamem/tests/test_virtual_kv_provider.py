from __future__ import annotations

import pytest
import torch

from deltamem.core.virtual_kv import ExplicitRWKVVirtualKV, VirtualKVShape
from deltamem.tests.test_delta_mem_regressions import (
    make_delta_module,
    make_position_embeddings,
)


def test_explicit_virtual_kv_zero_state_is_exactly_disabled() -> None:
    builder = ExplicitRWKVVirtualKV(
        VirtualKVShape(
            key_dim=6,
            state_heads=1,
            rank=4,
            slots=3,
            kv_heads=1,
            head_dim=4,
            probe_rank=3,
            value_hidden=7,
        )
    )
    kwargs = {
        "state": torch.zeros(1, 1, 3, 4, 4),
        "address_keys": torch.randn(1, 3, 6),
        "occupied": torch.ones(1, 3, dtype=torch.bool),
        "query_states": torch.randn(1, 2, 1, 4),
        "real_keys": torch.randn(1, 1, 2, 4),
        "real_values": torch.randn(1, 1, 2, 4),
        "attention_mask": None,
    }
    assert builder(**kwargs) is None


def test_explicit_virtual_kv_zero_address_is_exactly_disabled() -> None:
    builder = ExplicitRWKVVirtualKV(
        VirtualKVShape(
            key_dim=6,
            state_heads=1,
            rank=4,
            slots=1,
            kv_heads=1,
            head_dim=4,
            probe_rank=3,
            value_hidden=7,
        )
    )
    assert (
        builder(
            state=torch.randn(1, 1, 1, 4, 4),
            address_keys=torch.zeros(1, 1, 6),
            occupied=torch.ones(1, 1, dtype=torch.bool),
            query_states=torch.randn(1, 2, 1, 4),
            real_keys=torch.randn(1, 1, 2, 4),
            real_values=torch.randn(1, 1, 2, 4),
            attention_mask=None,
        )
        is None
    )


def test_explicit_virtual_kv_has_active_equal_norm_payload_and_mask() -> None:
    torch.manual_seed(3)
    builder = ExplicitRWKVVirtualKV(
        VirtualKVShape(
            key_dim=6,
            state_heads=1,
            rank=4,
            slots=3,
            kv_heads=1,
            head_dim=4,
            probe_rank=3,
            value_hidden=7,
        )
    )
    keys, values, mask = builder(
        state=torch.randn(1, 1, 3, 4, 4),
        address_keys=torch.randn(1, 3, 6),
        occupied=torch.tensor([[True, False, True]]),
        query_states=torch.randn(1, 2, 1, 4),
        real_keys=torch.randn(1, 1, 2, 4),
        real_values=torch.randn(1, 1, 2, 4),
        attention_mask=None,
    )
    assert keys.shape == (1, 1, 3, 4)
    assert values.shape == (1, 1, 3, 4)
    assert mask.shape == (1, 1, 1, 5)
    torch.testing.assert_close(
        keys[:, :, [0, 2]].float().square().mean(dim=-1),
        torch.ones((1, 1, 2)),
        atol=1e-5,
        rtol=1e-5,
    )
    assert torch.isneginf(mask[:, :, :, 3:]).logical_not().all()
    assert mask[0, 0, 0, 2].item() == 0.0
    assert mask[0, 0, 0, 3].item() < 0.0
    assert mask[0, 0, 0, 4].item() == 0.0


def test_explicit_virtual_kv_preserves_bool_mask_semantics() -> None:
    builder = ExplicitRWKVVirtualKV(
        VirtualKVShape(
            key_dim=6,
            state_heads=1,
            rank=4,
            slots=3,
            kv_heads=1,
            head_dim=4,
            probe_rank=3,
            value_hidden=7,
        )
    )
    real_mask = torch.tensor([[[[True, False]]]])
    _, _, mask = builder(
        state=torch.randn(1, 1, 3, 4, 4),
        address_keys=torch.randn(1, 3, 6),
        occupied=torch.tensor([[True, False, True]]),
        query_states=torch.randn(1, 2, 1, 4),
        real_keys=torch.randn(1, 1, 2, 4),
        real_values=torch.randn(1, 1, 2, 4),
        attention_mask=real_mask,
    )
    assert mask.dtype == torch.bool
    assert torch.equal(mask[..., :2], real_mask)
    assert torch.equal(mask[..., 2:], torch.tensor([[[[True, False, True]]]]))


def test_explicit_virtual_kv_bias_uses_zero_keys_and_only_changes_suffix() -> None:
    builder = ExplicitRWKVVirtualKV(
        VirtualKVShape(
            key_dim=6,
            state_heads=1,
            rank=4,
            slots=3,
            kv_heads=1,
            head_dim=4,
            probe_rank=3,
            value_hidden=7,
        )
    )
    real_mask = torch.tensor([[[[0.0, -7.0]]]])
    keys, values, mask = builder(
        state=torch.randn(1, 1, 3, 4, 4),
        address_keys=torch.randn(1, 3, 6),
        occupied=torch.tensor([[True, False, True]]),
        query_states=torch.randn(1, 2, 1, 4),
        real_keys=torch.randn(1, 1, 2, 4),
        real_values=torch.randn(1, 1, 2, 4),
        attention_mask=real_mask,
        attention_bias=torch.tensor([[1.25, 99.0, -0.75]]),
    )
    assert torch.equal(keys, torch.zeros_like(keys))
    assert values[:, :, 0].square().sum().item() > 0.0
    assert values[:, :, 1].square().sum().item() == 0.0
    assert values[:, :, 2].square().sum().item() > 0.0
    assert torch.equal(mask[..., :2], real_mask)
    assert mask[0, 0, 0, 2].item() == 1.25
    assert mask[0, 0, 0, 3].item() == torch.finfo(mask.dtype).min
    assert mask[0, 0, 0, 4].item() == -0.75


def test_explicit_virtual_kv_bias_rejects_bool_mask() -> None:
    builder = ExplicitRWKVVirtualKV(
        VirtualKVShape(
            key_dim=6,
            state_heads=1,
            rank=4,
            slots=1,
            kv_heads=1,
            head_dim=4,
            probe_rank=3,
            value_hidden=7,
        )
    )
    with pytest.raises(ValueError, match="additive floating mask"):
        builder(
            state=torch.randn(1, 1, 1, 4, 4),
            address_keys=torch.randn(1, 1, 6),
            occupied=torch.ones(1, 1, dtype=torch.bool),
            query_states=torch.randn(1, 2, 1, 4),
            real_keys=torch.randn(1, 1, 2, 4),
            real_values=torch.randn(1, 1, 2, 4),
            attention_mask=torch.ones(1, 1, 1, 2, dtype=torch.bool),
            attention_bias=torch.ones(1, 1),
        )


def test_rwkv_read_basis_publishes_receptance_only_for_provider_lifecycle() -> None:
    module = make_delta_module(
        output_init="zero",
        rank=4,
        num_state_heads=1,
        memory_backend="rwkv_ms",
        rwkv_ms_num_states=3,
    )
    state = module._ensure_state(1, torch.device("cpu"), torch.float32)
    source = torch.randn(1, 2, module.state_read_dim)
    without_provider, _, _ = module._rwkv_ms_token_state_read_basis(state, source, None)
    assert module.rwkv_virtual_router_receptance is None
    module.set_virtual_kv_provider(lambda **kwargs: None)
    receptance, _, _ = module._rwkv_ms_token_state_read_basis(state, source, None)
    assert module.rwkv_virtual_router_receptance is receptance
    assert tuple(receptance.shape) == (1, 2, 1, 4)
    torch.testing.assert_close(receptance, without_provider)
    module.clear_virtual_kv_provider()
    assert module.rwkv_virtual_router_receptance is None


def test_explicit_virtual_kv_normalizes_each_kv_head_independently() -> None:
    builder = ExplicitRWKVVirtualKV(
        VirtualKVShape(
            key_dim=6,
            state_heads=1,
            rank=4,
            slots=3,
            kv_heads=2,
            head_dim=4,
            probe_rank=3,
            value_hidden=7,
        )
    )
    keys, values, _ = builder(
        state=torch.randn(1, 1, 3, 4, 4),
        address_keys=torch.randn(1, 3, 6),
        occupied=torch.ones(1, 3, dtype=torch.bool),
        query_states=torch.randn(1, 2, 1, 4),
        real_keys=torch.randn(1, 2, 2, 4),
        real_values=torch.randn(1, 2, 2, 4),
        attention_mask=None,
    )
    torch.testing.assert_close(
        keys.float().square().mean(dim=-1),
        torch.ones(1, 2, 3),
        atol=1e-5,
        rtol=1e-5,
    )
    torch.testing.assert_close(
        values.float().square().mean(dim=-1),
        torch.ones(1, 2, 3),
        atol=1e-5,
        rtol=1e-5,
    )


def test_provider_appends_ephemeral_kv_without_changing_default_path() -> None:
    torch.manual_seed(5)
    module = make_delta_module(output_init="zero", rank=2)
    x = torch.randn(1, 1, 8)
    position_embeddings = make_position_embeddings(
        batch_size=1,
        seq_len=1,
        head_dim=module.head_dim,
        device=x.device,
        dtype=x.dtype,
    )
    baseline, _ = module(x, position_embeddings, None)

    def provider(**kwargs):
        real_keys = kwargs["key_states"]
        real_values = kwargs["value_states"]
        virtual_keys = torch.ones_like(real_keys)
        virtual_values = torch.ones_like(real_values)
        mask = torch.zeros(
            1,
            1,
            1,
            real_keys.size(2) + 1,
            dtype=real_keys.dtype,
        )
        return virtual_keys, virtual_values, mask

    module.reset_state()
    module.set_virtual_kv_provider(provider)
    output, weights = module(x, position_embeddings, None)
    assert output.shape == baseline.shape
    assert weights is not None
    assert weights.shape[-1] == 2
    module.set_virtual_kv_provider(None)
    module.reset_state()
    restored, _ = module(x, position_embeddings, None)
    assert torch.equal(restored, baseline)


def test_provider_rejects_non_single_query() -> None:
    module = make_delta_module(output_init="zero", rank=2)
    module.set_virtual_kv_provider(
        lambda **kwargs: (
            torch.ones_like(kwargs["key_states"]),
            torch.ones_like(kwargs["value_states"]),
            torch.zeros(1, 1, 2, 2),
        )
    )
    x = torch.randn(1, 2, 8)
    position_embeddings = make_position_embeddings(
        batch_size=1,
        seq_len=2,
        head_dim=module.head_dim,
        device=x.device,
        dtype=x.dtype,
    )
    with pytest.raises(ValueError, match="query length 1"):
        module(x, position_embeddings, None)


def test_provider_rejects_flash_attention_two() -> None:
    module = make_delta_module(output_init="zero", rank=2)
    module.base.config._attn_implementation = "flash_attention_2"
    with pytest.raises(ValueError, match="require eager attention"):
        module.set_virtual_kv_provider(lambda **kwargs: None)


@pytest.mark.parametrize(
    ("attribute", "initial", "message"),
    (
        ("delta_state", torch.ones(1, 1, 1, 2, 2), "mutated RWKV state"),
        ("projected_kv_keys", torch.ones(1, 1, 2), "mutated projected address keys"),
        (
            "projected_kv_occupied",
            torch.ones(1, 1, dtype=torch.bool),
            "mutated projected occupancy",
        ),
    ),
)
def test_provider_rejects_removing_audited_memory_sidecar(
    attribute: str,
    initial: torch.Tensor,
    message: str,
) -> None:
    module = make_delta_module(output_init="zero", rank=2)
    setattr(module, attribute, initial)

    def provider(**kwargs):
        setattr(kwargs["module"], attribute, None)
        return None

    module.set_virtual_kv_provider(provider)
    module.rwkv_virtual_router_receptance = torch.ones(1, 1, 1, 2)
    module.rwkv_virtual_router_receptance_calls = 2
    with pytest.raises(RuntimeError, match=message):
        module._append_rwkv_virtual_kv(
            torch.ones(1, 1, 1, 2),
            torch.ones(1, 1, 1, 2),
            torch.ones(1, 1, 1, 2),
            None,
            position_embeddings=(torch.ones(1, 1, 2), torch.zeros(1, 1, 2)),
        )
    assert module.rwkv_virtual_router_receptance is None
    assert module.rwkv_virtual_router_receptance_calls == 0


def test_co_rotated_virtual_key_has_prompt_shift_invariant_logit() -> None:
    torch.manual_seed(13)
    builder = ExplicitRWKVVirtualKV(
        VirtualKVShape(
            key_dim=6,
            state_heads=1,
            rank=4,
            slots=1,
            kv_heads=1,
            head_dim=4,
            probe_rank=3,
            value_hidden=7,
            co_rotate_keys=True,
        )
    )
    state = torch.randn(1, 1, 1, 4, 4)
    address = torch.randn(1, 1, 6)
    occupied = torch.ones(1, 1, dtype=torch.bool)
    pre_rope_query = torch.randn(1, 1, 1, 4)
    real_keys = torch.randn(1, 1, 2, 4)
    real_values = torch.randn(1, 1, 2, 4)

    def rotated(angle: float) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        cos = torch.full((1, 1, 4), torch.cos(torch.tensor(angle)))
        sin = torch.full((1, 1, 4), torch.sin(torch.tensor(angle)))
        query = pre_rope_query * cos[:, None] + builder._rotate_half(
            pre_rope_query
        ) * sin[:, None]
        keys, _, _ = builder(
            state=state,
            address_keys=address,
            occupied=occupied,
            query_states=query,
            real_keys=real_keys,
            real_values=real_values,
            attention_mask=None,
            position_embeddings=(cos, sin),
        )
        return query, keys, torch.einsum("bhqd,bhkd->bhqk", query, keys)

    _, _, first_logit = rotated(0.2)
    _, _, shifted_logit = rotated(1.1)
    torch.testing.assert_close(first_logit, shifted_logit, atol=1e-5, rtol=1e-5)


def test_co_rotated_virtual_key_requires_position_embeddings() -> None:
    builder = ExplicitRWKVVirtualKV(
        VirtualKVShape(
            key_dim=6,
            state_heads=1,
            rank=4,
            slots=1,
            kv_heads=1,
            head_dim=4,
            probe_rank=4,
            co_rotate_keys=True,
        )
    )
    with pytest.raises(ValueError, match="require query position embeddings"):
        builder(
            state=torch.randn(1, 1, 1, 4, 4),
            address_keys=torch.randn(1, 1, 6),
            occupied=torch.ones(1, 1, dtype=torch.bool),
            query_states=torch.randn(1, 1, 1, 4),
            real_keys=torch.randn(1, 1, 1, 4),
            real_values=torch.randn(1, 1, 1, 4),
            attention_mask=None,
        )
