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
    with pytest.raises(ValueError, match="flash_attention_2"):
        module.set_virtual_kv_provider(lambda **kwargs: None)
