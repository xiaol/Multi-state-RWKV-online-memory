from __future__ import annotations

from types import MethodType

import pytest
import torch

from deltamem.core.delta import DeltaMemAttention, HFDeltaMemConfig
from deltamem.tests.test_delta_mem_regressions import make_qwen3_attention
from experiments.rethinking_rwkv_ms_gemma import rwkv_continuous_write_integration as continuous


def _module() -> DeltaMemAttention:
    torch.manual_seed(0)
    return DeltaMemAttention(
        make_qwen3_attention(),
        HFDeltaMemConfig(
            rank=2,
            memory_backend="rwkv_ms",
            rwkv_ms_num_states=2,
            rwkv_ms_chunk_size=2,
            memory_readout_mode="projected_kv_rwkv_hybrid",
            memory_fusion_mode="content_gated_add",
            memory_fusion_gate_init=0.25,
            projected_kv_key_dim=4,
            projected_kv_temperature=4.0,
            projected_kv_update_cosine_threshold=1.0,
            memory_write_granularity="token",
            output_init="base_slice_fixed",
            base_slice_ref_width=2,
            rwkv_ms_output_init_scale=0.02,
            rwkv_ms_hybrid_mode="address_keyed_moe_deepembed_ffn",
            rwkv_ms_hybrid_gain=1.0 / 64.0,
            rwkv_ms_write_address_gain=0.25,
            rwkv_ms_outer_ffn_gain=1.0 / 128.0,
            rwkv_ms_outer_ffn_layers=(0,),
        ),
    )


def _byte_equal(left: torch.Tensor, right: torch.Tensor) -> bool:
    return torch.equal(
        left.detach().contiguous().view(torch.uint8),
        right.detach().contiguous().view(torch.uint8),
    )


def test_post_write_latch_uses_new_key_and_never_recomputes_live_address() -> None:
    module = _module()
    old_keys = torch.tensor(
        [[[-1.0, -2.0, -3.0, -4.0], [10.0, 11.0, 12.0, 13.0]]]
    )
    new_keys = torch.tensor(
        [[[2.0, 3.0, 4.0, 5.0], [10.0, 11.0, 12.0, 13.0]]]
    )
    write_routes = torch.tensor([[[1.0, 0.0]]])
    module.projected_kv_keys = old_keys.clone()

    def overwrite(self, hidden_states, token_mask) -> None:
        self.projected_kv_keys = new_keys.clone()
        self.last_write_routes = write_routes.clone()

    module._write_projected_kv_slots = MethodType(overwrite, module)
    audit = continuous.install(
        module,
        rank=2,
        seed=7,
        k_gain=0.5,
        a_gain=0.5,
        b_gain=0.5,
    )
    continuous.set_capture(module, True)
    hidden = torch.randn(1, 3, module.hidden_size)
    token_mask = torch.tensor([[True, True, False]])

    module._write_projected_kv_slots(hidden, token_mask)
    latch = module.rwkv_continuous_write_latch
    assert latch is not None
    assert latch.address_seq.requires_grad is False
    assert torch.equal(latch.keys, new_keys)
    assert torch.equal(latch.routes, write_routes)
    assert torch.equal(latch.selected_keys, new_keys[:, :1])
    expected_address = new_keys[:, :1].expand(-1, 3, -1).clone()
    expected_address[:, 2] = 0.0
    assert torch.equal(latch.address_seq, expected_address)

    module.projected_kv_keys = old_keys.clone()
    module.last_write_routes = torch.tensor([[[0.0, 1.0]]])
    address = module._projected_rwkv_write_address_sequence(hidden, token_mask)
    assert address is latch.address_seq
    assert torch.equal(address, expected_address)

    shape = (1, 3, module.state_read_dim)
    k = torch.tensor([[[1.0, -2.0], [3.0, -4.0], [5.0, -6.0]]])
    signed_zero = torch.tensor(-0.0)
    v = torch.randn(shape)
    v[0, 0, 0] = signed_zero
    a = torch.full(shape, 0.25)
    b = torch.full(shape, -0.5)
    outputs = module._rwkv_ms_address_conditioned_write_features(
        k,
        v,
        a,
        b,
        address,
        token_mask,
    )

    assert audit["conditioned_features"] == ("k", "a", "b")
    assert outputs[1] is v
    assert _byte_equal(outputs[1], v)
    assert not torch.equal(outputs[0][:, :2], k[:, :2])
    assert not torch.equal(outputs[2][:, :2], a[:, :2])
    assert not torch.equal(outputs[3][:, :2], b[:, :2])
    for output, feature in zip(outputs, (k, v, a, b)):
        assert _byte_equal(output[:, 2], feature[:, 2])

    capture = module.rwkv_continuous_write_audit
    assert capture is not None
    assert capture["conditioner_address"] is address
    assert capture["conditioner_address_object_id"] == id(address)
    assert capture["latched_address_object_id"] == id(address)
    assert torch.equal(capture["conditioner_address_value"], expected_address)
    assert capture["value_object_id"] == capture["returned_value_object_id"]


def test_zero_latched_address_is_exact_same_object_noop() -> None:
    module = _module()

    def no_write(self, hidden_states, token_mask) -> None:
        self.last_write_routes = None
        self.projected_kv_keys = torch.randn(
            hidden_states.shape[0],
            self.rwkv_ms_num_states,
            self.projected_kv_key_dim,
        )

    module._write_projected_kv_slots = MethodType(no_write, module)
    continuous.install(module, rank=2, seed=11)
    hidden = torch.randn(2, 4, module.hidden_size)
    token_mask = torch.tensor(
        [[True, True, True, True], [True, True, True, False]]
    )
    module._write_projected_kv_slots(hidden, token_mask)
    address = module._projected_rwkv_write_address_sequence(hidden, token_mask)
    assert torch.equal(address, torch.zeros_like(address))

    shape = (2, 4, module.state_read_dim)
    features = tuple(torch.randn(shape, dtype=torch.bfloat16) for _ in range(4))
    outputs = module._rwkv_ms_address_conditioned_write_features(
        *features,
        address,
        token_mask,
    )
    assert all(output is feature for output, feature in zip(outputs, features))
    assert all(_byte_equal(output, feature) for output, feature in zip(outputs, features))


def test_mixed_batch_zero_address_row_is_byte_exact_noop() -> None:
    conditioner = continuous.ContinuousWriteConditioner(
        address_dim=4,
        feature_dim=2,
        rank=2,
        seed=19,
        k_gain=0.5,
        a_gain=0.5,
        b_gain=0.5,
        trainable_map=False,
    )
    shape = (2, 3, 2)
    features = tuple(torch.randn(shape, dtype=torch.bfloat16) for _ in range(4))
    address = torch.randn(2, 3, 4)
    address[0] = 0.0
    token_mask = torch.ones(2, 3, dtype=torch.bool)

    outputs = conditioner(*features, address, token_mask)

    for output, feature in zip(outputs, features):
        assert _byte_equal(output[0], feature[0])
    assert outputs[1] is features[1]
    assert not _byte_equal(outputs[0][1], features[0][1])


def test_active_address_rejects_zero_or_nonfinite_direction() -> None:
    conditioner = continuous.ContinuousWriteConditioner(
        address_dim=4,
        feature_dim=2,
        rank=2,
        seed=23,
        k_gain=0.5,
        a_gain=0.5,
        b_gain=0.5,
        trainable_map=False,
    )
    with torch.no_grad():
        conditioner.down.zero_()
    with pytest.raises(RuntimeError, match="zero direction"):
        conditioner.direction(torch.ones(1, 4))

    with pytest.raises(ValueError, match="address is nonfinite"):
        conditioner.direction(torch.tensor([[float("nan"), 0.0, 0.0, 0.0]]))


def test_disabled_mode_is_exact_inherited_v5_baseline() -> None:
    module = _module()
    continuous.install(module, rank=2, seed=29)
    hidden = torch.randn(2, 3, module.hidden_size, dtype=torch.bfloat16)
    token_mask = torch.tensor([[True, True, True], [True, True, False]])
    module._write_projected_kv_slots(hidden, token_mask)
    latch = module.rwkv_continuous_write_latch
    assert latch is not None
    live_folded = module.rwkv_continuous_write_original_address_sequence(
        hidden,
        token_mask,
    )
    assert _byte_equal(latch.folded_address_seq, live_folded)

    shape = (2, 3, module.state_read_dim)
    features = tuple(torch.randn(shape, dtype=torch.bfloat16) for _ in range(4))
    expected = module.rwkv_continuous_write_original_conditioner(
        *features,
        live_folded,
        token_mask,
    )
    continuous.set_enabled(module, False)
    actual = module._rwkv_ms_address_conditioned_write_features(
        *features,
        latch.address_seq,
        token_mask,
    )

    assert module.rwkv_continuous_write_mode == continuous.INHERITED_EXACT_V5_MODE
    assert all(_byte_equal(left, right) for left, right in zip(actual, expected))
    assert actual[1] is not features[1]

    continuous.set_mode(module, continuous.RAW_UNCONDITIONED_MODE)
    raw = module._rwkv_ms_address_conditioned_write_features(
        *features,
        latch.address_seq,
        token_mask,
    )
    assert all(output is feature for output, feature in zip(raw, features))


def test_latched_address_rejects_in_place_mutation() -> None:
    module = _module()

    def write(self, hidden_states, token_mask) -> None:
        self.projected_kv_keys = torch.ones(
            hidden_states.shape[0],
            self.rwkv_ms_num_states,
            self.projected_kv_key_dim,
        )
        self.last_write_routes = torch.tensor([[[1.0, 0.0]]])

    module._write_projected_kv_slots = MethodType(write, module)
    continuous.install(module, rank=2, seed=13)
    hidden = torch.randn(1, 2, module.hidden_size)
    token_mask = torch.ones(1, 2, dtype=torch.bool)
    module._write_projected_kv_slots(hidden, token_mask)
    module.rwkv_continuous_write_latch.address_seq.add_(1.0)

    with pytest.raises(RuntimeError, match="immutable address was mutated"):
        module._projected_rwkv_write_address_sequence(hidden, token_mask)


def test_native_write_step_consumes_latched_full_address_and_reset_clears_it() -> None:
    module = _module()
    continuous.install(module, rank=2, seed=17)
    continuous.set_capture(module, True)
    module.set_write_enabled(True)
    hidden = torch.randn(2, 3, module.hidden_size)
    token_mask = torch.tensor([[True, True, True], [True, True, False]])
    state = torch.zeros(
        hidden.shape[0],
        module.num_state_heads,
        module.rwkv_ms_num_states,
        module.rank,
        module.rank,
    )

    next_state, reads, _, _ = module._projected_rwkv_hybrid_step(
        state,
        hidden,
        token_mask,
    )

    latch = module.rwkv_continuous_write_latch
    capture = module.rwkv_continuous_write_audit
    assert latch is not None
    assert capture is not None
    assert capture["conditioner_address"] is latch.address_seq
    assert capture["value_object_id"] == capture["returned_value_object_id"]
    assert latch.address_seq.shape == (2, 3, module.projected_kv_key_dim)
    assert next_state.shape == state.shape
    assert reads.shape == (2, 3, module.state_read_dim)
    assert torch.isfinite(next_state).all()
    assert torch.isfinite(reads).all()

    module.reset_state()
    assert module.rwkv_continuous_write_latch is None
    assert module.rwkv_continuous_write_audit is None
