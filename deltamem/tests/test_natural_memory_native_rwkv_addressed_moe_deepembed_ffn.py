from __future__ import annotations

import torch

from experiments.rethinking_rwkv_ms_gemma import (
    run_natural_memory_native_rwkv_addressed_moe_deepembed_ffn_sparse_causal_train as training,
)
from experiments.rethinking_rwkv_ms_gemma import (
    run_natural_memory_native_rwkv_addressed_moe_deepembed_ffn_sparse_screen as screen,
)


class _FakeRWKVCore(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.output = torch.nn.Linear(2, 2, bias=False)


class _FakeSparseDeepEmbedLayer(torch.nn.Module):
    def __init__(self, *, outer_ffn: bool) -> None:
        super().__init__()
        self.hrm_rwkv7_core = _FakeRWKVCore()
        self.delta_o_proj = torch.nn.Parameter(torch.zeros(2, 2))
        self.memory_fusion_hidden_weight = torch.nn.Parameter(torch.zeros(1, 2))
        self.memory_fusion_read_weight = torch.nn.Parameter(torch.zeros(1, 2))
        self.memory_fusion_bias = torch.nn.Parameter(torch.zeros(1))
        self.rwkv_moe_hidden_weight = torch.nn.Parameter(torch.zeros(3, 2))
        self.rwkv_moe_addressed_weight = torch.nn.Parameter(torch.zeros(3, 2))
        self.rwkv_moe_global_weight = torch.nn.Parameter(torch.zeros(3, 2))
        self.rwkv_moe_bias = torch.nn.Parameter(torch.zeros(3))
        if outer_ffn:
            self.rwkv_outer_ffn_down_weight = torch.nn.Parameter(torch.zeros(2, 2))
            self.rwkv_outer_ffn_gate_weight = torch.nn.Parameter(torch.zeros(2, 2))
            self.rwkv_outer_ffn_up_weight = torch.nn.Parameter(torch.zeros(2, 2))
        self.projected_kv_key_proj = torch.nn.Parameter(torch.zeros(2, 2))
        self.unrelated = torch.nn.Parameter(torch.zeros(2, 2))


def test_sparse_deepembed_protocol_locks_clean_four_gpu_endpoint() -> None:
    protocol = training.validate_protocol()

    assert protocol["training"]["hardware"] == "exactly four distinct A100 GPUs"
    assert protocol["architecture"]["attention_layers"] == "0..41"
    assert protocol["architecture"]["outer_ffn_layers"] == [10, 21, 31, 41]
    assert protocol["architecture"]["expected_trainable_parameter_tensors"] == 390
    assert protocol["heldout_causal_endpoint"]["rows"] == 11
    assert 718 not in protocol["heldout_causal_endpoint"]["source_ordinals"]
    assert protocol["protected_splits_opened_by_this_protocol"] == []


def test_sparse_deepembed_trainable_contract_keeps_only_four_channelmix_anchors() -> None:
    model = torch.nn.ModuleList(
        [
            _FakeSparseDeepEmbedLayer(outer_ffn=index in screen.OUTER_FFN_LAYERS)
            for index in range(42)
        ]
    )

    selected, audit = training.configure_sparse_deepembed_parameters(model)

    assert len(selected) == 390
    assert audit["passed"] is True
    assert audit["attention_trainable_tensors"] == 378
    assert audit["moe_router_trainable_tensors"] == 168
    assert audit["outer_ffn_trainable_tensors"] == 12
    assert audit["outer_ffn_active_layers"] == [10, 21, 31, 41]
    assert all(
        audit["family_counts"][suffix] == 4
        for suffix in training.base.OUTER_FFN_SUFFIXES[-3:]
    )
    assert not any(
        parameter.requires_grad
        for name, parameter in model.named_parameters()
        if name.endswith((".projected_kv_key_proj", ".unrelated"))
    )


def test_sparse_deepembed_bindings_install_and_restore_training_contract() -> None:
    shared = training.base.base.affine_train.shared
    original_screen = shared.screen
    original_configurer = shared.TRAINABLE_CONFIGURER

    with training.bindings():
        assert shared.screen is screen
        assert shared.TRAINABLE_CONFIGURER is training.configure_sparse_deepembed_parameters

    assert shared.screen is original_screen
    assert shared.TRAINABLE_CONFIGURER is original_configurer
