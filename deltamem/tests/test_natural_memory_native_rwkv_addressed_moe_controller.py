from __future__ import annotations

import json
from pathlib import Path

import torch

from experiments.rethinking_rwkv_ms_gemma import (
    run_natural_memory_native_rwkv_addressed_moe_controller_causal_train as training,
)
from experiments.rethinking_rwkv_ms_gemma import (
    run_natural_memory_native_rwkv_addressed_moe_controller_screen as screen,
)


class _FakeRWKVCore(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.output = torch.nn.Linear(2, 2, bias=False)


class _FakeMoELayer(torch.nn.Module):
    def __init__(self) -> None:
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
        self.projected_kv_key_proj = torch.nn.Parameter(torch.zeros(2, 2))
        self.unrelated = torch.nn.Parameter(torch.zeros(2, 2))


def test_protocols_lock_four_gpu_moe_causal_contract() -> None:
    screen_protocol = screen.validate_protocol()
    causal_protocol = training.validate_protocol()

    assert screen_protocol["architecture"]["hybrid_mode"] == "addressed_moe_controller"
    assert screen_protocol["execution"]["hardware"] == "exactly four distinct A100 GPUs"
    assert causal_protocol["training"]["hardware"] == "exactly four distinct A100 GPUs"
    assert causal_protocol["architecture"]["trainable_parameter_families"] == [
        suffix.removeprefix(".") for suffix in training.MOE_SUFFIXES
    ]
    assert causal_protocol["protected_splits_opened_by_this_protocol"] == []


def test_moe_trainable_contract_includes_every_router_family() -> None:
    model = torch.nn.ModuleList([_FakeMoELayer() for _ in range(42)])

    selected, audit = training.configure_moe_parameters(model)

    assert len(selected) == 378
    assert audit["passed"] is True
    assert audit["moe_router_trainable_tensors"] == 168
    assert all(count == 42 for count in audit["family_counts"].values())
    assert not any(
        parameter.requires_grad
        for name, parameter in model.named_parameters()
        if name.endswith((".projected_kv_key_proj", ".unrelated"))
    )


def test_moe_bindings_install_and_restore_trainable_contract() -> None:
    shared = training.base.affine_train.shared
    original_screen = shared.screen
    original_configurer = shared.TRAINABLE_CONFIGURER

    with training.bindings():
        assert shared.screen is screen
        assert shared.TRAINABLE_CONFIGURER is training.configure_moe_parameters

    assert shared.screen is original_screen
    assert shared.TRAINABLE_CONFIGURER is original_configurer


def test_signed_screen_authorizes_training_only() -> None:
    result_path = (
        Path(screen.__file__).resolve().parent
        / "local_artifacts/natural_memory_native_rwkv_addressed_moe_controller_screen_v4/"
        "result.json"
    )
    result = json.loads(result_path.read_text(encoding="utf-8"))
    unsigned = dict(result)
    receipt = unsigned.pop("receipt")

    assert screen.sha256_file(result_path) == training.SCREEN_RESULT_FILE_SHA256
    assert screen.canonical_sha256(unsigned) == receipt["payload_sha256"]
    assert receipt["payload_sha256"] == training.SCREEN_RESULT_RECEIPT
    assert result["status"] == screen.PASS_STATUS
    assert result["passed"] is True
    assert result["training_authorized"] is True
    assert result["native_generation_authorized"] is False
    assert result["protected_splits_opened"] == []
    assert len(result["rank_evidence"]) == 4
