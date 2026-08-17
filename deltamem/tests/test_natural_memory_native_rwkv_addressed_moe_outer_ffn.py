from __future__ import annotations

import json
from pathlib import Path

import torch

from experiments.rethinking_rwkv_ms_gemma import (
    run_natural_memory_native_rwkv_addressed_moe_outer_ffn_causal_train as training,
)
from experiments.rethinking_rwkv_ms_gemma import (
    run_natural_memory_native_rwkv_addressed_moe_outer_ffn_gain_ablation_screen as gain_screen,
)
from experiments.rethinking_rwkv_ms_gemma import (
    run_natural_memory_native_rwkv_addressed_moe_outer_ffn_scaled_screen as screen,
)


class _FakeRWKVCore(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.output = torch.nn.Linear(2, 2, bias=False)


class _FakeOuterFFNLayer(torch.nn.Module):
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
        self.rwkv_outer_ffn_down_weight = torch.nn.Parameter(torch.zeros(2, 2))
        self.rwkv_outer_ffn_gate_weight = torch.nn.Parameter(torch.zeros(2, 2))
        self.rwkv_outer_ffn_up_weight = torch.nn.Parameter(torch.zeros(2, 2))
        self.projected_kv_key_proj = torch.nn.Parameter(torch.zeros(2, 2))
        self.unrelated = torch.nn.Parameter(torch.zeros(2, 2))


def test_outer_ffn_protocols_lock_four_gpu_causal_contract() -> None:
    screen_protocol = screen.validate_protocol()
    causal_protocol = training.validate_protocol()

    assert screen_protocol["architecture"]["outer_ffn_gain"] == 1.0 / 2048.0
    assert screen_protocol["execution"]["hardware"] == "exactly four distinct A100 GPUs"
    assert causal_protocol["training"]["hardware"] == "exactly four distinct A100 GPUs"
    assert causal_protocol["architecture"]["expected_trainable_parameter_tensors"] == 504
    assert causal_protocol["heldout_causal_endpoint"]["rows"] == 12
    assert causal_protocol["protected_splits_opened_by_this_protocol"] == []


def test_outer_ffn_trainable_contract_includes_all_post_mlp_families() -> None:
    model = torch.nn.ModuleList([_FakeOuterFFNLayer() for _ in range(42)])

    selected, audit = training.configure_outer_ffn_parameters(model)

    assert len(selected) == 504
    assert audit["passed"] is True
    assert audit["moe_router_trainable_tensors"] == 168
    assert audit["outer_ffn_trainable_tensors"] == 126
    assert all(count == 42 for count in audit["family_counts"].values())
    assert not any(
        parameter.requires_grad
        for name, parameter in model.named_parameters()
        if name.endswith((".projected_kv_key_proj", ".unrelated"))
    )


def test_outer_ffn_bindings_install_and_restore_trainable_contract() -> None:
    shared = training.base.affine_train.shared
    original_screen = shared.screen
    original_configurer = shared.TRAINABLE_CONFIGURER

    with training.bindings():
        assert shared.screen is screen
        assert shared.TRAINABLE_CONFIGURER is training.configure_outer_ffn_parameters

    assert shared.screen is original_screen
    assert shared.TRAINABLE_CONFIGURER is original_configurer


def test_signed_scaled_screen_authorizes_training_only() -> None:
    result_path = (
        Path(screen.__file__).resolve().parent
        / "local_artifacts/natural_memory_native_rwkv_addressed_moe_outer_ffn_scaled_screen_v1/"
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
    assert result["model_audit"]["outer_ffn_hook_count"] == 42
    assert result["protected_splits_opened"] == []


def test_signed_same_mode_gain_ablation_stops_outer_ffn_branch() -> None:
    protocol = gain_screen.validate_protocol()
    result_path = (
        Path(gain_screen.__file__).resolve().parent
        / "local_artifacts/"
        "natural_memory_native_rwkv_addressed_moe_outer_ffn_gain_ablation_screen_v1/"
        "result.json"
    )
    result = json.loads(result_path.read_text(encoding="utf-8"))
    unsigned = dict(result)
    receipt = unsigned.pop("receipt")

    assert protocol["architecture"]["hybrid_mode"] == "addressed_moe_outer_ffn"
    assert protocol["architecture"]["outer_ffn_gain"] == 1.0 / 8192.0
    assert "only rwkv_ms_outer_ffn_gain" in protocol["architecture"]["direct_ablation"]
    assert gain_screen.sha256_file(result_path) == (
        "645f2dbe5098b23366797de154065677cd5fb57ad7e7bc5d733380ef0168c285"
    )
    assert gain_screen.canonical_sha256(unsigned) == receipt["payload_sha256"]
    assert receipt["payload_sha256"] == (
        "5c0e2babcd68aaeeb7db381f137c9abca183f716ddfe723fb35d42061c293278"
    )
    assert result["status"] == gain_screen.FAIL_STATUS
    assert result["passed"] is False
    assert result["training_authorized"] is False
    assert result["native_generation_authorized"] is False
    assert result["protected_splits_opened"] == []
    assert [
        row["comparisons"]["outer_ffn_gain_vs_zero_gain"]["max_abs_logit_delta"]
        for row in result["rank_evidence"]
    ] == [1.46875, 1.375, 1.65625, 1.125]
