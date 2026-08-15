from __future__ import annotations

import copy

import pytest
import torch

from deltamem.core.delta import DeltaMemAttention, HFDeltaMemConfig
from deltamem.core.delta import snapshot_delta_mem_weights
from deltamem.tests.test_delta_mem_regressions import make_qwen3_attention
from experiments.rethinking_rwkv_ms_gemma import (
    run_natural_memory_native_projected_rwkv_hybrid_benchmark_train as runner,
)


def _model(readout_mode: str, *, layers: int = 2) -> torch.nn.Module:
    model = torch.nn.Module()
    model.layers = torch.nn.ModuleList(
        [
            DeltaMemAttention(
                make_qwen3_attention(),
                HFDeltaMemConfig(
                    rank=2,
                    alpha=4,
                    memory_backend="rwkv_ms",
                    memory_readout_mode=readout_mode,
                    projected_kv_key_dim=2,
                    memory_write_granularity="token",
                    rwkv_ms_write_mode="recurrent",
                    rwkv_ms_hybrid_mode="scalar_gate",
                    rwkv_ms_hybrid_gain=0.03125,
                ),
            )
            for _ in range(layers)
        ]
    )
    return model


def test_protocol_binds_paired_native_benchmark_contract() -> None:
    protocol = runner.validate_protocol()

    assert protocol["receipt"]["payload_sha256"] == runner.PROTOCOL_PAYLOAD_SHA256
    assert protocol["training"]["seeds"] == list(runner.SEEDS)
    assert protocol["training"]["optimizer_updates"] == runner.TRAIN_UPDATES
    assert protocol["training"]["global_batch_rows"] == runner.GLOBAL_BATCH_SIZE
    assert protocol["frozen_inputs"]["evaluation_rows"] == 220
    assert protocol["frozen_inputs"]["evaluation_rows_payload_sha256"] == (
        "0493e75da858d4ddebba580cc7b5aaaa32249527e5e44502e6ff06591cd82d09"
    )
    assert protocol["protected_splits_opened_by_this_protocol"] == []


def test_protocol_validation_rejects_receipt_drift(monkeypatch, tmp_path) -> None:
    protocol = copy.deepcopy(runner.validate_protocol())
    protocol["training"]["seeds"] = [1]
    path = tmp_path / "protocol.json"
    path.write_text(runner.json.dumps(protocol), encoding="utf-8")
    monkeypatch.setattr(runner, "PROTOCOL", path)

    with pytest.raises(ValueError, match="payload hash differs"):
        runner.validate_protocol()


def test_paired_config_differs_only_in_memory_readout_mode() -> None:
    projected = runner.build_config("projected_control")
    hybrid = runner.build_config("hybrid_candidate")

    projected_values = vars(projected).copy()
    hybrid_values = vars(hybrid).copy()
    assert projected_values.pop("memory_readout_mode") == "projected_kv_slots"
    assert hybrid_values.pop("memory_readout_mode") == "projected_kv_rwkv_hybrid"
    assert projected_values == hybrid_values


def test_trainable_isolation_and_shared_initialization(monkeypatch) -> None:
    monkeypatch.setattr(runner.recurrent_preflight, "EXPECTED_LAYERS", 2)
    torch.manual_seed(57)
    projected = _model("projected_kv_slots")
    torch.manual_seed(57)
    hybrid = _model("projected_kv_rwkv_hybrid")

    projected_state = snapshot_delta_mem_weights(projected)
    hybrid_state = snapshot_delta_mem_weights(hybrid)
    assert runner.state_subset_sha256(
        projected_state, recurrent_only=False
    ) == runner.state_subset_sha256(hybrid_state, recurrent_only=False)

    projected_named, projected_audit = runner.configure_trainable_parameters(
        projected,
        architecture="projected_control",
    )
    hybrid_named, hybrid_audit = runner.configure_trainable_parameters(
        hybrid,
        architecture="hybrid_candidate",
    )

    assert projected_audit["passed"] is True
    assert projected_audit["recurrent_only_trainable_tensors"] == 0
    assert projected_audit["recurrent_only_frozen_tensors"] > 0
    assert all(
        not runner.is_recurrent_only_parameter(name)
        for name, _ in projected_named
    )
    assert hybrid_audit["passed"] is True
    assert hybrid_audit["recurrent_only_trainable_tensors"] > 0
    assert sum(
        name.endswith(runner.recurrent_calibration.RECURRENT_READOUT_SUFFIX)
        for name, _ in hybrid_named
    ) == 2


def test_four_gpu_contract_requires_distinct_a100s() -> None:
    devices = [
        {"device_name": "NVIDIA A100-PCIE-40GB", "device_uuid": f"GPU-{index}"}
        for index in range(runner.WORLD_SIZE)
    ]

    assert runner.four_distinct_a100s(devices) is True
    duplicate = copy.deepcopy(devices)
    duplicate[-1]["device_uuid"] = duplicate[0]["device_uuid"]
    assert runner.four_distinct_a100s(duplicate) is False
    wrong_model = copy.deepcopy(devices)
    wrong_model[-1]["device_name"] = "NVIDIA H100 80GB HBM3"
    assert runner.four_distinct_a100s(wrong_model) is False
