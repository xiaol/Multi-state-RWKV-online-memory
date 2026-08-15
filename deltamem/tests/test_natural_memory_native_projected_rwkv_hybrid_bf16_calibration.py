from __future__ import annotations

import torch

from experiments.rethinking_rwkv_ms_gemma import (
    run_natural_memory_native_projected_rwkv_hybrid_bf16_calibration as calibration,
)


def test_protocol_screen_result_and_runtime_config_are_bound() -> None:
    protocol = calibration.validate_protocol()
    screen_result = calibration.validate_screen_result()
    config = calibration.build_config()

    assert protocol["receipt"]["payload_sha256"] == calibration.PROTOCOL_PAYLOAD_SHA256
    assert (
        protocol["authorization_basis"]["screen_result_receipt"]
        == calibration.SCREEN_RESULT_RECEIPT
    )
    assert screen_result["receipt"]["payload_sha256"] == calibration.SCREEN_RESULT_RECEIPT
    assert screen_result["one_update_calibration_authorized"] is True
    assert config.memory_backend == "rwkv_ms"
    assert config.memory_readout_mode == "projected_kv_rwkv_hybrid"
    assert config.rwkv_ms_write_mode == "recurrent"
    assert config.rwkv_ms_hybrid_mode == "scalar_gate"
    assert config.rwkv_ms_hybrid_gain == 0.03125
    assert protocol["protected_splits_opened_by_this_protocol"] == []


def test_recurrent_readout_hash_covers_every_layer_and_detects_change() -> None:
    named = []
    for layer in range(calibration.preflight.EXPECTED_LAYERS):
        parameter = torch.nn.Parameter(torch.full((2, 2), float(layer)))
        named.append(
            (
                f"model.layers.{layer}.self_attn"
                f"{calibration.recurrent_calibration.RECURRENT_READOUT_SUFFIX}",
                parameter,
            )
        )

    before = calibration.recurrent_readout_sha256(named)
    with torch.no_grad():
        named[17][1].add_(1.0)
    after = calibration.recurrent_readout_sha256(named)

    assert before != after


def test_hybrid_calibration_contract_is_four_gpu_one_update() -> None:
    protocol = calibration.validate_protocol()

    assert calibration.WORLD_SIZE == 4
    assert calibration.LEARNING_RATE == 2e-4
    assert protocol["training"]["optimizer_updates"] == 1
    assert protocol["training"]["logical_global_batch_rows"] == 4
    assert protocol["required_gates"][
        "optimizer_update_changes_recurrent_output_weights"
    ] is True
    assert protocol["required_gates"][
        "projected_carrier_hash_fixed_across_all_causal_conditions_on_every_rank"
    ] is True


def test_four_distinct_a100_gate_checks_uuid_and_model() -> None:
    devices = [
        {"device_uuid": f"uuid-{rank}", "device_name": "NVIDIA A100-PCIE-40GB"}
        for rank in range(calibration.WORLD_SIZE)
    ]

    assert calibration.four_distinct_a100s(devices) is True
    devices[3]["device_uuid"] = devices[2]["device_uuid"]
    assert calibration.four_distinct_a100s(devices) is False
