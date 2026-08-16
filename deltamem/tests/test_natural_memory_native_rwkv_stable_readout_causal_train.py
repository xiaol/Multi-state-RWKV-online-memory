from __future__ import annotations

from experiments.rethinking_rwkv_ms_gemma import (
    run_natural_memory_native_rwkv_stable_readout_causal_train as training,
)


def test_protocol_and_calibration_binding_validate() -> None:
    protocol = training.validate_protocol()
    calibration = training.validate_calibration_result()

    assert protocol["receipt"]["payload_sha256"] == training.PROTOCOL_PAYLOAD_SHA256
    assert calibration["passed"] is True


def test_stable_readout_parameter_families_are_narrow() -> None:
    assert training.is_stable_readout_parameter(
        "layers.0.self_attn.hrm_rwkv7_core.output.weight"
    )
    assert training.is_stable_readout_parameter(
        "layers.0.self_attn.memory_fusion_bias"
    )
    assert not training.is_stable_readout_parameter(
        "layers.0.self_attn.hrm_rwkv7_core.receptance.weight"
    )
    assert not training.is_stable_readout_parameter(
        "layers.0.self_attn.delta_q_proj"
    )
