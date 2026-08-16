from __future__ import annotations

from experiments.rethinking_rwkv_ms_gemma import (
    run_natural_memory_native_rwkv_top2_abstention_causal_train as training,
)


def test_protocol_and_calibration_binding_validate() -> None:
    protocol = training.validate_protocol()
    calibration = training.validate_calibration_result()

    assert protocol["receipt"]["payload_sha256"] == training.PROTOCOL_PAYLOAD_SHA256
    assert calibration["status"] == "calibration_passed_contrastive_training_authorized"
    assert calibration["training"]["maximum_global_inactive_parameter_tensors"] == 0


def test_heldout_endpoint_is_locked_to_32_rows() -> None:
    assert len(training.HELDOUT_ORDINALS) == 32
    assert len(set(training.HELDOUT_ORDINALS)) == 32
    assert training.HELDOUT_PAYLOAD_SHA256 == (
        "f9a3bbd244c2e60528a5a84749c75ae8495335caf73bffe79c38c94d50059dcd"
    )
