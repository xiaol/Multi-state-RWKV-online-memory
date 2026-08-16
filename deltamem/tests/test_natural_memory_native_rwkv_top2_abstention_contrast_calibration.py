from __future__ import annotations

from experiments.rethinking_rwkv_ms_gemma import (
    run_natural_memory_native_rwkv_top2_abstention_contrast_calibration as calibration,
)


def test_protocol_and_failed_screen_binding_validate() -> None:
    protocol = calibration.validate_protocol()
    screen_result = calibration.validate_screen_result()

    assert (
        protocol["receipt"]["payload_sha256"]
        == calibration.PROTOCOL_PAYLOAD_SHA256
    )
    assert screen_result["passed"] is False
    assert screen_result["selected_candidate"] is None


def test_selected_candidate_is_first_preregistered_top2_abstention() -> None:
    assert calibration.SELECTED_CANDIDATE == {
        "candidate_id": "recurrent_value_t16_k2_gate025",
        "hybrid_mode": "recurrent_value",
        "hybrid_gain": 0.03125,
        "read_temperature": 16.0,
        "read_top_k": 2,
        "fusion_gate_probability": 0.25,
    }
