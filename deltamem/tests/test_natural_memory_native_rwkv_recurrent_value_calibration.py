from __future__ import annotations

from experiments.rethinking_rwkv_ms_gemma import (
    run_natural_memory_native_rwkv_recurrent_value_calibration as calibration,
)


def test_protocol_and_screen_binding_validate() -> None:
    protocol = calibration.validate_protocol()
    screen_result = calibration.validate_screen_result()

    assert (
        protocol["receipt"]["payload_sha256"]
        == calibration.PROTOCOL_PAYLOAD_SHA256
    )
    assert screen_result["status"] == "screen_passed_causal_calibration_authorized"
    assert screen_result["selected_candidate"]["hybrid_mode"] == "recurrent_value"


def test_locked_candidate_uses_internal_recurrent_router() -> None:
    assert calibration.SELECTED_CANDIDATE == {
        "candidate_id": "recurrent_value_g003125",
        "hybrid_mode": "recurrent_value",
        "hybrid_gain": 0.03125,
    }
