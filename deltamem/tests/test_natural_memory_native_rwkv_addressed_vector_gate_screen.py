from __future__ import annotations

from experiments.rethinking_rwkv_ms_gemma import (  # noqa: E402
    run_natural_memory_native_rwkv_addressed_vector_gate_screen as screen,
)


def test_protocol_locks_projected_addressed_controller_and_open_rows() -> None:
    protocol = screen.validate_protocol()

    assert protocol["receipt"]["payload_sha256"] == screen.PROTOCOL_PAYLOAD_SHA256
    assert protocol["architecture"]["hybrid_mode"] == "addressed_vector_gate"
    assert protocol["architecture"]["material_carrier"] == "projected slot value"
    assert protocol["execution"]["hardware"] == "exactly four distinct A100 GPUs"
    assert protocol["protected_splits_opened_by_this_protocol"] == []


def test_candidate_locks_addressed_vector_gate() -> None:
    assert screen.SELECTED_CANDIDATE == {
        "candidate_id": "addressed_vector_gate_t16_k2_gate025_g0125",
        "hybrid_mode": "addressed_vector_gate",
        "hybrid_gain": 0.125,
        "read_temperature": 16.0,
        "read_top_k": 2,
        "fusion_gate_probability": 0.25,
        "detach_read_scores": True,
    }


def test_screen_bindings_replace_and_restore_shared_contract() -> None:
    original_mode = screen.shared.SELECTED_CANDIDATE["hybrid_mode"]
    original_validator = screen.shared.validate_protocol

    with screen.screen_bindings():
        assert screen.shared.SELECTED_CANDIDATE is screen.SELECTED_CANDIDATE
        assert screen.shared.validate_protocol is screen.validate_protocol
        assert str(screen.shared.RUNNER_BINDING_PATH) == screen.__file__

    assert screen.shared.SELECTED_CANDIDATE["hybrid_mode"] == original_mode
    assert screen.shared.validate_protocol is original_validator
