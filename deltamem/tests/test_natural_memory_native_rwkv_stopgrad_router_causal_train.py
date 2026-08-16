from __future__ import annotations

from experiments.rethinking_rwkv_ms_gemma import (
    run_natural_memory_native_rwkv_stopgrad_router_causal_train as training,
)


def test_protocol_and_failure_bindings_validate() -> None:
    protocol = training.validate_protocol()

    assert protocol["receipt"]["payload_sha256"] == training.PROTOCOL_PAYLOAD_SHA256


def test_selected_candidate_only_changes_router_backward() -> None:
    assert training.SELECTED_CANDIDATE["candidate_id"] == (
        "recurrent_value_t16_k2_gate025"
    )
    assert training.SELECTED_CANDIDATE["read_top_k"] == 2
    assert training.SELECTED_CANDIDATE["detach_read_scores"] is True
