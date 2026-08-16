from __future__ import annotations

from experiments.rethinking_rwkv_ms_gemma import (  # noqa: E402
    run_natural_memory_native_rwkv_addressed_route_agreement_causal_train as training,
)


def test_protocol_locks_screen_authorization_and_fresh_endpoint() -> None:
    protocol = training.validate_protocol()

    assert protocol["receipt"]["payload_sha256"] == training.PROTOCOL_PAYLOAD_SHA256
    assert protocol["architecture"]["selected_candidate"]["hybrid_mode"] == (
        "addressed_route_agreement"
    )
    assert protocol["training"]["optimizer_updates"] == 16
    assert protocol["training"]["hardware"] == "exactly four distinct A100 GPUs"
    assert protocol["heldout_causal_endpoint"]["source_ordinals"] == list(
        training.HELDOUT_ORDINALS
    )
    assert protocol["heldout_causal_endpoint"]["source_donor_payload_sha256"] == (
        training.HELDOUT_PAYLOAD_SHA256
    )
    assert protocol["protected_splits_opened_by_this_protocol"] == []


def test_training_bindings_replace_and_restore_nested_contract() -> None:
    original_protocol = training.affine_train.shared.PROTOCOL
    original_loader = training.affine_train.shared.MODEL_LOADER

    with training.training_bindings():
        assert training.affine_train.shared.PROTOCOL is training.PROTOCOL
        assert training.affine_train.shared.MODEL_LOADER is training.load_model
        assert (
            training.affine_train.causal_train.PROTOCOL_PAYLOAD_SHA256
            == training.PROTOCOL_PAYLOAD_SHA256
        )

    assert training.affine_train.shared.PROTOCOL == original_protocol
    assert training.affine_train.shared.MODEL_LOADER is original_loader
