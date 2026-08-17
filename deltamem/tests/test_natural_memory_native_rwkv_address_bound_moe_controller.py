from __future__ import annotations

import json
from pathlib import Path

from experiments.rethinking_rwkv_ms_gemma import (
    run_natural_memory_native_rwkv_address_bound_moe_controller_causal_train as training,
)
from experiments.rethinking_rwkv_ms_gemma import (
    run_natural_memory_native_rwkv_address_bound_moe_controller_screen as screen,
)


def test_protocols_lock_address_bound_moe_contract() -> None:
    screen_protocol = screen.validate_protocol()
    causal_protocol = training.validate_protocol()

    assert (
        screen_protocol["architecture"]["hybrid_mode"]
        == "address_bound_moe_controller"
    )
    assert causal_protocol["architecture"]["selected_candidate"] == (
        training.SELECTED_CANDIDATE
    )
    assert causal_protocol["architecture"]["trainable_parameter_families"] == [
        suffix.removeprefix(".") for suffix in training.MOE_SUFFIXES
    ]
    assert causal_protocol["heldout_causal_endpoint"]["rows"] == 12
    assert causal_protocol["training"]["hardware"] == "exactly four distinct A100 GPUs"
    assert causal_protocol["protected_splits_opened_by_this_protocol"] == []


def test_causal_bindings_install_and_restore_moe_contract() -> None:
    shared = training.base.affine_train.shared
    original_screen = shared.screen
    original_configurer = shared.TRAINABLE_CONFIGURER

    with training.bindings():
        assert shared.screen is screen
        assert shared.TRAINABLE_CONFIGURER is training.moe.configure_moe_parameters

    assert shared.screen is original_screen
    assert shared.TRAINABLE_CONFIGURER is original_configurer


def test_signed_screen_authorizes_training_only() -> None:
    result_path = (
        Path(screen.__file__).resolve().parent
        / "local_artifacts/natural_memory_native_rwkv_address_bound_moe_controller_screen_v1/"
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
    assert result["protected_splits_opened"] == []
    assert len(result["rank_evidence"]) == 4
