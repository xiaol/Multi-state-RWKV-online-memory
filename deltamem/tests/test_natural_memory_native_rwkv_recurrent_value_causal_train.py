from __future__ import annotations

from experiments.rethinking_rwkv_ms_gemma import (
    run_natural_memory_native_rwkv_recurrent_value_causal_train as causal_train,
)


def test_protocol_and_calibration_binding_validate() -> None:
    protocol = causal_train.validate_protocol()
    calibration = causal_train.validate_calibration_result()

    assert protocol["receipt"]["payload_sha256"] == causal_train.PROTOCOL_PAYLOAD_SHA256
    assert calibration["status"] == "calibration_passed_causal_training_authorized"
    assert calibration["causal_training_authorized"] is True


def test_training_schedule_matches_prior_value_runs() -> None:
    assert causal_train.SEED == 60
    assert causal_train.GLOBAL_BATCH_SIZE == 8
    assert causal_train.TRAIN_UPDATES == 8
    assert causal_train.LEARNING_RATE == 1e-4
    assert causal_train.MAX_GRAD_NORM == 0.1
