from __future__ import annotations

from experiments.rethinking_rwkv_ms_gemma import (
    analyze_natural_memory_native_rwkv_recurrent_value_eval as analyzer,
)
from experiments.rethinking_rwkv_ms_gemma import (
    run_natural_memory_native_rwkv_recurrent_value_eval as evaluation,
)


def test_training_result_and_adapter_binding_validate() -> None:
    result_path = (
        evaluation.SCRIPT_DIR
        / "local_artifacts/"
        "natural_memory_native_rwkv_recurrent_value_causal_train_v1/result.json"
    )
    adapter_dir = result_path.parent / "adapter"

    result = evaluation.validate_train_result(result_path, adapter_dir=adapter_dir)

    assert result["open_native_evaluation_authorized"] is True
    assert result["protocol_payload_sha256"] == evaluation.PROTOCOL_PAYLOAD_SHA256


def test_locked_native_thresholds_match_training_protocol() -> None:
    assert evaluation.EVALUATION_ROWS == 220
    assert analyzer.MARGIN_MINIMUM == 0.005
    assert analyzer.COVERAGE_MINIMUM == 0.95
    assert analyzer.PARTITIONS_PER_SHARD == 1
