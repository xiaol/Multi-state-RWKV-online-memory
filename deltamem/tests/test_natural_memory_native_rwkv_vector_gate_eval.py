from __future__ import annotations

import json
from types import SimpleNamespace

import torch

from experiments.rethinking_rwkv_ms_gemma import (
    analyze_natural_memory_native_rwkv_vector_gate_eval as analyzer,
)
from experiments.rethinking_rwkv_ms_gemma import (
    run_natural_memory_native_rwkv_addressed_value_eval as addressed,
)
from experiments.rethinking_rwkv_ms_gemma import (
    run_natural_memory_native_rwkv_vector_gate_eval as evaluation,
)


def _training_result():
    return (
        evaluation.SCRIPT_DIR
        / "local_artifacts/"
        "natural_memory_native_rwkv_vector_gate_causal_train_v1/result.json"
    )


def _native_result():
    return (
        evaluation.SCRIPT_DIR
        / "local_artifacts/"
        "natural_memory_native_rwkv_vector_gate_eval_v1/result.json"
    )


def test_protocol_locks_vector_generation_conditions_and_open_rows() -> None:
    protocol = evaluation.validate_protocol()

    assert protocol["receipt"]["payload_sha256"] == (
        evaluation.PROTOCOL_PAYLOAD_SHA256
    )
    assert protocol["generation"]["conditions"] == list(evaluation.CONDITIONS)
    assert protocol["frozen_inputs"]["authorized_rows"] == 220
    assert protocol["protected_splits_opened_by_this_protocol"] == []


def test_training_result_and_vector_adapter_binding_validate() -> None:
    result_path = _training_result()

    result = evaluation.validate_train_result(
        result_path,
        adapter_dir=result_path.parent / "adapter",
    )

    assert result["open_native_generation_authorized"] is True
    assert result["status"] == (
        "vector_gate_heldout_passed_generation_authorized"
    )


def test_evaluation_bindings_replace_and_restore_addressed_helpers() -> None:
    original_conditions = addressed.CONDITIONS
    original_generate = addressed.generate_row_conditions
    original_validator = addressed.validate_train_result

    with evaluation.evaluation_bindings():
        assert addressed.CONDITIONS == evaluation.CONDITIONS
        assert addressed.generate_row_conditions is evaluation.generate_row_conditions
        assert addressed.validate_train_result is evaluation.validate_train_result

    assert addressed.CONDITIONS is original_conditions
    assert addressed.generate_row_conditions is original_generate
    assert addressed.validate_train_result is original_validator


def test_locked_native_thresholds_and_identity_control() -> None:
    assert evaluation.EVALUATION_ROWS == 220
    assert evaluation.CONDITIONS[-1] == "projected_only_bypass"
    assert analyzer.MARGIN_MINIMUM == 0.005
    assert analyzer.COVERAGE_MINIMUM == 0.95
    assert analyzer.PARTITIONS_PER_SHARD == 1


def test_signed_result_discloses_protocol_runtime_mode_erratum(monkeypatch) -> None:
    empty_records = {
        condition: {
            index: {
                "prediction": [],
                "raw_generation": "",
                "projected_carrier_byte_identical": True,
            }
            for index in range(evaluation.EVALUATION_ROWS)
        }
        for condition in evaluation.CONDITIONS
    }
    metrics = {
        "micro_f1": 0.0,
        "coverage": 1.0,
    }
    monkeypatch.setattr(analyzer, "read_records", lambda root: (empty_records, []))
    monkeypatch.setattr(analyzer, "aggregate_condition", lambda records: metrics)
    monkeypatch.setattr(analyzer.Path, "resolve", lambda self, strict=False: self)

    result = analyzer.analyze(analyzer.Path("unused"))

    runtime_mode_erratum, restoration_erratum = result["protocol_errata"]
    assert runtime_mode_erratum["recorded_runtime_mode"] == "scalar_gate"
    assert runtime_mode_erratum["correct_runtime_mode"] == "vector_gate"
    assert "Non-operative prose error only" in runtime_mode_erratum["impact"]
    assert restoration_erratum["recorded_action"] == (
        "set runtime mode and hybrid gain"
    )
    assert "load rwkv_ms_hybrid_gain=0.125" in restoration_erratum[
        "executed_action"
    ]


def test_signed_native_result_locks_gain_without_causal_pass() -> None:
    result = json.loads(_native_result().read_text(encoding="utf-8"))
    unsigned = dict(result)
    receipt = unsigned.pop("receipt")

    assert analyzer.canonical_sha256(unsigned) == receipt["payload_sha256"]
    assert receipt["payload_sha256"] == (
        "9fcbbd11ba502fdab77bee6c1177a5f5296cd4ff6ebecd0acdb3ce02b4cd10af"
    )
    assert result["status"] == "vector_gate_native_gain_without_full_causal_pass"
    assert result["passed"] is False
    assert result["gates"][
        "correct_minus_projected_only_micro_f1_minimum"
    ] is True
    assert result["gates"]["correct_minus_matched_donor_micro_f1_minimum"] is False
    assert result["gates"][
        "correct_minus_layer_permuted_micro_f1_minimum"
    ] is False
    assert result["gates"][
        "zero_recurrent_exactly_matches_projected_only_predictions"
    ] is True
    assert result["scope"] == {
        "hard32_opened": False,
        "publisher_test_opened": False,
        "publisher_validation_predictions_opened": False,
        "split": "publisher-TRAIN-derived authorized development partition",
        "strength_holdout_opened": False,
    }


def test_runtime_mode_restoration_changes_no_learned_tensor(monkeypatch) -> None:
    modules = []
    for index in range(42):
        module = SimpleNamespace(
            memory_readout_mode="projected_kv_rwkv_hybrid",
            rwkv_ms_hybrid_mode="recurrent_value",
            rwkv_ms_hybrid_gain=0.125,
            rwkv_ms_read_temperature=16.0,
            rwkv_ms_read_top_k=2,
            rwkv_ms_detach_read_scores=True,
            learned=torch.tensor([float(index)]),
        )
        modules.append((f"layer.{index}", module))
    before = [module.learned.clone() for _, module in modules]
    monkeypatch.setattr(evaluation, "iter_delta_mem_modules", lambda model: modules)

    evaluation._restore_and_assert_vector_gate_modules(object())

    assert all(module.rwkv_ms_hybrid_mode == "vector_gate" for _, module in modules)
    assert all(
        torch.equal(module.learned, expected)
        for (_, module), expected in zip(modules, before)
    )
