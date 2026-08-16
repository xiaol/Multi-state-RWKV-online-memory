from __future__ import annotations

import json
from types import SimpleNamespace

import torch

from experiments.rethinking_rwkv_ms_gemma import (  # noqa: E402
    analyze_natural_memory_native_rwkv_addressed_affine_eval as analyzer,
)
from experiments.rethinking_rwkv_ms_gemma import (  # noqa: E402
    run_natural_memory_native_rwkv_addressed_affine_eval as evaluation,
)
from experiments.rethinking_rwkv_ms_gemma import (  # noqa: E402
    run_natural_memory_native_rwkv_addressed_value_eval as addressed,
)


def _training_result():
    return (
        evaluation.SCRIPT_DIR
        / "local_artifacts/"
        "natural_memory_native_rwkv_addressed_affine_causal_train_v1/"
        "result.json"
    )


def _native_result():
    return (
        evaluation.SCRIPT_DIR
        / "local_artifacts/"
        "natural_memory_native_rwkv_addressed_affine_eval_v1/"
        "result.json"
    )


def test_protocol_locks_addressed_affine_generation_and_open_rows() -> None:
    protocol = evaluation.validate_protocol()

    assert protocol["receipt"]["payload_sha256"] == (
        evaluation.PROTOCOL_PAYLOAD_SHA256
    )
    assert protocol["generation"]["conditions"] == list(evaluation.CONDITIONS)
    assert protocol["architecture"]["hybrid_mode"] == "addressed_affine"
    assert protocol["frozen_inputs"]["authorized_rows"] == 220
    assert protocol["protected_splits_opened_by_this_protocol"] == []


def test_training_result_and_adapter_binding_validate() -> None:
    result_path = _training_result()
    result = evaluation.validate_train_result(
        result_path,
        adapter_dir=result_path.parent / "adapter",
    )

    assert result["open_native_generation_authorized"] is True
    assert result["status"] == evaluation.TRAIN_RESULT_STATUS


def test_evaluation_bindings_use_addressed_affine_mode_then_restore() -> None:
    original_generate = addressed.generate_row_conditions
    original_validator = addressed.validate_train_result

    with evaluation.evaluation_bindings():
        assert addressed.CONDITIONS == evaluation.CONDITIONS
        assert addressed.validate_train_result is evaluation.base.validate_train_result
        assert evaluation.base.RUNTIME_HYBRID_MODE == "addressed_affine"

    assert addressed.generate_row_conditions is original_generate
    assert addressed.validate_train_result is original_validator


def test_runtime_restoration_changes_no_learned_tensor(monkeypatch) -> None:
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
    monkeypatch.setattr(evaluation.base, "iter_delta_mem_modules", lambda model: modules)

    evaluation.restore_and_assert_addressed_affine_modules(object())

    assert all(module.rwkv_ms_hybrid_mode == "addressed_affine" for _, module in modules)
    assert all(
        torch.equal(module.learned, expected)
        for (_, module), expected in zip(modules, before)
    )


def test_analyzer_locks_causal_thresholds_without_prior_errata() -> None:
    assert analyzer.MARGIN_MINIMUM == 0.005
    assert analyzer.COVERAGE_MINIMUM == 0.95
    assert analyzer.PARTITIONS_PER_SHARD == 1
    with analyzer.analysis_bindings():
        assert analyzer.base.evaluation is evaluation
        assert analyzer.base.INCLUDE_PROTOCOL_ERRATA is False


def test_signed_native_result_locks_generation_failure() -> None:
    result_path = _native_result()
    result = json.loads(result_path.read_text(encoding="utf-8"))
    unsigned = dict(result)
    receipt = unsigned.pop("receipt")

    assert evaluation.sha256_file(result_path) == (
        "43097d4bceef4eb5a4a760f146bf4e5f697e5294e3ba929b948c9fd12f4b6d73"
    )
    assert analyzer.canonical_sha256(unsigned) == receipt["payload_sha256"]
    assert receipt["payload_sha256"] == (
        "c18d190c8fffcbac142c4d95cce6899129df637685cc551ae0e78214710ecdde"
    )
    assert result["status"] == "addressed_affine_native_gain_not_established"
    assert result["passed"] is False
    assert result["native_recurrent_causal_gain_established"] is False
    assert result["causal_margins"] == {
        "correct_minus_layer_permuted_micro_f1": -0.006762605737164029,
        "correct_minus_matched_donor_micro_f1": -0.00549737278333523,
        "correct_minus_projected_only_micro_f1": -0.008576410867123158,
        "correct_minus_zero_micro_f1": -0.008576410867123158,
    }
    assert all(
        result["gates"][gate] is False
        for gate in (
            "correct_minus_projected_only_micro_f1_minimum",
            "correct_minus_zero_micro_f1_minimum",
            "correct_minus_matched_donor_micro_f1_minimum",
            "correct_minus_layer_permuted_micro_f1_minimum",
        )
    )
    assert result["gates"][
        "zero_recurrent_exactly_matches_projected_only_predictions"
    ] is True
    assert result["gates"][
        "projected_carrier_fixed_across_recurrent_interventions"
    ] is True
    assert result["protocol_errata"] == []
    assert result["scope"] == {
        "hard32_opened": False,
        "publisher_test_opened": False,
        "publisher_validation_predictions_opened": False,
        "split": "publisher-TRAIN-derived authorized development partition",
        "strength_holdout_opened": False,
    }
