from __future__ import annotations

import ast

from experiments.rethinking_rwkv_ms_gemma import (
    analyze_natural_memory_native_scene_strength_calibration as analysis,
)
from experiments.rethinking_rwkv_ms_gemma import (
    run_natural_memory_native_scene_strength_calibration as runner,
)
from experiments.rethinking_rwkv_ms_gemma import (
    run_natural_memory_native_scene_causal as causal_runner,
)


def test_strength_calibration_protocol_receipt_is_bound() -> None:
    protocol = runner.validate_protocol()

    assert protocol["authorization"]["publisher_validation_predictions_allowed_as_input"] is False
    assert protocol["authorization"]["publisher_test_authorized"] is False
    assert protocol["authorization"]["hard32_authorized"] is False
    assert protocol["intervention"]["candidate_strengths"] == [0.125, 0.25, 0.5, 0.75]


def test_strength_calibration_partition_receipts_are_bound() -> None:
    root = runner.PROJECT_ROOT / (
        "experiments/rethinking_rwkv_ms_gemma/local_artifacts/"
        "natural_memory_native_development_v1"
    )
    rows = causal_runner.load_rows(root)

    payload = runner.validate_partitions(rows)

    assert len(payload) == 357
    assert sum(record["partition"] == "fit" for record in payload) == 284
    assert sum(record["partition"] == "holdout" for record in payload) == 73


def test_strength_calibration_runner_hash_is_bound() -> None:
    assert runner.sha256_file(runner.Path(runner.__file__)) == analysis.EXPECTED_RUNNER_SHA256


def test_strength_calibration_executables_have_no_json_literal_names() -> None:
    for module in (runner, analysis):
        tree = ast.parse(module.Path(module.__file__).read_text(encoding="utf-8"))
        assert not {
            node.id for node in ast.walk(tree) if isinstance(node, ast.Name)
        } & {"false", "true", "null"}


def test_strength_calibration_sets_all_wrapped_layers(monkeypatch) -> None:
    class Module:
        def __init__(self, layer_index: int) -> None:
            self.layer_idx = layer_index
            self.memory_fusion_placement = "attention_output"
            self.memory_fusion_residual_scale = 1.0
            self.memory_fusion_residual_scale_max = 1.0

    modules = [(f"layer.{index}", Module(index)) for index in range(42)]
    monkeypatch.setattr(runner, "iter_delta_mem_modules", lambda model: iter(modules))

    settings = runner.set_strength(object(), 0.25)

    assert len(settings) == 42
    assert all(module.memory_fusion_residual_scale == 0.25 for _, module in modules)
    assert all(setting["memory_fusion_residual_scale"] == 0.25 for setting in settings)


def test_strength_calibration_holdout_requires_signed_selection(tmp_path) -> None:
    selection = tmp_path / "selection.json"
    selection.write_text("{}\n", encoding="utf-8")

    try:
        runner.validate_selection(selection, runner_sha256=analysis.EXPECTED_RUNNER_SHA256)
    except ValueError as error:
        assert "receipt is missing" in str(error)
    else:
        raise AssertionError("Unsigned strength holdout selection was accepted")
