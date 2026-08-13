from __future__ import annotations

import ast

import torch

from experiments.rethinking_rwkv_ms_gemma import (
    analyze_natural_memory_native_scene_strength_controller as analysis,
)
from experiments.rethinking_rwkv_ms_gemma import (
    run_natural_memory_native_scene_strength_controller as runner,
)


def test_strength_controller_protocol_receipt_is_bound() -> None:
    protocol = runner.validate_protocol()

    assert protocol["authorization"]["publisher_validation_predictions_allowed_as_input"] is False
    assert protocol["authorization"]["publisher_test_authorized"] is False
    assert protocol["authorization"]["hard32_authorized"] is False
    assert protocol["preflight"]["source_indices"] == [0, 1, 2, 3]


def test_strength_controller_runner_hash_is_bound() -> None:
    assert runner.sha256_file(runner.Path(runner.__file__)) == analysis.EXPECTED_RUNNER_SHA256


def test_strength_controller_executables_have_no_json_literal_names() -> None:
    for module in (runner, analysis):
        tree = ast.parse(module.Path(module.__file__).read_text(encoding="utf-8"))
        assert not {
            node.id for node in ast.walk(tree) if isinstance(node, ast.Name)
        } & {"false", "true", "null"}


def test_strength_controller_scales_q_and_o(monkeypatch) -> None:
    class Module:
        def __init__(self) -> None:
            self.training = False
            self._eval_memory_delta_controller = None

    modules = [(f"layer.{index}", Module()) for index in range(42)]
    monkeypatch.setattr(runner, "iter_delta_mem_modules", lambda model: iter(modules))
    calls, digest = runner.attach_controller(object(), 0.25)
    reference = torch.ones(2, 3)
    delta = torch.full((2, 3), 4.0)

    for head_name in ("q", "o"):
        output = modules[0][1]._eval_memory_delta_controller(
            modules[0][1],
            head_name,
            reference,
            delta,
            None,
        )
        assert torch.equal(output, torch.full((2, 3), 2.0))

    assert calls == {"q": 1, "o": 1}
    assert len(digest) == 64
    assert all(module._eval_memory_delta_controller is modules[0][1]._eval_memory_delta_controller for _, module in modules)


def test_strength_controller_zero_is_identity(monkeypatch) -> None:
    class Module:
        training = False
        _eval_memory_delta_controller = None

    modules = [(f"layer.{index}", Module()) for index in range(42)]
    monkeypatch.setattr(runner, "iter_delta_mem_modules", lambda model: iter(modules))
    runner.attach_controller(object(), 0.0)
    reference = torch.randn(2, 3)
    delta = torch.randn(2, 3)

    output = modules[0][1]._eval_memory_delta_controller(
        modules[0][1],
        "q",
        reference,
        delta,
        None,
    )

    assert torch.equal(output, reference)


def test_strength_controller_fit_requires_signed_preflight(tmp_path) -> None:
    receipt = tmp_path / "preflight.json"
    receipt.write_text("{}\n", encoding="utf-8")

    try:
        runner.validate_signed_receipt(
            receipt,
            schema=runner.PREFLIGHT_SCHEMA,
            runner_sha256=analysis.EXPECTED_RUNNER_SHA256,
            require_passed=True,
        )
    except ValueError as error:
        assert "receipt missing" in str(error)
    else:
        raise AssertionError("Unsigned strength-controller preflight was accepted")
