from __future__ import annotations

import torch

from experiments.rethinking_rwkv_ms_gemma import (
    run_natural_memory_native_recurrent_rwkv_bf16_calibration as calibration,
)


def test_materiality_metrics_require_a_bf16_visible_change() -> None:
    reference = torch.zeros(1, 2, 3)

    absent = calibration.materiality_metrics(reference, reference.clone())
    visible = calibration.materiality_metrics(
        reference,
        torch.full_like(reference, calibration.MIN_POST_UPDATE_MAX_ABS_LOGIT_DELTA),
    )

    assert absent["max_abs_logit_delta"] == 0.0
    assert absent["passed"] is False
    assert visible["passed"] is True


def test_recurrent_readout_gradient_audit_requires_all_layers() -> None:
    named = []
    for layer in range(calibration.preflight.EXPECTED_LAYERS):
        parameter = torch.nn.Parameter(torch.ones(2, 2))
        parameter.grad = torch.full_like(parameter, float(layer + 1))
        named.append(
            (
                f"model.layers.{layer}.self_attn"
                f"{calibration.RECURRENT_READOUT_SUFFIX}",
                parameter,
            )
        )

    audit = calibration.audit_recurrent_readout_gradients(named)

    assert audit["parameter_tensors"] == calibration.preflight.EXPECTED_LAYERS
    assert audit["all_42_finite_nonzero"] is True
    assert audit["passed"] is True

    named[7][1].grad.zero_()
    failed = calibration.audit_recurrent_readout_gradients(named)
    assert failed["passed"] is False


def test_calibration_contract_is_four_gpu_one_step() -> None:
    protocol = calibration.preflight.validate_protocol()

    assert calibration.WORLD_SIZE == 4
    assert len(calibration.CALIBRATION_SOURCE_ORDINALS) == calibration.WORLD_SIZE
    assert calibration.LEARNING_RATE == 2e-4
    assert calibration.MIN_POST_UPDATE_MAX_ABS_LOGIT_DELTA > 0.0
    assert protocol["training"]["bf16_calibration_gate"][
        "required_before_benchmark_training"
    ] is True
    assert protocol["protected_splits_opened_by_this_protocol"] == []
