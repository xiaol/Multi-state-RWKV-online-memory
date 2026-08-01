from __future__ import annotations

from pathlib import Path

import pytest
import torch

from experiments.rethinking_rwkv_ms_gemma import (
    scene_hard_failure_run_audit as audit,
)


def _tensor_shape(suffix: str) -> tuple[int, ...]:
    rank = 4
    hidden_size = 8
    if suffix in {
        "delta_scale_raw",
        "beta_bias",
        "hrm_rwkv7_core.x_r",
        "hrm_rwkv7_core.x_w",
        "hrm_rwkv7_core.x_k",
        "hrm_rwkv7_core.x_v",
        "hrm_rwkv7_core.x_a",
        "hrm_rwkv7_core.x_g",
        "hrm_rwkv7_core.w0",
        "hrm_rwkv7_core.a0",
        "hrm_rwkv7_core.k_k",
        "hrm_rwkv7_core.k_a",
        "hrm_rwkv7_core.ln_x.weight",
        "hrm_rwkv7_core.ln_x.bias",
    }:
        return (rank,)
    if suffix in {"memory_q_proj", "memory_k_proj", "memory_v_proj", "beta_proj"}:
        return (rank, hidden_size)
    if suffix == "delta_q_proj":
        return (12, rank)
    if suffix in {"delta_k_proj", "delta_v_proj"}:
        return (4, rank)
    if suffix == "delta_o_proj":
        return (hidden_size, rank)
    if suffix in {
        "hrm_rwkv7_core.w1",
        "hrm_rwkv7_core.a1",
        "hrm_rwkv7_core.g1",
    }:
        return (rank, 32)
    if suffix in {
        "hrm_rwkv7_core.w2",
        "hrm_rwkv7_core.a2",
        "hrm_rwkv7_core.g2",
    }:
        return (32, rank)
    return (rank, rank)


def _adapter_state(changed_suffixes: set[str] | None = None) -> dict[str, torch.Tensor]:
    changed_suffixes = set() if changed_suffixes is None else changed_suffixes
    state: dict[str, torch.Tensor] = {}
    frozen = set(audit.FROZEN_ADAPTER_TENSOR_SUFFIXES)
    for layer in range(42):
        prefix = f"model.language_model.layers.{layer}.self_attn."
        for suffix in audit.ADAPTER_TENSOR_SUFFIXES:
            dtype = torch.bfloat16 if suffix in frozen else torch.float32
            tensor = torch.zeros(_tensor_shape(suffix), dtype=dtype)
            if suffix in changed_suffixes:
                if suffix == "delta_scale_raw":
                    tensor[0] = 1.0
                    tensor[3] = 1.0
                else:
                    tensor.fill_(1.0)
            state[prefix + suffix] = tensor
    return state


def _trainable_names() -> list[str]:
    return audit._expected_tensor_names(audit.TRAINABLE_ADAPTER_TENSOR_SUFFIXES)


def _initial_topology(state: dict[str, torch.Tensor]) -> dict[str, object]:
    return {
        "replaced_modules": [
            f"model.language_model.layers.{layer}.self_attn"
            for layer in range(42)
        ],
        "trainable_names": _trainable_names(),
        "adapter_tensor_count": len(state),
        "adapter_parameter_count": sum(tensor.numel() for tensor in state.values()),
        "adapter_topology_sha256": audit._adapter_topology_sha256(state),
    }


def test_exact_qo_rank4_topology_has_27_trainable_families_on_42_layers() -> None:
    initial = _adapter_state()

    assert len(audit.ADAPTER_TENSOR_SUFFIXES) == 32
    assert len(audit.TRAINABLE_ADAPTER_TENSOR_SUFFIXES) == 27
    assert len(audit.FROZEN_ADAPTER_TENSOR_SUFFIXES) == 5
    assert audit.EXPECTED_ADAPTER_TENSOR_COUNT == 32 * 42 == 1344
    assert audit.EXPECTED_TRAINABLE_TENSOR_COUNT == 27 * 42 == 1134
    assert audit._validate_initial_adapter_topology(
        _initial_topology(initial),
        initial,
    ) == _trainable_names()


def test_step_one_proves_real_update_without_claiming_delayed_families() -> None:
    initial = _adapter_state()
    candidate = _adapter_state(set(audit.FIRST_UPDATE_REQUIRED_TENSOR_SUFFIXES))

    change = audit.adapter_change_record(
        initial,
        candidate,
        trainable_names=_trainable_names(),
        checkpoint_step=1,
        smoke=True,
    )

    assert change["changed_trainable_tensor_count"] == 22 * 42 == 924
    assert change["full_trainable_family_coverage"] is False
    assert set(change["missing_trainable_family_layers"]) == set(
        audit.FIRST_UPDATE_DELAYED_TENSOR_SUFFIXES
    )
    assert all(
        count == 42
        for count in change["first_update_required_family_layer_coverage"].values()
    )
    assert change["frozen_adapter_tensors_unchanged"] is True


def test_step_one_rejects_one_unchanged_required_family_layer() -> None:
    initial = _adapter_state()
    candidate = _adapter_state(set(audit.FIRST_UPDATE_REQUIRED_TENSOR_SUFFIXES))
    name = "model.language_model.layers.17.self_attn.delta_q_proj"
    candidate[name] = initial[name].clone()

    with pytest.raises(
        audit.AuditError,
        match="checkpoint_missing_required_family_layer_change:delta_q_proj",
    ):
        audit.adapter_change_record(
            initial,
            candidate,
            trainable_names=_trainable_names(),
            checkpoint_step=1,
            smoke=True,
        )


def test_final_checkpoint_rejects_one_unchanged_delayed_family_layer() -> None:
    initial = _adapter_state()
    candidate = _adapter_state(set(audit.TRAINABLE_ADAPTER_TENSOR_SUFFIXES))
    name = "model.language_model.layers.41.self_attn.hrm_rwkv7_core.x_w"
    candidate[name] = initial[name].clone()

    with pytest.raises(
        audit.AuditError,
        match="final_checkpoint_trainable_family_layer_coverage_incomplete",
    ):
        audit.adapter_change_record(
            initial,
            candidate,
            trainable_names=_trainable_names(),
            checkpoint_step=64,
            smoke=False,
        )


def test_final_checkpoint_proves_all_27_families_changed_on_all_42_layers() -> None:
    change = audit.adapter_change_record(
        _adapter_state(),
        _adapter_state(set(audit.TRAINABLE_ADAPTER_TENSOR_SUFFIXES)),
        trainable_names=_trainable_names(),
        checkpoint_step=64,
        smoke=False,
    )

    assert change["changed_trainable_tensor_count"] == 27 * 42 == 1134
    assert change["trainable_tensor_family_count"] == 27
    assert change["target_layer_count"] == 42
    assert change["trainable_family_layer_coverage"] == {
        suffix: 42 for suffix in audit.TRAINABLE_ADAPTER_TENSOR_SUFFIXES
    }
    assert change["missing_trainable_family_layers"] == {}
    assert change["full_trainable_family_coverage"] is True
    assert change["full_trainable_family_coverage_required"] is True


def test_checkpoint_rejects_changed_frozen_adapter_tensor() -> None:
    initial = _adapter_state()
    candidate = _adapter_state(set(audit.FIRST_UPDATE_REQUIRED_TENSOR_SUFFIXES))
    candidate[
        "model.language_model.layers.0.self_attn.memory_q_proj"
    ].fill_(1.0)

    with pytest.raises(
        audit.AuditError,
        match="checkpoint_changed_frozen_adapter_tensor",
    ):
        audit.adapter_change_record(
            initial,
            candidate,
            trainable_names=_trainable_names(),
            checkpoint_step=1,
            smoke=True,
        )


def test_checkpoint_rejects_missing_tensor_family_from_one_layer() -> None:
    initial = _adapter_state()
    initial.pop("model.language_model.layers.9.self_attn.hrm_rwkv7_core.a2")

    with pytest.raises(
        audit.AuditError,
        match="adapter_exact_qo_rank4_topology_differs",
    ):
        audit.adapter_change_record(
            initial,
            _adapter_state(set(audit.FIRST_UPDATE_REQUIRED_TENSOR_SUFFIXES)),
            trainable_names=_trainable_names(),
            checkpoint_step=1,
            smoke=True,
        )


def _optimizer_payload(step: int) -> dict[str, object]:
    states = {
        index: {
            "step": torch.tensor(float(step)),
            "exp_avg": torch.zeros(1),
            "exp_avg_sq": torch.zeros(1),
        }
        for index in range(audit.EXPECTED_TRAINABLE_TENSOR_COUNT)
    }
    return {
        "state": states,
        "param_groups": [
            {"params": list(range(audit.EXPECTED_TRAINABLE_TENSOR_COUNT))}
        ],
    }


def test_optimizer_audit_proves_every_declared_adapter_state_took_step_one(
    tmp_path: Path,
) -> None:
    path = tmp_path / "optimizer.pt"
    torch.save(_optimizer_payload(1), path)

    evidence = audit._validate_optimizer_state(path, checkpoint_step=1)

    assert evidence == {
        "optimizer_parameter_state_count": 1134,
        "declared_trainable_adapter_tensor_count": 1134,
        "all_optimizer_parameter_states_at_checkpoint_step": True,
    }


def test_optimizer_audit_rejects_one_parameter_at_wrong_step(tmp_path: Path) -> None:
    path = tmp_path / "optimizer.pt"
    payload = _optimizer_payload(1)
    payload["state"][17]["step"] = torch.tensor(0.0)  # type: ignore[index]
    torch.save(payload, path)

    with pytest.raises(audit.AuditError, match="checkpoint_optimizer_step_differs"):
        audit._validate_optimizer_state(path, checkpoint_step=1)
