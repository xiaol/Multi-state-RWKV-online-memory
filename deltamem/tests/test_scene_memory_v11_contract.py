from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path
import subprocess
from types import SimpleNamespace
from typing import Any

import pytest
import torch
from torch import nn

from experiments.rethinking_rwkv_ms_gemma import (
    scene_memory_v11_launch_contract as contract,
)
from experiments.rethinking_rwkv_ms_gemma import (
    scene_memory_v11_warm_start as warm_start,
)
from experiments.rethinking_rwkv_ms_gemma import run_scene_memory_v9_gate as v9_gate


class _FakeAdapterModel(nn.Module):
    def __init__(self, state: dict[str, torch.Tensor]) -> None:
        super().__init__()
        self.adapter_state = state


def _self_hash(payload: dict[str, Any], field: str) -> dict[str, Any]:
    result = dict(payload)
    result[field] = contract.canonical_sha256(result)
    return result


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")


def _first_cycle_data() -> dict[str, Any]:
    entries = [
        {
            "schedule_index": index,
            "canonical_pair_ordinals": list(pair),
            "entry_sha256": f"entry-{index}",
        }
        for index, pair in enumerate(contract.FIRST_CYCLE_PAIRS)
    ]
    return {
        "entries": entries,
        "checkpoint_steps": [1, 2, 3, 4],
        "source_presentation_checkpoint_steps": [7, 14, 21, 28],
        "optimizer_cycles": [
            {"presentation_start": 0, "presentation_stop": 7},
            {"presentation_start": 7, "presentation_stop": 14},
            {"presentation_start": 14, "presentation_stop": 21},
            {"presentation_start": 21, "presentation_stop": 28},
        ],
    }


def _v11_protocol() -> dict[str, Any]:
    return {
        "schema_version": contract.OBJECTIVE_SCHEMA_VERSION,
        "memory_objective_version": contract.OBJECTIVE_VERSION,
        "max_steps": 1,
        "max_grad_norm": contract.MAX_GRAD_NORM,
        "train_sampler_mode": contract.FIXED_SAMPLER_MODE,
        "gradient_accumulation_steps": contract.GRADIENT_ACCUMULATION_STEPS,
        "learning_rate": contract.LEARNING_RATE,
        "scene_generation_objective_formula": contract.OBJECTIVE_FORMULA,
        "scene_generation_backward_mode": contract.BACKWARD_MODE,
        "scene_generation_generated_prefix_correction_weight": (
            contract.SUFFIX_REPAIR_WEIGHT
        ),
        "scene_generation_generated_prefix_correction_mode": (
            contract.SUFFIX_REPAIR_MODE
        ),
        "scene_generation_generated_unlikelihood_max_wrong_tokens": (
            contract.GENERATED_MAX_CORRECTION_EVENTS
        ),
        "scene_generation_generated_prefix_max_correction_events": (
            contract.GENERATED_MAX_CORRECTION_EVENTS
        ),
        "scene_generation_generated_rollout_extra_tokens": (
            contract.GENERATED_ROLLOUT_EXTRA_TOKENS
        ),
        "scene_generation_generated_rollout_max_tokens": (
            contract.GENERATED_ROLLOUT_MAX_TOKENS
        ),
        "scene_generation_suffix_repair_mode": contract.SUFFIX_REPAIR_MODE,
        "scene_generation_suffix_repair_weight": contract.SUFFIX_REPAIR_WEIGHT,
        "scene_generation_suffix_repair_divergence": (
            contract.SUFFIX_REPAIR_DIVERGENCE
        ),
        "scene_generation_suffix_repair_gold_weighting": (
            contract.SUFFIX_REPAIR_GOLD_WEIGHTING
        ),
        "scene_generation_suffix_repair_first_wrong_unlikelihood": True,
        "scene_generation_suffix_repair_premature_termination_suppression": True,
        "scene_generation_suffix_repair_exact_rollout_loss": 0.0,
        "scene_generation_cycle_retention_mode": contract.CYCLE_RETENTION_MODE,
        "scene_generation_cycle_pair_presentations": 7,
        "scene_generation_gradient_accumulation_pair_cycle": 7,
        "train_schedule": {
            "checkpoint_steps": [7],
            "optimizer_checkpoint_steps": [1],
            "microbatch_cycle_size": 7,
            "continuation_policy": "forbidden",
        },
    }


def test_v11_constants_lock_one_fresh_suffix_repair_cycle() -> None:
    assert contract.OBJECTIVE_VERSION == (
        "scene_state_generation_ce_symmetric_cycle_suffix_repair_v5"
    )
    assert contract.OBJECTIVE_SCHEMA_VERSION == 14
    assert contract.CHECKPOINT_STEPS == (1,)
    assert contract.PRESENTATION_CHECKPOINTS == (7,)
    assert contract.CONTINUATION_POLICY == "forbidden"
    assert contract.TOTAL_OPTIMIZER_STEPS == 1
    assert contract.TOTAL_PAIR_PRESENTATIONS == 7
    assert contract.GRADIENT_ACCUMULATION_STEPS == 7
    assert contract.LEARNING_RATE == 2e-4
    assert contract.MAX_GRAD_NORM == 1.0
    assert contract.WARMUP_STEPS == 0
    assert contract.WARMUP_RATIO == 0.0
    assert contract.GENERATED_MAX_CORRECTION_EVENTS == 1
    assert contract.SUFFIX_REPAIR_WEIGHT == 0.5
    assert contract.TRAINING_CONTINUATION_POLICY == (
        "forbidden_one_cycle_only_regardless_of_gate_status"
    )
    assert [contract.presentation_cursor(step) for step in (0, 1)] == [0, 7]
    for invalid in (-1, 2, True):
        with pytest.raises(contract.LaunchContractError):
            contract.presentation_cursor(invalid)


def test_v11_suffix_contract_strings_pin_first_divergence_repair() -> None:
    assert contract.SUFFIX_REPAIR_MODE == (
        "first_raw_token_divergence_common_prefix_weighted_gold_suffix_ce_"
        "first_generated_wrong_unlikelihood_v5"
    )
    assert contract.SUFFIX_REPAIR_DIVERGENCE == (
        "first_raw_token_divergence_including_length_mismatch_v1"
    )
    assert contract.SUFFIX_REPAIR_GOLD_WEIGHTING == (
        "schema_2_decision_4_termination_1_v1"
    )
    assert "first_divergence_suffix_repair" in contract.OBJECTIVE_FORMULA
    assert "weighted_gold_suffix_ce(schema=2,decision=4,termination=1)" in (
        contract.OBJECTIVE_FORMULA
    )
    assert "first_generated_wrong_unlikelihood" in contract.OBJECTIVE_FORMULA
    assert contract.BACKWARD_MODE.endswith(
        "first_divergence_gold_suffix_replay_v6"
    )


@pytest.mark.parametrize(
    "field",
    (
        "max_grad_norm",
        "max_steps",
        "scene_generation_generated_prefix_correction_mode",
        "scene_generation_generated_prefix_max_correction_events",
        "scene_generation_suffix_repair_mode",
        "scene_generation_suffix_repair_weight",
        "scene_generation_suffix_repair_divergence",
        "scene_generation_suffix_repair_gold_weighting",
        "scene_generation_suffix_repair_first_wrong_unlikelihood",
        "scene_generation_suffix_repair_premature_termination_suppression",
        "scene_generation_suffix_repair_exact_rollout_loss",
    ),
)
def test_v11_protocol_rejects_suffix_or_runtime_drift(
    field: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        contract.v10,
        "_validate_checkpoint_protocol",
        lambda *_args, **_kwargs: None,
    )
    protocol = _v11_protocol()
    current = protocol[field]
    if isinstance(current, bool):
        protocol[field] = not current
    elif isinstance(current, str):
        protocol[field] = current + "_drift"
    else:
        protocol[field] = current + 1

    with pytest.raises(contract.LaunchContractError, match=field):
        contract._validate_checkpoint_protocol(protocol, data={})


@pytest.mark.parametrize(
    "field",
    (
        "checkpoint_steps",
        "optimizer_checkpoint_steps",
        "continuation_policy",
        "resume_schedule_cursor_formula",
    ),
)
def test_v11_protocol_rejects_continuation_schedule_drift(
    field: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        contract.v10,
        "_validate_checkpoint_protocol",
        lambda *_args, **_kwargs: None,
    )
    protocol = _v11_protocol()
    schedule = protocol["train_schedule"]
    if field == "resume_schedule_cursor_formula":
        schedule[field] = "global_step_times_7_v1"
    else:
        schedule[field] = [1, 2, 3, 4] if field.endswith("steps") else "allowed"

    with pytest.raises(contract.LaunchContractError, match=field):
        contract._validate_checkpoint_protocol(protocol, data={})


def test_v11_protocol_reuses_v10_validation_only_as_compatibility_layer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, Any] = {}

    def validate(
        protocol: dict[str, Any],
        *,
        checkpoint_step: int,
        data: dict[str, Any],
    ) -> None:
        captured.update(protocol)
        assert checkpoint_step == 1
        assert data == {"source": "v10-data"}

    monkeypatch.setattr(contract.v10, "_validate_checkpoint_protocol", validate)
    protocol = _v11_protocol()
    contract._validate_checkpoint_protocol(protocol, data={"source": "v10-data"})

    assert captured["memory_objective_version"] == contract.v10.OBJECTIVE_VERSION
    assert captured["scene_generation_objective_formula"] == (
        contract.v10.OBJECTIVE_FORMULA
    )
    assert captured["train_schedule"]["checkpoint_steps"] == [7, 14, 21, 28]
    assert captured["train_schedule"]["optimizer_checkpoint_steps"] == [
        1,
        2,
        3,
        4,
    ]
    assert captured["train_schedule"]["resume_schedule_cursor_formula"] == (
        "global_step_times_7_v1"
    )
    assert "continuation_policy" not in captured["train_schedule"]
    assert protocol["memory_objective_version"] == contract.OBJECTIVE_VERSION
    assert protocol["scene_generation_suffix_repair_mode"] == (
        contract.SUFFIX_REPAIR_MODE
    )
    assert protocol["train_schedule"]["checkpoint_steps"] == [7]
    assert protocol["train_schedule"]["optimizer_checkpoint_steps"] == [1]
    assert protocol["train_schedule"]["continuation_policy"] == "forbidden"
    assert "resume_schedule_cursor_formula" not in protocol["train_schedule"]


def test_v11_data_contract_uses_exact_pinned_first_cycle(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    reused = _first_cycle_data()
    monkeypatch.setattr(
        contract.v10,
        "validate_data_contract",
        lambda **_kwargs: reused,
    )

    result = contract.validate_data_contract()

    assert contract.FIRST_CYCLE_PAIRS == (
        (3, 24),
        (19, 28),
        (20, 31),
        (10, 23),
        (1, 14),
        (5, 9),
        (22, 26),
    )
    assert contract.canonical_sha256(
        [list(pair) for pair in contract.FIRST_CYCLE_PAIRS]
    ) == contract.FIRST_CYCLE_PAIRS_SHA256
    assert result["checkpoint_steps"] == [1]
    assert result["presentation_checkpoint_steps"] == [7]
    assert result["optimizer_cycles"] == [reused["optimizer_cycles"][0]]
    assert result["first_cycle_pairs"] == [
        list(pair) for pair in contract.FIRST_CYCLE_PAIRS
    ]

    drifted = _first_cycle_data()
    drifted["entries"] = list(drifted["entries"])
    drifted["entries"][0], drifted["entries"][1] = (
        drifted["entries"][1],
        drifted["entries"][0],
    )
    monkeypatch.setattr(
        contract.v10,
        "validate_data_contract",
        lambda **_kwargs: drifted,
    )
    with pytest.raises(contract.LaunchContractError, match="first_cycle_pair_order"):
        contract.validate_data_contract()


def test_v11_fresh_start_forbids_resume_and_imported_training_state() -> None:
    fresh = warm_start.V11FreshStartContract(
        resume_from_checkpoint=None,
        initial_global_step=0,
        optimizer_created=False,
        scheduler_created=False,
        trainer_state_imported=False,
        rng_state_imported=False,
        optim="adamw_torch_fused",
    )
    assert warm_start.validate_v11_fresh_start_contract(fresh) == {
        "initial_global_step": 0,
        "optimizer_implementation": "adamw_torch_fused",
        "optimizer_created_after_adapter_load": True,
        "optimizer_state": "fresh",
        "scheduler_state": "fresh",
        "trainer_state": "fresh",
        "rng_state": "fresh_from_v11_seed",
    }
    with pytest.raises(ValueError, match="forbids checkpoint resume"):
        warm_start.validate_v11_fresh_start_contract(
            replace(fresh, resume_from_checkpoint="checkpoint-1")
        )
    with pytest.raises(ValueError, match="global step 0"):
        warm_start.validate_v11_fresh_start_contract(
            replace(fresh, initial_global_step=56)
        )
    for field in (
        "optimizer_created",
        "scheduler_created",
        "trainer_state_imported",
        "rng_state_imported",
    ):
        with pytest.raises(ValueError, match="preloaded training state"):
            warm_start.validate_v11_fresh_start_contract(
                replace(fresh, **{field: True})
            )


def test_v11_warm_start_identity_is_v8_only(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    source = SimpleNamespace(
        checkpoint=tmp_path / "v8" / "checkpoint-56",
        lock_path=tmp_path / "v8-lock.json",
        lock={},
        source_config={},
        source_trainer_state={},
        source_training_protocol={},
        source_pairing_manifest={},
        continuation_lineage=(),
    )
    monkeypatch.setattr(
        warm_start.v9,
        "prepare_v9_v8_checkpoint56_warm_start",
        lambda *_args, **_kwargs: source,
    )

    result = warm_start.prepare_v11_v8_checkpoint56_warm_start(
        source.checkpoint,
        lock_path=source.lock_path,
    )

    assert result.checkpoint == source.checkpoint
    assert warm_start.WARM_START_MODE == (
        "scene_memory_v11_v8_checkpoint56_adapter_only"
    )
    assert warm_start.RECEIPT_SCHEMA == (
        "rwkv_ms_scene_memory_v11_adapter_warm_start_receipt.v1"
    )
    assert warm_start.TARGET_FRESH_START_POLICY["rng_state"] == (
        "fresh_from_v11_seed"
    )
    assert warm_start.SOURCE_IMPORT_POLICY == {
        "adapter": True,
        "optimizer": False,
        "scheduler": False,
        "trainer_state": False,
        "rng": False,
        "global_step": False,
    }


def test_v11_launch_is_fresh_only_and_v10_is_diagnostic_only(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    data = _first_cycle_data()
    data.update(
        {
            "train_file": "/ssd/train32.jsonl",
            "entries": _first_cycle_data()["entries"],
        }
    )
    warm = {
        "warm_start_checkpoint": "/ssd/v8/checkpoint-56",
        "warm_start_lock": "/ssd/v8-lock.json",
        "warm_start_lock_sha256": "a" * 64,
        "lock": {},
    }
    baseline = {
        "role": "frozen_diagnostic_only_never_warm_start",
        "checkpoint": "/ssd/v10/checkpoint-1",
    }
    monkeypatch.setattr(contract, "validate_data_contract", lambda **_kwargs: data)
    monkeypatch.setattr(
        contract,
        "validate_warm_start_contract",
        lambda **_kwargs: warm,
    )
    monkeypatch.setattr(
        contract,
        "validate_v10_diagnostic_baseline",
        lambda **_kwargs: baseline,
    )
    monkeypatch.setattr(
        contract,
        "validate_base_model_contract",
        lambda **_kwargs: {"path": str(contract.PINNED_BASE_MODEL)},
    )
    monkeypatch.setattr(
        contract,
        "critical_training_code_bindings",
        lambda **_kwargs: {"trainer": {"sha256": "f" * 64}},
    )

    result = contract.validate_launch_contract(target_step=1)

    assert result["source_step"] == 0
    assert result["target_step"] == 1
    assert result["resume_checkpoint"] is None
    assert result["resume_schedule_cursor"] == 0
    assert result["warm_start_checkpoint"] == warm["warm_start_checkpoint"]
    assert result["v10_diagnostic_baseline"] == baseline
    assert result["v10_diagnostic_baseline"]["checkpoint"] != (
        result["warm_start_checkpoint"]
    )
    assert result["max_grad_norm"] == 1.0
    assert result["training_continuation_policy"] == (
        "forbidden_one_cycle_only_regardless_of_gate_status"
    )

    for kwargs, message in (
        ({"target_step": 2}, "target_step_must_be_one"),
        (
            {"target_step": 1, "resume_checkpoint": Path("checkpoint-1")},
            "resume_is_forbidden",
        ),
        (
            {"target_step": 1, "gate_receipt": Path("gate_receipt.json")},
            "gate_receipt_cannot_authorize_training",
        ),
    ):
        with pytest.raises(contract.LaunchContractError, match=message):
            contract.validate_launch_contract(**kwargs)
    with pytest.raises(contract.LaunchContractError, match="forbids resume"):
        contract.validate_resume_contract()


def test_v11_validates_pinned_v10_failure_as_diagnostic_only(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    checkpoint = tmp_path / "v10" / "trainer" / "checkpoint-1"
    summary_path = tmp_path / "v10" / "gate" / "summary.json"
    receipt_path = tmp_path / "v10" / "gate" / "gate_receipt.json"
    manifest_path = tmp_path / "v10" / "gate" / "manifest.json"
    fingerprint = "f" * 64
    metrics = dict(contract.V10_DIAGNOSTIC_BASELINE_METRICS)
    summary = _self_hash(
        {
            "schema": "rwkv_ms_scene_memory_v10_train32_gate_summary.v1",
            "fingerprint": fingerprint,
            "gate": {
                "status": "fail",
                "training_continuation_authorized": False,
                "hard32_authorized": False,
                "full170_authorized": False,
                "metrics": {
                    "value14_generation": {
                        name: metrics[name]
                        for name in (
                            "canonical_correct_outputs",
                            "correct_strict_exact_rows",
                            "donor_identity_strict_exact_rows",
                            "correct_strict_micro_f1",
                        )
                    },
                    "value14_selected_token_identity": {
                        "overall": {
                            name: metrics[name]
                            for name in metrics
                            if name
                            not in {
                                "canonical_correct_outputs",
                                "correct_strict_exact_rows",
                                "donor_identity_strict_exact_rows",
                                "correct_strict_micro_f1",
                            }
                        }
                    },
                },
            },
        },
        "summary_sha256",
    )
    receipt = _self_hash(
        {
            "schema": "rwkv_ms_scene_memory_v10_train32_gate_receipt.v1",
            "evaluation_fingerprint": fingerprint,
            "checkpoint": {"memory_dir": str(checkpoint), "global_step": 1},
        },
        "receipt_sha256",
    )
    model_identity = {
        "weights": {"combined_sha256": "1" * 64, "files": []},
        "prompt_artifacts": {"combined_sha256": "2" * 64, "files": []},
    }
    manifest = {
        "schema": "rwkv_ms_scene_memory_v10_train32_gate_manifest.v1",
        "fingerprint": fingerprint,
        "hard32_access": contract.HARD32_ACCESS_POLICY,
        "fingerprint_payload": {
            "base_model": str(contract.PINNED_BASE_MODEL),
            "base_model_weights": model_identity["weights"],
            "base_model_prompt_artifacts": model_identity["prompt_artifacts"],
        },
    }
    _write_json(summary_path, summary)
    _write_json(receipt_path, receipt)
    _write_json(manifest_path, manifest)
    monkeypatch.setattr(contract, "SSD_ROOT", tmp_path)
    monkeypatch.setattr(contract, "PINNED_V10_DIAGNOSTIC_GATE_DIR", summary_path.parent)
    monkeypatch.setattr(contract, "PINNED_V10_DIAGNOSTIC_SUMMARY", summary_path)
    monkeypatch.setattr(contract, "PINNED_V10_DIAGNOSTIC_RECEIPT", receipt_path)
    monkeypatch.setattr(contract, "PINNED_V10_DIAGNOSTIC_MANIFEST", manifest_path)
    monkeypatch.setattr(contract, "PINNED_V10_DIAGNOSTIC_CHECKPOINT", checkpoint)
    monkeypatch.setattr(
        contract,
        "PINNED_V10_DIAGNOSTIC_SUMMARY_FILE_SHA256",
        contract.sha256_file(summary_path),
    )
    monkeypatch.setattr(
        contract,
        "PINNED_V10_DIAGNOSTIC_RECEIPT_FILE_SHA256",
        contract.sha256_file(receipt_path),
    )
    monkeypatch.setattr(
        contract,
        "PINNED_V10_DIAGNOSTIC_MANIFEST_FILE_SHA256",
        contract.sha256_file(manifest_path),
    )
    monkeypatch.setattr(contract, "PINNED_V10_DIAGNOSTIC_FINGERPRINT", fingerprint)
    monkeypatch.setattr(
        contract,
        "PINNED_V10_DIAGNOSTIC_SUMMARY_SHA256",
        summary["summary_sha256"],
    )
    monkeypatch.setattr(
        contract,
        "PINNED_V10_DIAGNOSTIC_RECEIPT_SHA256",
        receipt["receipt_sha256"],
    )

    result = contract.validate_v10_diagnostic_baseline(ssd_root=tmp_path)

    assert result["role"] == "frozen_diagnostic_only_never_warm_start"
    assert result["checkpoint"] == str(checkpoint)
    assert result["metrics"] == contract.V10_DIAGNOSTIC_BASELINE_METRICS


def test_v11_paths_are_locked_and_hard32_is_rejected_before_resolve(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run_root = contract.v11_run_root_for(tmp_path)
    gate_root = contract.v11_gates_root_for(tmp_path)
    checkpoint = run_root / "run" / "trainer" / "checkpoint-1"
    receipt = gate_root / "cycle1" / "gate_receipt.json"
    assert contract.require_v11_run_path(
        checkpoint,
        description="checkpoint",
        ssd_root=tmp_path,
    ) == checkpoint
    assert contract.require_v11_gate_path(
        receipt,
        description="receipt",
        ssd_root=tmp_path,
    ) == receipt
    inherited_error = contract.v10.LaunchContractError
    with pytest.raises(inherited_error, match="outside_locked_root"):
        contract.require_v11_run_path(
            tmp_path / "other" / "checkpoint-1",
            description="checkpoint",
            ssd_root=tmp_path,
        )
    with pytest.raises(inherited_error, match="outside_locked_root"):
        contract.require_v11_gate_path(
            run_root / "not-gates" / "gate_receipt.json",
            description="receipt",
            ssd_root=tmp_path,
        )

    protected = tmp_path / "hard32-copy" / "artifact.json"
    original_resolve = Path.resolve

    def guarded_resolve(path: Path, *args: Any, **kwargs: Any) -> Path:
        if any("hard32" in part.lower() for part in path.parts):
            raise AssertionError("protected path was resolved")
        return original_resolve(path, *args, **kwargs)

    monkeypatch.setattr(Path, "resolve", guarded_resolve)
    with pytest.raises(inherited_error, match="forbidden_hard32"):
        contract.require_v11_run_path(
            protected,
            description="protected checkpoint",
            ssd_root=tmp_path,
        )
    assert contract.HARD32_ACCESS_POLICY == (
        "forbidden_not_resolved_opened_or_hashed"
    )


def test_v11_cycle_telemetry_proves_exact_order_and_rejects_drift() -> None:
    log = {
        "step": 1,
        "delta/scene_generation_v11_cycle_pair_presentations": 7.0,
    }
    for index, (low, high) in enumerate(contract.FIRST_CYCLE_PAIRS):
        log[f"delta/scene_generation_v11_cycle_pair_{index}_low_ordinal"] = float(low)
        log[f"delta/scene_generation_v11_cycle_pair_{index}_high_ordinal"] = float(high)
    state = {"log_history": [log]}

    telemetry = contract.validate_v11_cycle_pair_telemetry(state)

    assert telemetry["ordered_pairs"] == [
        list(pair) for pair in contract.FIRST_CYCLE_PAIRS
    ]
    assert telemetry["ordered_pairs_sha256"] == contract.FIRST_CYCLE_PAIRS_SHA256
    drifted = json.loads(json.dumps(state))
    drifted["log_history"][0][
        "delta/scene_generation_v11_cycle_pair_3_low_ordinal"
    ] = 1.0
    with pytest.raises(contract.LaunchContractError, match="identity_or_order"):
        contract.validate_v11_cycle_pair_telemetry(drifted)


def test_v11_base_model_must_equal_pinned_v10_manifest(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    identity = {
        "path": str(contract.PINNED_BASE_MODEL),
        "weights": {"combined_sha256": "a" * 64, "files": []},
        "prompt_artifacts": {"combined_sha256": "b" * 64, "files": []},
    }
    baseline = {"base_model_identity": identity}
    monkeypatch.setattr(
        contract,
        "require_exact_path",
        lambda path, *_args, **_kwargs: Path(path),
    )
    monkeypatch.setattr(Path, "is_dir", lambda _self: True)
    monkeypatch.setattr(Path, "is_symlink", lambda _self: False)
    monkeypatch.setattr(
        v9_gate,
        "base_model_weight_identity",
        lambda _path: identity["weights"],
    )
    monkeypatch.setattr(
        v9_gate,
        "base_model_prompt_identity",
        lambda _path: identity["prompt_artifacts"],
    )

    assert contract.validate_base_model_contract(baseline=baseline) == identity
    monkeypatch.setattr(
        v9_gate,
        "base_model_prompt_identity",
        lambda _path: {"combined_sha256": "c" * 64, "files": []},
    )
    with pytest.raises(contract.LaunchContractError, match="pinned_v10_manifest"):
        contract.validate_base_model_contract(baseline=baseline)


def test_v11_apply_warm_start_loads_only_pinned_adapter(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    checkpoint = tmp_path / "checkpoint-56"
    checkpoint.mkdir()
    source_state = {"adapter": torch.arange(4, dtype=torch.float32)}
    torch.save(source_state, checkpoint / "delta_mem_adapter.pt")
    _, topology_sha256, tensor_elements = warm_start.v9.ordered_adapter_topology(
        source_state
    )
    context = warm_start.V11WarmStartContext(
        checkpoint=checkpoint,
        lock_path=tmp_path / "lock.json",
        lock={
            "lock_sha256": "a" * 64,
            "artifacts": {},
            "adapter_topology": {
                "sha256": topology_sha256,
                "tensor_count": 1,
                "tensor_elements": tensor_elements,
            },
            "source_state_imports": warm_start.SOURCE_IMPORT_POLICY,
        },
        source_config={},
        source_trainer_state={"global_step": 56, "epoch": 1.0},
        source_training_protocol={"memory_objective_version": "v8"},
        source_pairing_manifest={"objective_version": "v8-pairing"},
        continuation_lineage=(),
    )
    model = _FakeAdapterModel({"adapter": torch.zeros(4)})
    monkeypatch.setattr(
        warm_start,
        "get_delta_mem_state_dict",
        lambda target: {
            name: tensor.clone() for name, tensor in target.adapter_state.items()
        },
    )
    monkeypatch.setattr(
        warm_start,
        "load_delta_mem_state_dict",
        lambda target, state: setattr(
            target,
            "adapter_state",
            {name: tensor.clone() for name, tensor in state.items()},
        ),
    )
    fresh = warm_start.V11FreshStartContract(
        resume_from_checkpoint=None,
        initial_global_step=0,
        optimizer_created=False,
        scheduler_created=False,
        trainer_state_imported=False,
        rng_state_imported=False,
        optim="adamw_torch_fused",
    )

    receipt = warm_start.apply_v11_v8_checkpoint56_adapter_only_warm_start(
        model,
        context,
        fresh_start=fresh,
    )

    assert torch.equal(model.adapter_state["adapter"], source_state["adapter"])
    assert receipt["target_fresh_start"]["rng_state"] == "fresh_from_v11_seed"
    assert receipt["loaded_source_artifacts"] == ["delta_mem_adapter.pt"]


def test_v11_launch_completion_provenance_chain_rejects_telemetry_drift(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    run_root = tmp_path / "scene_memory_v11"
    checkpoint = run_root / "run" / "trainer" / "checkpoint-1"
    logs = run_root / "logs"
    checkpoint.mkdir(parents=True)
    logs.mkdir(parents=True)
    for filename in contract.REQUIRED_CHECKPOINT_ARTIFACTS:
        (checkpoint / filename).write_bytes(filename.encode())
    (checkpoint / "rng_state.pth").write_bytes(b"rng")
    summary_path = checkpoint.parents[1] / "training_summary.json"
    protocol_sha = "d" * 64
    _write_json(
        summary_path,
        {
            "memory_objective_version": contract.OBJECTIVE_VERSION,
            "warm_start_mode": warm_start.WARM_START_MODE,
            "training_protocol_sha256": protocol_sha,
        },
    )
    log_path = logs / "run.log"
    log_path.write_text("completed\n", encoding="utf-8")
    launch_path = logs / "run.launch.json"
    completion_path = logs / "run.completion.json"
    baseline = {
        "checkpoint": "/ssd/v10/checkpoint-1",
        "base_model_identity": {"path": "/ssd/model"},
    }
    model_identity = baseline["base_model_identity"]
    critical = {"trainer": {"sha256": "a" * 64}}
    monkeypatch.setattr(
        contract,
        "require_v11_run_path",
        lambda path, **_kwargs: Path(path).resolve(),
    )
    monkeypatch.setattr(contract, "v11_run_root_for", lambda _root: run_root)
    monkeypatch.setattr(contract, "_git_head", lambda _root=contract.PROJECT_ROOT: "b" * 64)
    monkeypatch.setattr(
        contract,
        "critical_training_code_bindings",
        lambda **_kwargs: critical,
    )
    launch_payload = {
        "schema": contract.LAUNCH_RECEIPT_SCHEMA,
        "attached_foreground_execution": True,
        "launch_mode": "warm_start",
        "source_step": 0,
        "target_step": 1,
        "resume_checkpoint": None,
        "trainer_output": str(checkpoint.parents[1]),
        "checkpoint": str(checkpoint),
        "objective": contract.OBJECTIVE_VERSION,
        "gradient_accumulation_steps": 7,
        "max_grad_norm": 1.0,
        "first_cycle_pairs": [list(pair) for pair in contract.FIRST_CYCLE_PAIRS],
        "first_cycle_pairs_sha256": contract.FIRST_CYCLE_PAIRS_SHA256,
        "warm_start_checkpoint": str(contract.PINNED_WARM_START_CHECKPOINT),
        "v10_diagnostic_baseline": baseline,
        "base_model_identity": model_identity,
        "training_continuation": contract.TRAINING_CONTINUATION_POLICY,
        "hard32_access": contract.HARD32_ACCESS_POLICY,
        "evaluation_access": "forbidden",
        "tracked_worktree_clean": True,
        "git_commit": "b" * 64,
        "critical_files": critical,
    }
    launch_payload["receipt_sha256"] = contract.canonical_sha256(launch_payload)
    _write_json(launch_path, launch_payload)
    launch_validation = contract.validate_launch_receipt(
        launch_path,
        checkpoint=checkpoint,
        baseline=baseline,
        base_model_identity=model_identity,
        ssd_root=tmp_path,
    )
    telemetry = {
        "schema": "rwkv_ms_scene_memory_v11_cycle_pair_telemetry.v1",
        "pair_presentations": 7,
        "ordered_pairs": [list(pair) for pair in contract.FIRST_CYCLE_PAIRS],
        "ordered_pairs_sha256": contract.FIRST_CYCLE_PAIRS_SHA256,
    }
    checkpoint_contract = {
        "training_protocol_sha256": protocol_sha,
        "cycle_pair_telemetry": telemetry,
    }
    completion_payload = {
        "schema": contract.COMPLETION_RECEIPT_SCHEMA,
        "status": "completed",
        "optimizer_step": 1,
        "consumed_pair_presentations": 7,
        "checkpoint": str(checkpoint),
        "launch_receipt": launch_validation["artifact"],
        "launch_receipt_sha256": launch_validation["receipt_sha256"],
        "checkpoint_artifacts": {
            name: contract.artifact_binding(
                checkpoint / name,
                description=name,
            )
            for name in contract.REQUIRED_CHECKPOINT_ARTIFACTS
        },
        "rng_state_artifacts": {
            "rng_state.pth": contract.artifact_binding(
                checkpoint / "rng_state.pth",
                description="rng",
            )
        },
        "training_summary": contract.artifact_binding(
            summary_path,
            description="summary",
        ),
        "log": contract.artifact_binding(log_path, description="log"),
        "cycle_pair_telemetry": telemetry,
        "training_continuation": contract.TRAINING_CONTINUATION_POLICY,
        "hard32_access": contract.HARD32_ACCESS_POLICY,
        "evaluation_access": "forbidden",
    }
    completion_payload["receipt_sha256"] = contract.canonical_sha256(
        completion_payload
    )
    _write_json(completion_path, completion_payload)

    validated = contract.validate_completion_receipt(
        completion_path,
        checkpoint=checkpoint,
        checkpoint_contract=checkpoint_contract,
        launch=launch_validation,
        ssd_root=tmp_path,
    )

    assert validated["payload"]["cycle_pair_telemetry"] == telemetry
    completion_payload["cycle_pair_telemetry"] = {**telemetry, "pair_presentations": 6}
    completion_payload["receipt_sha256"] = contract.canonical_sha256(
        {key: value for key, value in completion_payload.items() if key != "receipt_sha256"}
    )
    _write_json(completion_path, completion_payload)
    with pytest.raises(contract.LaunchContractError, match="cycle_pair_telemetry"):
        contract.validate_completion_receipt(
            completion_path,
            checkpoint=checkpoint,
            checkpoint_contract=checkpoint_contract,
            launch=launch_validation,
            ssd_root=tmp_path,
        )


def test_v11_launcher_dry_run_parses_and_exits_before_training() -> None:
    launcher = (
        Path(contract.__file__).resolve().parent / "train_scene_memory_v11.sh"
    )
    result = subprocess.run(
        ["bash", "-n", str(launcher)],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    source = launcher.read_text(encoding="utf-8")
    dry_run = source.index('if [[ "${DRY_RUN}" == "1" ]]')
    training_start = source.index("mkdir -p", dry_run)
    assert source.index("exit 0", dry_run) < training_start
    assert "--max-steps 1" in source
    assert "--max-grad-norm 1.0" in source
    assert "--resume-from-checkpoint" not in source
