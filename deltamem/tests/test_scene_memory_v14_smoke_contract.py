from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
import torch

from experiments.rethinking_rwkv_ms_gemma import (
    scene_memory_v14_launch_contract as contract,
)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_receipt(path: Path, payload: dict[str, Any]) -> dict[str, Any]:
    unsigned = {key: value for key, value in payload.items() if key != "receipt_sha256"}
    result = {**unsigned, "receipt_sha256": contract.canonical_sha256(unsigned)}
    _write_json(path, result)
    return result


def _row_hash(ordinal: int) -> str:
    return f"{ordinal:064x}"


def _data() -> dict[str, Any]:
    low, high = contract.ONE_PAIR_SMOKE_PAIR
    return {
        "train_file": "/synthetic/train32.jsonl",
        "entries": [
            {
                "canonical_pair_ordinals": [low, high],
                "members": [
                    {
                        "train_row_ordinal": low,
                        "row_sha256": _row_hash(low),
                    },
                    {
                        "train_row_ordinal": high,
                        "row_sha256": _row_hash(high),
                    },
                ],
            }
        ],
    }


def _row_observation(
    *,
    role: str,
    ordinal: int,
    paired: int,
    parsed_exact: bool,
) -> dict[str, Any]:
    cached_ce = 0.0 if parsed_exact else 0.4
    failed_hinge = 0.0 if parsed_exact else 0.5
    exact_hinge = 0.2 if parsed_exact else 0.0
    loss = exact_hinge if parsed_exact else cached_ce + failed_hinge
    return {
        "phase": "smoke_input",
        "cycle": 1,
        "adapter_optimizer_step_before_update": 0,
        "presentation": 1,
        "pair_role": role,
        "row_ordinal": ordinal,
        "paired_row_ordinal": paired,
        "row_sha256": _row_hash(ordinal),
        "paired_row_sha256": _row_hash(paired),
        "parsed_boundary_exact": parsed_exact,
        "raw_token_exact": parsed_exact,
        "first_divergence": 3,
        "rollout_token_count": 3,
        "cached_branch_kind": (
            "cached_gold_prefix"
            if parsed_exact
            else "cached_actual_greedy_prefix"
        ),
        "cached_branch_kind_code": 0 if parsed_exact else 1,
        "cached_replay_use_cache": True,
        "cached_replay_logits_to_keep": 1,
        "cached_replay_token_count": 3,
        "cached_replay_selected_cursor": 1 if parsed_exact else 2,
        "cached_decision_token_count": 2,
        "cached_selected_decision_ordinal": 0,
        "cached_selected_label_position": 3,
        "cached_selected_gold_token_id": 5,
        "cached_selected_competitor_id": 6,
        "cached_competitor_is_actual_greedy": not parsed_exact,
        "cached_replay_top1_matches_actual": not parsed_exact,
        "cached_replay_top1_match_count": 0 if parsed_exact else 3,
        "cached_ce": cached_ce,
        "cached_failed_competitor_hinge": failed_hinge,
        "cached_exact_retention_hinge": exact_hinge,
        "cached_selected_gold_vs_competitor_margin": 0.5,
        "cached_gold_top1_fraction": 1.0 if parsed_exact else 0.5,
        "cached_alignment_kind_code": -1 if parsed_exact else 0,
        "cached_selected_is_termination": False,
        "cached_branch_loss": loss,
        "auxiliary_optimization_loss": 0.0,
        "auxiliary_telemetry_loss": 0.7,
        "selected_top_competitor_hinge_telemetry": 0.3,
        "selected_correct_vs_zero_hinge_telemetry": 0.4,
        "total_side_loss": loss,
    }


def _row_audit() -> dict[str, Any]:
    low, high = contract.ONE_PAIR_SMOKE_PAIR
    source = _row_observation(
        role="source",
        ordinal=low,
        paired=high,
        parsed_exact=True,
    )
    donor = _row_observation(
        role="donor",
        ordinal=high,
        paired=low,
        parsed_exact=False,
    )
    return {
        "schema": contract.ROW_OBJECTIVE_AUDIT_SCHEMA,
        "memory_objective_version": contract.OBJECTIVE_VERSION,
        "run_mode": contract.ONE_PAIR_SMOKE_RUN_MODE,
        "production_eligible": False,
        "checkpoint_optimizer_step": 1,
        "completed_pair_presentations": 1,
        "phases": ["smoke_input"],
        "pair_schedule": [
            {"source_row_ordinal": low, "donor_row_ordinal": high}
        ],
        "pair_presentations": [
            {
                "phase": "smoke_input",
                "cycle": 1,
                "adapter_optimizer_step_before_update": 0,
                "presentation": 1,
                "source_row_ordinal": low,
                "donor_row_ordinal": high,
                "source_row_sha256": _row_hash(low),
                "donor_row_sha256": _row_hash(high),
                "pair_mean_cached_branch_loss": 0.55,
                "pair_mean_cached_exact_retention_hinge": 0.1,
                "pair_mean_cached_failed_ce": 0.2,
                "pair_mean_cached_failed_competitor_hinge": 0.25,
                "pair_mean_auxiliary_optimization_loss": 0.0,
                "pair_mean_selected_top_competitor_hinge_telemetry": 0.3,
                "pair_mean_selected_correct_vs_zero_hinge_telemetry": 0.4,
                "pair_mean_total_side_loss": 0.55,
                "reported_objective_total_loss": 0.55,
                "recomputed_objective_total_loss": 0.55,
            }
        ],
        "rows": [
            {"row_ordinal": low, "smoke_input": source},
            {"row_ordinal": high, "smoke_input": donor},
        ],
    }


def _trainer_state() -> dict[str, Any]:
    low, high = contract.ONE_PAIR_SMOKE_PAIR
    return {
        "global_step": 1,
        "max_steps": 1,
        "log_history": [
            {
                "step": 1,
                "loss": 0.55,
                "grad_norm": 1.25,
                "learning_rate": contract.LEARNING_RATE,
                "delta/scene_generation_v14_cycle_index": 1,
                "delta/scene_generation_v14_cycle_pair_presentations": 1,
                "delta/scene_generation_v14_cycle_pair_0_low_ordinal": low,
                "delta/scene_generation_v14_cycle_pair_0_high_ordinal": high,
            }
        ],
    }


def _protocol() -> dict[str, Any]:
    return {
        "schema_version": contract.OBJECTIVE_SCHEMA_VERSION,
        "memory_objective_version": contract.OBJECTIVE_VERSION,
        "max_steps": 1,
        "max_grad_norm": contract.MAX_GRAD_NORM,
        "train_sampler_mode": contract.ONE_PAIR_SMOKE_SAMPLER_MODE,
        "gradient_accumulation_steps": 1,
        "learning_rate": contract.LEARNING_RATE,
        "lr_scheduler_type": "constant",
        "warmup_steps": contract.WARMUP_STEPS,
        "warmup_ratio": contract.WARMUP_RATIO,
        "optim": contract.OPTIMIZER_IMPLEMENTATION,
        "weight_decay": contract.WEIGHT_DECAY,
        "logging_steps": contract.LOGGING_STEPS,
        "save_steps": 1,
        "save_total_limit": 1,
        "seed": contract.SEED,
        "data_seed": contract.DATA_SEED,
        "ignore_data_skip": False,
        "scene_generation_v14_run_mode": contract.ONE_PAIR_SMOKE_RUN_MODE,
        "scene_generation_v14_production_eligible": False,
        "scene_generation_cycle_pair_presentations": 1,
        "scene_generation_gradient_accumulation_pair_cycle": 1,
        "scene_generation_raw_token_exact_optimization_weight": 0.0,
        "scene_generation_schema_ce_optimization_scope": (
            "standalone_schema_mask_partition_only_v1"
        ),
        "scene_generation_failed_prefix_replay": contract.FAILED_REPLAY_MODE,
        "train_schedule": {
            "schema": contract.PAIR_CURRICULUM_BINDING_SCHEMA,
            "checkpoint_steps": [1],
            "optimizer_checkpoint_steps": [1],
            "microbatch_cycle_size": 1,
            "continuation_policy": contract.CONTINUATION_POLICY,
            "source_total_steps": 28,
            "source_checkpoint_steps": [7, 14, 21, 28],
            "source_ordered_pairs_sha256": contract.FOUR_CYCLE_PAIRS_SHA256,
            "schedule_selection_mode": (
                contract.ONE_PAIR_SMOKE_SCHEDULE_SELECTION_MODE
            ),
            "active_ordered_pairs_sha256": contract.canonical_sha256(
                [list(contract.ONE_PAIR_SMOKE_PAIR)]
            ),
            "total_steps": 1,
            "pair_indices": [list(contract.ONE_PAIR_SMOKE_PAIR)],
        },
    }


def _checkpoint_fixture(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[Path, dict[str, Any], dict[str, Any]]:
    output = (
        contract.v14_run_root_for(tmp_path)
        / "scene_memory_v14_smoke_unit_step1"
    )
    checkpoint = output / "trainer/checkpoint-1"
    checkpoint.mkdir(parents=True)
    source_checkpoint = tmp_path / "warm/checkpoint-4"
    source_checkpoint.mkdir(parents=True)
    torch.save({"weight": torch.tensor([1.0, 2.0])}, source_checkpoint / "delta_mem_adapter.pt")
    torch.save({"weight": torch.tensor([1.0, 2.25])}, checkpoint / "delta_mem_adapter.pt")
    _write_json(checkpoint / "trainer_state.json", _trainer_state())
    _write_json(checkpoint / "training_protocol.json", _protocol())
    _write_json(checkpoint / "delta_mem_config.json", {})
    _write_json(checkpoint / "scene_state_identity_pairing_manifest.json", {})
    _write_json(checkpoint / contract.ROW_OBJECTIVE_AUDIT_FILENAME, _row_audit())
    _write_json(checkpoint / contract.warm_module_lineage_filename(), {})
    for name in ("optimizer.pt", "scheduler.pt", "rng_state.pth"):
        (checkpoint / name).write_bytes(b"artifact\n")
    monkeypatch.setattr(contract, "_validate_checkpoint_protocol", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(contract.v13.v10.v9, "_validate_checkpoint_config", lambda _value: None)
    monkeypatch.setattr(contract, "_validate_pairing_manifest", lambda _value: "pairing")
    monkeypatch.setattr(contract, "_validate_warm_lineage", lambda *_args, **_kwargs: "lineage")
    warm = {"warm_start_checkpoint": str(source_checkpoint)}
    return checkpoint, _data(), warm


def test_one_pair_smoke_checkpoint_requires_real_update_and_cached_parity(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    checkpoint, data, warm = _checkpoint_fixture(tmp_path, monkeypatch)

    result = contract.validate_one_pair_smoke_checkpoint_contract(
        checkpoint,
        data=data,
        warm=warm,
        ssd_root=tmp_path,
    )

    assert result["consumed_pair_presentations"] == 1
    assert result["cycle_pair_telemetry"]["grad_norm"] == 1.25
    assert result["row_objective_audit"]["cached_replay_top1_parity_verified"]
    assert result["adapter_update"]["changed_tensor_count"] == 1

    state = _trainer_state()
    state["log_history"][0]["grad_norm"] = 0.0
    _write_json(checkpoint / "trainer_state.json", state)
    with pytest.raises(contract.LaunchContractError, match="grad_norm_must_be_positive"):
        contract.validate_one_pair_smoke_checkpoint_contract(
            checkpoint,
            data=data,
            warm=warm,
            ssd_root=tmp_path,
        )


def _launch_payload(
    output: Path,
    *,
    baseline: dict[str, Any],
    base_identity: dict[str, Any],
    critical_files: dict[str, Any],
) -> dict[str, Any]:
    checkpoint = output / "trainer/checkpoint-1"
    logs = output.parent / "logs"
    return {
        "schema": contract.ONE_PAIR_SMOKE_LAUNCH_RECEIPT_SCHEMA,
        "run_name": "unit",
        "attached_foreground_execution": True,
        "launch_mode": "warm_start_smoke",
        "run_mode": contract.ONE_PAIR_SMOKE_RUN_MODE,
        "production_eligible": False,
        "source_step": 0,
        "source_checkpoint_step": 4,
        "target_step": 1,
        "resume_checkpoint": None,
        "trainer_output": str(output),
        "checkpoints": {"checkpoint-1": str(checkpoint)},
        "log_file": str(logs / f"{output.name}.log"),
        "objective": contract.OBJECTIVE_VERSION,
        "objective_schema_version": contract.OBJECTIVE_SCHEMA_VERSION,
        "gradient_accumulation_steps": 1,
        "max_grad_norm": contract.MAX_GRAD_NORM,
        "max_steps": 1,
        "learning_rate": contract.LEARNING_RATE,
        "optim": contract.OPTIMIZER_IMPLEMENTATION,
        "weight_decay": contract.WEIGHT_DECAY,
        "lr_scheduler_type": "constant",
        "warmup_steps": contract.WARMUP_STEPS,
        "warmup_ratio": contract.WARMUP_RATIO,
        "logging_steps": contract.LOGGING_STEPS,
        "save_steps": 1,
        "save_total_limit": 1,
        "seed": contract.SEED,
        "data_seed": contract.DATA_SEED,
        "total_pair_presentations": 1,
        "scheduled_pairs": [list(contract.ONE_PAIR_SMOKE_PAIR)],
        "four_cycle_pairs": [list(pair) for pair in contract.FOUR_CYCLE_PAIRS],
        "four_cycle_pairs_sha256": contract.FOUR_CYCLE_PAIRS_SHA256,
        "warm_start_checkpoint": str(contract.PINNED_WARM_START_CHECKPOINT),
        "warm_start_adapter_sha256": contract.PINNED_WARM_START_ADAPTER_SHA256,
        "warm_start_mode": contract.warm.WARM_START_MODE,
        "warm_start_lock": "/synthetic/warm-lock.json",
        "warm_start_lock_sha256": "a" * 64,
        "v10_diagnostic_baseline": baseline,
        "base_model_identity": base_identity,
        "critical_files": critical_files,
        "tracked_worktree_clean": True,
        "training_continuation": contract.TRAINING_CONTINUATION_POLICY,
        "hard32_access": contract.HARD32_ACCESS_POLICY,
        "evaluation_access": "forbidden",
        "git_commit": "b" * 40,
    }


def _validated_launch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    checkpoint: Path,
) -> tuple[Path, dict[str, Any]]:
    output = checkpoint.parents[1]
    logs = output.parent / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    baseline = {"baseline": True}
    base_identity = {"model": True}
    critical_files = {"contract.py": {"sha256": "c" * 64}}
    launch_path = logs / f"{output.name}.launch.json"
    _write_receipt(
        launch_path,
        _launch_payload(
            output,
            baseline=baseline,
            base_identity=base_identity,
            critical_files=critical_files,
        ),
    )
    monkeypatch.setattr(
        contract,
        "validate_warm_start_contract",
        lambda **_kwargs: {
            "warm_start_lock": "/synthetic/warm-lock.json",
            "warm_start_lock_sha256": "a" * 64,
        },
    )
    monkeypatch.setattr(contract, "_resolve_git_commit", lambda *_args, **_kwargs: "b" * 40)
    monkeypatch.setattr(contract, "_git_head", lambda *_args, **_kwargs: "b" * 40)
    monkeypatch.setattr(contract, "_require_git_ancestor", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        contract,
        "critical_training_code_bindings_at_commit",
        lambda *_args, **_kwargs: critical_files,
    )
    validated = contract.validate_one_pair_smoke_launch_receipt(
        launch_path,
        checkpoint=checkpoint,
        baseline=baseline,
        base_model_identity=base_identity,
        ssd_root=tmp_path,
    )
    return launch_path, validated


def test_one_pair_smoke_launch_receipt_is_nonproduction_and_one_pair_only(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    checkpoint = (
        contract.v14_run_root_for(tmp_path)
        / "scene_memory_v14_smoke_unit_step1/trainer/checkpoint-1"
    )
    checkpoint.mkdir(parents=True)
    launch_path, validated = _validated_launch(tmp_path, monkeypatch, checkpoint)

    assert validated["payload"]["scheduled_pairs"] == [
        list(contract.ONE_PAIR_SMOKE_PAIR)
    ]
    payload = dict(validated["payload"])
    payload["production_eligible"] = True
    _write_receipt(launch_path, payload)
    with pytest.raises(contract.LaunchContractError, match="mode_or_horizon"):
        contract.validate_one_pair_smoke_launch_receipt(
            launch_path,
            checkpoint=checkpoint,
            baseline={"baseline": True},
            base_model_identity={"model": True},
            ssd_root=tmp_path,
        )


def test_one_pair_smoke_completion_binds_artifacts_cuda_and_forbids_evaluation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    checkpoint, data, warm = _checkpoint_fixture(tmp_path, monkeypatch)
    checkpoint_contract = contract.validate_one_pair_smoke_checkpoint_contract(
        checkpoint,
        data=data,
        warm=warm,
        ssd_root=tmp_path,
    )
    launch_path, launch = _validated_launch(tmp_path, monkeypatch, checkpoint)
    output = checkpoint.parents[1]
    log_path = Path(launch["payload"]["log_file"])
    log_path.write_text("completed smoke\n", encoding="utf-8")
    summary_path = output / "training_summary.json"
    _write_json(
        summary_path,
        {
            "memory_objective_version": contract.OBJECTIVE_VERSION,
            "warm_start_mode": contract.warm.WARM_START_MODE,
            "warm_start_from_checkpoint": str(contract.PINNED_WARM_START_CHECKPOINT),
            "resume_from_checkpoint": None,
            "training_protocol_sha256": checkpoint_contract[
                "training_protocol_sha256"
            ],
            "train_sampler_mode": contract.ONE_PAIR_SMOKE_SAMPLER_MODE,
            "training_mode": "episode",
            "save_steps": 1,
            "save_total_limit": 1,
            "train_schedule": {
                "schedule_selection_mode": (
                    contract.ONE_PAIR_SMOKE_SCHEDULE_SELECTION_MODE
                ),
                "active_ordered_pairs_sha256": contract.canonical_sha256(
                    [list(contract.ONE_PAIR_SMOKE_PAIR)]
                ),
                "total_steps": 1,
            },
            "cuda_memory": {
                "device": "cuda:0",
                "baseline_allocated_bytes": 100,
                "baseline_reserved_bytes": 200,
                "peak_allocated_bytes": 500,
                "peak_reserved_bytes": 600,
                "post_train_allocated_bytes": 150,
                "post_train_reserved_bytes": 300,
            },
        },
    )
    entry = {
        "path": str(checkpoint),
        "optimizer_step": 1,
        "consumed_pair_presentations": 1,
        "checkpoint_artifacts": {
            name: contract.artifact_binding(checkpoint / name, description=name)
            for name in contract.REQUIRED_CHECKPOINT_ARTIFACTS
        },
        "rng_state_artifacts": {
            "rng_state.pth": contract.artifact_binding(
                checkpoint / "rng_state.pth",
                description="rng",
            )
        },
        "cycle_pair_telemetry": checkpoint_contract["cycle_pair_telemetry"],
        "row_objective_audit_file_sha256": checkpoint_contract[
            "row_objective_audit_file_sha256"
        ],
    }
    completion_path = launch_path.with_name(
        launch_path.name.removesuffix(".launch.json") + ".completion.json"
    )
    payload = {
        "schema": contract.ONE_PAIR_SMOKE_COMPLETION_RECEIPT_SCHEMA,
        "status": "completed",
        "optimizer_step": 1,
        "consumed_pair_presentations": 1,
        "launch_receipt": launch["artifact"],
        "launch_receipt_sha256": launch["receipt_sha256"],
        "log": contract.artifact_binding(log_path, description="log"),
        "training_summary": contract.artifact_binding(
            summary_path,
            description="summary",
        ),
        "checkpoints": {"checkpoint-1": entry},
        "training_continuation": contract.TRAINING_CONTINUATION_POLICY,
        "hard32_access": contract.HARD32_ACCESS_POLICY,
        "evaluation_access": "forbidden",
        "production_eligible": False,
    }
    _write_receipt(completion_path, payload)

    validated = contract.validate_one_pair_smoke_completion_receipt(
        completion_path,
        checkpoint=checkpoint,
        checkpoint_contract=checkpoint_contract,
        launch=launch,
        ssd_root=tmp_path,
    )

    assert validated["cuda_memory"]["peak_allocated_bytes"] == 500
    assert validated["adapter_update"]["changed_tensor_count"] == 1
    payload["evaluation_access"] = "value14_gate_only"
    _write_receipt(completion_path, payload)
    with pytest.raises(contract.LaunchContractError, match="authorization"):
        contract.validate_one_pair_smoke_completion_receipt(
            completion_path,
            checkpoint=checkpoint,
            checkpoint_contract=checkpoint_contract,
            launch=launch,
            ssd_root=tmp_path,
        )
