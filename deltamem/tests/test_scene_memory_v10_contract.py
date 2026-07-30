from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path

import pytest
import torch
from torch import nn

from experiments.rethinking_rwkv_ms_gemma import (
    scene_memory_v10_launch_contract as contract,
)
from experiments.rethinking_rwkv_ms_gemma import (
    scene_memory_v10_warm_start as warm_start,
)


class _FakeAdapterModel(nn.Module):
    def __init__(self, state: dict[str, torch.Tensor]) -> None:
        super().__init__()
        self.adapter_state = {
            name: tensor.detach().clone() for name, tensor in state.items()
        }


def _entries() -> list[dict[str, object]]:
    pairs = list(contract.CANONICAL_VALUE14_PAIRS)
    entries: list[dict[str, object]] = []
    for cycle in range(4):
        for position, pair in enumerate(pairs[cycle:] + pairs[:cycle]):
            entries.append(
                {
                    "schedule_index": len(entries),
                    "canonical_pair_ordinals": list(pair),
                    "entry_sha256": f"entry-{cycle}-{position}",
                }
            )
    return entries


def _data() -> dict[str, object]:
    return {
        "train_file": "/ssd/train32.jsonl",
        "source_manifest": "/ssd/source.json",
        "source_manifest_file_sha256": "source-file",
        "schedule": "/ssd/schedule.jsonl",
        "schedule_file_sha256": "schedule-file",
        "schedule_entries_sha256": "schedule-entries",
        "schedule_manifest": "/ssd/schedule-manifest.json",
        "schedule_manifest_file_sha256": "schedule-manifest-file",
        "schedule_manifest_sha256": "schedule-manifest",
        "ordered_pairs_sha256": "ordered-pairs",
        "source_presentation_checkpoint_steps": [7, 14, 21, 28],
        "entries": _entries(),
    }


def _protocol(data: dict[str, object], *, step: int = 1) -> dict[str, object]:
    return {
        "schema_version": contract.OBJECTIVE_SCHEMA_VERSION,
        "memory_objective_version": contract.OBJECTIVE_VERSION,
        "memory_loss_mode": "scene_state_generation_ce",
        "train_file": data["train_file"],
        "eval_samples": 0,
        "training_mode": "episode",
        "assistant_loss_mode": "final_assistant_only",
        "episode_recent_messages": 0,
        "episode_read_write_enabled": False,
        "max_length": 256,
        "max_write_length": 2048,
        "teacher_max_length": 2304,
        "per_device_train_batch_size": 1,
        "per_device_eval_batch_size": 1,
        "gradient_accumulation_steps": 7,
        "learning_rate": 2e-4,
        "lr_scheduler_type": "constant",
        "warmup_ratio": 0.0,
        "warmup_steps": 0,
        "weight_decay": 0.0,
        "optim": "adamw_torch_fused",
        "save_steps": 1,
        "logging_steps": 1,
        "eval_steps": 1000,
        "save_total_limit": 1,
        "num_train_epochs": 1.0,
        "max_steps": step,
        "validation_split_ratio": 0.0,
        "load_best_model_at_end": False,
        "dataset_num_proc": 1,
        "dataloader_num_workers": 0,
        "frozen_mlp_activation_checkpointing": True,
        "seed": 42,
        "data_seed": 42,
        "dtype": "bfloat16",
        "bf16": True,
        "tf32": True,
        "train_sampler_seed": None,
        "train_sampler_mode": contract.FIXED_SAMPLER_MODE,
        "ignore_data_skip": False,
        "scene_boundary_payload_ce_weight": 0.0,
        "memory_dropout_no_memory_prob": 0.0,
        "memory_dropout_state_only_prob": 0.0,
        "memory_contrast_weight": 0.0,
        "memory_kl_weight": 0.0,
        "memory_base_kl_weight": 0.0,
        "memory_representation_weight": 0.0,
        "memory_causal_weight": 0.0,
        "memory_anchor_weight": 0.0,
        "memory_recover_weight": 0.0,
        "write_sparsity_weight": 0.0,
        "memory_partition_alignment_weight": 0.0,
        "memory_partition_entropy_weight": 0.0,
        "memory_partition_balance_weight": 0.0,
        "scene_generation_objective_formula": contract.OBJECTIVE_FORMULA,
        "scene_generation_backward_mode": contract.BACKWARD_MODE,
        "scene_generation_generated_unlikelihood_weight": 0.0,
        "scene_generation_generated_unlikelihood_max_wrong_tokens": 4,
        "scene_generation_generated_prefix_correction_weight": 0.5,
        "scene_generation_generated_prefix_correction_mode": (
            contract.GENERATED_PREFIX_MODE
        ),
        "scene_generation_generated_prefix_max_correction_events": 4,
        "scene_generation_generated_rollout_extra_tokens": 4,
        "scene_generation_generated_rollout_max_tokens": 24,
        "scene_generation_generated_rollout_decoding": (
            "greedy_use_cache_true_exact_system_only_prompt_v1"
        ),
        "scene_generation_generated_replay_state_gradient": True,
        "scene_generation_generated_replay_read_path_gradient": True,
        "scene_generation_pair_physical_batch_size": 1,
        "scene_generation_pair_directional_exposures": 2,
        "scene_generation_first_error_top1_hinge_weight": 1.0,
        "scene_generation_all_target_top1_retention_weight": 1.0,
        "scene_generation_all_target_top1_retention_margin": 0.2,
        "scene_generation_selected_full_vocab_ce_in_total": False,
        "scene_generation_selected_full_vocab_ce_optimization_weight": 0.0,
        "scene_generation_cycle_retention_mode": contract.CYCLE_RETENTION_MODE,
        "scene_generation_cycle_pair_presentations": 7,
        "scene_generation_gradient_accumulation_pair_cycle": 7,
        "train_schedule": contract._expected_schedule_protocol(data),
    }


def _write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")


def _self_hash(payload: dict[str, object], field: str) -> dict[str, object]:
    result = dict(payload)
    result[field] = contract.canonical_sha256(result)
    return result


def _write_checkpoint_payload(
    checkpoint: Path,
    *,
    step: int,
    protocol: dict[str, object],
    pairing: dict[str, object],
) -> None:
    checkpoint.mkdir(parents=True)
    for name in ("delta_mem_adapter.pt", "optimizer.pt", "scheduler.pt", "rng_state.pth"):
        (checkpoint / name).write_bytes(name.encode("ascii"))
    _write_json(checkpoint / "delta_mem_config.json", {})
    _write_json(
        checkpoint / "trainer_state.json",
        {"global_step": step, "max_steps": step},
    )
    _write_json(checkpoint / "training_protocol.json", protocol)
    _write_json(
        checkpoint / "scene_state_identity_pairing_manifest.json",
        pairing,
    )


def test_v10_optimizer_endpoints_bind_complete_seven_pair_cycles() -> None:
    assert contract.CHECKPOINT_STEPS == (1, 2, 3, 4)
    assert contract.PRESENTATION_CHECKPOINTS == (7, 14, 21, 28)
    assert contract.GRADIENT_ACCUMULATION_STEPS == 7
    assert [contract.presentation_cursor(step) for step in range(5)] == [
        0,
        7,
        14,
        21,
        28,
    ]
    canonical = set(contract.CANONICAL_VALUE14_PAIRS)
    entries = _entries()
    for step in contract.CHECKPOINT_STEPS:
        start = contract.presentation_cursor(step - 1)
        stop = contract.presentation_cursor(step)
        pairs = {
            tuple(entry["canonical_pair_ordinals"])
            for entry in entries[start:stop]
        }
        assert len(entries[start:stop]) == len(pairs) == 7
        assert pairs == canonical


def test_v10_real_reused_schedule_is_four_exact_pair_permutations() -> None:
    data = contract.validate_data_contract()

    assert data["checkpoint_steps"] == [1, 2, 3, 4]
    assert data["presentation_checkpoint_steps"] == [7, 14, 21, 28]
    assert len(data["optimizer_cycles"]) == 4
    assert [cycle["presentation_stop"] for cycle in data["optimizer_cycles"]] == [
        7,
        14,
        21,
        28,
    ]


def test_v10_protocol_binds_objective_accumulation_and_resume_cursor() -> None:
    data = _data()
    protocol = _protocol(data)

    contract._validate_checkpoint_protocol(protocol, checkpoint_step=1, data=data)
    assert "selected_full_vocab_ce=telemetry_only" in contract.OBJECTIVE_FORMULA
    assert "first_error_top1_hinge(0.2)" in contract.OBJECTIVE_FORMULA
    assert "all_target_top1_retention_hinge(0.2)" in contract.OBJECTIVE_FORMULA
    assert "generated_prefix_per_event_mean" in contract.OBJECTIVE_FORMULA
    assert contract.GENERATED_PREFIX_MODE == (
        "levenshtein_raw_generated_prefix_per_event_mean_gold_ce_safe_wrong_"
        "unlikelihood_v4"
    )
    assert contract.CYCLE_RETENTION_MODE == (
        "teacher_forced_all_target_top1_margin_detached_competitor_v1"
    )
    assert protocol["scene_generation_generated_rollout_decoding"] == (
        "greedy_use_cache_true_exact_system_only_prompt_v1"
    )
    assert protocol["scene_generation_generated_replay_state_gradient"] is True
    assert protocol["scene_generation_generated_replay_read_path_gradient"] is True
    assert protocol["scene_generation_selected_full_vocab_ce_in_total"] is False
    assert protocol["scene_generation_selected_full_vocab_ce_optimization_weight"] == 0.0
    assert protocol["ignore_data_skip"] is False
    assert protocol["train_schedule"]["optimizer_checkpoint_steps"] == [1, 2, 3, 4]
    assert protocol["train_schedule"]["microbatch_cycle_size"] == 7
    assert protocol["train_schedule"]["resume_schedule_cursor_formula"] == (
        "global_step_times_7_v1"
    )

    drifted = dict(protocol)
    drifted["ignore_data_skip"] = True
    with pytest.raises(contract.LaunchContractError, match="ignore_data_skip"):
        contract._validate_checkpoint_protocol(
            drifted,
            checkpoint_step=1,
            data=data,
        )


@pytest.mark.parametrize(
    "field",
    (
        "max_length",
        "max_write_length",
        "teacher_max_length",
        "per_device_eval_batch_size",
        "weight_decay",
        "optim",
        "logging_steps",
        "eval_steps",
        "save_total_limit",
        "validation_split_ratio",
        "load_best_model_at_end",
        "dataset_num_proc",
        "dataloader_num_workers",
        "frozen_mlp_activation_checkpointing",
        "seed",
        "data_seed",
        "dtype",
        "bf16",
        "tf32",
        "scene_boundary_payload_ce_weight",
        "memory_dropout_no_memory_prob",
        "memory_dropout_state_only_prob",
        "memory_contrast_weight",
        "memory_causal_weight",
        "memory_anchor_weight",
        "memory_recover_weight",
        "memory_partition_alignment_weight",
        "memory_partition_entropy_weight",
        "memory_partition_balance_weight",
        "scene_generation_generated_unlikelihood_max_wrong_tokens",
        "scene_generation_generated_prefix_max_correction_events",
        "scene_generation_generated_rollout_extra_tokens",
        "scene_generation_generated_rollout_max_tokens",
    ),
)
def test_v10_protocol_rejects_live_launcher_parameter_drift(field: str) -> None:
    data = _data()
    protocol = _protocol(data)
    current = protocol[field]
    if isinstance(current, bool):
        protocol[field] = not current
    elif isinstance(current, str):
        protocol[field] = current + "_drift"
    else:
        protocol[field] = current + 1

    with pytest.raises(contract.LaunchContractError, match=field):
        contract._validate_checkpoint_protocol(
            protocol,
            checkpoint_step=1,
            data=data,
        )


def test_v10_checkpoint_lineage_binds_root_continuation_and_cycle_cursor(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    data = _data()
    pairing = _self_hash(
        {"objective_version": contract.PAIRING_OBJECTIVE_VERSION},
        "manifest_sha256",
    )
    config_sha256 = contract.canonical_sha256({})
    warm = {
        "warm_start_checkpoint": str(tmp_path / "v8" / "checkpoint-56"),
        "warm_start_lock": str(tmp_path / "v8-lock.json"),
        "warm_start_lock_sha256": "a" * 64,
    }
    run_root = contract.v10_run_root_for(tmp_path)
    checkpoint1 = run_root / "cycle1" / "trainer" / "checkpoint-1"
    protocol1 = _protocol(data, step=1)
    _write_checkpoint_payload(
        checkpoint1,
        step=1,
        protocol=protocol1,
        pairing=pairing,
    )
    root_lineage = _self_hash(
        {
            "schema": warm_start.RECEIPT_SCHEMA,
            "schema_version": 1,
            "mode": warm_start.WARM_START_MODE,
            "source_checkpoint": warm["warm_start_checkpoint"],
            "source_lock": {
                "path": warm["warm_start_lock"],
                "lock_sha256": warm["warm_start_lock_sha256"],
            },
            "source_state_imports": warm_start.SOURCE_IMPORT_POLICY,
            "post_load_bit_equal": True,
            "target_fresh_start": {
                "initial_global_step": 0,
                "optimizer_implementation": "adamw_torch_fused",
                "optimizer_created_after_adapter_load": True,
                "optimizer_state": "fresh",
                "scheduler_state": "fresh",
                "trainer_state": "fresh",
                "rng_state": "fresh_from_v10_seed",
            },
            "trainer_resume_from_checkpoint": None,
            "target_initial_global_step": 0,
            "pre_train_global_step": 0,
            "fresh_optimizer_created": True,
            "fresh_optimizer_state_entries_before_train": 0,
            "fresh_scheduler_created_before_train": False,
            "fresh_optimizer_class": "torch.optim.AdamW",
            "target_delta_config_sha256": config_sha256,
            "target_training_protocol_sha256": contract.canonical_sha256(protocol1),
            "target_scene_state_pairing_manifest_sha256": pairing["manifest_sha256"],
        },
        "receipt_sha256",
    )
    root_lineage_path = checkpoint1 / warm_start.WARM_START_LINEAGE_FILENAME
    _write_json(root_lineage_path, root_lineage)

    checkpoint2 = run_root / "cycle2" / "trainer" / "checkpoint-2"
    protocol2 = _protocol(data, step=2)
    _write_checkpoint_payload(
        checkpoint2,
        step=2,
        protocol=protocol2,
        pairing=pairing,
    )
    continuation = _self_hash(
        {
            "schema_version": contract.CONTINUATION_LINEAGE_SCHEMA_VERSION,
            "mode": "extend",
            "source_checkpoint": str(checkpoint1.resolve()),
            "source_global_step": 1,
            "target_max_steps": 2,
            "source_schedule_cursor": 7,
            "target_schedule_cursor": 14,
            "source_training_protocol_sha256": contract.canonical_sha256(protocol1),
            "target_training_protocol_sha256": contract.canonical_sha256(protocol2),
            "source_lineage_filename": warm_start.WARM_START_LINEAGE_FILENAME,
            "source_lineage_file_sha256": contract.sha256_file(root_lineage_path),
            "root_warm_start_receipt_sha256": root_lineage["receipt_sha256"],
        },
        "manifest_sha256",
    )
    continuation_path = checkpoint2 / warm_start.CONTINUATION_LINEAGE_FILENAME
    _write_json(continuation_path, continuation)
    monkeypatch.setattr(contract.v9, "_validate_checkpoint_config", lambda _config: None)

    result = contract.validate_checkpoint_contract(
        checkpoint2,
        data=data,
        warm=warm,
        ssd_root=tmp_path,
    )

    assert result["checkpoint_step"] == 2
    assert result["consumed_pair_presentations"] == 14
    assert protocol1["ignore_data_skip"] is False
    assert protocol2["ignore_data_skip"] is False

    drifted_lineage = dict(continuation)
    drifted_lineage["source_schedule_cursor"] = 1
    drifted_lineage.pop("manifest_sha256")
    drifted_lineage = _self_hash(drifted_lineage, "manifest_sha256")
    _write_json(continuation_path, drifted_lineage)
    with pytest.raises(contract.LaunchContractError, match="horizon_or_protocol"):
        contract.validate_checkpoint_contract(
            checkpoint2,
            data=data,
            warm=warm,
            ssd_root=tmp_path,
        )

    _write_json(continuation_path, continuation)
    drifted_protocol = dict(protocol2)
    drifted_protocol["ignore_data_skip"] = True
    _write_json(checkpoint2 / "training_protocol.json", drifted_protocol)
    with pytest.raises(contract.LaunchContractError, match="ignore_data_skip"):
        contract.validate_checkpoint_contract(
            checkpoint2,
            data=data,
            warm=warm,
            ssd_root=tmp_path,
        )


def test_v10_cycle_validation_rejects_duplicate_pair(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    reused = contract.validate_data_contract()
    reused.update(
        {
            "checkpoint_steps": [7, 14, 21, 28],
            "schedule_entries_sha256": "schedule-entries",
        }
    )
    entries = list(reused["entries"])
    entries[6] = dict(entries[0])
    reused["entries"] = entries
    monkeypatch.setattr(contract.v9, "validate_data_contract", lambda **_kwargs: reused)

    with pytest.raises(contract.LaunchContractError, match="cycle_not_complete"):
        contract.validate_data_contract(
            data_root=contract.DATA_ROOT,
            ssd_root=contract.SSD_ROOT,
        )


def test_v10_rejects_protected_path_before_resolve(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    protected = tmp_path / "hard32" / "receipt.json"
    original_resolve = Path.resolve

    def guarded_resolve(path: Path, *args, **kwargs):
        if "hard32" in {part.lower() for part in path.parts}:
            raise AssertionError("protected path was resolved")
        return original_resolve(path, *args, **kwargs)

    monkeypatch.setattr(Path, "resolve", guarded_resolve)
    with pytest.raises(contract.LaunchContractError, match="forbidden_hard32"):
        contract.require_ssd(
            protected,
            description="protected receipt",
            ssd_root=tmp_path,
        )


@pytest.mark.parametrize("marker", ("hard32-copy", "my_eval", "validation-set", "holdout"))
def test_v10_path_guard_rejects_protected_substrings(marker: str, tmp_path: Path) -> None:
    with pytest.raises(contract.LaunchContractError, match="forbidden_hard32"):
        contract.require_ssd(
            tmp_path / marker / "artifact.json",
            description="protected artifact",
            ssd_root=tmp_path,
        )


def test_v10_path_guard_rejects_parent_alias_and_parent_symlink(tmp_path: Path) -> None:
    with pytest.raises(contract.LaunchContractError, match="parent_alias"):
        contract.require_ssd(
            tmp_path / "data" / ".." / "artifact.json",
            description="aliased artifact",
            ssd_root=tmp_path,
        )

    real = tmp_path / "real"
    real.mkdir()
    alias = tmp_path / "alias"
    alias.symlink_to(real, target_is_directory=True)
    with pytest.raises(contract.LaunchContractError, match="symlink_component"):
        contract.require_ssd(
            alias / "artifact.json",
            description="symlinked artifact",
            ssd_root=tmp_path,
        )


def test_v10_exact_roots_and_historical_exception_are_fail_closed(tmp_path: Path) -> None:
    run_root = contract.v10_run_root_for(tmp_path)
    gate_root = contract.v10_gates_root_for(tmp_path)
    assert contract.require_v10_run_path(
        run_root / "run" / "trainer" / "checkpoint-1",
        description="checkpoint",
        ssd_root=tmp_path,
    ) == run_root / "run" / "trainer" / "checkpoint-1"
    assert contract.require_v10_gate_path(
        gate_root / "checkpoint-1" / "gate_receipt.json",
        description="receipt",
        ssd_root=tmp_path,
    ) == gate_root / "checkpoint-1" / "gate_receipt.json"
    with pytest.raises(contract.LaunchContractError, match="outside_locked_root"):
        contract.require_v10_run_path(
            tmp_path / "other" / "checkpoint-1",
            description="checkpoint",
            ssd_root=tmp_path,
        )
    with pytest.raises(contract.LaunchContractError, match="outside_locked_root"):
        contract.require_v10_gate_path(
            run_root / "not-gates" / "gate_receipt.json",
            description="receipt",
            ssd_root=tmp_path,
        )

    pinned = contract.PINNED_HISTORICAL_TRAIN32_ARTIFACTS["train32"]["path"]
    assert contract._lexically_guard_path(
        pinned,
        description="pinned historical Train32",
        protected_exact_allowlist=(pinned,),
    ) == pinned
    with pytest.raises(contract.LaunchContractError, match="forbidden_hard32"):
        contract._lexically_guard_path(
            pinned.with_name("train32-copy.jsonl"),
            description="historical Train32 copy",
            protected_exact_allowlist=(pinned,),
        )


def test_v10_warm_start_has_distinct_target_identity(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    source = type(
        "Context",
        (),
        {
            "checkpoint": tmp_path / "checkpoint-56",
            "lock_path": tmp_path / "lock.json",
            "lock": {},
            "source_config": {},
            "source_trainer_state": {},
            "source_training_protocol": {},
            "source_pairing_manifest": {},
            "continuation_lineage": (),
        },
    )()
    monkeypatch.setattr(
        warm_start.v9,
        "prepare_v9_v8_checkpoint56_warm_start",
        lambda *_args, **_kwargs: source,
    )

    result = warm_start.prepare_v10_v8_checkpoint56_warm_start(
        source.checkpoint,
        lock_path=source.lock_path,
    )

    assert result.checkpoint == source.checkpoint
    assert warm_start.WARM_START_MODE == "scene_memory_v10_v8_checkpoint56_adapter_only"
    assert warm_start.RECEIPT_SCHEMA == (
        "rwkv_ms_scene_memory_v10_adapter_warm_start_receipt.v1"
    )
    assert warm_start.TARGET_FRESH_START_POLICY["rng_state"] == (
        "fresh_from_v10_seed"
    )
    assert warm_start.TARGET_FRESH_START_POLICY != (
        warm_start.v9.TARGET_FRESH_START_POLICY
    )


def test_v10_apply_emits_native_fresh_start_receipt(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    checkpoint = tmp_path / "checkpoint-56"
    checkpoint.mkdir()
    source_state = {
        "first": torch.arange(4, dtype=torch.float32),
        "second": torch.arange(3, dtype=torch.bfloat16),
    }
    torch.save(source_state, checkpoint / "delta_mem_adapter.pt")
    _, topology_sha256, tensor_elements = warm_start.v9.ordered_adapter_topology(
        source_state
    )
    context = warm_start.V10WarmStartContext(
        checkpoint=checkpoint,
        lock_path=tmp_path / "v8-lock.json",
        lock={
            "lock_sha256": "a" * 64,
            "artifacts": {"delta_mem_adapter.pt": {"sha256": "b" * 64}},
            "adapter_topology": {
                "sha256": topology_sha256,
                "tensor_count": len(source_state),
                "tensor_elements": tensor_elements,
            },
            "source_state_imports": warm_start.SOURCE_IMPORT_POLICY,
        },
        source_config={},
        source_trainer_state={"global_step": 56, "epoch": 1.0},
        source_training_protocol={"memory_objective_version": "v8-objective"},
        source_pairing_manifest={"objective_version": "v8-pairing"},
        continuation_lineage=(),
    )
    model = _FakeAdapterModel(
        {name: torch.zeros_like(tensor) for name, tensor in source_state.items()}
    )

    def get_state(target: _FakeAdapterModel) -> dict[str, torch.Tensor]:
        return {
            name: tensor.detach().clone()
            for name, tensor in target.adapter_state.items()
        }

    def load_state(
        target: _FakeAdapterModel,
        state: dict[str, torch.Tensor],
    ) -> None:
        target.adapter_state = {
            name: tensor.detach().clone() for name, tensor in state.items()
        }

    monkeypatch.setattr(warm_start, "get_delta_mem_state_dict", get_state)
    monkeypatch.setattr(warm_start, "load_delta_mem_state_dict", load_state)
    fresh = warm_start.V10FreshStartContract(
        resume_from_checkpoint=None,
        initial_global_step=0,
        optimizer_created=False,
        scheduler_created=False,
        trainer_state_imported=False,
        rng_state_imported=False,
        optim="adamw_torch_fused",
    )

    receipt = warm_start.apply_v10_v8_checkpoint56_adapter_only_warm_start(
        model,
        context,
        fresh_start=fresh,
    )

    assert receipt["schema"] == warm_start.RECEIPT_SCHEMA
    assert receipt["mode"] == warm_start.WARM_START_MODE
    assert receipt["target_fresh_start"]["rng_state"] == "fresh_from_v10_seed"
    assert receipt["receipt_sha256"] == warm_start.v9.canonical_sha256(
        {key: value for key, value in receipt.items() if key != "receipt_sha256"}
    )
    for name, tensor in source_state.items():
        assert torch.equal(model.adapter_state[name], tensor)

    with pytest.raises(ValueError, match="global step 0"):
        warm_start.validate_v10_fresh_start_contract(
            replace(fresh, initial_global_step=56)
        )


def test_v10_launcher_locks_runtime_and_ssd_policy() -> None:
    launcher = Path(
        "experiments/rethinking_rwkv_ms_gemma/train_scene_memory_v10.sh"
    ).read_text(encoding="utf-8")

    for fragment in (
        "--gradient-accumulation-steps 7",
        "--max-steps \"${TARGET_STEP}\"",
        "--save-steps \"${SAVE_STEPS}\"",
        "--lr-scheduler-type constant",
        "--warmup-steps 0",
        "--no-ignore-data-skip",
        "scene_state_generation_ce_symmetric_cycle_retention_v4",
        "scene_memory_v10_v8_checkpoint56_adapter_only",
        '[[ "${RESUME_SCHEDULE_CURSOR}" -eq $((SOURCE_STEP * 7)) ]]',
            "*hard32*|*eval*|*validation*|*holdout*",
            "PINNED_HISTORICAL_TRAIN32",
            "PINNED_WARM_START_CHECKPOINT",
            'require_under_root "${GATE_RECEIPT}" "${GATES_ROOT}"',
            'export PYTHONPYCACHEPREFIX="${PYTHONPYCACHEPREFIX_LOCKED}"',
            'export TORCHINDUCTOR_CACHE_DIR="${TORCHINDUCTOR_CACHE_DIR_LOCKED}"',
            'export CUDA_CACHE_PATH="${CUDA_CACHE_PATH_LOCKED}"',
        ):
        assert fragment in launcher
