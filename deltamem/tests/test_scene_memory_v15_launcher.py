from __future__ import annotations

import os
import json
from pathlib import Path
import shutil
import shlex
import subprocess
import sys

import pytest

from experiments.rethinking_rwkv_ms_gemma import (
    prepare_scene_memory_v15_data as data_prep,
)
from experiments.rethinking_rwkv_ms_gemma import (
    scene_memory_v15_launch_contract as launch,
)
from deltamem.train import delta_sft


REPO_ROOT = Path(__file__).resolve().parents[2]
LAUNCHER = (
    REPO_ROOT
    / "experiments/rethinking_rwkv_ms_gemma/train_scene_memory_v15.sh"
)


def _output(run_name: str, *, smoke: bool = False) -> Path:
    kind = "smoke" if smoke else "production"
    step = 1 if smoke else 4
    return launch.RUN_ROOT / f"scene_memory_v15_{kind}_{run_name}_step{step}"


def test_v15_locked_schedule_objective_and_architecture() -> None:
    assert launch.OBJECTIVE_VERSION == (
        "scene_state_generation_ce_symmetric_cached_prefix_identity_v15"
    )
    assert launch.OBJECTIVE_SCHEMA_VERSION == 18
    assert launch.FIXED_SAMPLER_MODE == (
        "explicit_ordered_v15_full_pair_cycle_v1"
    )
    assert launch.PRODUCTION_RUN_MODE == "production_four_all32_pair_cycles_v15"
    assert launch.PAIR_PRESENTATIONS_PER_OPTIMIZER_STEP == 16
    assert launch.GRADIENT_ACCUMULATION_STEPS == 16
    assert launch.TOTAL_PAIR_PRESENTATIONS == 64
    assert launch.TOTAL_OPTIMIZER_STEPS == 4
    assert launch.CHECKPOINT_STEPS == (1, 2, 3, 4)
    assert launch.PRESENTATION_CHECKPOINTS == (16, 32, 48, 64)
    assert launch.TARGET_LAYERS == tuple(range(42))
    assert launch.DELTA_HEADS == ("q", "o")
    assert launch.ONE_PAIR_SMOKE_PAIR == (5, 9)
    assert launch.FULL_PAIR_CYCLES == data_prep.FULL_PAIR_CYCLES
    assert launch.FOUR_CYCLE_PAIRS_SHA256 == launch.canonical_sha256(
        [list(pair) for pair in launch.FOUR_CYCLE_PAIRS]
    )
    assert launch.PINNED_PAIR_SCHEDULE_SHA256 == (
        "d59e239e4f29f7981175783dafb8a8e4f34c9c95399000e73f2c3dbd26347421"
    )


def test_v15_each_cycle_covers_all_32_rows_once() -> None:
    expected_pairs = {tuple(pair) for pair in data_prep.CANONICAL_ALL32_PAIRS}
    for cycle in launch.FULL_PAIR_CYCLES:
        assert len(cycle) == 16
        assert set(cycle) == expected_pairs
        assert sorted(ordinal for pair in cycle for ordinal in pair) == list(range(32))
    assert [launch.presentation_cursor(step) for step in range(5)] == [
        0,
        16,
        32,
        48,
        64,
    ]
    with pytest.raises(launch.LaunchContractError, match="outside_v15_schedule"):
        launch.presentation_cursor(5)


def test_v15_live_data_and_v13_checkpoint4_warm_start_contracts() -> None:
    data = launch.validate_data_contract()
    warm = launch.validate_warm_start_contract()

    assert data["train_file"] == str(launch.PINNED_TRAIN_FILE)
    assert data["train_file_sha256"] == launch.PINNED_TRAIN_FILE_SHA256
    assert data["scheduled_train_rows"] == 32
    assert data["hard32_rows_in_schedule"] == 0
    assert data["pair_presentations"] == 64
    assert warm["warm_start_checkpoint"] == str(
        launch.PINNED_WARM_START_CHECKPOINT
    )
    assert warm["source_global_step"] == 4
    assert warm["warm_start_adapter_sha256"] == (
        launch.PINNED_WARM_START_ADAPTER_SHA256
    )
    assert warm["warm_start_mode"] == (
        "scene_memory_v14_v13_checkpoint4_adapter_only"
    )


def test_v15_live_production_and_smoke_preflight() -> None:
    production = launch.validate_launch_contract(
        target_step=4,
        run_name="pytest_contract_production",
        output_dir=_output("pytest_contract_production"),
    )
    smoke = launch.validate_launch_contract(
        target_step=1,
        run_name="pytest_contract_smoke",
        output_dir=_output("pytest_contract_smoke", smoke=True),
        smoke=True,
    )

    assert production["gradient_accumulation_steps"] == 16
    assert production["max_steps"] == 4
    assert production["save_total_limit"] == 4
    assert production["total_pair_presentations"] == 64
    assert len(production["scheduled_pairs"]) == 64
    assert production["train_sampler_mode"] == launch.FIXED_SAMPLER_MODE
    assert production["production_eligible"] is True

    assert smoke["gradient_accumulation_steps"] == 1
    assert smoke["max_steps"] == 1
    assert smoke["save_total_limit"] == 1
    assert smoke["total_pair_presentations"] == 1
    assert smoke["scheduled_pairs"] == [[5, 9]]
    assert smoke["train_sampler_mode"] == launch.ONE_PAIR_SMOKE_SAMPLER_MODE
    assert smoke["production_eligible"] is False


def test_v15_preflight_rejects_resume_benchmark_and_wrong_storage() -> None:
    kwargs = {
        "target_step": 4,
        "run_name": "pytest_rejection",
        "output_dir": _output("pytest_rejection"),
    }
    with pytest.raises(launch.LaunchContractError, match="resume_is_forbidden"):
        launch.validate_launch_contract(
            **kwargs,
            resume_checkpoint=launch.PINNED_WARM_START_CHECKPOINT,
        )
    with pytest.raises(launch.LaunchContractError, match="benchmark_or_validation"):
        launch.validate_launch_contract(
            **kwargs,
            benchmark_paths=[Path("/tmp/hard32.jsonl")],
        )
    with pytest.raises(launch.LaunchContractError, match="outside_locked_root"):
        launch.validate_storage_contract(
            output_dir=Path("/tmp/scene_memory_v15_production_pytest_step4"),
            cache_root=launch.CACHE_ROOT,
            run_name="pytest",
            target_step=4,
            smoke=False,
        )


def _dry_run(run_name: str, *, smoke: bool = False) -> subprocess.CompletedProcess[str]:
    environment = os.environ.copy()
    for name in (
        "WORLD_SIZE",
        "LOCAL_RANK",
        "RANK",
        "MASTER_ADDR",
        "MASTER_PORT",
        "HARD32",
        "FULL170",
        "BENCHMARK_PATH",
        "EVAL_FILE",
        "VALIDATION_FILE",
        "TEST_FILE",
    ):
        environment.pop(name, None)
    environment.update(
        {
            "RUN_NAME": run_name,
            "DRY_RUN": "1",
            "SMOKE_RUN": "1" if smoke else "0",
        }
    )
    return subprocess.run(
        [str(LAUNCHER)],
        cwd=REPO_ROOT,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
    )


@pytest.mark.parametrize("smoke", (False, True))
def test_v15_launcher_dry_run_emits_locked_command_without_starting(
    smoke: bool,
) -> None:
    run_name = f"pytest_dry_run_{'smoke' if smoke else 'production'}"
    result = _dry_run(run_name, smoke=smoke)
    assert result.returncode == 0, result.stderr
    assert "Validated V15 training command (not started):" in result.stdout
    assert not _output(run_name, smoke=smoke).exists()
    assert "--scene-state-generation-objective-version" in result.stdout
    assert launch.OBJECTIVE_VERSION in result.stdout
    assert "--warm-start-mode" in result.stdout
    assert "scene_memory_v14_v13_checkpoint4_adapter_only" in result.stdout
    assert "--target-layers" in result.stdout
    rendered = result.stdout.replace("\\,", ",")
    assert ",".join(str(layer) for layer in range(42)) in rendered
    assert "--delta-heads" in result.stdout
    assert "q,o" in rendered
    assert f"--gradient-accumulation-steps {1 if smoke else 16}" in result.stdout
    assert f"--max-steps {1 if smoke else 4}" in result.stdout
    assert (launch.ONE_PAIR_SMOKE_FLAG in result.stdout) is smoke
    assert "--resume-from-checkpoint" not in result.stdout
    assert "--validation-split-ratio 0" in result.stdout
    assert "--per-device-eval-batch-size 1" in result.stdout
    assert "--eval-steps 1000" in result.stdout


@pytest.mark.parametrize("smoke", (False, True))
def test_v15_launcher_command_passes_real_training_argument_validation(
    smoke: bool,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run_name = f"pytest_parse_{'smoke' if smoke else 'production'}"
    result = _dry_run(run_name, smoke=smoke)
    assert result.returncode == 0, result.stderr
    marker = "Validated V15 training command (not started):\n"
    command = shlex.split(result.stdout.split(marker, 1)[1].splitlines()[0])
    assert command[1:3] == ["-m", "deltamem.train.delta_sft"]
    monkeypatch.setattr(sys, "argv", [command[2], *command[3:]])

    parsed = delta_sft.parse_args()

    assert parsed.scene_state_generation_objective_version == launch.OBJECTIVE_VERSION
    assert parsed.per_device_eval_batch_size == 1
    assert parsed.eval_steps == 1000


def test_v15_launcher_locks_every_cache_and_temp_to_ssd() -> None:
    source = LAUNCHER.read_text(encoding="utf-8")
    assert 'RUN_ROOT="${SSD_ROOT}/delta_mem_outputs/' in source
    assert 'CACHE_ROOT="${SSD_ROOT}/delta_mem_cache/scene_memory_v15"' in source
    for variable in (
        "HF_HOME",
        "HF_HUB_CACHE",
        "HF_DATASETS_CACHE",
        "HF_ASSETS_CACHE",
        "TRANSFORMERS_CACHE",
        "XDG_CACHE_HOME",
        "TMPDIR",
        "TMP",
        "TEMP",
        "TORCH_HOME",
        "TORCH_EXTENSIONS_DIR",
        "TRITON_CACHE_DIR",
        "PYTHONPYCACHEPREFIX",
        "TORCHINDUCTOR_CACHE_DIR",
        "CUDA_CACHE_PATH",
        "NUMBA_CACHE_DIR",
        "MPLCONFIGDIR",
        "WANDB_DIR",
    ):
        assert f"export {variable}=" in source
    assert "tracked_worktree_must_be_clean_before_v15_training" in source
    assert "critical_v15_source_must_be_tracked" in source
    assert source.index("launch.write_launch_receipt(") < source.index(
        'mkdir "${OUTPUT_DIR}"'
    )


def test_v15_launcher_forbids_every_evaluation_surface() -> None:
    source = LAUNCHER.read_text(encoding="utf-8")
    for token in (
        "HARD32_FILE",
        "FULL170_FILE",
        "BENCHMARK_PATH",
        "EVAL_FILE",
        "VALIDATION_FILE",
        "TEST_FILE",
    ):
        assert token in source
    assert "--benchmark-path" not in source
    assert "--resume-from-checkpoint" not in source
    assert "HF_HUB_OFFLINE=1" in source


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")


def _locked_schedule_entries() -> list[dict[str, object]]:
    return [
        json.loads(line)
        for line in launch.PAIR_SCHEDULE.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _row_objective_audit(*, step: int, smoke: bool) -> dict[str, object]:
    count = 1 if smoke else launch.presentation_cursor(step)
    entries = _locked_schedule_entries()[:count]
    phases = (
        ["smoke_input"]
        if smoke
        else [f"cycle{cycle}_input" for cycle in range(1, step + 1)]
    )
    pair_presentations: list[dict[str, object]] = []
    observations: dict[tuple[str, int], dict[str, object]] = {}
    for index, entry in enumerate(entries):
        pair = entry["canonical_pair_ordinals"]
        members = entry["members"]
        assert isinstance(pair, list)
        assert isinstance(members, list)
        cycle = int(entry["cycle_index"])
        phase = "smoke_input" if smoke else f"cycle{cycle}_input"
        pair_presentations.append(
            {
                "phase": phase,
                "cycle": cycle,
                "adapter_optimizer_step_before_update": cycle - 1,
                "presentation": index + 1,
                "source_row_ordinal": pair[0],
                "donor_row_ordinal": pair[1],
                "source_row_sha256": members[0]["row_sha256"],
                "donor_row_sha256": members[1]["row_sha256"],
                "pair_mean_pair_identity_hinge": 0.75,
                "pair_mean_pair_identity_logit_margin": 0.25,
                "pair_mean_pair_identity_own_beats_paired_fraction": 1.0,
                "pair_mean_pair_identity_margin_satisfied_fraction": 0.0,
            }
        )
        for role, row_position, paired_position in (
            ("source", 0, 1),
            ("donor", 1, 0),
        ):
            row_ordinal = int(pair[row_position])
            observations[(phase, row_ordinal)] = {
                "phase": phase,
                "cycle": cycle,
                "adapter_optimizer_step_before_update": cycle - 1,
                "presentation": index + 1,
                "pair_role": role,
                "row_ordinal": row_ordinal,
                "paired_row_ordinal": int(pair[paired_position]),
                "row_sha256": members[row_position]["row_sha256"],
                "paired_row_sha256": members[paired_position]["row_sha256"],
                "pair_identity_hinge": 0.75,
                "pair_identity_logit_margin": 0.25,
                "pair_identity_own_beats_paired_fraction": 1.0,
                "pair_identity_margin_satisfied_fraction": 0.0,
            }
    cycle_size = 1 if smoke else launch.PAIR_PRESENTATIONS_PER_OPTIMIZER_STEP
    row_order = [
        int(ordinal)
        for entry in entries[:cycle_size]
        for ordinal in entry["canonical_pair_ordinals"]
    ]
    return {
        "schema": launch.ROW_OBJECTIVE_AUDIT_SCHEMA,
        "memory_objective_version": launch.OBJECTIVE_VERSION,
        "run_mode": (
            launch.ONE_PAIR_SMOKE_RUN_MODE
            if smoke
            else launch.PRODUCTION_RUN_MODE
        ),
        "production_eligible": not smoke,
        "checkpoint_optimizer_step": step,
        "completed_pair_presentations": count,
        "phases": phases,
        "pair_schedule": [
            {
                "source_row_ordinal": entry["canonical_pair_ordinals"][0],
                "donor_row_ordinal": entry["canonical_pair_ordinals"][1],
            }
            for entry in entries
        ],
        "pair_presentations": pair_presentations,
        "rows": [
            {
                "row_ordinal": row_ordinal,
                **{
                    phase: observations[(phase, row_ordinal)]
                    for phase in phases
                },
            }
            for row_ordinal in row_order
        ],
    }


def _source_manifest_identity() -> dict[str, object]:
    data = launch.validate_data_contract()
    return {
        "path": data["source_manifest"],
        "file_sha256": data["source_manifest_sha256"],
        "schema": data_prep.SOURCE_SCHEMA,
        "train_file": data["train_file"],
        "train_file_sha256": data["train_file_sha256"],
        "train_rows": 32,
        "train_source_split": "train",
        "episode_contract": {
            "episode_recent_messages": 0,
            "write_phase": "system + user",
            "read_supervision": "system + assistant",
        },
    }


def _pairing_manifest() -> dict[str, object]:
    pairing: dict[str, object] = {
        "objective_version": launch.PAIRING_OBJECTIVE_VERSION,
    }
    pairing["manifest_sha256"] = launch.canonical_sha256(pairing)
    return pairing


def _warm_start_lineage(
    *,
    config: dict[str, object],
    protocol: dict[str, object],
    pairing: dict[str, object],
) -> dict[str, object]:
    warm_contract = launch.validate_warm_start_contract()
    source = warm_contract["lineage_source"]
    assert isinstance(source, dict)
    topology = source["source_adapter_topology"]
    assert isinstance(topology, dict)
    fresh = launch.warm.validate_v14_fresh_start_contract(
        launch.warm.V14FreshStartContract(
            resume_from_checkpoint=None,
            initial_global_step=0,
            optimizer_created=False,
            scheduler_created=False,
            trainer_state_imported=False,
            rng_state_imported=False,
            optim=launch.OPTIMIZER_IMPLEMENTATION,
        )
    )
    lineage: dict[str, object] = {
        "schema": launch.warm.RECEIPT_SCHEMA,
        "schema_version": 1,
        "mode": launch.warm.WARM_START_MODE,
        "source_checkpoint": warm_contract["warm_start_checkpoint"],
        "source_lock": {
            "path": warm_contract["warm_start_lock"],
            "lock_sha256": warm_contract["warm_start_lock_sha256"],
        },
        "source_artifacts": source["source_artifacts"],
        "source_global_step": warm_contract["source_global_step"],
        "source_epoch": source["source_epoch"],
        "source_protocol_objective_version": source[
            "source_protocol_objective_version"
        ],
        "source_pairing_objective_version": source[
            "source_pairing_objective_version"
        ],
        "source_row_objective_audit_schema": source[
            "source_row_objective_audit_schema"
        ],
        "source_v13_warm_start_receipt_sha256": source[
            "source_v13_warm_start_receipt_sha256"
        ],
        "source_state_imports": launch.warm.SOURCE_IMPORT_POLICY,
        "loaded_source_artifacts": ["delta_mem_adapter.pt"],
        "validated_not_imported_source_artifacts": [
            "optimizer.pt",
            "scheduler.pt",
            "rng_state.pth",
            "trainer_state.json",
        ],
        "topology": {
            "adapter_tensor_count": topology["tensor_count"],
            "adapter_tensor_elements": topology["tensor_elements"],
            "adapter_topology_sha256": topology["sha256"],
            "ordered_dtypes_equal": True,
            "ordered_parameter_names_equal": True,
            "ordered_shapes_equal": True,
        },
        "post_load_topology_sha256": topology["sha256"],
        "post_load_bit_equal": True,
        "target_fresh_start": fresh,
        "trainer_resume_from_checkpoint": None,
        "target_initial_global_step": 0,
        "pre_train_global_step": 0,
        "fresh_optimizer_created": True,
        "fresh_optimizer_class": "torch.optim.adamw.AdamW",
        "fresh_optimizer_state_entries_before_train": 0,
        "fresh_scheduler_created_before_train": False,
        "fresh_adamw_creation_required_after_adapter_load": True,
        "target_delta_config_sha256": launch.canonical_sha256(config),
        "target_training_protocol_sha256": launch.canonical_sha256(protocol),
        "target_scene_state_pairing_manifest_sha256": pairing["manifest_sha256"],
    }
    lineage["receipt_sha256"] = launch.canonical_sha256(lineage)
    return lineage


def _write_bound_checkpoint_manifests(
    checkpoint: Path,
    *,
    smoke: bool,
) -> None:
    config: dict[str, object] = {
        "target_layers": list(range(42)),
        "delta_heads": ["q", "o"],
        "rank": 4,
    }
    pairing = _pairing_manifest()
    protocol: dict[str, object] = {
        "schema_version": 18,
        "memory_objective_version": launch.OBJECTIVE_VERSION,
        "max_steps": 1 if smoke else 4,
        "gradient_accumulation_steps": 1 if smoke else 16,
        "save_steps": 1,
        "scene_state_source_manifest": _source_manifest_identity(),
        "scene_state_identity_pairing": {
            "manifest_sha256": pairing["manifest_sha256"],
        },
    }
    _write_json(checkpoint / "delta_mem_config.json", config)
    _write_json(checkpoint / "training_protocol.json", protocol)
    _write_json(checkpoint / "scene_state_identity_pairing_manifest.json", pairing)
    _write_json(
        checkpoint / launch.WARM_START_LINEAGE_FILENAME,
        _warm_start_lineage(config=config, protocol=protocol, pairing=pairing),
    )


def _smoke_checkpoint(
    tmp_path: Path,
    *,
    trainer_output: Path | None = None,
) -> Path:
    output = trainer_output or tmp_path / "scene_memory_v15_smoke_unit_step1"
    checkpoint = output / "trainer/checkpoint-1"
    checkpoint.mkdir(parents=True)
    (checkpoint / "delta_mem_adapter.pt").write_bytes(b"changed-adapter\n")
    for name in ("optimizer.pt", "scheduler.pt"):
        (checkpoint / name).write_bytes(b"state\n")
    _write_json(
        checkpoint / "trainer_state.json",
        {
            "global_step": 1,
            "log_history": [
                {
                    "step": 1,
                    "loss": 1.25,
                    "grad_norm": 0.5,
                    "delta/scene_generation_v15_objective_total_loss": 1.25,
                    "delta/scene_generation_v15_pair_mean_pair_identity_hinge": 0.75,
                    "delta/scene_generation_v15_pair_mean_pair_identity_logit_margin": 0.25,
                    "delta/scene_generation_v15_pair_mean_pair_identity_own_beats_paired_fraction": 1.0,
                    "delta/scene_generation_v15_pair_mean_pair_identity_margin_satisfied_fraction": 0.0,
                    "delta/scene_generation_v15_cycle_pair_presentations": 1.0,
                    "delta/scene_generation_v15_cycle_index": 1.0,
                    "delta/scene_generation_v15_cycle_pair_0_low_ordinal": 5.0,
                    "delta/scene_generation_v15_cycle_pair_0_high_ordinal": 9.0,
                }
            ],
        },
    )
    _write_bound_checkpoint_manifests(checkpoint, smoke=True)
    _write_json(
        checkpoint / launch.ROW_OBJECTIVE_AUDIT_FILENAME,
        _row_objective_audit(step=1, smoke=True),
    )
    return checkpoint


def _production_checkpoint(
    tmp_path: Path,
    *,
    step: int,
    trainer_output: Path | None = None,
) -> Path:
    output = trainer_output or (
        tmp_path / f"scene_memory_v15_production_unit_step{step}"
    )
    checkpoint = output / f"trainer/checkpoint-{step}"
    checkpoint.mkdir(parents=True)
    (checkpoint / "delta_mem_adapter.pt").write_bytes(b"production-adapter\n")
    for name in ("optimizer.pt", "scheduler.pt"):
        (checkpoint / name).write_bytes(b"state\n")
    _write_json(
        checkpoint / "trainer_state.json",
        {"global_step": step, "log_history": []},
    )
    _write_bound_checkpoint_manifests(checkpoint, smoke=False)
    _write_json(
        checkpoint / launch.ROW_OBJECTIVE_AUDIT_FILENAME,
        _row_objective_audit(step=step, smoke=False),
    )
    return checkpoint


def _launch_receipt_fixture(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    smoke: bool,
) -> dict[str, Path]:
    monkeypatch.setattr(launch, "RUN_ROOT", tmp_path)
    monkeypatch.setattr(
        launch,
        "critical_training_code_bindings_at_commit",
        lambda *_args, **_kwargs: {},
    )
    data = launch.validate_data_contract()
    warm_contract = launch.validate_warm_start_contract()
    kind = "smoke" if smoke else "production"
    step = 1 if smoke else 4
    run_id = f"scene_memory_v15_{kind}_unit_step{step}"
    output = tmp_path / run_id
    log = tmp_path / "logs" / f"{run_id}.log"
    launch_receipt = tmp_path / "logs" / f"{run_id}.launch.json"
    completion_receipt = tmp_path / "logs" / f"{run_id}.completion.json"
    output.mkdir(parents=True)
    log.parent.mkdir(parents=True)
    log.write_text("training completed\n", encoding="utf-8")
    summary = output / "training_summary.json"
    _write_json(summary, {"global_step": step})
    scheduled_pairs = (
        [list(launch.ONE_PAIR_SMOKE_PAIR)]
        if smoke
        else [list(pair) for pair in launch.FOUR_CYCLE_PAIRS]
    )
    launch_contract: dict[str, object] = {
        **data,
        **warm_contract,
        "production_eligible": not smoke,
        "output_dir": str(output),
        "log_file": str(log),
        "launch_mode": "warm_start_smoke" if smoke else "warm_start",
        "run_mode": (
            launch.ONE_PAIR_SMOKE_RUN_MODE if smoke else launch.PRODUCTION_RUN_MODE
        ),
        "target_step": step,
        "scheduled_pairs": scheduled_pairs,
        "scheduled_pairs_sha256": launch.canonical_sha256(scheduled_pairs),
        "total_pair_presentations": 1 if smoke else launch.TOTAL_PAIR_PRESENTATIONS,
        "gradient_accumulation_steps": (
            1 if smoke else launch.GRADIENT_ACCUMULATION_STEPS
        ),
        "max_steps": step,
        "save_total_limit": 1 if smoke else len(launch.CHECKPOINT_STEPS),
        "base_model_identity": {},
    }
    receipt = launch.build_launch_receipt(
        launch_contract,
        git_commit="0" * 40,
        critical_files={},
    )
    _write_json(launch_receipt, receipt)
    return {
        "output": output,
        "log": log,
        "summary": summary,
        "launch": launch_receipt,
        "completion": completion_receipt,
    }


def test_v15_smoke_checkpoint_proves_real_optimizer_update(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    checkpoint = _smoke_checkpoint(tmp_path)
    monkeypatch.setattr(launch, "RUN_ROOT", tmp_path)

    result = launch.validate_checkpoint_contract(checkpoint, smoke=True)
    update = result["smoke_real_optimizer_update"]
    assert update["adapter_changed"] is True
    assert update["grad_norm"] == 0.5
    assert update["pair_identity_hinge"] == 0.75

    shutil.copyfile(
        launch.PINNED_WARM_START_CHECKPOINT / "delta_mem_adapter.pt",
        checkpoint / "delta_mem_adapter.pt",
    )
    with pytest.raises(
        launch.LaunchContractError,
        match="adapter_did_not_change_from_warm_start",
    ):
        launch.validate_checkpoint_contract(checkpoint, smoke=True)


def test_v15_checkpoint_rejects_non_rank4_adapter_topology(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    checkpoint = _smoke_checkpoint(tmp_path)
    monkeypatch.setattr(launch, "RUN_ROOT", tmp_path)
    config_path = checkpoint / "delta_mem_config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    config["rank"] = 8
    _write_json(config_path, config)

    with pytest.raises(
        launch.LaunchContractError,
        match="v15_checkpoint_adapter_topology_differs",
    ):
        launch.validate_checkpoint_contract(checkpoint, smoke=True)


def test_v15_production_checkpoint_audit_binds_exact_pair_prefix_and_hashes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    checkpoint = _production_checkpoint(tmp_path, step=2)
    monkeypatch.setattr(launch, "RUN_ROOT", tmp_path)
    audit_path = checkpoint / launch.ROW_OBJECTIVE_AUDIT_FILENAME
    original = json.loads(audit_path.read_text(encoding="utf-8"))

    result = launch.validate_checkpoint_contract(checkpoint, smoke=False)
    binding = result["row_objective_audit_binding"]
    assert binding["pair_prefix_sha256"] == (
        launch.PAIR_PREFIX_SHA256_BY_CHECKPOINT[2]
    )
    assert binding["pair_presentations"] == 32
    assert binding["row_phase_observations"] == 64

    wrong_order = json.loads(json.dumps(original))
    wrong_order["pair_schedule"][0], wrong_order["pair_schedule"][1] = (
        wrong_order["pair_schedule"][1],
        wrong_order["pair_schedule"][0],
    )
    _write_json(audit_path, wrong_order)
    with pytest.raises(
        launch.LaunchContractError,
        match="pair_schedule_prefix_or_order_differs",
    ):
        launch.validate_checkpoint_contract(checkpoint, smoke=False)

    wrong_pair_hash = json.loads(json.dumps(original))
    wrong_pair_hash["pair_presentations"][0]["source_row_sha256"] = "0" * 64
    _write_json(audit_path, wrong_pair_hash)
    with pytest.raises(
        launch.LaunchContractError,
        match="pair_presentation_order_or_hash_differs",
    ):
        launch.validate_checkpoint_contract(checkpoint, smoke=False)

    wrong_row_hash = json.loads(json.dumps(original))
    wrong_row_hash["rows"][0]["cycle1_input"]["row_sha256"] = "0" * 64
    _write_json(audit_path, wrong_row_hash)
    with pytest.raises(
        launch.LaunchContractError,
        match="row_phase_order_or_hash_differs",
    ):
        launch.validate_checkpoint_contract(checkpoint, smoke=False)


def test_v15_smoke_completion_revalidates_bound_checkpoint_and_launch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _launch_receipt_fixture(tmp_path, monkeypatch, smoke=True)
    checkpoint = _smoke_checkpoint(
        tmp_path,
        trainer_output=fixture["output"],
    )
    completion = launch.build_completion_receipt(
        launch_receipt=fixture["launch"],
        training_summary=fixture["summary"],
        log_file=fixture["log"],
        checkpoints=[checkpoint],
        smoke=True,
    )
    _write_json(fixture["completion"], completion)

    validated = launch.validate_completion_receipt(fixture["completion"])

    assert validated["payload"]["production_eligible"] is False
    assert list(validated["checkpoint_contracts"]) == ["checkpoint-1"]
    assert validated["checkpoint_contracts"]["checkpoint-1"][
        "warm_start_lineage_receipt_sha256"
    ]


def test_v15_completion_rejects_cross_run_checkpoint_paths(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _launch_receipt_fixture(tmp_path, monkeypatch, smoke=False)
    wrong_output = tmp_path / "scene_memory_v15_production_other_step4"
    wrong_checkpoints = [
        wrong_output / f"trainer/checkpoint-{step}"
        for step in launch.CHECKPOINT_STEPS
    ]

    with pytest.raises(
        launch.LaunchContractError,
        match="checkpoint_paths_or_order_differ_from_launch",
    ):
        launch.build_completion_receipt(
            launch_receipt=fixture["launch"],
            training_summary=fixture["summary"],
            log_file=fixture["log"],
            checkpoints=wrong_checkpoints,
            smoke=False,
        )


def test_v15_completion_rejects_smoke_launch_as_production(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _launch_receipt_fixture(tmp_path, monkeypatch, smoke=True)

    with pytest.raises(
        launch.LaunchContractError,
        match="launch_receipt_mode_differs_from_completion",
    ):
        launch.build_completion_receipt(
            launch_receipt=fixture["launch"],
            training_summary=fixture["summary"],
            log_file=fixture["log"],
            checkpoints=[],
            smoke=False,
        )


def test_v15_checkpoint_rejects_missing_nested_source_manifest_binding(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    checkpoint = _smoke_checkpoint(tmp_path)
    monkeypatch.setattr(launch, "RUN_ROOT", tmp_path)
    protocol_path = checkpoint / "training_protocol.json"
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    protocol.pop("scene_state_source_manifest")
    _write_json(protocol_path, protocol)

    with pytest.raises(
        launch.LaunchContractError,
        match="checkpoint_data_binding_differs",
    ):
        launch.validate_checkpoint_contract(checkpoint, smoke=True)


@pytest.mark.parametrize(
    ("mutation", "message"),
    (
        ("empty", "warm_lineage_self_hash_differs"),
        ("source", "warm_lineage_identity_differs"),
        ("target", "warm_lineage_target_binding_differs"),
    ),
)
def test_v15_checkpoint_rejects_empty_or_forged_warm_lineage(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mutation: str,
    message: str,
) -> None:
    checkpoint = _smoke_checkpoint(tmp_path)
    monkeypatch.setattr(launch, "RUN_ROOT", tmp_path)
    lineage_path = checkpoint / launch.WARM_START_LINEAGE_FILENAME
    lineage = json.loads(lineage_path.read_text(encoding="utf-8"))
    if mutation == "empty":
        lineage = {}
    elif mutation == "source":
        lineage["source_checkpoint"] = str(checkpoint)
        lineage.pop("receipt_sha256")
        lineage["receipt_sha256"] = launch.canonical_sha256(lineage)
    else:
        lineage["target_training_protocol_sha256"] = "0" * 64
        lineage.pop("receipt_sha256")
        lineage["receipt_sha256"] = launch.canonical_sha256(lineage)
    _write_json(lineage_path, lineage)

    with pytest.raises(launch.LaunchContractError, match=message):
        launch.validate_checkpoint_contract(checkpoint, smoke=True)
