from __future__ import annotations

import json
import os
from pathlib import Path
import shlex
import shutil
from types import SimpleNamespace
import subprocess
import sys
import uuid

import pytest
from transformers.trainer_callback import (
    DefaultFlowCallback,
    TrainerControl,
    TrainerState,
)
from transformers.training_args import IntervalStrategy, SaveStrategy

from experiments.rethinking_rwkv_ms_gemma import scene_memory_v8_launch_contract as contract


REPO_ROOT = Path(__file__).resolve().parents[2]
LAUNCHER = REPO_ROOT / "experiments/rethinking_rwkv_ms_gemma/train_scene_memory_v8.sh"
SOURCE_LOCK = REPO_ROOT / "experiments/rethinking_rwkv_ms_gemma/scene_memory_v8_source_lock.json"
SSD_ROOT = Path("/run/media/xiaol/B214449214445C0B")
RUN_ROOT = SSD_ROOT / "delta_mem_outputs/novel_rwkv_ms_memory/scene_memory_v8"
TARGET_LAYERS = ",".join(str(index) for index in range(42))
EMPTY_ENV = (
    "WORLD_SIZE",
    "LOCAL_RANK",
    "RANK",
    "MASTER_ADDR",
    "MASTER_PORT",
    "SLURM_PROCID",
    "PMI_RANK",
    "OMPI_COMM_WORLD_RANK",
    "RESUME_FROM_CHECKPOINT",
    "RESUME_MODE",
    "TARGET_STEP",
    "SMOKE_RUN",
    "HARD32",
    "HARD32_FILE",
    "HARD32_PATH",
    "HARD32_DIR",
    "EVAL_DATASET",
    "EVAL_FILE",
    "VALIDATION_DATASET",
    "VALIDATION_FILE",
    "TEST_DATASET",
    "TEST_FILE",
)


@pytest.fixture
def fake_git_bin(tmp_path: Path) -> Path:
    binary_dir = tmp_path / "bin"
    binary_dir.mkdir()
    git = binary_dir / "git"
    git.write_text(
        """#!/usr/bin/env bash
set -euo pipefail
args="$*"
if [[ "$args" == *"status --porcelain --untracked-files=no"* ]]; then
  if [[ "${FAKE_GIT_DIRTY:-0}" == "1" ]]; then
    printf ' M tracked_file.py\\n'
  fi
  exit 0
fi
if [[ "$args" == *"rev-parse HEAD"* ]]; then
  printf '2c3f00a000000000000000000000000000000000\\n'
  exit 0
fi
if [[ "$args" == *"ls-files --error-unmatch"* ]]; then
  if [[ "${FAKE_GIT_UNTRACKED_CRITICAL:-0}" == "1" ]]; then
    exit 1
  fi
  exit 0
fi
printf 'unexpected fake git invocation: %s\\n' "$args" >&2
exit 3
""",
        encoding="utf-8",
    )
    git.chmod(0o755)
    return binary_dir


def launcher_environment(
    fake_git_bin: Path,
    run_name: str,
    **updates: str,
) -> dict[str, str]:
    environment = os.environ.copy()
    for variable in EMPTY_ENV:
        environment.pop(variable, None)
    environment.update(
        {
            "RUN_NAME": run_name,
            "DRY_RUN": "1",
            "PYTHON_BIN": "/bin/true",
            "VALIDATION_PYTHON_BIN": sys.executable,
            "PATH": f"{fake_git_bin}:{environment['PATH']}",
        }
    )
    environment.update(updates)
    return environment


def run_launcher(environment: dict[str, str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [str(LAUNCHER)],
        cwd=REPO_ROOT,
        env=environment,
        capture_output=True,
        text=True,
    )


def dry_run_arguments(result: subprocess.CompletedProcess[str]) -> list[str]:
    assert result.returncode == 0, result.stderr
    return shlex.split(result.stdout.strip().splitlines()[-1])


def argument_value(arguments: list[str], flag: str) -> str:
    return arguments[arguments.index(flag) + 1]


def source_lock() -> dict:
    return json.loads(SOURCE_LOCK.read_text(encoding="utf-8"))


def test_fresh_v8_dry_run_binds_exact_schedule_warm_start_and_objective(
    fake_git_bin: Path,
) -> None:
    run_name = f"pytest_fresh_{uuid.uuid4().hex}"
    arguments = dry_run_arguments(
        run_launcher(launcher_environment(fake_git_bin, run_name))
    )
    lock = source_lock()

    assert arguments[:3] == ["/bin/true", "-m", "deltamem.train.delta_sft"]
    assert argument_value(arguments, "--train-file") == lock["parent_v7"]["train32"][
        "path"
    ]
    assert argument_value(arguments, "--scene-state-source-manifest") == lock[
        "artifacts"
    ]["source_manifest"]["path"]
    assert argument_value(
        arguments,
        "--expected-scene-state-source-manifest-sha256",
    ) == lock["artifacts"]["source_manifest"]["sha256"]
    assert argument_value(arguments, "--warm-start-from-checkpoint") == json.loads(
        contract.WARM_START_LOCK.read_text(encoding="utf-8")
    )["source_checkpoint"]
    assert argument_value(arguments, "--warm-start-mode") == contract.WARM_START_MODE
    assert argument_value(arguments, "--resume-mode") == "exact"
    assert "--resume-from-checkpoint" not in arguments
    assert "--initial-adapter-output-dir" not in arguments
    assert argument_value(arguments, "--target-layers") == TARGET_LAYERS
    assert argument_value(arguments, "--delta-heads") == "q,o"
    assert argument_value(arguments, "--rank") == "4"
    assert argument_value(arguments, "--learning-rate") == "2e-4"
    assert argument_value(arguments, "--lr-scheduler-type") == "constant_with_warmup"
    assert argument_value(arguments, "--warmup-ratio") == "0"
    assert argument_value(arguments, "--warmup-steps") == "4"
    assert argument_value(arguments, "--per-device-train-batch-size") == "1"
    assert argument_value(arguments, "--gradient-accumulation-steps") == "1"
    assert argument_value(arguments, "--max-steps") == "14"
    assert argument_value(arguments, "--save-steps") == "14"
    assert argument_value(arguments, "--scene-state-generated-unlikelihood-weight") == "0.5"
    assert argument_value(
        arguments,
        "--scene-state-generated-unlikelihood-max-wrong-tokens",
    ) == "4"
    assert argument_value(arguments, "--scene-state-generated-rollout-extra-tokens") == "4"
    assert argument_value(arguments, "--scene-state-generated-rollout-max-tokens") == "24"
    assert argument_value(arguments, "--memory-kl-weight") == "0"
    assert argument_value(arguments, "--memory-base-kl-weight") == "0"
    assert "--train-sampler-seed" not in arguments
    assert "--frozen-mlp-activation-checkpointing" in arguments
    assert not Path(argument_value(arguments, "--output-dir")).exists()


def test_one_step_smoke_keeps_exact_four_step_warmup(
    fake_git_bin: Path,
) -> None:
    run_name = f"pytest_smoke_{uuid.uuid4().hex}"
    arguments = dry_run_arguments(
        run_launcher(
            launcher_environment(fake_git_bin, run_name, SMOKE_RUN="1")
        )
    )

    assert argument_value(arguments, "--max-steps") == "1"
    assert argument_value(arguments, "--save-steps") == "1"
    assert argument_value(arguments, "--warmup-ratio") == "0"
    assert argument_value(arguments, "--warmup-steps") == "4"
    assert "scene_memory_v8_smoke_" in argument_value(arguments, "--output-dir")
    assert "--warm-start-from-checkpoint" in arguments
    assert "--resume-from-checkpoint" not in arguments


def _resume_protocol(data: dict[str, object], source_step: int) -> dict[str, object]:
    return {
        "schema_version": 11,
        "memory_objective_version": (
            "scene_state_generation_ce_generated_prefix_unlikelihood_v2"
        ),
        "memory_loss_mode": "scene_state_generation_ce",
        "train_file": data["train_file"],
        "train_samples": 32,
        "eval_samples": 0,
        "training_mode": "episode",
        "assistant_loss_mode": "final_assistant_only",
        "episode_recent_messages": 0,
        "episode_read_write_enabled": False,
        "per_device_train_batch_size": 1,
        "gradient_accumulation_steps": 1,
        "learning_rate": contract.LEARNING_RATE,
        "lr_scheduler_type": "constant_with_warmup",
        "warmup_ratio": contract.WARMUP_RATIO,
        "warmup_steps": contract.WARMUP_STEPS,
        "save_steps": contract.SAVE_STEPS,
        "num_train_epochs": 1.0,
        "max_steps": source_step,
        "train_sampler_seed": None,
        "train_sampler_mode": contract.FIXED_SAMPLER_MODE,
        "memory_kl_weight": 0.0,
        "memory_base_kl_weight": 0.0,
        "memory_representation_weight": 0.0,
        "write_sparsity_weight": 0.0,
        "scene_generation_generated_unlikelihood_weight": 0.5,
        "scene_generation_generated_unlikelihood_max_wrong_tokens": 4,
        "scene_generation_generated_rollout_extra_tokens": 4,
        "scene_generation_generated_rollout_max_tokens": 24,
        "scene_state_source_manifest": {
            "path": data["source_manifest"],
            "file_sha256": data["source_manifest_file_sha256"],
            "schema": contract.SOURCE_SCHEMA,
            "train_file": data["train_file"],
            "train_file_sha256": contract.TRAIN32_SHA256,
            "train_rows": 32,
            "train_source_split": "train",
        },
        "train_schedule": {
            "schema": contract.CURRICULUM_SCHEMA,
            "source_manifest_path": data["source_manifest"],
            "source_manifest_file_sha256": data["source_manifest_file_sha256"],
            "schedule_path": data["schedule"],
            "schedule_file_sha256": contract.SCHEDULE_FILE_SHA256,
            "schedule_entries_sha256": contract.SCHEDULE_ENTRIES_SHA256,
            "schedule_manifest_path": data["schedule_manifest"],
            "schedule_manifest_file_sha256": contract.SCHEDULE_MANIFEST_FILE_SHA256,
            "schedule_manifest_sha256": contract.SCHEDULE_MANIFEST_CANONICAL_SHA256,
            "ordered_train_row_ordinals_sha256": contract.SCHEDULE_ORDINALS_SHA256,
            "total_steps": contract.TOTAL_STEPS,
            "checkpoint_steps": list(contract.CHECKPOINT_STEPS),
            "value14_ordinals": list(contract.VALUE14_ORDINALS),
        },
    }


def _resume_delta_config() -> dict[str, object]:
    return {
        "rank": 4,
        "alpha": 8.0,
        "memory_backend": "rwkv_ms",
        "target_layers": list(range(42)),
        "delta_heads": ["q", "o"],
        "memory_fusion_mode": "add",
        "memory_fusion_placement": "attention_output",
        "memory_fusion_residual_scale": 1.0,
        "memory_fusion_residual_scale_max": 1.0,
        "trainable_delta_scale": True,
        "delta_scale_init": 0.1,
        "delta_scale_max": 0.5,
        "delta_scale_granularity": "head",
        "delta_scale_parameterization": "alpha_over_rank",
        "output_init": "base_slice_fixed",
        "base_slice_ref_width": 8,
        "online_gain": 0.2,
        "rwkv_ms_num_states": 4,
        "rwkv_ms_chunk_size": 128,
        "rwkv_ms_semantics_version": 2,
    }


def _write_json_object(path: Path, payload: dict[str, object]) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=True, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _rewrite_self_hashed_manifest(
    path: Path,
    *,
    hash_field: str,
    updates: dict[str, object],
) -> dict[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload.update(updates)
    payload.pop(hash_field, None)
    payload[hash_field] = contract.canonical_sha256(payload)
    _write_json_object(path, payload)
    return payload


def _write_completed_v8_checkpoint(
    fixture_root: Path,
    *,
    step: int,
    source_checkpoint: Path | None = None,
) -> Path:
    checkpoint = fixture_root / f"block-{step}/trainer/checkpoint-{step}"
    checkpoint.mkdir(parents=True)
    data = contract.validate_data_contract()
    protocol = _resume_protocol(data, step)
    config = _resume_delta_config()
    pairing_unsigned: dict[str, object] = {
        "schema": "rwkv_ms_scene_state_identity_pairing.v1",
        "rows": 32,
    }
    pairing = {
        **pairing_unsigned,
        "manifest_sha256": contract.canonical_sha256(pairing_unsigned),
    }
    _write_json_object(
        checkpoint / "trainer_state.json",
        {"global_step": step, "max_steps": step, "epoch": step / 152},
    )
    _write_json_object(checkpoint / "training_protocol.json", protocol)
    _write_json_object(checkpoint / "delta_mem_config.json", config)
    _write_json_object(
        checkpoint / "scene_state_identity_pairing_manifest.json",
        pairing,
    )
    for filename in (
        "delta_mem_adapter.pt",
        "optimizer.pt",
        "scheduler.pt",
        "rng_state.pth",
    ):
        (checkpoint / filename).write_bytes(b"V8 launcher test fixture\n")

    if step == contract.CHECKPOINT_STEPS[0]:
        assert source_checkpoint is None
        warm = contract.validate_warm_start_contract()
        warm_lock = json.loads(
            contract.WARM_START_LOCK.read_text(encoding="utf-8")
        )
        lineage: dict[str, object] = {
            "schema": contract.WARM_START_RECEIPT_SCHEMA,
            "schema_version": contract.WARM_START_LINEAGE_SCHEMA_VERSION,
            "mode": contract.WARM_START_MODE,
            "source_checkpoint": warm["warm_start_checkpoint"],
            "source_lock": {
                "path": warm["warm_start_lock"],
                "lock_sha256": contract.WARM_START_LOCK_CANONICAL_SHA256,
            },
            "source_artifacts": warm_lock["artifacts"],
            "source_global_step": 256,
            "source_epoch": 8.0,
            "source_state_imports": {
                "adapter": True,
                "optimizer": False,
                "scheduler": False,
                "trainer_state": False,
                "rng": False,
                "global_step": False,
            },
            "post_load_bit_equal": True,
            "target_fresh_start": {
                "initial_global_step": 0,
                "optimizer_implementation": "adamw_torch_fused",
                "optimizer_created_after_adapter_load": True,
                "optimizer_state": "fresh",
                "scheduler_state": "fresh",
                "trainer_state": "fresh",
                "rng_state": "fresh_from_v8_seed",
            },
            "target_delta_config_sha256": contract.canonical_sha256(config),
            "target_training_protocol_sha256": contract.canonical_sha256(protocol),
            "target_scene_state_pairing_manifest_sha256": pairing[
                "manifest_sha256"
            ],
            "trainer_resume_from_checkpoint": None,
            "target_initial_global_step": 0,
            "pre_train_global_step": 0,
            "fresh_optimizer_created": True,
            "fresh_optimizer_class": "torch.optim.adamw.AdamW",
            "fresh_optimizer_state_entries_before_train": 0,
            "fresh_scheduler_created_before_train": False,
        }
        lineage["receipt_sha256"] = contract.canonical_sha256(lineage)
        _write_json_object(
            checkpoint / contract.WARM_START_LINEAGE_FILENAME,
            lineage,
        )
        return checkpoint

    assert source_checkpoint is not None
    source_state = json.loads(
        (source_checkpoint / "trainer_state.json").read_text(encoding="utf-8")
    )
    source_step = int(source_state["global_step"])
    expected_source_step = contract.CHECKPOINT_STEPS[
        contract.CHECKPOINT_STEPS.index(step) - 1
    ]
    assert source_step == expected_source_step
    source_protocol = json.loads(
        (source_checkpoint / "training_protocol.json").read_text(encoding="utf-8")
    )
    source_lineage_filename = (
        contract.WARM_START_LINEAGE_FILENAME
        if source_step == contract.CHECKPOINT_STEPS[0]
        else contract.CONTINUATION_LINEAGE_FILENAME
    )
    source_lineage_path = source_checkpoint / source_lineage_filename
    source_lineage = json.loads(source_lineage_path.read_text(encoding="utf-8"))
    root_receipt_sha256 = (
        source_lineage["receipt_sha256"]
        if source_step == contract.CHECKPOINT_STEPS[0]
        else source_lineage["root_warm_start_receipt_sha256"]
    )
    continuation: dict[str, object] = {
        "schema_version": contract.CONTINUATION_LINEAGE_SCHEMA_VERSION,
        "mode": "extend",
        "source_checkpoint": str(source_checkpoint.resolve()),
        "source_global_step": source_step,
        "source_effective_max_steps": source_step,
        "source_max_steps": source_step,
        "source_num_train_epochs": float(source_protocol["num_train_epochs"]),
        "source_training_protocol_sha256": contract.canonical_sha256(
            source_protocol
        ),
        "source_rng_state_files": ["rng_state.pth"],
        "source_lineage_filename": source_lineage_filename,
        "source_lineage_file_sha256": contract.sha256_file(source_lineage_path),
        "root_warm_start_receipt_sha256": root_receipt_sha256,
        "target_max_steps": step,
        "target_num_train_epochs": float(protocol["num_train_epochs"]),
        "target_training_protocol_sha256": contract.canonical_sha256(protocol),
        "lr_scheduler_type": protocol["lr_scheduler_type"],
        "warmup_steps": protocol["warmup_steps"],
    }
    continuation["manifest_sha256"] = contract.canonical_sha256(continuation)
    _write_json_object(
        checkpoint / contract.CONTINUATION_LINEAGE_FILENAME,
        continuation,
    )
    return checkpoint


@pytest.fixture
def completed_v8_step14() -> Path:
    fixture_root = RUN_ROOT / f"pytest_resume_source_{uuid.uuid4().hex}"
    checkpoint = _write_completed_v8_checkpoint(fixture_root, step=14)
    try:
        yield checkpoint
    finally:
        shutil.rmtree(fixture_root, ignore_errors=True)


@pytest.fixture
def completed_v8_step28() -> Path:
    fixture_root = RUN_ROOT / f"pytest_resume_chain_{uuid.uuid4().hex}"
    step14 = _write_completed_v8_checkpoint(fixture_root, step=14)
    step28 = _write_completed_v8_checkpoint(
        fixture_root,
        step=28,
        source_checkpoint=step14,
    )
    try:
        yield step28
    finally:
        shutil.rmtree(fixture_root, ignore_errors=True)


def test_resume_advances_only_to_next_locked_endpoint_and_exact_cursor(
    fake_git_bin: Path,
    completed_v8_step14: Path,
) -> None:
    run_name = f"pytest_resume_{uuid.uuid4().hex}"
    result = run_launcher(
        launcher_environment(
            fake_git_bin,
            run_name,
            TARGET_STEP="28",
            RESUME_FROM_CHECKPOINT=str(completed_v8_step14),
        )
    )
    arguments = dry_run_arguments(result)

    assert argument_value(arguments, "--resume-from-checkpoint") == str(
        completed_v8_step14.resolve()
    )
    assert argument_value(arguments, "--resume-mode") == "extend"
    assert argument_value(arguments, "--max-steps") == "28"
    assert argument_value(arguments, "--save-steps") == "14"
    assert argument_value(arguments, "--warmup-ratio") == "0"
    assert argument_value(arguments, "--warmup-steps") == "4"
    assert "--warm-start-from-checkpoint" not in arguments
    assert "source_step=14 target_step=28 cursor=14" in result.stdout


def test_resume_accepts_recursive_continuation_chain(
    fake_git_bin: Path,
    completed_v8_step28: Path,
) -> None:
    result = run_launcher(
        launcher_environment(
            fake_git_bin,
            f"pytest_resume_chain_{uuid.uuid4().hex}",
            TARGET_STEP="42",
            RESUME_FROM_CHECKPOINT=str(completed_v8_step28),
        )
    )
    arguments = dry_run_arguments(result)
    validated = contract.validate_launch_contract(
        target_step=42,
        resume_checkpoint=completed_v8_step28,
    )
    continuation = json.loads(
        (
            completed_v8_step28 / contract.CONTINUATION_LINEAGE_FILENAME
        ).read_text(encoding="utf-8")
    )

    assert argument_value(arguments, "--max-steps") == "42"
    assert validated["root_warm_start_receipt_sha256"] == continuation[
        "root_warm_start_receipt_sha256"
    ]
    assert validated["source_lineage_filename"] == (
        contract.CONTINUATION_LINEAGE_FILENAME
    )
    assert validated["source_lineage_file_sha256"] == contract.sha256_file(
        completed_v8_step28 / contract.CONTINUATION_LINEAGE_FILENAME
    )


def test_resume_requires_warm_start_lineage_at_step14(
    completed_v8_step14: Path,
) -> None:
    (completed_v8_step14 / contract.WARM_START_LINEAGE_FILENAME).unlink()

    with pytest.raises(
        contract.LaunchContractError,
        match="resume_lineage_missing_or_symlink",
    ):
        contract.validate_launch_contract(
            target_step=28,
            resume_checkpoint=completed_v8_step14,
        )


def test_resume_rejects_ambiguous_lineage_files(
    completed_v8_step14: Path,
) -> None:
    _write_json_object(
        completed_v8_step14 / contract.CONTINUATION_LINEAGE_FILENAME,
        {"mode": "extend"},
    )

    with pytest.raises(
        contract.LaunchContractError,
        match="resume_lineage_unexpected",
    ):
        contract.validate_launch_contract(
            target_step=28,
            resume_checkpoint=completed_v8_step14,
        )


def test_resume_rejects_noncanonical_warm_start_receipt(
    completed_v8_step14: Path,
) -> None:
    lineage_path = completed_v8_step14 / contract.WARM_START_LINEAGE_FILENAME
    lineage = json.loads(lineage_path.read_text(encoding="utf-8"))
    lineage["source_global_step"] = 255
    _write_json_object(lineage_path, lineage)

    with pytest.raises(
        contract.LaunchContractError,
        match="resume_warm_start_lineage_self_hash_differs",
    ):
        contract.validate_launch_contract(
            target_step=28,
            resume_checkpoint=completed_v8_step14,
        )


@pytest.mark.parametrize(
    ("field", "value", "message"),
    (
        (
            "source_checkpoint",
            "/run/media/xiaol/B214449214445C0B/not-the-locked-v7-checkpoint",
            "resume_warm_start_source_checkpoint_differs",
        ),
        (
            "target_training_protocol_sha256",
            "0" * 64,
            "resume_warm_start_target_protocol_differs",
        ),
    ),
)
def test_resume_rejects_semantically_drifted_warm_start_lineage(
    completed_v8_step14: Path,
    field: str,
    value: object,
    message: str,
) -> None:
    lineage_path = completed_v8_step14 / contract.WARM_START_LINEAGE_FILENAME
    _rewrite_self_hashed_manifest(
        lineage_path,
        hash_field="receipt_sha256",
        updates={field: value},
    )

    with pytest.raises(contract.LaunchContractError, match=message):
        contract.validate_launch_contract(
            target_step=28,
            resume_checkpoint=completed_v8_step14,
        )


def test_resume_requires_continuation_lineage_after_step14(
    completed_v8_step28: Path,
) -> None:
    (completed_v8_step28 / contract.CONTINUATION_LINEAGE_FILENAME).unlink()

    with pytest.raises(
        contract.LaunchContractError,
        match="resume_lineage_missing_or_symlink",
    ):
        contract.validate_launch_contract(
            target_step=42,
            resume_checkpoint=completed_v8_step28,
        )


def test_resume_rejects_noncanonical_continuation_self_hash(
    completed_v8_step28: Path,
) -> None:
    lineage_path = completed_v8_step28 / contract.CONTINUATION_LINEAGE_FILENAME
    lineage = json.loads(lineage_path.read_text(encoding="utf-8"))
    lineage["target_max_steps"] = 29
    _write_json_object(lineage_path, lineage)

    with pytest.raises(
        contract.LaunchContractError,
        match="resume_continuation_lineage_self_hash_differs",
    ):
        contract.validate_launch_contract(
            target_step=42,
            resume_checkpoint=completed_v8_step28,
        )


@pytest.mark.parametrize(
    ("field", "value", "message"),
    (
        (
            "root_warm_start_receipt_sha256",
            "0" * 64,
            "resume_continuation_root_warm_start_receipt_differs",
        ),
        (
            "source_lineage_filename",
            contract.CONTINUATION_LINEAGE_FILENAME,
            "resume_continuation_source_lineage_filename_differs",
        ),
        (
            "source_lineage_file_sha256",
            "0" * 64,
            "resume_continuation_source_lineage_file_hash_differs",
        ),
        (
            "source_training_protocol_sha256",
            "0" * 64,
            "resume_continuation_source_protocol_differs",
        ),
        (
            "target_training_protocol_sha256",
            "0" * 64,
            "resume_continuation_target_protocol_differs",
        ),
    ),
)
def test_resume_rejects_drifted_continuation_chain_fields(
    completed_v8_step28: Path,
    field: str,
    value: object,
    message: str,
) -> None:
    lineage_path = completed_v8_step28 / contract.CONTINUATION_LINEAGE_FILENAME
    _rewrite_self_hashed_manifest(
        lineage_path,
        hash_field="manifest_sha256",
        updates={field: value},
    )

    with pytest.raises(contract.LaunchContractError, match=message):
        contract.validate_launch_contract(
            target_step=42,
            resume_checkpoint=completed_v8_step28,
        )


def test_resume_recursively_revalidates_immediate_source_lineage(
    completed_v8_step28: Path,
) -> None:
    continuation_path = completed_v8_step28 / contract.CONTINUATION_LINEAGE_FILENAME
    continuation = json.loads(continuation_path.read_text(encoding="utf-8"))
    source_checkpoint = Path(str(continuation["source_checkpoint"]))
    warm_lineage_path = source_checkpoint / contract.WARM_START_LINEAGE_FILENAME
    warm_lineage = _rewrite_self_hashed_manifest(
        warm_lineage_path,
        hash_field="receipt_sha256",
        updates={"target_training_protocol_sha256": "0" * 64},
    )
    _rewrite_self_hashed_manifest(
        continuation_path,
        hash_field="manifest_sha256",
        updates={
            "source_lineage_file_sha256": contract.sha256_file(warm_lineage_path),
            "root_warm_start_receipt_sha256": warm_lineage["receipt_sha256"],
        },
    )

    with pytest.raises(
        contract.LaunchContractError,
        match="resume_warm_start_target_protocol_differs",
    ):
        contract.validate_launch_contract(
            target_step=42,
            resume_checkpoint=completed_v8_step28,
        )


@pytest.mark.parametrize("target_step", contract.CHECKPOINT_STEPS)
def test_transformers_saves_every_locked_block_endpoint(target_step: int) -> None:
    args = SimpleNamespace(
        logging_first_step=False,
        logging_strategy=IntervalStrategy.NO,
        eval_strategy=IntervalStrategy.NO,
        eval_delay=0,
        save_strategy=SaveStrategy.STEPS,
    )
    state = TrainerState(
        global_step=target_step,
        max_steps=target_step,
        logging_steps=1,
        eval_steps=1000,
        save_steps=contract.SAVE_STEPS,
    )

    control = DefaultFlowCallback().on_step_end(
        args,
        state,
        TrainerControl(),
    )

    assert control.should_training_stop is True
    assert control.should_save is True


def test_resume_rejects_skipped_endpoint_and_schedule_drift(
    fake_git_bin: Path,
    completed_v8_step14: Path,
) -> None:
    skipped = run_launcher(
        launcher_environment(
            fake_git_bin,
            f"pytest_skip_{uuid.uuid4().hex}",
            TARGET_STEP="42",
            RESUME_FROM_CHECKPOINT=str(completed_v8_step14),
        )
    )
    assert skipped.returncode != 0
    assert "resume_target_is_not_next_locked_checkpoint" in skipped.stderr

    protocol_path = completed_v8_step14 / "training_protocol.json"
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    protocol["train_schedule"]["schedule_entries_sha256"] = "0" * 64
    protocol_path.write_text(json.dumps(protocol), encoding="utf-8")
    drifted = run_launcher(
        launcher_environment(
            fake_git_bin,
            f"pytest_drift_{uuid.uuid4().hex}",
            TARGET_STEP="28",
            RESUME_FROM_CHECKPOINT=str(completed_v8_step14),
        )
    )
    assert drifted.returncode != 0
    assert "resume_protocol_schedule_differs" in drifted.stderr


@pytest.mark.parametrize(
    ("updates", "message"),
    (
        ({"TARGET_STEP": "15"}, "target_step_not_locked_checkpoint"),
        ({"TARGET_STEP": "28"}, "fresh_launch_must_target_step14"),
        ({"SMOKE_RUN": "1", "TARGET_STEP": "2"}, "smoke_launch_must_target_step1"),
        ({"RESUME_FROM_CHECKPOINT": "latest", "TARGET_STEP": "28"}, "resume_checkpoint_must_be_explicit"),
    ),
)
def test_launcher_rejects_invalid_horizons(
    fake_git_bin: Path,
    updates: dict[str, str],
    message: str,
) -> None:
    result = run_launcher(
        launcher_environment(
            fake_git_bin,
            f"pytest_horizon_{uuid.uuid4().hex}",
            **updates,
        )
    )

    assert result.returncode != 0
    assert message in result.stderr


def test_launcher_rejects_hard32_or_evaluation_environment(
    fake_git_bin: Path,
) -> None:
    result = run_launcher(
        launcher_environment(
            fake_git_bin,
            f"pytest_hard32_{uuid.uuid4().hex}",
            HARD32_FILE="/tmp/forbidden.jsonl",
        )
    )

    assert result.returncode != 0
    assert "hard32_or_evaluation_access_is_forbidden variable=HARD32_FILE" in result.stderr


def test_launch_contract_never_opens_hard32(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    hard32 = Path(source_lock()["fixed_hard32"]["path"]).resolve()
    original_open = Path.open
    original_resolve = Path.resolve

    def guarded_open(path: Path, *args, **kwargs):
        if path.resolve() == hard32:
            raise AssertionError("launch contract attempted to open Hard32")
        return original_open(path, *args, **kwargs)

    def guarded_resolve(path: Path, *args, **kwargs):
        if str(path.expanduser()) == str(hard32):
            raise AssertionError("launch contract attempted to resolve Hard32")
        return original_resolve(path, *args, **kwargs)

    monkeypatch.setattr(Path, "open", guarded_open)
    monkeypatch.setattr(Path, "resolve", guarded_resolve)
    result = contract.validate_launch_contract(target_step=14)

    assert result["hard32_access"] == "forbidden_not_opened_by_launch_contract"


def test_launch_contract_rejects_lock_and_schedule_tampering(
    tmp_path: Path,
) -> None:
    tampered_lock = tmp_path / "source_lock.json"
    payload = source_lock()
    payload["curriculum"]["total_steps"] = 151
    tampered_lock.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(contract.LaunchContractError, match="source_lock_file_hash_differs"):
        contract.validate_data_contract(source_lock_path=tampered_lock)

    tampered_schedule = tmp_path / "schedule.jsonl"
    schedule = Path(source_lock()["artifacts"]["schedule"]["path"])
    lines = schedule.read_text(encoding="utf-8").splitlines()
    lines[0], lines[1] = lines[1], lines[0]
    tampered_schedule.write_text("\n".join(lines) + "\n", encoding="utf-8")
    with pytest.raises(contract.LaunchContractError, match="schedule_indexing_differs"):
        contract._validate_schedule(tampered_schedule)


def test_launcher_collision_and_tracked_dirty_checks(
    fake_git_bin: Path,
) -> None:
    run_name = f"pytest_collision_{uuid.uuid4().hex}"
    output = RUN_ROOT / f"scene_memory_v8_production_{run_name}_step14"
    output.mkdir(parents=True)
    try:
        collision = run_launcher(launcher_environment(fake_git_bin, run_name))
        assert collision.returncode != 0
        assert "fresh_output_collision" in collision.stderr
    finally:
        shutil.rmtree(output, ignore_errors=True)

    dirty = run_launcher(
        launcher_environment(
            fake_git_bin,
            f"pytest_dirty_{uuid.uuid4().hex}",
            FAKE_GIT_DIRTY="1",
        )
    )
    assert dirty.returncode != 0
    assert "tracked_worktree_must_be_clean_before_v8_training" in dirty.stderr

    untracked_critical = run_launcher(
        launcher_environment(
            fake_git_bin,
            f"pytest_untracked_critical_{uuid.uuid4().hex}",
            FAKE_GIT_UNTRACKED_CRITICAL="1",
        )
    )
    assert untracked_critical.returncode != 0
    assert "critical_v8_source_must_be_tracked" in untracked_critical.stderr


def test_launcher_locks_mutable_paths_to_ssd_and_ignores_untracked() -> None:
    source = LAUNCHER.read_text(encoding="utf-8")

    assert 'SSD_ROOT="/run/media/xiaol/B214449214445C0B"' in source
    assert 'LOG_DIR="${RUN_ROOT}/logs"' in source
    assert 'HF_HOME_LOCKED="${CACHE_ROOT}/huggingface"' in source
    assert 'TOKENIZED_DATASET_ROOT="${CACHE_ROOT}/tokenized"' in source
    assert 'TMPDIR_LOCKED="${CACHE_ROOT}/tmp/${run_kind}_${RUN_NAME}_step${TARGET_STEP}"' in source
    assert "path_must_stay_on_2t_ssd" in source
    assert "status --porcelain --untracked-files=no" in source
    assert "tracked_worktree_must_be_clean_before_v8_training" in source
    assert "ls-files --error-unmatch" in source
    assert "critical_v8_source_must_be_tracked" in source
    assert '"deltamem/train/scene_state_generation_alignment.py"' in source
    assert 'EXECUTION_METADATA="${LOG_DIR}/' in source
    assert 'EXECUTION_METADATA="${OUTPUT_DIR}/' not in source
    assert "trainer_output_must_be_empty_before_entry" in source
    assert "hard32_or_evaluation_access_is_forbidden" in source
    assert "--warmup-steps 4" in source


def test_non_dry_launcher_keeps_output_empty_until_trainer_entry(
    fake_git_bin: Path,
    tmp_path: Path,
) -> None:
    run_name = f"pytest_nondry_{uuid.uuid4().hex}"
    output = RUN_ROOT / f"scene_memory_v8_production_{run_name}_step14"
    log = RUN_ROOT / "logs" / f"scene_memory_v8_production_{run_name}_step14.log"
    metadata = (
        RUN_ROOT
        / "logs"
        / f"scene_memory_v8_production_{run_name}_step14.execution.json"
    )
    fake_python = tmp_path / "fake_python"
    fake_python.write_text(
        """#!/usr/bin/env python3
from pathlib import Path
import sys

args = sys.argv[1:]
output = Path(args[args.index("--output-dir") + 1])
entries = list(output.iterdir())
if entries:
    raise SystemExit(f"output was not empty at trainer entry: {entries}")
target = args[args.index("--max-steps") + 1]
(output / "training_summary.json").write_text("{}\\n", encoding="utf-8")
(output / "trainer" / f"checkpoint-{target}").mkdir(parents=True)
""",
        encoding="utf-8",
    )
    fake_python.chmod(0o755)
    environment = launcher_environment(
        fake_git_bin,
        run_name,
        DRY_RUN="0",
        PYTHON_BIN=str(fake_python),
    )

    try:
        result = run_launcher(environment)

        assert result.returncode == 0, result.stderr
        assert (output / "training_summary.json").is_file()
        assert (output / "trainer/checkpoint-14").is_dir()
        assert metadata.is_file()
        assert metadata.parent != output
        assert log.is_file()
        receipt = json.loads(metadata.read_text(encoding="utf-8"))
        assert receipt["fixed_schedule_cursor"]["consumed_steps"] == 0
        assert receipt["optimization"]["warmup_steps"] == 4
        assert receipt["hard32_access"] == "forbidden"
    finally:
        shutil.rmtree(output, ignore_errors=True)
        log.unlink(missing_ok=True)
        metadata.unlink(missing_ok=True)
