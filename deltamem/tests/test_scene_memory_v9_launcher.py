from __future__ import annotations

import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys
import uuid

import pytest

from experiments.rethinking_rwkv_ms_gemma import scene_memory_v9_launch_contract as contract


REPO_ROOT = Path(__file__).resolve().parents[2]
LAUNCHER = REPO_ROOT / "experiments/rethinking_rwkv_ms_gemma/train_scene_memory_v9.sh"
SSD_ROOT = Path("/run/media/xiaol/B214449214445C0B")
RUN_ROOT = SSD_ROOT / "delta_mem_outputs/novel_rwkv_ms_memory/scene_memory_v9"
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
    "GATE_RECEIPT",
    "RESUME_MODE",
    "TARGET_STEP",
    "SMOKE_RUN",
    "HARD32",
    "HARD32_FILE",
    "HARD32_PATH",
    "HARD32_DIR",
    "EVAL_DATASET",
    "EVAL_FILE",
    "EVAL_PATH",
    "EVAL_DIR",
    "DO_EVAL",
    "VALIDATION_DATASET",
    "VALIDATION_FILE",
    "VALIDATION_PATH",
    "VALIDATION_DIR",
    "TEST_DATASET",
    "TEST_FILE",
    "TEST_PATH",
    "TEST_DIR",
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
  printf '8511662000000000000000000000000000000000\\n'
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


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=True, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _protocol(data: dict[str, object], step: int) -> dict[str, object]:
    return {
        "schema_version": contract.OBJECTIVE_SCHEMA_VERSION,
        "memory_objective_version": contract.OBJECTIVE_VERSION,
        "memory_loss_mode": "scene_state_generation_ce",
        "train_file": data["train_file"],
        "train_samples": 32,
        "eval_samples": 0,
        "training_mode": "episode",
        "assistant_loss_mode": "final_assistant_only",
        "episode_recent_messages": 0,
        "episode_read_write_enabled": False,
        "per_device_train_batch_size": contract.PAIR_PHYSICAL_BATCH_SIZE,
        "gradient_accumulation_steps": 1,
        "learning_rate": contract.LEARNING_RATE,
        "lr_scheduler_type": "constant_with_warmup",
        "warmup_ratio": contract.WARMUP_RATIO,
        "warmup_steps": contract.WARMUP_STEPS,
        "save_steps": contract.SAVE_STEPS,
        "num_train_epochs": 1.0,
        "max_steps": step,
        "train_sampler_seed": None,
        "train_sampler_mode": contract.FIXED_SAMPLER_MODE,
        "memory_kl_weight": 0.0,
        "memory_base_kl_weight": 0.0,
        "memory_representation_weight": 0.0,
        "write_sparsity_weight": 0.0,
        "scene_generation_generated_unlikelihood_weight": 0.0,
        "scene_generation_generated_prefix_correction_weight": (
            contract.PREFIX_CORRECTION_WEIGHT
        ),
        "scene_generation_pair_physical_batch_size": (
            contract.PAIR_PHYSICAL_BATCH_SIZE
        ),
        "scene_generation_pair_directional_exposures": (
            contract.PAIR_DIRECTIONAL_EXPOSURES
        ),
        "train_schedule": contract._expected_schedule_protocol(data),
    }


def _config() -> dict[str, object]:
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


def _pairing() -> dict[str, object]:
    payload: dict[str, object] = {
        "schema_version": 2,
        "objective_version": contract.PAIRING_OBJECTIVE_VERSION,
        "pairing_version": "v9_launcher_test_reciprocal_v1",
    }
    payload["manifest_sha256"] = contract.canonical_sha256(payload)
    return payload


def _write_checkpoint(
    fixture_root: Path,
    *,
    step: int,
    data: dict[str, object],
    warm: dict[str, object],
    source_checkpoint: Path | None = None,
) -> Path:
    checkpoint = fixture_root / f"block-{step}/trainer/checkpoint-{step}"
    checkpoint.mkdir(parents=True)
    protocol = _protocol(data, step)
    config = _config()
    pairing = _pairing()
    _write_json(
        checkpoint / "trainer_state.json",
        {"global_step": step, "max_steps": step, "epoch": step / 28},
    )
    _write_json(checkpoint / "training_protocol.json", protocol)
    _write_json(checkpoint / "delta_mem_config.json", config)
    _write_json(checkpoint / "scene_state_identity_pairing_manifest.json", pairing)
    for filename in (
        "delta_mem_adapter.pt",
        "optimizer.pt",
        "scheduler.pt",
        "rng_state.pth",
    ):
        (checkpoint / filename).write_bytes(b"V9 launcher test fixture\n")

    if step == contract.CHECKPOINT_STEPS[0]:
        assert source_checkpoint is None
        lineage: dict[str, object] = {
            "schema": contract.WARM_START_RECEIPT_SCHEMA,
            "schema_version": 1,
            "mode": contract.WARM_START_MODE,
            "source_checkpoint": warm["warm_start_checkpoint"],
            "source_lock": {
                "path": warm["warm_start_lock"],
                "lock_sha256": warm["warm_start_lock_sha256"],
            },
            "source_state_imports": contract.SOURCE_IMPORT_POLICY,
            "post_load_bit_equal": True,
            "target_fresh_start": {
                "initial_global_step": 0,
                "optimizer_implementation": "adamw_torch_fused",
                "optimizer_created_after_adapter_load": True,
                "optimizer_state": "fresh",
                "scheduler_state": "fresh",
                "trainer_state": "fresh",
                "rng_state": "fresh_from_v9_seed",
            },
            "target_delta_config_sha256": contract.canonical_sha256(config),
            "target_training_protocol_sha256": contract.canonical_sha256(protocol),
            "target_scene_state_pairing_manifest_sha256": pairing["manifest_sha256"],
            "trainer_resume_from_checkpoint": None,
            "target_initial_global_step": 0,
            "pre_train_global_step": 0,
            "fresh_optimizer_created": True,
            "fresh_optimizer_class": "torch.optim.adamw.AdamW",
            "fresh_optimizer_state_entries_before_train": 0,
            "fresh_scheduler_created_before_train": False,
        }
        lineage["receipt_sha256"] = contract.canonical_sha256(lineage)
        _write_json(checkpoint / contract.WARM_START_LINEAGE_FILENAME, lineage)
        return checkpoint

    assert source_checkpoint is not None
    source_step = int(source_checkpoint.name.removeprefix("checkpoint-"))
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
    root_receipt = source_lineage.get(
        "root_warm_start_receipt_sha256",
        source_lineage.get("receipt_sha256"),
    )
    continuation: dict[str, object] = {
        "schema_version": contract.CONTINUATION_LINEAGE_SCHEMA_VERSION,
        "mode": "extend",
        "source_checkpoint": str(source_checkpoint.resolve()),
        "source_global_step": source_step,
        "source_effective_max_steps": source_step,
        "source_max_steps": source_step,
        "source_num_train_epochs": 1.0,
        "source_training_protocol_sha256": contract.canonical_sha256(source_protocol),
        "source_rng_state_files": ["rng_state.pth"],
        "source_lineage_filename": source_lineage_filename,
        "source_lineage_file_sha256": contract.sha256_file(source_lineage_path),
        "root_warm_start_receipt_sha256": root_receipt,
        "target_max_steps": step,
        "target_num_train_epochs": 1.0,
        "target_training_protocol_sha256": contract.canonical_sha256(protocol),
        "lr_scheduler_type": "constant_with_warmup",
        "warmup_steps": contract.WARMUP_STEPS,
    }
    continuation["manifest_sha256"] = contract.canonical_sha256(continuation)
    _write_json(checkpoint / contract.CONTINUATION_LINEAGE_FILENAME, continuation)
    return checkpoint


@pytest.fixture
def completed_step7() -> Path:
    fixture_root = RUN_ROOT / f"pytest_v9_step7_{uuid.uuid4().hex}"
    data = contract.validate_data_contract()
    warm = contract.validate_warm_start_contract()
    checkpoint = _write_checkpoint(
        fixture_root,
        step=7,
        data=data,
        warm=warm,
    )
    try:
        yield checkpoint
    finally:
        shutil.rmtree(fixture_root, ignore_errors=True)


def test_fresh_v9_contract_and_dry_run_bind_pair_objective(
    fake_git_bin: Path,
) -> None:
    result = contract.validate_launch_contract(target_step=7)
    assert result["launch_mode"] == "warm_start"
    assert result["resume_schedule_cursor"] == 0
    assert result["pair_physical_batch_size"] == 1
    assert result["pair_logical_batch_size"] == 2
    assert result["checkpoint_steps"] == [7, 14, 21, 28]
    assert result["hard32_access"] == "forbidden_not_resolved_opened_or_hashed"

    arguments = dry_run_arguments(
        run_launcher(
            launcher_environment(
                fake_git_bin,
                f"pytest_v9_fresh_{uuid.uuid4().hex}",
            )
        )
    )
    assert arguments[:3] == ["/bin/true", "-m", "deltamem.train.delta_sft"]
    assert argument_value(arguments, "--warm-start-mode") == contract.WARM_START_MODE
    assert argument_value(arguments, "--resume-mode") == "exact"
    assert "--resume-from-checkpoint" not in arguments
    assert argument_value(arguments, "--target-layers") == TARGET_LAYERS
    assert argument_value(arguments, "--delta-heads") == "q,o"
    assert argument_value(arguments, "--per-device-train-batch-size") == "1"
    assert argument_value(arguments, "--max-steps") == "7"
    assert argument_value(arguments, "--save-steps") == "7"
    assert argument_value(
        arguments,
        "--scene-state-generation-objective-version",
    ) == contract.OBJECTIVE_VERSION
    assert argument_value(
        arguments,
        "--scene-state-generated-prefix-correction-weight",
    ) == "0.5"
    assert argument_value(
        arguments,
        "--scene-state-generated-unlikelihood-weight",
    ) == "0"


def test_v9_smoke_is_fresh_step1(fake_git_bin: Path) -> None:
    arguments = dry_run_arguments(
        run_launcher(
            launcher_environment(
                fake_git_bin,
                f"pytest_v9_smoke_{uuid.uuid4().hex}",
                SMOKE_RUN="1",
            )
        )
    )
    assert argument_value(arguments, "--max-steps") == "1"
    assert argument_value(arguments, "--save-steps") == "1"
    assert "--warm-start-from-checkpoint" in arguments
    assert "--resume-from-checkpoint" not in arguments
    assert "scene_memory_v9_smoke_" in argument_value(arguments, "--output-dir")


def test_v9_resume_is_immediate_exact_state_continuation(
    fake_git_bin: Path,
    completed_step7: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    data = contract.validate_data_contract()
    warm = contract.validate_warm_start_contract()
    checkpoint = contract.validate_checkpoint_contract(
        completed_step7,
        data=data,
        warm=warm,
    )
    assert checkpoint["checkpoint_step"] == 7
    gate_receipt = completed_step7.parents[2] / "step7-gate-receipt.json"
    gate_receipt.write_text("{}\n", encoding="utf-8")
    from experiments.rethinking_rwkv_ms_gemma import run_scene_memory_v9_gate as gate

    monkeypatch.setattr(
        gate,
        "validate_continuation_authorization",
        lambda *_args, **_kwargs: {
            "authorization_kind": gate.CONTINUATION_AUTHORIZATION_KIND,
            "gate_receipt": str(gate_receipt.resolve()),
            "gate_receipt_file_sha256": contract.sha256_file(gate_receipt),
            "gate_receipt_sha256": "a" * 64,
            "source_checkpoint": str(completed_step7.resolve()),
            "source_step": 7,
            "target_step": 14,
            "hard32_access": "forbidden_not_resolved_opened_or_hashed",
            "hard32_authorized": False,
        },
    )
    result = contract.validate_resume_contract(
        resume_checkpoint=completed_step7,
        target_step=14,
        gate_receipt=gate_receipt,
        data=data,
        warm=warm,
    )
    assert result["resume_schedule_cursor"] == 7
    assert result["source_step"] == 7
    assert result["gate_receipt_sha256"] == "a" * 64

    missing_gate = run_launcher(
        launcher_environment(
            fake_git_bin,
            f"pytest_v9_resume_missing_gate_{uuid.uuid4().hex}",
            TARGET_STEP="14",
            RESUME_FROM_CHECKPOINT=str(completed_step7),
        )
    )
    assert missing_gate.returncode != 0
    assert "resume_requires_explicit_v9_gate_receipt" in missing_gate.stderr


@pytest.mark.parametrize(
    ("updates", "message"),
    (
        ({"TARGET_STEP": "8"}, "target_step_not_locked_v9_endpoint"),
        ({"TARGET_STEP": "14"}, "fresh_launch_must_target_step7"),
        ({"SMOKE_RUN": "1", "TARGET_STEP": "2"}, "smoke_launch_must_target_step1"),
        ({"RESUME_FROM_CHECKPOINT": "latest", "TARGET_STEP": "14"}, "resume_checkpoint_must_be_explicit"),
    ),
)
def test_v9_launcher_rejects_unlocked_horizons(
    fake_git_bin: Path,
    updates: dict[str, str],
    message: str,
) -> None:
    result = run_launcher(
        launcher_environment(
            fake_git_bin,
            f"pytest_v9_horizon_{uuid.uuid4().hex}",
            **updates,
        )
    )
    assert result.returncode != 0
    assert message in result.stderr


def test_v9_launcher_rejects_hard32_eval_and_dirty_tree(
    fake_git_bin: Path,
) -> None:
    forbidden = run_launcher(
        launcher_environment(
            fake_git_bin,
            f"pytest_v9_forbidden_{uuid.uuid4().hex}",
            EVAL_FILE="/tmp/forbidden.jsonl",
        )
    )
    assert forbidden.returncode != 0
    assert "hard32_or_evaluation_access_is_forbidden variable=EVAL_FILE" in forbidden.stderr

    dirty = run_launcher(
        launcher_environment(
            fake_git_bin,
            f"pytest_v9_dirty_{uuid.uuid4().hex}",
            FAKE_GIT_DIRTY="1",
        )
    )
    assert dirty.returncode != 0
    assert "tracked_worktree_must_be_clean_before_v9_training" in dirty.stderr


def test_v9_launch_contract_never_resolves_or_opens_hard32(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original_open = Path.open
    original_resolve = Path.resolve

    def guarded_open(path: Path, *args, **kwargs):
        if "hard32" in {part.lower() for part in path.parts}:
            raise AssertionError("V9 launch contract attempted to open Hard32")
        return original_open(path, *args, **kwargs)

    def guarded_resolve(path: Path, *args, **kwargs):
        if "hard32" in {part.lower() for part in path.parts}:
            raise AssertionError("V9 launch contract attempted to resolve Hard32")
        return original_resolve(path, *args, **kwargs)

    monkeypatch.setattr(Path, "open", guarded_open)
    monkeypatch.setattr(Path, "resolve", guarded_resolve)
    result = contract.validate_launch_contract(target_step=7)
    assert result["hard32_access"] == "forbidden_not_resolved_opened_or_hashed"


def test_v9_attached_fake_run_writes_launch_log_and_completion_receipts(
    fake_git_bin: Path,
    tmp_path: Path,
) -> None:
    run_name = f"pytest_v9_attached_{uuid.uuid4().hex}"
    run_id = f"scene_memory_v9_production_{run_name}_step7"
    output = RUN_ROOT / run_id
    log = RUN_ROOT / "logs" / f"{run_id}.log"
    launch_receipt = RUN_ROOT / "logs" / f"{run_id}.launch.json"
    completion_receipt = RUN_ROOT / "logs" / f"{run_id}.completion.json"
    fake_python = tmp_path / "fake_python"
    fake_python.write_text(
        """#!/usr/bin/env python3
from pathlib import Path
import sys

args = sys.argv[1:]
output = Path(args[args.index("--output-dir") + 1])
if list(output.iterdir()):
    raise SystemExit("trainer output was not empty at entry")
target = args[args.index("--max-steps") + 1]
(output / "training_summary.json").write_text("{}\\n", encoding="utf-8")
checkpoint = output / "trainer" / f"checkpoint-{target}"
checkpoint.mkdir(parents=True)
for name in (
    "delta_mem_adapter.pt", "delta_mem_config.json", "optimizer.pt",
    "scheduler.pt", "trainer_state.json", "training_protocol.json",
    "scene_state_identity_pairing_manifest.json", "rng_state.pth",
):
    (checkpoint / name).write_bytes(b"fake attached V9 artifact\\n")
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
        assert output.is_dir()
        assert log.is_file()
        assert launch_receipt.is_file()
        assert completion_receipt.is_file()
        launch = json.loads(launch_receipt.read_text(encoding="utf-8"))
        completion = json.loads(completion_receipt.read_text(encoding="utf-8"))
        assert launch["attached_foreground_execution"] is True
        assert launch["fixed_schedule_cursor"]["consumed_pair_steps"] == 0
        assert launch["pair_execution"]["physical_batch_size"] == 1
        assert launch["pair_execution"]["logical_reciprocal_pair_size"] == 2
        gate_binding = launch["training_code"]["progression_gate"]
        assert gate_binding["path"].endswith("run_scene_memory_v9_gate.py")
        assert len(gate_binding["sha256"]) == 64
        assert completion["status"] == "completed"
        assert completion["hard32_access"] == "forbidden"
        assert completion["evaluation_access"] == "forbidden"
    finally:
        shutil.rmtree(output, ignore_errors=True)
        log.unlink(missing_ok=True)
        launch_receipt.unlink(missing_ok=True)
        completion_receipt.unlink(missing_ok=True)
