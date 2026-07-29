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


REPO_ROOT = Path(__file__).resolve().parents[2]
LAUNCHER = (
    REPO_ROOT
    / "experiments/rethinking_rwkv_ms_gemma/train_scene_memory_v7.sh"
)
SOURCE_LOCK = (
    REPO_ROOT
    / "experiments/rethinking_rwkv_ms_gemma/scene_memory_v7_source_lock.json"
)
SSD_ROOT = Path("/run/media/xiaol/B214449214445C0B")
RUN_ROOT = (
    SSD_ROOT
    / "delta_mem_outputs/novel_rwkv_ms_memory/scene_memory_v7"
)
TARGET_LAYERS = ",".join(str(index) for index in range(42))
EMPTY_DISTRIBUTED_ENV = (
    "WORLD_SIZE",
    "LOCAL_RANK",
    "RANK",
    "MASTER_ADDR",
    "MASTER_PORT",
    "SLURM_PROCID",
    "PMI_RANK",
    "OMPI_COMM_WORLD_RANK",
)


def _source_lock() -> dict[str, object]:
    return json.loads(SOURCE_LOCK.read_text(encoding="utf-8"))


def _launcher_environment(dataset_kind: str, run_name: str) -> dict[str, str]:
    environment = os.environ.copy()
    for variable in (*EMPTY_DISTRIBUTED_ENV, "RESUME_FROM_CHECKPOINT", "RESUME_MODE"):
        environment.pop(variable, None)
    environment.update(
        {
            "DATASET_KIND": dataset_kind,
            "RUN_NAME": run_name,
            "DRY_RUN": "1",
            "PYTHON_BIN": "/bin/true",
            "VALIDATION_PYTHON_BIN": sys.executable,
        }
    )
    return environment


def _run_launcher(environment: dict[str, str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [str(LAUNCHER)],
        cwd=REPO_ROOT,
        env=environment,
        capture_output=True,
        text=True,
    )


def _dry_run_arguments(result: subprocess.CompletedProcess[str]) -> list[str]:
    assert result.returncode == 0, result.stderr
    return shlex.split(result.stdout.strip().splitlines()[-1])


def _argument_value(arguments: list[str], flag: str) -> str:
    return arguments[arguments.index(flag) + 1]


@pytest.mark.parametrize("dataset_kind", ["tiny2", "train32"])
def test_fresh_dry_run_binds_exact_v7_dataset_and_objective(
    dataset_kind: str,
) -> None:
    run_name = f"pytest_fresh_{dataset_kind}_{uuid.uuid4().hex}"
    arguments = _dry_run_arguments(
        _run_launcher(_launcher_environment(dataset_kind, run_name))
    )
    artifacts = _source_lock()["artifacts"]
    data_binding = artifacts[dataset_kind]
    source_binding = artifacts[f"{dataset_kind}_source_manifest"]

    assert arguments[:3] == ["/bin/true", "-m", "deltamem.train.delta_sft"]
    assert _argument_value(arguments, "--train-file") == data_binding["path"]
    assert _argument_value(arguments, "--scene-state-source-manifest") == (
        source_binding["path"]
    )
    assert _argument_value(
        arguments,
        "--expected-scene-state-source-manifest-sha256",
    ) == source_binding["sha256"]
    assert _argument_value(arguments, "--memory-loss-mode") == (
        "scene_state_generation_ce"
    )
    assert _argument_value(arguments, "--target-layers") == TARGET_LAYERS
    assert _argument_value(arguments, "--delta-heads") == "q,o"
    assert _argument_value(arguments, "--rank") == "4"
    assert _argument_value(arguments, "--max-write-length") == "2048"
    assert _argument_value(arguments, "--per-device-train-batch-size") == "1"
    assert _argument_value(arguments, "--gradient-accumulation-steps") == "1"
    assert _argument_value(arguments, "--max-steps") == "32"
    assert _argument_value(arguments, "--save-steps") == "32"
    assert "--frozen-mlp-activation-checkpointing" in arguments
    assert "--initial-adapter-output-dir" in arguments
    assert "--resume-from-checkpoint" not in arguments
    assert "--resume-mode" not in arguments
    assert not Path(_argument_value(arguments, "--output-dir")).exists()


@pytest.fixture
def completed_tiny2_checkpoint() -> Path:
    fixture_root = RUN_ROOT / f"pytest_source_{uuid.uuid4().hex}"
    checkpoint = fixture_root / "trainer/checkpoint-32"
    checkpoint.mkdir(parents=True)
    lock = _source_lock()
    artifacts = lock["artifacts"]
    protocol = {
        "max_steps": 32,
        "save_steps": 32,
        "memory_loss_mode": "scene_state_generation_ce",
        "train_file": artifacts["tiny2"]["path"],
        "scene_state_source_manifest": {
            "file_sha256": artifacts["tiny2_source_manifest"]["sha256"]
        },
        "per_device_train_batch_size": 1,
        "gradient_accumulation_steps": 1,
        "frozen_mlp_activation_checkpointing": True,
    }
    (checkpoint / "trainer_state.json").write_text(
        json.dumps({"global_step": 32, "max_steps": 32}),
        encoding="utf-8",
    )
    (checkpoint / "training_protocol.json").write_text(
        json.dumps(protocol),
        encoding="utf-8",
    )
    for filename in (
        "delta_mem_adapter.pt",
        "delta_mem_config.json",
        "optimizer.pt",
        "scheduler.pt",
        "scene_state_identity_pairing_manifest.json",
        "rng_state.pth",
    ):
        (checkpoint / filename).write_bytes(b"test fixture\n")
    try:
        yield checkpoint
    finally:
        shutil.rmtree(fixture_root, ignore_errors=True)


def test_extend_requires_and_emits_one_completed_checkpoint_block(
    completed_tiny2_checkpoint: Path,
) -> None:
    environment = _launcher_environment(
        "tiny2",
        f"pytest_extend_{uuid.uuid4().hex}",
    )
    environment.update(
        {
            "RESUME_FROM_CHECKPOINT": str(completed_tiny2_checkpoint),
            "RESUME_MODE": "extend",
        }
    )

    arguments = _dry_run_arguments(_run_launcher(environment))

    assert _argument_value(arguments, "--resume-from-checkpoint") == str(
        completed_tiny2_checkpoint.resolve()
    )
    assert _argument_value(arguments, "--resume-mode") == "extend"
    assert _argument_value(arguments, "--max-steps") == "64"
    assert _argument_value(arguments, "--save-steps") == "32"
    assert "--initial-adapter-output-dir" not in arguments


def test_launcher_rejects_nonexplicit_resume_and_distributed_environment() -> None:
    environment = _launcher_environment(
        "tiny2",
        f"pytest_reject_{uuid.uuid4().hex}",
    )
    environment["RESUME_MODE"] = "extend"
    missing_checkpoint = _run_launcher(environment)
    assert missing_checkpoint.returncode != 0
    assert "resume_mode_requires_explicit_checkpoint" in missing_checkpoint.stderr

    environment = _launcher_environment(
        "tiny2",
        f"pytest_distributed_{uuid.uuid4().hex}",
    )
    environment["WORLD_SIZE"] = "2"
    distributed = _run_launcher(environment)
    assert distributed.returncode != 0
    assert "distributed_environment_is_forbidden variable=WORLD_SIZE" in (
        distributed.stderr
    )


def test_launcher_rejects_output_collision() -> None:
    run_name = f"pytest_collision_{uuid.uuid4().hex}"
    output = RUN_ROOT / f"scene_memory_v7_tiny2_{run_name}"
    output.mkdir(parents=True)
    try:
        result = _run_launcher(_launcher_environment("tiny2", run_name))
        assert result.returncode != 0
        assert "fresh_output_collision" in result.stderr
    finally:
        shutil.rmtree(output, ignore_errors=True)


def test_launcher_hard_locks_all_mutable_paths_to_the_2t_ssd() -> None:
    source = LAUNCHER.read_text(encoding="utf-8")

    assert 'SSD_ROOT="/run/media/xiaol/B214449214445C0B"' in source
    assert 'LOG_DIR="${RUN_ROOT}/logs"' in source
    assert 'LOG_FILE="${LOG_DIR}/scene_memory_v7_${DATASET_KIND}_${RUN_NAME}.log"' in source
    assert 'HF_HOME_LOCKED="${CACHE_ROOT}/huggingface"' in source
    assert 'XDG_CACHE_HOME_LOCKED="${CACHE_ROOT}/xdg"' in source
    assert 'TOKENIZED_DATASET_ROOT="${CACHE_ROOT}/tokenized"' in source
    assert 'TMPDIR_LOCKED="${CACHE_ROOT}/tmp/${DATASET_KIND}_${RUN_NAME}"' in source
    assert "path_must_stay_on_2t_ssd" in source
    assert "source_lock_self_hash_differs" in source
    assert "locked_artifact_hash_differs" in source
    assert "resume_checkpoint_is_not_a_completed_horizon" in source
    assert '"mode": "no_builtin_ce_plus_target_selected_shared_ce_v1"' in source
    assert (
        '"model_builtin_causal_lm_loss": "disabled_labels_omitted_v1"'
        in source
    )
    assert '"target_selected_before_fp32_cast_scatter_v1"' in source
    assert '"generation_margin_materialization": "separate_target_decision_v1"' in source
    assert '"objective_math_changed": False' in source


def test_non_dry_launcher_writes_only_execution_receipt_before_trainer(
    tmp_path: Path,
) -> None:
    run_name = f"pytest_nondry_{uuid.uuid4().hex}"
    output = RUN_ROOT / f"scene_memory_v7_tiny2_{run_name}"
    log = RUN_ROOT / "logs" / f"scene_memory_v7_tiny2_{run_name}.log"
    fake_python = tmp_path / "fake_python"
    fake_python.write_text(
        """#!/usr/bin/env python3
from pathlib import Path
import sys

args = sys.argv[1:]
output = Path(args[args.index("--output-dir") + 1])
entries = sorted(path.name for path in output.iterdir())
if entries != ["execution_metadata.json"]:
    raise SystemExit(f"unexpected files at trainer entry: {entries}")
max_steps = args[args.index("--max-steps") + 1]
(output / "training_summary.json").write_text("{}\\n", encoding="utf-8")
(output / "trainer" / f"checkpoint-{max_steps}").mkdir(parents=True)
""",
        encoding="utf-8",
    )
    fake_python.chmod(0o755)
    environment = _launcher_environment("tiny2", run_name)
    environment.update({"DRY_RUN": "0", "PYTHON_BIN": str(fake_python)})

    try:
        result = _run_launcher(environment)

        assert result.returncode == 0, result.stderr
        assert (output / "training_summary.json").is_file()
        assert (output / "trainer/checkpoint-32").is_dir()
        assert log.is_file()
        assert log.parent != output
    finally:
        shutil.rmtree(output, ignore_errors=True)
        log.unlink(missing_ok=True)
