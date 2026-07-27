from __future__ import annotations

import os
from pathlib import Path
import subprocess


REPO_ROOT = Path(__file__).resolve().parents[2]
BASE_LAUNCHER = (
    REPO_ROOT / "experiments/rethinking_rwkv_ms_gemma/train_all42_gated_memory.sh"
)
HYBRID_LAUNCHER = (
    REPO_ROOT
    / "experiments/rethinking_rwkv_ms_gemma/train_all42_residual_hybrid_w8_ablation.sh"
)


def _launcher_environment(tmp_path: Path) -> tuple[dict[str, str], Path]:
    capture_path = tmp_path / "args.txt"
    fake_python = tmp_path / "python"
    fake_python.write_text(
        "#!/usr/bin/env bash\nprintf '%s\\n' \"$@\" > \"$CAPTURE_PATH\"\n"
    )
    fake_python.chmod(0o755)
    environment = os.environ.copy()
    environment.update(
        {
            "PYTHON_BIN": str(fake_python),
            "MODEL_PATH": str(tmp_path / "model"),
            "TRAIN_FILE": str(tmp_path / "train.jsonl"),
            "OUTPUT_DIR": str(tmp_path / "output"),
            "HF_CACHE_DIR": str(tmp_path / "hf-cache"),
            "TOKENIZED_DATASET_ROOT": str(tmp_path / "tokenized"),
            "CAPTURE_PATH": str(capture_path),
        }
    )
    return environment, capture_path


def _argument_value(arguments: list[str], name: str) -> str:
    return arguments[arguments.index(name) + 1]


def test_hybrid_launcher_emits_ability_first_warm_start_contract(tmp_path: Path) -> None:
    environment, capture_path = _launcher_environment(tmp_path)
    environment["WARM_START_FROM_CHECKPOINT"] = str(
        tmp_path / "source" / "checkpoint-416"
    )

    subprocess.run(
        [str(HYBRID_LAUNCHER)],
        cwd=REPO_ROOT,
        env=environment,
        check=True,
        capture_output=True,
        text=True,
    )

    arguments = capture_path.read_text().splitlines()
    assert "--resume-from-checkpoint" not in arguments
    assert _argument_value(arguments, "--warm-start-from-checkpoint").endswith(
        "source/checkpoint-416"
    )
    assert _argument_value(arguments, "--warm-start-mode") == (
        "residual_hybrid_w8_ablation"
    )
    assert _argument_value(arguments, "--memory-fusion-placement") == (
        "post_attention_residual_hybrid"
    )
    assert _argument_value(arguments, "--memory-fusion-residual-scale") == "0.01"
    assert _argument_value(arguments, "--memory-fusion-residual-scale-max") == "0.02"
    assert _argument_value(arguments, "--memory-loss-mode") == "content_contrast_ce"
    assert _argument_value(arguments, "--memory-contrast-weight") == "0.25"
    assert _argument_value(arguments, "--memory-margin") == "0.5"
    assert _argument_value(arguments, "--memory-representation-weight") == "0.1"
    assert _argument_value(arguments, "--max-steps") == "32"
    assert _argument_value(arguments, "--num-train-epochs") == "1"
    assert _argument_value(arguments, "--target-layers") == ",".join(
        str(layer_index) for layer_index in range(42)
    )


def test_existing_output_is_allowed_only_for_exact_resume(tmp_path: Path) -> None:
    environment, capture_path = _launcher_environment(tmp_path)
    output_dir = Path(environment["OUTPUT_DIR"])
    output_dir.mkdir()

    rejected = subprocess.run(
        [str(BASE_LAUNCHER)],
        cwd=REPO_ROOT,
        env=environment,
        capture_output=True,
        text=True,
    )
    assert rejected.returncode != 0
    assert "only for an exact checkpoint resume" in rejected.stderr

    environment["RESUME_FROM_CHECKPOINT"] = str(output_dir / "trainer/checkpoint-16")
    environment["RESUME_MODE"] = "exact"
    subprocess.run(
        [str(BASE_LAUNCHER)],
        cwd=REPO_ROOT,
        env=environment,
        check=True,
        capture_output=True,
        text=True,
    )
    arguments = capture_path.read_text().splitlines()
    assert _argument_value(arguments, "--resume-from-checkpoint").endswith(
        "trainer/checkpoint-16"
    )
    assert _argument_value(arguments, "--resume-mode") == "exact"
