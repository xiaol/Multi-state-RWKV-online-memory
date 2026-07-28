from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
from typing import Any
from unittest.mock import patch

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
LAUNCHER = (
    REPO_ROOT
    / "experiments/rethinking_rwkv_ms_gemma/train_scene_failure_state.sh"
)


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _canonical_json_sha256(value: Any) -> str:
    serialized = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return _sha256_text(serialized)


def _model_artifacts(model_path: Path) -> dict[str, Any]:
    def record(path: Path) -> dict[str, Any]:
        return {
            "relative_path": path.relative_to(model_path).as_posix(),
            "size_bytes": path.stat().st_size,
            "sha256": _sha256_file(path),
        }

    weights = [record(model_path / "model.safetensors")]
    runtime_artifacts = [
        record(model_path / name) for name in ("config.json", "tokenizer.json")
    ]
    aggregate_payload = {
        "weights": weights,
        "runtime_artifacts": runtime_artifacts,
    }
    return {
        "root": str(model_path.resolve()),
        **aggregate_payload,
        "aggregate_sha256": _canonical_json_sha256(aggregate_payload),
    }


def _write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _write_pair_artifacts(
    tmp_path: Path,
    *,
    model_path: Path,
    row_count: int = 32,
) -> tuple[Path, Path]:
    pair_dir = tmp_path / "pairs"
    pair_dir.mkdir()
    rows = [
        {
            "messages": [
                {"role": "system", "content": "Find scene boundaries."},
                {"role": "user", "content": f"[P1] row {index} a\n[P2] row {index} b"},
                {"role": "assistant", "content": '{"boundaries":[1]}'},
            ]
        }
        for index in range(row_count)
    ]
    raw_rows = [
        json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        for row in rows
    ]
    train_path = pair_dir / "train.jsonl"
    train_path.write_text("\n".join(raw_rows) + "\n", encoding="utf-8")
    row_hashes = [_sha256_text(row) for row in raw_rows]
    prompt_hashes = [_sha256_text(row["messages"][1]["content"]) for row in rows]

    row_manifest_path = pair_dir / "train_manifest.jsonl"
    row_manifest_rows = [
        {
            "partition": "train",
            "source_split": "train",
            "source_line_index": index,
            "row_sha256": row_hashes[index],
            "prompt_sha256": prompt_hashes[index],
        }
        for index in range(len(rows))
    ]
    row_manifest_path.write_text(
        "\n".join(
            json.dumps(row, sort_keys=True, separators=(",", ":"))
            for row in row_manifest_rows
        )
        + "\n",
        encoding="utf-8",
    )

    base_model_artifacts = _model_artifacts(model_path)
    producer_fingerprint_payload = {
        "base_model": str(model_path.resolve()),
        "base_model_artifacts": base_model_artifacts,
    }
    producer_manifest = {
        "schema": "rwkv_ms_scene_train_base_eval.v1",
        "fingerprint": _canonical_json_sha256(producer_fingerprint_payload),
        "fingerprint_payload": producer_fingerprint_payload,
    }
    producer_manifest_path = pair_dir / "base_producer_manifest.json"
    _write_json(producer_manifest_path, producer_manifest)

    manifest = {
        "schema": "rwkv_ms_scene_failure_pairs.v1",
        "task": "scene-v4-current",
        "contract": {
            "failure_mining_split": "train",
            "holdout_source_split": "val",
            "test_policy": "provenance_and_overlap_audit_only; never emitted",
            "candidate_count": 64,
            "train_failure_count": 32,
            "episode_contract": {
                "messages": ["system", "user", "assistant"],
                "episode_recent_messages": 0,
            },
        },
        "config": {
            "candidate_count": 64,
            "train_failure_count": 32,
            "holdout_count": 1,
            "selection_seed": 42,
        },
        "sources": {
            "train": {"emitted_for_training": True, "emitted_for_holdout": False},
            "val": {"emitted_for_training": False, "emitted_for_holdout": True},
            "test": {"emitted_for_training": False, "emitted_for_holdout": False},
        },
        "partitions": {
            "train": {
                "source_split": "train",
                "rows": len(rows),
                "data": {"path": str(train_path), "sha256": _sha256_file(train_path)},
                "row_manifest": {
                    "path": str(row_manifest_path),
                    "sha256": _sha256_file(row_manifest_path),
                },
                "row_hashes_sha256": _canonical_json_sha256(row_hashes),
                "prompt_hashes_sha256": _canonical_json_sha256(prompt_hashes),
            },
            "holdout": {"source_split": "val", "rows": 1},
        },
        "validation": {
            "row_sha256_pairwise_disjoint": True,
            "exact_user_prompt_sha256_pairwise_disjoint": True,
            "all_base_records_joined_to_train_by_row_sha256": True,
            "base_gold_matches_train_source": True,
            "train_holdout_row_sha256_disjoint": True,
            "train_holdout_exact_user_prompt_sha256_disjoint": True,
            "output_rows_preserve_source_serialization": True,
            "output_rows_have_exactly_three_messages": True,
            "candidate_count_matches_protocol": True,
            "train_failure_count_matches_protocol": True,
            "base_records_match_producer_selection": True,
            "base_records_share_producer_fingerprint": True,
            "producer_summary_complete": True,
            "failure_selection_uses_eval_record_order": False,
            "holdout_selection_uses_model_output": False,
            "test_rows_emitted": 0,
        },
        "base_train_evaluation": {
            "selected_task_records": 64,
            "eligible_failures": 40,
            "selected_failures": len(rows),
            "producer_bundle": {
                "base_model": {
                    "path": str(model_path.resolve()),
                    "artifact_aggregate_sha256": base_model_artifacts[
                        "aggregate_sha256"
                    ],
                },
                "manifest": {
                    "path": str(producer_manifest_path),
                    "sha256": _sha256_file(producer_manifest_path),
                },
            },
        },
    }
    manifest_path = pair_dir / "manifest.json"
    _write_json(manifest_path, manifest)
    return train_path, manifest_path


def _launcher_environment(
    tmp_path: Path,
    *,
    row_count: int = 32,
) -> tuple[dict[str, str], Path, Path]:
    model_path = tmp_path / "model"
    model_path.mkdir()
    _write_json(model_path / "config.json", {"text_config": {"num_hidden_layers": 42}})
    (model_path / "model.safetensors").write_bytes(b"local test model weights\n")
    _write_json(model_path / "tokenizer.json", {"version": "test"})
    train_path, manifest_path = _write_pair_artifacts(
        tmp_path,
        model_path=model_path,
        row_count=row_count,
    )
    output_path = tmp_path / "output"
    environment = os.environ.copy()
    for name in ("RESUME_FROM_CHECKPOINT", "WARM_START_FROM_CHECKPOINT"):
        environment.pop(name, None)
    environment.update(
        {
            "PYTHON_BIN": "/bin/true",
            "VALIDATION_PYTHON_BIN": sys.executable,
            "MODEL_PATH": str(model_path),
            "TRAIN_FILE": str(train_path),
            "TRAIN_SHA256": _sha256_file(train_path),
            "PAIR_MANIFEST": str(manifest_path),
            "OUTPUT_DIR": str(output_path),
            "RUN_ROOT": str(tmp_path / "runs"),
            "LOG_FILE": str(tmp_path / "scene-failure.log"),
            "HF_CACHE_DIR": str(tmp_path / "hf-cache"),
            "TOKENIZED_DATASET_ROOT": str(tmp_path / "tokenized"),
            "CACHE_ROOT": str(tmp_path / "cache"),
            "DRY_RUN": "1",
        }
    )
    return environment, output_path, manifest_path


def _dry_run_arguments(result: subprocess.CompletedProcess[str]) -> list[str]:
    command_line = result.stdout.strip().splitlines()[-1]
    return shlex.split(command_line)


def _argument_value(arguments: list[str], name: str) -> str:
    return arguments[arguments.index(name) + 1]


def _write_fake_trainer(tmp_path: Path) -> Path:
    fake_trainer = tmp_path / "fake-python"
    fake_trainer.write_text(
        f"""#!{sys.executable}
from __future__ import annotations

import json
import os
from pathlib import Path
import sys

Path(os.environ["FAKE_INVOCATION_PATH"]).write_text(
    json.dumps(sys.argv) + "\\n",
    encoding="utf-8",
)
arguments = sys.argv[1:]
snapshot_dir = Path(
    arguments[arguments.index("--initial-adapter-output-dir") + 1]
)
snapshot_dir.mkdir(parents=True, exist_ok=False)
for filename in (
    "delta_mem_adapter.pt",
    "delta_mem_config.json",
    "training_protocol.json",
    "initial_adapter_manifest.json",
):
    (snapshot_dir / filename).write_text("non-training placeholder\\n", encoding="utf-8")
""",
        encoding="utf-8",
    )
    fake_trainer.chmod(0o755)
    return fake_trainer


def test_dry_run_emits_locked_all42_state_training_contract(tmp_path: Path) -> None:
    environment, output_path, _ = _launcher_environment(tmp_path)

    result = subprocess.run(
        ["bash", str(LAUNCHER)],
        cwd=REPO_ROOT,
        env=environment,
        check=True,
        capture_output=True,
        text=True,
    )

    arguments = _dry_run_arguments(result)
    assert arguments[:3] == ["/bin/true", "-m", "deltamem.train.delta_sft"]
    assert _argument_value(arguments, "--target-layers") == ",".join(
        str(index) for index in range(42)
    )
    assert _argument_value(arguments, "--delta-heads") == "q,o"
    assert _argument_value(arguments, "--rwkv-ms-semantics-version") == "2"
    assert _argument_value(arguments, "--rank") == "4"
    assert _argument_value(arguments, "--episode-recent-messages") == "0"
    assert _argument_value(arguments, "--max-write-length") == "1280"
    assert "--no-episode-read-write-enabled" in arguments
    assert _argument_value(arguments, "--memory-loss-mode") == "context_dropout_ce"
    assert _argument_value(arguments, "--memory-dropout-no-memory-prob") == "0"
    assert _argument_value(arguments, "--memory-dropout-state-only-prob") == "0"
    assert _argument_value(arguments, "--memory-base-kl-weight") == "0"
    assert _argument_value(arguments, "--memory-kl-weight") == "0"
    assert _argument_value(arguments, "--memory-contrast-weight") == "0"
    assert _argument_value(arguments, "--memory-representation-weight") == "0"
    assert _argument_value(arguments, "--learning-rate") == "5e-4"
    assert _argument_value(arguments, "--max-steps") == "128"
    assert _argument_value(arguments, "--save-steps") == "32"
    assert _argument_value(arguments, "--save-total-limit") == "4"
    assert _argument_value(arguments, "--validation-split-ratio") == "0"
    assert "--no-load-best-model-at-end" in arguments
    assert "--frozen-mlp-activation-checkpointing" in arguments
    assert _argument_value(arguments, "--initial-adapter-output-dir") == str(
        output_path / "initial_adapter"
    )
    assert not output_path.exists()

    from deltamem.train import delta_sft

    with patch.object(sys, "argv", ["delta_sft", *arguments[3:]]):
        parsed = delta_sft.parse_args()
    assert parsed.target_layers == ",".join(str(index) for index in range(42))
    assert parsed.delta_heads == "q,o"
    assert parsed.frozen_mlp_activation_checkpointing is True
    assert parsed.initial_adapter_output_dir == output_path / "initial_adapter"


def test_smoke_mode_changes_only_horizon_and_zeroes_warmup(tmp_path: Path) -> None:
    environment, _, _ = _launcher_environment(tmp_path)
    production = subprocess.run(
        ["bash", str(LAUNCHER)],
        cwd=REPO_ROOT,
        env=environment,
        check=True,
        capture_output=True,
        text=True,
    )
    smoke_environment = dict(environment)
    smoke_environment["RUN_MODE"] = "smoke"
    smoke = subprocess.run(
        ["bash", str(LAUNCHER)],
        cwd=REPO_ROOT,
        env=smoke_environment,
        check=True,
        capture_output=True,
        text=True,
    )

    production_arguments = _dry_run_arguments(production)
    smoke_arguments = _dry_run_arguments(smoke)
    assert _argument_value(production_arguments, "--max-steps") == "128"
    assert _argument_value(smoke_arguments, "--max-steps") == "1"
    assert _argument_value(production_arguments, "--warmup-ratio") == "0.0625"
    assert _argument_value(smoke_arguments, "--warmup-ratio") == "0"
    assert _argument_value(smoke_arguments, "--save-steps") == "32"
    normalized_smoke = list(smoke_arguments)
    normalized_smoke[normalized_smoke.index("--max-steps") + 1] = "128"
    normalized_smoke[normalized_smoke.index("--warmup-ratio") + 1] = "0.0625"
    assert normalized_smoke == production_arguments


def test_smoke_can_override_data_seed_but_production_cannot(tmp_path: Path) -> None:
    environment, _, _ = _launcher_environment(tmp_path)
    environment["DATA_SEED"] = "28"

    production = subprocess.run(
        ["bash", str(LAUNCHER)],
        cwd=REPO_ROOT,
        env=environment,
        capture_output=True,
        text=True,
    )
    assert production.returncode != 0
    assert "production_data_seed_must_be_42" in production.stderr

    smoke_environment = dict(environment)
    smoke_environment["RUN_MODE"] = "smoke"
    smoke = subprocess.run(
        ["bash", str(LAUNCHER)],
        cwd=REPO_ROOT,
        env=smoke_environment,
        check=True,
        capture_output=True,
        text=True,
    )
    smoke_arguments = _dry_run_arguments(smoke)
    assert _argument_value(smoke_arguments, "--data-seed") == "28"


def test_non_training_fake_run_writes_atomic_launch_provenance(tmp_path: Path) -> None:
    environment, output_path, manifest_path = _launcher_environment(tmp_path)
    fake_trainer = _write_fake_trainer(tmp_path)
    invocation_path = tmp_path / "fake-invocation.json"
    environment.update(
        {
            "DRY_RUN": "0",
            "PYTHON_BIN": str(fake_trainer),
            "FAKE_INVOCATION_PATH": str(invocation_path),
        }
    )

    subprocess.run(
        ["bash", str(LAUNCHER)],
        cwd=REPO_ROOT,
        env=environment,
        check=True,
        capture_output=True,
        text=True,
    )

    launch_manifest_path = output_path / "launch_manifest.json"
    launch_manifest = json.loads(launch_manifest_path.read_text(encoding="utf-8"))
    assert launch_manifest["schema"] == "rwkv_ms_scene_failure_launch.v1"
    assert launch_manifest["run_mode"] == "production"
    assert launch_manifest["fresh_run"] is True
    assert launch_manifest["train_rows"] == 32
    assert launch_manifest["max_steps"] == 128
    assert launch_manifest["warmup_ratio"] == 0.0625
    assert launch_manifest["effective_passes"] == 4
    assert launch_manifest["production_reference"] == {
        "max_steps": 128,
        "warmup_ratio": 0.0625,
        "save_steps": 32,
        "checkpoint_steps": [32, 64, 96, 128],
        "effective_passes": 4,
    }
    assert launch_manifest["initial_adapter"] == {
        "required": True,
        "path": str((output_path / "initial_adapter").resolve()),
        "expected_global_step": 0,
    }

    invoked_argv = json.loads(invocation_path.read_text(encoding="utf-8"))
    assert launch_manifest["command"]["argv"] == invoked_argv
    assert launch_manifest["command"]["argv_sha256"] == _canonical_json_sha256(
        invoked_argv
    )
    assert launch_manifest["command"]["shell"] == shlex.join(invoked_argv)
    assert _argument_value(invoked_argv, "--initial-adapter-output-dir") == str(
        output_path / "initial_adapter"
    )

    model_weight_path = Path(environment["MODEL_PATH"]) / "model.safetensors"
    model_weights = launch_manifest["artifacts"]["model_weights"]
    expected_weight_content = [
        {
            "relative_path": "model.safetensors",
            "bytes": model_weight_path.stat().st_size,
            "sha256": _sha256_file(model_weight_path),
        }
    ]
    assert model_weights["file_count"] == 1
    assert model_weights["files"] == [
        {
            "path": str(model_weight_path.resolve()),
            **expected_weight_content[0],
        }
    ]
    assert model_weights["aggregate_sha256"] == _canonical_json_sha256(
        expected_weight_content
    )
    assert launch_manifest["artifacts"]["pair_manifest"]["sha256"] == _sha256_file(
        manifest_path
    )
    expected_behavior_sources = {
        "delta_entrypoint": REPO_ROOT / "deltamem/core/delta.py",
        "delta_implementation": REPO_ROOT / "deltamem/core/delta_impl.py",
        "rwkv_ms_core": REPO_ROOT / "deltamem/core/hrm_rwkv7.py",
        "backbone_compatibility": REPO_ROOT / "deltamem/core/backbone_compat.py",
        "affine_scan": REPO_ROOT / "deltamem/kernels/affine_scan.py",
        "chat_templates": REPO_ROOT / "deltamem/chat_templates.py",
    }
    assert launch_manifest["artifacts"]["behavior_sources"] == {
        label: {"path": str(path.resolve()), "sha256": _sha256_file(path)}
        for label, path in expected_behavior_sources.items()
    }

    recorded_manifest_sha256 = launch_manifest.pop("manifest_sha256")
    assert recorded_manifest_sha256 == _canonical_json_sha256(launch_manifest)
    assert sorted(path.name for path in output_path.iterdir()) == [
        "initial_adapter",
        "launch_manifest.json",
    ]


def test_launcher_rejects_nonempty_output_even_in_dry_run(tmp_path: Path) -> None:
    environment, output_path, _ = _launcher_environment(tmp_path)
    output_path.mkdir()
    (output_path / "partial.txt").write_text("partial", encoding="utf-8")

    result = subprocess.run(
        ["bash", str(LAUNCHER)],
        cwd=REPO_ROOT,
        env=environment,
        capture_output=True,
        text=True,
    )

    assert result.returncode != 0
    assert "output_directory_must_be_empty" in result.stderr


def test_launcher_rejects_train_manifest_source_leakage(tmp_path: Path) -> None:
    environment, _, manifest_path = _launcher_environment(tmp_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    row_manifest_path = Path(manifest["partitions"]["train"]["row_manifest"]["path"])
    records = [json.loads(line) for line in row_manifest_path.read_text().splitlines()]
    records[0]["source_split"] = "val"
    row_manifest_path.write_text(
        "\n".join(json.dumps(row, sort_keys=True, separators=(",", ":")) for row in records)
        + "\n",
        encoding="utf-8",
    )
    manifest["partitions"]["train"]["row_manifest"]["sha256"] = _sha256_file(
        row_manifest_path
    )
    _write_json(manifest_path, manifest)

    result = subprocess.run(
        ["bash", str(LAUNCHER)],
        cwd=REPO_ROOT,
        env=environment,
        capture_output=True,
        text=True,
    )

    assert result.returncode != 0
    assert "non-train source_split" in result.stderr


def test_launcher_rejects_dataset_checksum_drift(tmp_path: Path) -> None:
    environment, _, _ = _launcher_environment(tmp_path)
    environment["TRAIN_SHA256"] = "0" * 64

    result = subprocess.run(
        ["bash", str(LAUNCHER)],
        cwd=REPO_ROOT,
        env=environment,
        capture_output=True,
        text=True,
    )

    assert result.returncode != 0
    assert "dataset_checksum_mismatch" in result.stderr


def test_launcher_rejects_training_model_drift_from_failure_mining_base(
    tmp_path: Path,
) -> None:
    environment, _, _ = _launcher_environment(tmp_path)
    model_weight_path = Path(environment["MODEL_PATH"]) / "model.safetensors"
    model_weight_path.write_bytes(b"different local model weights\n")

    result = subprocess.run(
        ["bash", str(LAUNCHER)],
        cwd=REPO_ROOT,
        env=environment,
        capture_output=True,
        text=True,
    )

    assert result.returncode != 0
    assert "artifact aggregate differs from the failure-mining base model" in result.stderr


def test_launcher_rejects_any_train_partition_other_than_32_rows(
    tmp_path: Path,
) -> None:
    environment, _, _ = _launcher_environment(tmp_path, row_count=31)

    result = subprocess.run(
        ["bash", str(LAUNCHER)],
        cwd=REPO_ROOT,
        env=environment,
        capture_output=True,
        text=True,
    )

    assert result.returncode != 0
    assert "train partition must contain exactly 32 failures" in result.stderr


@pytest.mark.parametrize(
    ("field_path", "invalid_value", "expected_error"),
    [
        (("config", "candidate_count"), 63, "config.candidate_count must be 64"),
        (
            ("config", "train_failure_count"),
            31,
            "config.train_failure_count must be 32",
        ),
        (
            ("base_train_evaluation", "selected_task_records"),
            63,
            "base evaluation must contain exactly 64 selected task rows",
        ),
        (
            ("base_train_evaluation", "eligible_failures"),
            31,
            "base evaluation must contain at least 32 eligible failures",
        ),
    ],
)
def test_launcher_rejects_invalid_64_by_32_pair_provenance(
    tmp_path: Path,
    field_path: tuple[str, str],
    invalid_value: int,
    expected_error: str,
) -> None:
    environment, _, manifest_path = _launcher_environment(tmp_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest[field_path[0]][field_path[1]] = invalid_value
    _write_json(manifest_path, manifest)

    result = subprocess.run(
        ["bash", str(LAUNCHER)],
        cwd=REPO_ROOT,
        env=environment,
        capture_output=True,
        text=True,
    )

    assert result.returncode != 0
    assert expected_error in result.stderr
