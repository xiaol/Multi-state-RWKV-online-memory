from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess

import pytest

from experiments.rethinking_rwkv_ms_gemma import scene_memory_v6_data_contract as data_contract
from experiments.rethinking_rwkv_ms_gemma import scene_memory_v6_launch_contract as launch
from experiments.rethinking_rwkv_ms_gemma import scene_memory_v6_run_audit as audit


REPO_ROOT = Path(__file__).resolve().parents[2]
LAUNCHER = REPO_ROOT / "experiments/rethinking_rwkv_ms_gemma/train_scene_memory_v6.sh"
TOKENIZATION_LOCK = (
    REPO_ROOT
    / "experiments/rethinking_rwkv_ms_gemma/scene_memory_v6_tokenized_cache_lock.json"
)


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=True, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def locked_command(run_mode: str, run_attempt: str = "run1") -> list[str]:
    stage = launch.STAGES[run_mode]
    output = launch.EXPECTED_RUN_ROOT / launch.expected_run_name(
        run_mode,
        run_attempt,
    )
    values = {
        "--model-path": str(launch.EXPECTED_MODEL),
        "--train-file": str(launch.EXPECTED_TRAIN),
        "--output-dir": str(output),
        "--initial-adapter-output-dir": str(output / "initial_adapter"),
        "--hf-cache-dir": str(launch.EXPECTED_HF_CACHE_DIR),
        "--device": "cuda:0",
        "--dtype": "bfloat16",
        "--attn-implementation": "sdpa",
        "--memory-backend": "rwkv_ms",
        "--rwkv-ms-num-states": "4",
        "--rwkv-ms-chunk-size": "128",
        "--rwkv-ms-boundary-mode": "fixed_chunk",
        "--rwkv-ms-erase-gate": "1.0",
        "--rwkv-ms-read-top-k": "0",
        "--rwkv-ms-output-init-scale": "0.02",
        "--rwkv-ms-semantics-version": "2",
        "--rank": "4",
        "--alpha": "8",
        "--num-state-heads": "1",
        "--beta-bias-init": "0.0",
        "--state-update-mode": "standard",
        "--output-init": "base_slice_fixed",
        "--base-slice-ref-width": "8",
        "--delta-heads": "q,o",
        "--memory-fusion-mode": "add",
        "--memory-fusion-placement": "attention_output",
        "--delta-scale-init": "0.1",
        "--delta-scale-max": "0.5",
        "--delta-scale-granularity": "head",
        "--delta-scale-parameterization": "alpha_over_rank",
        "--online-gain": "0.2",
        "--target-layers": launch.EXPECTED_TARGET_LAYERS,
        "--memory-readout-mode": "delta",
        "--memory-write-source": "learned_hidden",
        "--memory-write-granularity": "token",
        "--training-mode": "episode",
        "--assistant-loss-mode": "final_assistant_only",
        "--episode-recent-messages": "0",
        "--max-length": "256",
        "--max-write-length": "1280",
        "--memory-loss-mode": "scene_state_identity_ce",
        "--scene-state-identity-margin": "0.5",
        "--scene-state-source-manifest": str(launch.EXPECTED_PAIR_MANIFEST),
        "--expected-scene-state-source-manifest-sha256": (
            launch.EXPECTED_PAIR_MANIFEST_SHA256
        ),
        "--scene-boundary-payload-ce-weight": "0",
        "--memory-base-kl-weight": "0",
        "--memory-contrast-weight": "0",
        "--memory-representation-weight": "0",
        "--memory-kl-weight": "0",
        "--memory-causal-weight": "0",
        "--memory-anchor-weight": "0",
        "--memory-recover-weight": "0",
        "--memory-dropout-no-memory-prob": "0",
        "--memory-dropout-state-only-prob": "0",
        "--write-sparsity-weight": "0",
        "--learning-rate": "5e-4",
        "--lr-scheduler-type": "constant_with_warmup",
        "--weight-decay": "0",
        "--max-steps": str(stage["max_steps"]),
        "--save-steps": str(stage["save_steps"]),
        "--save-total-limit": str(stage["save_total_limit"]),
        "--warmup-ratio": str(stage["warmup_ratio"]),
        "--validation-split-ratio": "0",
        "--seed": "42",
        "--data-seed": "42",
        "--train-sampler-seed": "42",
        "--per-device-train-batch-size": "1",
        "--gradient-accumulation-steps": "1",
        "--dataloader-num-workers": "0",
    }
    command = [str(launch.EXPECTED_PYTHON_BIN), "-m", "deltamem.train.delta_sft"]
    for flag, value in values.items():
        command.extend((flag, value))
    command.extend(
        (
            "--bf16",
            "--couple-lambda",
            "--trainable-delta-scale",
            "--no-episode-read-write-enabled",
            "--no-load-best-model-at-end",
            "--frozen-mlp-activation-checkpointing",
            "--tf32",
            "--rankwise-gates",
            "--no-delta-o-rmsnorm",
            "--no-tokenized-cache",
            "--log-delta-debug-stats",
        )
    )
    if run_mode == "prepare":
        command.append("--prepare-only")
    return command


def validate_command(command: list[str], run_mode: str) -> None:
    output = launch.EXPECTED_RUN_ROOT / launch.expected_run_name(run_mode, "run1")
    launch.validate_command(
        command,
        python_bin=launch.EXPECTED_PYTHON_BIN,
        run_mode=run_mode,
        output_dir=output,
        initial_adapter_dir=output / "initial_adapter",
        hf_cache_dir=launch.EXPECTED_HF_CACHE_DIR,
    )


def synthetic_pairing_manifest() -> dict[str, object]:
    pairs: list[dict[str, object]] = []
    for source in range(32):
        donor = source + 1 if source % 2 == 0 else source - 1
        pair_number = source // 2
        if pair_number < 12:
            source_boundary = source % 2
            donor_boundary = donor % 2
            stratum = "presence"
        else:
            source_boundary = donor_boundary = 1
            stratum = "same_cardinality_value"
        pairs.append(
            {
                "source_index": source,
                "donor_index": donor,
                "source_row_sha256": f"{source + 1:064x}",
                "donor_row_sha256": f"{donor + 1:064x}",
                "source_label_sha256": f"{source + 101:064x}",
                "donor_label_sha256": f"{donor + 101:064x}",
                "source_write_sha256": f"{source + 201:064x}",
                "donor_write_sha256": f"{donor + 201:064x}",
                "source_write_token_count": 100 + source,
                "donor_write_token_count": 100 + donor,
                "source_boundary_count": source_boundary,
                "donor_boundary_count": donor_boundary,
                "target_stratum": stratum,
                "write_token_count_delta": 1,
                "target_mode": "first_pair_distinguishing_semantic_token_v1",
                "causal_prefix_mode": launch.SCENE_STATE_IDENTITY_CAUSAL_PREFIX_MODE,
                "target_span_tokens": 1,
                "first_differing_semantic_ordinal": 0,
                "target_label_positions": [2],
                "donor_target_label_positions": [2],
                "target_predictor_positions": [1],
                "donor_target_predictor_positions": [1],
                "target_token_ids": [source + 1000],
                "donor_token_ids": [donor + 1000],
                "causal_prefix_token_count": 2,
                "causal_prefix_sha256": f"{pair_number + 401:064x}",
                "target_mask_sha256": f"{source + 301:064x}",
            }
        )
    pair_sha = launch.canonical_sha256(pairs)
    counts = {"presence": 24, "same_cardinality_value": 8, "cross_cardinality_value": 0}
    histogram = {"0": 12, "1": 20}
    train: dict[str, object] = {
        "split": "train",
        "pairing_version": launch.SCENE_STATE_IDENTITY_PAIRING_VERSION,
        "pairing_refinement": launch.SCENE_STATE_IDENTITY_PAIRING_REFINEMENT,
        "pairing_refinement_applied": True,
        "target_mode": "first_pair_distinguishing_semantic_token_v1",
        "causal_prefix_mode": launch.SCENE_STATE_IDENTITY_CAUSAL_PREFIX_MODE,
        "sample_count": 32,
        "pair_count": 16,
        "target_token_count": 32,
        "target_stratum_row_counts": counts,
        "source_boundary_count_histogram": histogram,
        "write_token_count_delta_max": 1,
        "write_token_count_delta_mean": 1.0,
        "write_token_count_delta_total": 16,
        "nearest_baseline_write_token_count_delta_max": 1,
        "nearest_baseline_write_token_count_delta_total": 16,
        "source_fingerprint": "source",
        "paired_fingerprint": "paired",
        "pairs_sha256": pair_sha,
        "pairs": pairs,
    }
    train["manifest_sha256"] = launch.canonical_sha256(train)
    manifest: dict[str, object] = {
        "schema_version": 2,
        "objective_version": launch.SCENE_STATE_IDENTITY_OBJECTIVE_VERSION,
        "pairing_version": launch.SCENE_STATE_IDENTITY_PAIRING_VERSION,
        "pairing_refinement": launch.SCENE_STATE_IDENTITY_PAIRING_REFINEMENT,
        "pairing_refinement_applied": True,
        "pairing_scope": "within_post_split_partition",
        "target_mode": "first_pair_distinguishing_semantic_token_v1",
        "causal_prefix_mode": launch.SCENE_STATE_IDENTITY_CAUSAL_PREFIX_MODE,
        "semantic_mask_mode": "top_level_boundaries_nonwhitespace_offset_overlap_v1",
        "semantic_loss_normalization": "selected_tokens_per_row_then_batch_mean_v1",
        "target_token_count": 32,
        "target_stratum_row_counts": counts,
        "source_boundary_count_histogram": histogram,
        "write_token_count_delta_max": 1,
        "write_token_count_delta_mean": 1.0,
        "write_token_count_delta_total": 16,
        "nearest_baseline_write_token_count_delta_max": 1,
        "nearest_baseline_write_token_count_delta_total": 16,
        "data_seed": 42,
        "tokenized_fingerprint": "tokenized",
        "tokenized_dataset_sha256": "f" * 64,
        "splits": {"train": train},
    }
    manifest["manifest_sha256"] = launch.canonical_sha256(manifest)
    return manifest


def bind_synthetic_pairing(
    monkeypatch: pytest.MonkeyPatch,
    manifest: dict[str, object],
) -> None:
    train = manifest["splits"]["train"]
    monkeypatch.setattr(
        launch,
        "EXPECTED_TARGET_STRATUM_ROW_COUNTS",
        manifest["target_stratum_row_counts"],
    )
    monkeypatch.setattr(
        launch,
        "EXPECTED_SOURCE_BOUNDARY_COUNT_HISTOGRAM",
        manifest["source_boundary_count_histogram"],
    )
    monkeypatch.setattr(
        launch,
        "EXPECTED_WRITE_TOKEN_COUNT_DELTA_MAX",
        manifest["write_token_count_delta_max"],
    )
    monkeypatch.setattr(
        launch,
        "EXPECTED_WRITE_TOKEN_COUNT_DELTA_MEAN",
        manifest["write_token_count_delta_mean"],
    )
    monkeypatch.setattr(
        launch,
        "EXPECTED_WRITE_TOKEN_COUNT_DELTA_TOTAL",
        manifest["write_token_count_delta_total"],
    )
    monkeypatch.setattr(
        launch,
        "EXPECTED_IDENTITY_PAIRS_SHA256",
        train["pairs_sha256"],
    )
    monkeypatch.setattr(
        launch,
        "EXPECTED_CAUSAL_PREFIX_TOKEN_COUNT_HISTOGRAM",
        {"2": 32},
    )
    monkeypatch.setattr(
        launch,
        "EXPECTED_CAUSAL_PREFIX_SHA256_SET",
        {row["causal_prefix_sha256"] for row in train["pairs"]},
    )


@pytest.fixture
def isolated_source_lock_contract(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> dict[str, object]:
    repo = (tmp_path / "repo").resolve()
    repo.mkdir()
    source_paths = {
        "launcher": "experiments/launcher.sh",
        "trainer": "deltamem/train/trainer.py",
    }
    sources: dict[str, Path] = {}
    for label, relative_path in source_paths.items():
        path = repo / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"source:{label}\n", encoding="utf-8")
        sources[label] = path

    data_root = tmp_path / "data"
    data_root.mkdir()
    data_paths: dict[str, Path] = {}
    data_artifacts: dict[str, tuple[Path, str]] = {}
    for label in ("train32", "hard32"):
        path = data_root / f"{label}.jsonl"
        path.write_text(f'{{"artifact":"{label}"}}\n', encoding="utf-8")
        data_paths[label] = path
        data_artifacts[label] = (path, launch.sha256_file(path))

    model = tmp_path / "model"
    model.mkdir()
    config = model / "config.json"
    config.write_text('{"model_type":"test"}\n', encoding="utf-8")
    weight = model / "model.safetensors"
    weight.write_bytes(b"test model weights\n")
    model_weight = {
        "relative_path": weight.name,
        "bytes": weight.stat().st_size,
        "sha256": launch.sha256_file(weight),
    }
    tokenizer_paths: dict[str, Path] = {}
    tokenizer_artifacts: dict[str, dict[str, object]] = {}
    for label, filename in (
        ("tokenizer_json", "tokenizer.json"),
        ("chat_template", "chat_template.jinja"),
    ):
        path = model / filename
        path.write_text(f"tokenizer:{label}\n", encoding="utf-8")
        tokenizer_paths[label] = path
        tokenizer_artifacts[label] = {
            "relative_path": filename,
            "bytes": path.stat().st_size,
            "sha256": launch.sha256_file(path),
        }

    source_lock = repo / "source-lock.json"
    runtime = {"python": "test-runtime"}
    monkeypatch.setattr(launch, "EXPECTED_REPO", repo)
    monkeypatch.setattr(launch, "EXPECTED_SOURCE_LOCK", source_lock)
    monkeypatch.setattr(launch, "EXPECTED_SOURCE_PATHS", source_paths)
    monkeypatch.setattr(launch, "EXPECTED_DATA_ARTIFACTS", data_artifacts)
    monkeypatch.setattr(launch, "EXPECTED_MODEL", model)
    monkeypatch.setattr(
        launch,
        "EXPECTED_MODEL_CONFIG_SHA256",
        launch.sha256_file(config),
    )
    monkeypatch.setattr(launch, "EXPECTED_MODEL_WEIGHT", model_weight)
    monkeypatch.setattr(
        launch,
        "EXPECTED_TOKENIZER_ARTIFACTS",
        tokenizer_artifacts,
    )
    monkeypatch.setattr(launch, "EXPECTED_RUNTIME_VERSIONS", runtime)
    monkeypatch.setattr(
        launch,
        "EXPECTED_PYTHON_BIN",
        Path(launch.sys.executable).resolve(),
    )
    monkeypatch.setattr(launch, "current_runtime_versions", lambda: dict(runtime))
    return {
        "repo": repo,
        "source_lock": source_lock,
        "sources": sources,
        "data": data_paths,
        "config": config,
        "weight": weight,
        "tokenizers": tokenizer_paths,
    }


def replace_with_symlink(path: Path) -> None:
    target = path.with_name(f"{path.name}.target")
    path.rename(target)
    path.symlink_to(target)


def test_source_lock_writer_is_deterministic_and_self_validating(
    isolated_source_lock_contract: dict[str, object],
) -> None:
    repo = isolated_source_lock_contract["repo"]
    path = isolated_source_lock_contract["source_lock"]
    assert isinstance(repo, Path)
    assert isinstance(path, Path)

    first = launch.write_source_lock(repo, path)
    first_bytes = path.read_bytes()
    second = launch.write_source_lock(repo, path)
    assert second == first
    assert path.read_bytes() == first_bytes
    assert launch.validate_source_lock(
        repo,
        path,
        verify_model_weight=True,
        verify_runtime=True,
    ) == first

    assert set(first) == {
        "schema",
        "experiment",
        "repository",
        "runtime_versions",
        "sources",
        "data_artifacts",
        "model",
        "lock_sha256",
    }
    assert all(
        set(record) == {"relative_path", "bytes", "sha256"}
        for record in first["sources"].values()
    )
    assert all(
        set(record) == {"path", "bytes", "sha256"}
        for record in first["data_artifacts"].values()
    )
    model = first["model"]
    assert set(model) == {"path", "config", "weight", "tokenizer_artifacts"}
    assert set(model["config"]) == {"relative_path", "bytes", "sha256"}
    assert set(model["weight"]) == {"relative_path", "bytes", "sha256"}
    assert all(
        set(record) == {"relative_path", "bytes", "sha256"}
        for record in model["tokenizer_artifacts"].values()
    )
    unsigned = dict(first)
    assert unsigned.pop("lock_sha256") == launch.canonical_sha256(unsigned)


def test_source_lock_writer_rejects_missing_source(
    isolated_source_lock_contract: dict[str, object],
) -> None:
    source = next(iter(isolated_source_lock_contract["sources"].values()))
    source.unlink()
    with pytest.raises(launch.ContractError, match="behavior source .*missing"):
        launch.build_source_lock(isolated_source_lock_contract["repo"])


def test_source_lock_writer_rejects_source_symlink(
    isolated_source_lock_contract: dict[str, object],
) -> None:
    source = next(iter(isolated_source_lock_contract["sources"].values()))
    replace_with_symlink(source)
    with pytest.raises(launch.ContractError, match="behavior source .*symlink"):
        launch.build_source_lock(isolated_source_lock_contract["repo"])


@pytest.mark.parametrize("mutation", ["content", "symlink"])
def test_source_lock_writer_rejects_changed_or_symlinked_data(
    isolated_source_lock_contract: dict[str, object],
    mutation: str,
) -> None:
    data = next(iter(isolated_source_lock_contract["data"].values()))
    if mutation == "content":
        data.write_text("changed\n", encoding="utf-8")
    else:
        replace_with_symlink(data)
    with pytest.raises(launch.ContractError, match="(differs from lock|symlink)"):
        launch.build_source_lock(isolated_source_lock_contract["repo"])


@pytest.mark.parametrize("artifact", ["config", "weight", "tokenizer"])
@pytest.mark.parametrize("mutation", ["content", "symlink"])
def test_source_lock_writer_rejects_changed_or_symlinked_model_artifacts(
    isolated_source_lock_contract: dict[str, object],
    artifact: str,
    mutation: str,
) -> None:
    if artifact == "tokenizer":
        path = next(iter(isolated_source_lock_contract["tokenizers"].values()))
    else:
        path = isolated_source_lock_contract[artifact]
    if mutation == "content":
        path.write_bytes(path.read_bytes() + b"changed\n")
    else:
        replace_with_symlink(path)
    with pytest.raises(launch.ContractError, match="(differs from lock|symlink)"):
        launch.build_source_lock(isolated_source_lock_contract["repo"])


def test_source_lock_writer_rejects_symlinked_output(
    isolated_source_lock_contract: dict[str, object],
) -> None:
    output = isolated_source_lock_contract["source_lock"]
    target = output.with_name("unrelated.json")
    target.write_text("unchanged\n", encoding="utf-8")
    output.symlink_to(target)
    with pytest.raises(launch.ContractError, match="source-lock output is a symlink"):
        launch.write_source_lock(isolated_source_lock_contract["repo"], output)
    assert target.read_text(encoding="utf-8") == "unchanged\n"


def test_write_source_lock_cli_uses_only_the_bound_output(
    isolated_source_lock_contract: dict[str, object],
    capsys: pytest.CaptureFixture[str],
) -> None:
    repo = isolated_source_lock_contract["repo"]
    output = isolated_source_lock_contract["source_lock"]
    real_lock = REPO_ROOT / (
        "experiments/rethinking_rwkv_ms_gemma/scene_memory_v6_source_lock.json"
    )
    real_lock_before = real_lock.read_bytes()
    assert launch.main(
        [
            "write-source-lock",
            "--repo",
            str(repo),
            "--source-lock",
            str(output),
        ]
    ) == 0
    payload = json.loads(output.read_text(encoding="utf-8"))
    stdout = capsys.readouterr().out
    assert f"source_lock={output}" in stdout
    assert f"lock_sha256={payload['lock_sha256']}" in stdout
    assert real_lock.read_bytes() == real_lock_before


def test_write_source_lock_cli_has_no_weight_hash_bypass() -> None:
    parser = launch.build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(
            [
                "write-source-lock",
                "--repo",
                "/tmp/repo",
                "--source-lock",
                "/tmp/source-lock.json",
                "--skip-model-weight-hash",
            ]
        )


@pytest.mark.parametrize("run_mode", ["prepare", "smoke", "proof"])
def test_each_mode_has_a_valid_v2_command(run_mode: str) -> None:
    validate_command(locked_command(run_mode), run_mode)


def test_prepare_only_is_exclusive_to_prepare_mode() -> None:
    assert "--prepare-only" in locked_command("prepare")
    for run_mode in ("smoke", "proof"):
        command = locked_command(run_mode)
        assert "--prepare-only" not in command
        with pytest.raises(launch.ContractError, match="prepare-only switch differs"):
            validate_command([*command, "--prepare-only"], run_mode)


@pytest.mark.parametrize(
    ("flag", "bad_value"),
    [
        ("--delta-heads", "o"),
        ("--rank", "8"),
        ("--rwkv-ms-semantics-version", "1"),
        ("--target-layers", "0,1,2,3"),
        ("--memory-loss-mode", "context_dropout_ce"),
        ("--scene-state-identity-margin", "0.1"),
        ("--scene-boundary-payload-ce-weight", "4"),
        ("--gradient-accumulation-steps", "2"),
    ],
)
def test_command_rejects_topology_or_objective_drift(
    flag: str,
    bad_value: str,
) -> None:
    command = locked_command("proof")
    command[command.index(flag) + 1] = bad_value
    with pytest.raises(launch.ContractError, match="locked trainer option differs"):
        validate_command(command, "proof")


@pytest.mark.parametrize(
    "forbidden",
    [
        "--resume-from-checkpoint",
        "--warm-start-from-checkpoint",
        "--tokenized-cache",
        "--tokenized-dataset-dir",
        "--tokenized-dataset-root",
        "--expected-tokenized-dataset-sha256",
    ],
)
def test_command_rejects_reuse(forbidden: str) -> None:
    command = [*locked_command("proof"), forbidden, "/tmp/reused"]
    with pytest.raises(launch.ContractError, match="identity proof forbids"):
        validate_command(command, "proof")


def test_v2_stages_are_one_fresh_pass_with_two_proof_checkpoints() -> None:
    assert set(launch.STAGES) == {"prepare", "smoke", "proof"}
    assert launch.STAGES["prepare"]["optimization_updates"] == 0
    assert launch.STAGES["smoke"]["checkpoint_steps"] == [1]
    assert launch.STAGES["proof"]["source_rows_consumed"] == 32
    assert launch.STAGES["proof"]["checkpoint_steps"] == [16, 32]
    assert launch.expected_run_name("proof", "run3").endswith("_s32_run3")


def test_external_paths_are_hard_locked() -> None:
    output = launch.EXPECTED_RUN_ROOT / launch.expected_run_name("proof", "run2")
    launch.validate_external_paths(
        run_mode="proof",
        run_attempt="run2",
        python_bin=launch.EXPECTED_PYTHON_BIN,
        output_dir=output,
        initial_adapter_dir=output / "initial_adapter",
        hf_cache_dir=launch.EXPECTED_HF_CACHE_DIR,
    )
    with pytest.raises(launch.ContractError, match="output directory differs"):
        launch.validate_external_paths(
            run_mode="proof",
            run_attempt="run2",
            python_bin=launch.EXPECTED_PYTHON_BIN,
            output_dir=Path("/tmp/v6"),
            initial_adapter_dir=Path("/tmp/v6/initial_adapter"),
            hf_cache_dir=launch.EXPECTED_HF_CACHE_DIR,
        )


def test_tokenization_lock_requires_fresh_v2_identity_columns() -> None:
    payload = launch.validate_tokenization_lock(TOKENIZATION_LOCK)
    assert payload["persisted_cache_enabled"] is False
    assert payload["rebuild_each_fresh_run"] is True
    assert {
        "scene_state_boundary_count",
        "scene_state_donor_boundary_count",
        "scene_state_identity_target_stratum",
    } <= set(payload["required_generated_columns"])


def test_official_data_manifest_binds_zero_overlap_selected_slices(
    tmp_path: Path,
) -> None:
    payload = data_contract.build_official_contract()
    path = tmp_path / "data_contract_manifest.json"
    write_json(path, payload)
    validated = launch.validate_data_manifest(path)
    overlap = validated["selected_slice_overlap_audit"]
    assert overlap["passage_disjoint"] is True
    assert overlap["comparison"]["exact_normalized_paragraphs_shared"] == 0


def test_pairing_validator_checks_symmetric_strata_and_length_deltas(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest = synthetic_pairing_manifest()
    bind_synthetic_pairing(monkeypatch, manifest)
    path = tmp_path / "scene_state_identity_pairing_manifest.json"
    write_json(path, manifest)
    assert launch.validate_identity_pairing_manifest(path) == manifest


def test_pairing_validator_rejects_relabelled_stratum_even_when_rehashed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest = synthetic_pairing_manifest()
    bind_synthetic_pairing(monkeypatch, manifest)
    train = manifest["splits"]["train"]
    train["pairs"][0]["target_stratum"] = "cross_cardinality_value"
    train["pairs_sha256"] = launch.canonical_sha256(train["pairs"])
    train.pop("manifest_sha256")
    train["manifest_sha256"] = launch.canonical_sha256(train)
    manifest.pop("manifest_sha256")
    manifest["manifest_sha256"] = launch.canonical_sha256(manifest)
    monkeypatch.setattr(launch, "EXPECTED_IDENTITY_PAIRS_SHA256", train["pairs_sha256"])
    path = tmp_path / "scene_state_identity_pairing_manifest.json"
    write_json(path, manifest)
    with pytest.raises(launch.ContractError, match="target stratum differs"):
        launch.validate_identity_pairing_manifest(path)


def test_smoke_authorization_reads_adapter_change_field(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run_root = (tmp_path / "runs").resolve()
    monkeypatch.setattr(launch, "EXPECTED_RUN_ROOT", run_root)
    path = (
        run_root
        / launch.expected_run_name("smoke", "run4")
        / "run_audit_receipt.json"
    )
    path.parent.mkdir(parents=True)
    path.write_text("{}\n", encoding="utf-8")
    receipt = {
        "checkpoints": [
            {
                "step": 1,
                "history": {"records": 1, "identity_metrics_finite": True},
                "adapter_change": {
                    "changed_trainable_tensor_count": 1,
                    "changed_nontrainable_tensor_count": 0,
                    "inactive_kv_projection_tensors_unchanged": 168,
                    "inactive_kv_delta_scale_entries_unchanged": 84,
                    "required_changed_layer_coverage": {
                        f"category_{index}": 42 for index in range(22)
                    },
                },
            }
        ]
    }
    monkeypatch.setattr(audit, "validate_existing_run_receipt", lambda *args, **kwargs: receipt)
    assert launch.validate_smoke_authorization(path) == receipt


def test_launcher_is_the_failure32_v2_identity_proof() -> None:
    source = LAUNCHER.read_text(encoding="utf-8")
    assert "source1804" not in source
    assert "pairs_candidate64_failure32_holdout32_v1" in source
    assert "--memory-loss-mode scene_state_identity_ce" in source
    assert "--scene-boundary-payload-ce-weight 0" in source
    assert "--no-tokenized-cache" in source
    assert "--delta-heads q,o" in source
    assert "--target-layers" in source
    assert "validated_smoke_receipt_missing" in source
    pending = (
        "scene_memory_v6 identity %s trainer exited successfully; "
        "final audit pending."
    )
    success = "scene_memory_v6 identity %s completed successfully."
    assert source.index(pending) < source.index('"${RUN_AUDIT_TOOL}" audit-run')
    assert source.index('"${RUN_AUDIT_TOOL}" audit-run') < source.index(success)
    post_receipt = source[source.index('[[ -s "${RUN_AUDIT_RECEIPT}" ]]') :]
    assert 'tee -a "${LOG_FILE}"' not in post_receipt
    prepare_success = "Prepare-only identity proof completed: %s"
    prepare_receipt = source[source.index('[[ -s "${PREPARE_RECEIPT}" ]]') :]
    assert source.index('"${RUN_AUDIT_TOOL}" audit-prepare') < source.index(
        prepare_success
    )
    assert 'tee -a "${LOG_FILE}"' not in prepare_receipt.split("exit 0", 1)[0]


@pytest.mark.parametrize("audit_failure", [False, True], ids=["success", "audit-failure"])
def test_smoke_launcher_log_lifecycle_is_receipt_stable(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    audit_failure: bool,
) -> None:
    repo = tmp_path / "repo"
    script_dir = repo / "experiments/rethinking_rwkv_ms_gemma"
    script_dir.mkdir(parents=True)
    pair_root = tmp_path / "pairs"
    pair_root.mkdir()
    pair_manifest = pair_root / "manifest.json"
    train_file = pair_root / "train.jsonl"
    pair_manifest.write_text("{}\n", encoding="utf-8")
    train_file.write_text('{}\n', encoding="utf-8")
    model = tmp_path / "model"
    model.mkdir()
    external = tmp_path / "external"
    run_root = external / "delta_mem_outputs/novel_rwkv_ms_memory"
    run_root.mkdir(parents=True)
    prepare_root = run_root / (
        "scene_memory_v6_identityproof_all42_qo_r4_fail32_s32_run1_prepare"
    )
    prepare_root.mkdir()
    (prepare_root / "prepare_receipt.json").write_text("{}\n", encoding="utf-8")

    fake_python = tmp_path / "fake_python"
    objective_json = json.dumps(dict(audit.OBJECTIVE_PROTOCOL), sort_keys=True)
    fake_python.write_text(
        f'''#!/usr/bin/env python3
import hashlib
import json
import os
from pathlib import Path
import sys
import time

ARGS = sys.argv[1:]
OBJECTIVE = json.loads({objective_json!r})

def value(flag):
    return ARGS[ARGS.index(flag) + 1]

def canonical(payload):
    return hashlib.sha256(json.dumps(
        payload,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")).hexdigest()

def write_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\\n")

if ARGS[:2] == ["-m", "deltamem.train.delta_sft"]:
    if "--help" in ARGS:
        print(" ".join([
            "--prepare-only", "--initial-adapter-output-dir",
            "--scene-state-identity-margin", "--scene-state-source-manifest",
            "--expected-scene-state-source-manifest-sha256",
            "--scene-boundary-payload-ce-weight", "--train-sampler-seed",
            "--rwkv-ms-semantics-version", "--delta-heads", "--target-layers",
        ]))
    else:
        root = Path(value("--output-dir"))
        write_json(root / "trainer/checkpoint-1/checkpoint_receipt.json", {{"complete": True}})
        write_json(root / "training_summary.json", {{"complete": True}})
        print("synthetic trainer completed")
elif Path(ARGS[0]).name == "scene_memory_v6_data_contract.py":
    if "--summary" in ARGS:
        print("synthetic-data-contract")
    else:
        write_json(value("--output"), {{"complete": True}})
elif Path(ARGS[0]).name == "scene_memory_v6_launch_contract.py":
    if ARGS[1] == "write-launch-manifest":
        write_json(value("--launch-manifest"), {{"complete": True}})
elif Path(ARGS[0]).name == "scene_memory_v6_run_audit.py":
    action = ARGS[1]
    if action == "watch-checkpoints":
        checkpoint = Path(value("--run-root")) / "trainer/checkpoint-1/checkpoint_receipt.json"
        deadline = time.time() + 5
        while not checkpoint.is_file() and time.time() < deadline:
            time.sleep(0.01)
        if not checkpoint.is_file():
            raise SystemExit(2)
    elif action == "audit-run":
        if os.environ.get("FAKE_AUDIT_FAILURE") == "1":
            print("synthetic audit failure", file=sys.stderr)
            raise SystemExit(2)
        root = Path(value("--run-root")).resolve()
        log = Path(value("--log-file")).resolve()
        receipt = Path(value("--receipt"))
        log_record = {{
            "path": str(log),
            "bytes": log.stat().st_size,
            "file_sha256": hashlib.sha256(log.read_bytes()).hexdigest(),
        }}
        checkpoint_history = {{"synthetic": True}}
        checkpoint_change = {{"synthetic": True}}
        payload = {{
            "schema": {audit.RUN_RECEIPT_SCHEMA!r},
            "experiment": {audit.EXPERIMENT!r},
            "run_mode": "smoke",
            "run_root": str(root),
            "complete": True,
            "trainer_exit_code": 0,
            "tee_exit_code": 0,
            "training_processes_active": [],
            "hard32_only": True,
            "full170_authorized": False,
            "test_forbidden": True,
            "auditor": {{}},
            "launch": {{}},
            "data_contract": {{}},
            "source_lock": {{}},
            "pair_manifest": {{}},
            "identity_pairing_manifest": {{}},
            "log": log_record,
            "initial_adapter": {{
                "manifest": {{}}, "adapter": {{}}, "config": {{}}, "protocol": {{}},
            }},
            "checkpoints": [{{
                "step": 1,
                "receipt": {{
                    "path": str(root / "trainer/checkpoint-1/checkpoint_receipt.json"),
                }},
                "history": checkpoint_history,
                "adapter_change": checkpoint_change,
            }}],
            "completed_artifacts": {{
                "adapter": {{}}, "config": {{}}, "protocol": {{}},
                "training_summary": {{}},
            }},
            "objective": OBJECTIVE,
        }}
        payload["receipt_sha256"] = canonical(payload)
        write_json(receipt, payload)
        print("synthetic run audit completed")
''',
        encoding="utf-8",
    )
    fake_python.chmod(0o755)

    launcher = script_dir / LAUNCHER.name
    source = LAUNCHER.read_text(encoding="utf-8")
    replacements = {
        'REPO="/home/xiaol/X/Multi-state-RWKV-online-memory"': f'REPO="{repo}"',
        'PYTHON_BIN="/home/xiaol/X/delta-Mem/.venv/bin/python"': (
            f'PYTHON_BIN="{fake_python}"'
        ),
        'VALIDATION_PYTHON_BIN="python3"': (
            f'VALIDATION_PYTHON_BIN="{fake_python}"'
        ),
        'MODEL_PATH="/run/media/xiaol/B214449214445C0B/models/gemma/gemma-4-E4B-it"': (
            f'MODEL_PATH="{model}"'
        ),
        'PAIR_ROOT="/run/media/xiaol/B214449214445C0B/delta_mem_data/scene_failure_state/pairs_candidate64_failure32_holdout32_v1"': (
            f'PAIR_ROOT="{pair_root}"'
        ),
        'PAIR_MANIFEST_SHA256="2ceb291b9c21063164e30ca0b8b052798f8ba42d9a089a5abc78d1cb321dc008"': (
            f'PAIR_MANIFEST_SHA256="{launch.sha256_file(pair_manifest)}"'
        ),
        'TRAIN_SHA256="5f35f6ed41a2edaf88afee83626f17c34da38f5cb61cf4b6796a03eaae38f897"': (
            f'TRAIN_SHA256="{launch.sha256_file(train_file)}"'
        ),
        'EXTERNAL_ROOT="/run/media/xiaol/B214449214445C0B"': (
            f'EXTERNAL_ROOT="{external}"'
        ),
    }
    for original, replacement in replacements.items():
        assert original in source
        source = source.replace(original, replacement, 1)
    launcher.write_text(source, encoding="utf-8")
    launcher.chmod(0o755)
    for filename in (
        "scene_memory_v6_source_lock.json",
        "scene_memory_v6_tokenized_cache_lock.json",
        "scene_memory_v6_data_contract.py",
        "scene_memory_v6_launch_contract.py",
        "scene_memory_v6_run_audit.py",
    ):
        path = script_dir / filename
        if not path.exists():
            path.write_text("stub\n", encoding="utf-8")

    environment = dict(os.environ)
    environment.update(
        {
            "RUN_MODE": "smoke",
            "RUN_ATTEMPT": "run1",
            "PREPARE_AUTH_ATTEMPT": "run1",
            "DRY_RUN": "0",
            "FAKE_AUDIT_FAILURE": "1" if audit_failure else "0",
        }
    )
    completed = subprocess.run(
        [str(launcher)],
        cwd=repo,
        env=environment,
        capture_output=True,
        text=True,
        timeout=20,
        check=False,
    )
    run_name = "scene_memory_v6_identityproof_all42_qo_r4_fail32_smoke1_run1"
    output = run_root / run_name
    log = run_root / f"{run_name}.log"
    receipt_path = output / "run_audit_receipt.json"
    if audit_failure:
        assert completed.returncode == 2
        assert not receipt_path.exists()
        assert "completed successfully" not in completed.stdout
        assert log.read_text(encoding="utf-8").endswith(
            f"ERROR: run_audit_failed mode=smoke path={receipt_path}\n"
        )
        return

    assert completed.returncode == 0, completed.stderr
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    bound_log_bytes = log.read_bytes()
    assert receipt["log"]["bytes"] == len(bound_log_bytes)
    assert receipt["log"]["file_sha256"] == hashlib.sha256(
        bound_log_bytes
    ).hexdigest()
    assert "completed successfully" in completed.stdout
    assert "completed successfully" not in log.read_text(encoding="utf-8")
    assert log.read_text(encoding="utf-8").endswith("final audit pending.\n")

    real_validate_file_record = audit._validate_file_record

    def validate_only_bound_log(record: object, *, description: str) -> None:
        if description == "run log":
            real_validate_file_record(record, description=description)

    monkeypatch.setattr(audit, "_validate_file_record", validate_only_bound_log)
    monkeypatch.setattr(
        audit,
        "validate_existing_checkpoint_receipt",
        lambda *args, **kwargs: {
            "history": {"synthetic": True},
            "adapter_change": {"synthetic": True},
        },
    )
    assert audit.validate_existing_run_receipt(
        receipt_path,
        expected_run_mode="smoke",
        expected_run_root=output,
    ) == receipt
    assert log.read_bytes() == bound_log_bytes


def test_launch_manifest_copies_selected_overlap_evidence(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output = tmp_path / "run"
    for name in ("source.json", "token.json", "data.json"):
        (tmp_path / name).write_text("{}\n", encoding="utf-8")
    selected_overlap = {
        "passage_disjoint": True,
        "comparison": {"exact_normalized_paragraphs_shared": 0},
    }
    data = {
        "manifest_sha256": "data",
        "pair_manifest": {},
        "training_partition": {},
        "hard_evaluation_selection": {},
        "test_policy": {},
        "overlap_audit": {},
        "selected_slice_overlap_audit": selected_overlap,
    }
    monkeypatch.setattr(launch, "validate_external_paths", lambda **kwargs: None)
    monkeypatch.setattr(launch, "validate_source_lock", lambda *args, **kwargs: {"source": True})
    monkeypatch.setattr(launch, "validate_tokenization_lock", lambda path: {"token": True})
    monkeypatch.setattr(launch, "validate_data_manifest", lambda path: data)
    monkeypatch.setattr(launch, "validate_command", lambda *args, **kwargs: None)
    monkeypatch.setattr(launch, "current_runtime_versions", lambda: {})
    args = argparse.Namespace(
        output_dir=output,
        launch_manifest=output / "launch_manifest.json",
        run_mode="prepare",
        run_attempt="run1",
        python_bin=launch.EXPECTED_PYTHON_BIN,
        initial_adapter_dir=output / "initial_adapter",
        hf_cache_dir=launch.EXPECTED_HF_CACHE_DIR,
        repo=REPO_ROOT,
        source_lock=tmp_path / "source.json",
        tokenization_lock=tmp_path / "token.json",
        data_manifest=tmp_path / "data.json",
        prepare_receipt=tmp_path / "unused-prepare.json",
        smoke_receipt=tmp_path / "unused-smoke.json",
        skip_model_weight_hash=True,
        command=["trainer"],
    )
    payload = launch.write_launch_manifest(args)
    assert payload["data_contract"]["selected_slice_overlap_audit"] == selected_overlap
