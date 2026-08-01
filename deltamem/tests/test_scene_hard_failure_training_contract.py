from __future__ import annotations

from argparse import Namespace
from collections.abc import Mapping
import json
import os
from pathlib import Path
import shlex
import stat
import subprocess
import sys
import uuid

from datasets import Dataset
import pytest
import torch

from deltamem.train import delta_sft_experimental as trainer
from experiments.rethinking_rwkv_ms_gemma import (
    scene_hard_failure_train_contract as contract,
)
from experiments.rethinking_rwkv_ms_gemma import (
    scene_memory_v15_launch_contract as legacy_v15,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
HARD_FAILURE_LAUNCHER = (
    REPO_ROOT
    / "experiments/rethinking_rwkv_ms_gemma/train_scene_hard_failure.sh"
)
_DISTRIBUTED_ENVIRONMENT = (
    "WORLD_SIZE",
    "RANK",
    "LOCAL_RANK",
    "MASTER_ADDR",
    "MASTER_PORT",
)
_PROTECTED_EVALUATION_ENVIRONMENT_PREFIXES = (
    "HARD32",
    "VALIDATION",
    "TEST",
    "BENCHMARK",
)


@pytest.fixture(scope="module")
def live_data_contract() -> Mapping[str, object]:
    return contract.validate_data_contract()


@pytest.fixture(scope="module")
def trainer_schedule_binding() -> dict[str, object]:
    args = Namespace(
        scene_state_source_manifest=contract.SOURCE_MANIFEST,
        expected_scene_state_source_manifest_sha256=contract.FILE_SHA256[
            "source_manifest.json"
        ],
        train_file=contract.TRAIN_FILE,
    )
    binding = trainer._scene_state_hard_failure_curriculum_binding(args)
    assert binding is not None
    return binding


def test_hard_failure_pairs_are_reciprocal_and_cover_32_rows_exactly(
    live_data_contract: Mapping[str, object],
) -> None:
    canonical_pairs = {
        tuple(pair) for pair in live_data_contract["canonical_pairs"]
    }
    pair_manifest = json.loads(
        (contract.DATA_ROOT / "pair_manifest.json").read_text(encoding="utf-8")
    )
    donor_by_row = {
        int(entry["train_row_ordinal"]): int(entry["donor_train_row_ordinal"])
        for entry in pair_manifest["directed_pairs"]
    }

    assert len(canonical_pairs) == contract.PAIR_COUNT == 16
    assert sorted(row for pair in canonical_pairs for row in pair) == list(range(32))
    assert sorted(donor_by_row) == list(range(32))
    assert all(donor != row for row, donor in donor_by_row.items())
    assert all(donor_by_row[donor] == row for row, donor in donor_by_row.items())
    assert {
        tuple(sorted((row, donor))) for row, donor in donor_by_row.items()
    } == canonical_pairs
    assert live_data_contract["scheduled_ordinals"] == list(range(32))


def test_hard_failure_schedule_has_four_bound_deterministic_cycles(
    live_data_contract: Mapping[str, object],
) -> None:
    expected_schedule_sha256 = (
        "bd12f021fc238f644972758047e7850cd22301be93b484dbf9f38f2203adb249"
    )
    canonical_pairs = {
        tuple(pair) for pair in live_data_contract["canonical_pairs"]
    }
    cycles = [
        [tuple(pair) for pair in cycle]
        for cycle in live_data_contract["full_pair_cycles"]
    ]
    scheduled_pairs = [
        tuple(pair) for pair in live_data_contract["scheduled_pairs"]
    ]

    assert len(cycles) == contract.PAIR_CYCLES == 4
    assert len(scheduled_pairs) == contract.TOTAL_PAIR_PRESENTATIONS == 64
    assert scheduled_pairs == [pair for cycle in cycles for pair in cycle]
    assert len({tuple(cycle) for cycle in cycles}) == 4
    for cycle in cycles:
        assert len(cycle) == contract.PAIR_COUNT == 16
        assert len(set(cycle)) == 16
        assert set(cycle) == canonical_pairs
        assert sorted(row for pair in cycle for row in pair) == list(range(32))

    assert contract.load_pair_schedule() == contract.load_pair_schedule()
    assert contract.PAIR_SCHEDULE_SHA256 == expected_schedule_sha256
    assert live_data_contract["expected_schedule_sha256"] == expected_schedule_sha256
    assert live_data_contract["actual_schedule_sha256"] == expected_schedule_sha256


def test_one_pair_per_update_means_64_steps_and_64_checkpoints() -> None:
    assert contract.PAIR_PRESENTATIONS_PER_OPTIMIZER_STEP == 1
    assert contract.GRADIENT_ACCUMULATION_STEPS == 1
    assert contract.TOTAL_PAIR_PRESENTATIONS == 64
    assert contract.TOTAL_OPTIMIZER_STEPS == 64
    assert contract.CHECKPOINT_STEPS == tuple(range(1, 65))
    assert len(contract.CHECKPOINT_STEPS) == contract.TOTAL_OPTIMIZER_STEPS
    assert contract.SAVE_STEPS == 1
    assert contract.SAVE_TOTAL_LIMIT == 64


def test_generation_endpoints_bind_cycle_prefixes(
    live_data_contract: Mapping[str, object],
) -> None:
    endpoints = (16, 32, 48, 64)
    endpoint_map = {16: 1, 32: 2, 48: 3, 64: 4}
    scheduled_pairs = [
        tuple(pair) for pair in live_data_contract["scheduled_pairs"]
    ]
    cycles = [
        [tuple(pair) for pair in cycle]
        for cycle in live_data_contract["full_pair_cycles"]
    ]

    assert contract.GENERATION_ENDPOINT_STEPS == endpoints
    assert live_data_contract["generation_endpoint_by_step"] == endpoint_map
    for step, cycle_number in endpoint_map.items():
        assert scheduled_pairs[step - 16 : step] == cycles[cycle_number - 1]


def test_trainer_binds_the_fresh_64_update_schedule(
    trainer_schedule_binding: Mapping[str, object],
) -> None:
    assert trainer._SCENE_STATE_HARD_FAILURE_OBJECTIVE_VERSION == (
        contract.OBJECTIVE_VERSION
    )
    assert trainer._SCENE_STATE_HARD_FAILURE_OBJECTIVE_VERSION in (
        trainer._SCENE_STATE_RECIPROCAL_OBJECTIVE_VERSIONS
    )
    assert trainer._SCENE_STATE_HARD_FAILURE_OBJECTIVE_VERSION in (
        trainer._SCENE_STATE_CYCLE_OBJECTIVE_VERSIONS
    )
    assert trainer_schedule_binding["schema"] == (
        trainer._SCENE_MEMORY_HARD_FAILURE_CURRICULUM_SCHEMA
    )
    assert trainer_schedule_binding["total_steps"] == 64
    assert trainer_schedule_binding["checkpoint_steps"] == list(range(1, 65))
    assert trainer_schedule_binding["generation_endpoint_steps"] == [16, 32, 48, 64]
    assert len(trainer_schedule_binding["pair_indices"]) == 64
    assert len(trainer_schedule_binding["indices"]) == 64
    trainer._validate_scene_state_hard_failure_schedule(
        dict(trainer_schedule_binding)
    )


def test_trainer_smoke_selects_exactly_the_first_bound_pair(
    trainer_schedule_binding: Mapping[str, object],
) -> None:
    full_binding = dict(trainer_schedule_binding)
    smoke = trainer._scene_state_hard_failure_one_pair_smoke_binding(full_binding)

    assert smoke["total_steps"] == 1
    assert smoke["pair_indices"] == full_binding["pair_indices"][:1]
    assert smoke["indices"] == full_binding["indices"][:1]
    assert smoke["schedule_selection_mode"] == (
        trainer._SCENE_STATE_HARD_FAILURE_ONE_PAIR_SMOKE_SCHEDULE_SELECTION
    )
    trainer._validate_scene_state_hard_failure_one_pair_smoke_schedule(smoke)


def test_hard_failure_objective_backpropagates_pair_identity_hinge(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from deltamem.tests.test_scene_memory_cached_identity_training import (
        _run_cached_pair,
    )

    monkeypatch.setattr(
        trainer.DeltaMemTrainer,
        "_scene_state_hard_failure_record_pair_presentation",
        lambda *args, **kwargs: None,
        raising=False,
    )
    model, stats, backward_calls = _run_cached_pair(
        monkeypatch,
        objective_version=trainer._SCENE_STATE_HARD_FAILURE_OBJECTIVE_VERSION,
    )

    assert backward_calls == 4
    assert model.parameters_by_side.grad is not None
    assert model.parameters_by_side.grad.tolist() == pytest.approx([-1.5, -1.5])
    assert any(
        "pair_identity_hinge" in key and value > 0.0
        for key, value in stats.items()
    )


def test_first_hard_failure_update_produces_complete_checkpoint_audit(
    trainer_schedule_binding: Mapping[str, object],
) -> None:
    from deltamem.tests.test_scene_memory_cached_identity_training import (
        _v15_exact_audit_stats,
    )

    active = object.__new__(trainer.DeltaMemTrainer)
    active.scene_state_generation_objective_version = (
        trainer._SCENE_STATE_HARD_FAILURE_OBJECTIVE_VERSION
    )
    active.scene_state_hard_failure_one_pair_smoke = False
    active.scene_state_v15_one_pair_smoke = False
    active.train_schedule_binding = dict(trainer_schedule_binding)
    active._scene_state_cycle_retention_metric_sums = {}
    active._scene_state_cycle_retention_metric_presentations = 0
    active._scene_state_hard_failure_cycle_pairs = []
    active._scene_state_hard_failure_completed_updates = 0
    active._scene_state_hard_failure_row_observations = []
    active._scene_state_hard_failure_pair_observations = []
    active.state = Namespace(global_step=1)
    first_pair = trainer_schedule_binding["pair_indices"][0]
    source, donor = first_pair
    manifest_pairs: list[dict[str, object]] = [{} for _ in range(32)]
    manifest_pairs[source] = {
        "source_index": source,
        "donor_index": donor,
        "source_row_sha256": bytes([source] * 32).hex(),
        "donor_row_sha256": bytes([donor] * 32).hex(),
    }
    active.scene_state_identity_pairing_manifest = {
        "splits": {"train": {"pairs": manifest_pairs}}
    }
    row_hash = lambda ordinal: torch.full(
        (1, 32), ordinal, dtype=torch.uint8
    )
    stats = _v15_exact_audit_stats()

    active._scene_state_hard_failure_record_pair_presentation(
        torch.tensor([source]),
        torch.tensor([donor]),
        row_hash(source),
        row_hash(donor),
        stats,
    )
    averaged = active._scene_state_cycle_retention_aggregate_memory_stats(stats)
    payload = active._scene_state_hard_failure_row_audit_payload()

    assert averaged["scene_generation_hard_failure_cycle_pair_presentations"] == 1.0
    assert averaged["scene_generation_hard_failure_optimizer_step"] == 1.0
    assert "scene_generation_v10_cycle_pair_presentations" not in averaged
    assert active._scene_state_hard_failure_completed_updates == 1
    assert active._scene_state_hard_failure_cycle_pairs == []
    assert active._scene_state_cycle_retention_metric_presentations == 0
    assert payload["schema"] == trainer._SCENE_STATE_HARD_FAILURE_ROW_AUDIT_SCHEMA
    assert payload["checkpoint_optimizer_step"] == 1
    assert payload["completed_pair_presentations"] == 1
    assert payload["generation_endpoint"] is False
    assert payload["pair_schedule"] == [
        {"source_row_ordinal": source, "donor_row_ordinal": donor}
    ]
    assert len(payload["rows"]) == 2

    second_source, second_donor = trainer_schedule_binding["pair_indices"][1]
    manifest_pairs[second_source] = {
        "source_index": second_source,
        "donor_index": second_donor,
        "source_row_sha256": bytes([second_source] * 32).hex(),
        "donor_row_sha256": bytes([second_donor] * 32).hex(),
    }
    active.state.global_step = 2
    active._scene_state_hard_failure_record_pair_presentation(
        torch.tensor([second_source]),
        torch.tensor([second_donor]),
        row_hash(second_source),
        row_hash(second_donor),
        stats,
    )
    second_averaged = active._scene_state_cycle_retention_aggregate_memory_stats(
        stats
    )
    second_payload = active._scene_state_hard_failure_row_audit_payload()

    assert second_averaged["scene_generation_hard_failure_optimizer_step"] == 2.0
    assert second_averaged["scene_generation_hard_failure_cycle_index"] == 1.0
    assert second_payload["checkpoint_optimizer_step"] == 2
    assert second_payload["pair_presentations"][1]["cycle"] == 1
    assert all(row["cycle"] == 1 for row in second_payload["rows"])


def _parse_hard_failure_args(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    *,
    extra_argv: tuple[str, ...] = (),
) -> Namespace:
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "delta-sft",
            "--model-path",
            "model",
            "--output-dir",
            str(tmp_path / "run"),
            "--initial-adapter-output-dir",
            str(tmp_path / "initial-adapter"),
            "--train-file",
            str(contract.TRAIN_FILE),
            "--memory-loss-mode",
            "scene_state_generation_ce",
            "--training-mode",
            "episode",
            "--assistant-loss-mode",
            "final_assistant_only",
            "--episode-recent-messages",
            "0",
            "--no-episode-read-write-enabled",
            "--memory-kl-weight",
            "0",
            "--memory-base-kl-weight",
            "0",
            "--scene-state-source-manifest",
            str(contract.SOURCE_MANIFEST),
            "--expected-scene-state-source-manifest-sha256",
            contract.FILE_SHA256["source_manifest.json"],
            "--scene-state-generation-objective-version",
            contract.OBJECTIVE_VERSION,
            "--scene-state-generated-prefix-correction-weight",
            "0",
            "--scene-state-generated-unlikelihood-max-wrong-tokens",
            "0",
            "--gradient-accumulation-steps",
            "1",
            "--learning-rate",
            "1e-4",
            "--lr-scheduler-type",
            "constant",
            "--warmup-ratio",
            "0",
            "--warmup-steps",
            "0",
            "--max-steps",
            "64",
            "--save-steps",
            "1",
            "--save-total-limit",
            "64",
            "--no-ignore-data-skip",
            *extra_argv,
        ],
    )

    return trainer.parse_args()


def test_real_parser_accepts_non_smoke_accumulation_one(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    parsed = _parse_hard_failure_args(monkeypatch, tmp_path)

    assert parsed.scene_state_generation_objective_version == contract.OBJECTIVE_VERSION
    assert parsed.gradient_accumulation_steps == 1
    assert parsed.max_steps == 64


def test_real_parser_rejects_v15_accumulation_sixteen_fallback(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    with pytest.raises(
        ValueError,
        match="requires gradient-accumulation-steps=1",
    ):
        _parse_hard_failure_args(
            monkeypatch,
            tmp_path,
            extra_argv=("--gradient-accumulation-steps", "16"),
        )


def test_real_protocol_builder_uses_hard_failure_64_update_contract(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    trainer_schedule_binding: Mapping[str, object],
) -> None:
    args = _parse_hard_failure_args(monkeypatch, tmp_path)
    monkeypatch.setattr(
        trainer,
        "_scene_state_identity_protocol_pairing_summary",
        lambda _manifest: {},
    )
    protocol = trainer.build_training_protocol(
        args,
        Dataset.from_dict({"input_ids": [[1]] for _ in range(32)}),
        effective_training_mode="episode",
        train_samples=32,
        eval_samples=0,
        warmup_steps=0,
        scene_state_identity_pairing_manifest={},
        train_schedule_binding=dict(trainer_schedule_binding),
    )

    assert protocol["schema_version"] == contract.OBJECTIVE_SCHEMA_VERSION == 19
    assert protocol["memory_objective_version"] == contract.OBJECTIVE_VERSION
    assert protocol["gradient_accumulation_steps"] == 1
    assert protocol["max_steps"] == 64
    assert protocol["save_steps"] == 1
    assert protocol["save_total_limit"] == 64
    assert protocol["train_sampler_mode"] == contract.FIXED_SAMPLER_MODE
    assert protocol["scene_generation_hard_failure_run_mode"] == (
        contract.PRODUCTION_RUN_MODE
    )
    assert protocol["scene_generation_hard_failure_production_eligible"] is True
    assert protocol["scene_generation_cycle_pair_presentations"] == 1
    assert protocol["scene_generation_gradient_accumulation_pair_cycle"] == 1

    schedule = protocol["train_schedule"]
    assert isinstance(schedule, dict)
    assert schedule["schema"] == trainer._SCENE_MEMORY_HARD_FAILURE_CURRICULUM_SCHEMA
    assert schedule["total_steps"] == 64
    assert schedule["checkpoint_steps"] == list(range(1, 65))
    assert schedule["optimizer_checkpoint_steps"] == list(range(1, 65))
    assert schedule["generation_endpoint_steps"] == [16, 32, 48, 64]
    assert schedule["microbatch_cycle_size"] == 1
    assert trainer._scene_hard_failure_protocol_checkpoint_steps(protocol) == tuple(
        range(1, 65)
    )

    assert not any(key.startswith("scene_generation_v15_run") for key in protocol)
    assert "scene_generation_v15_production_eligible" not in protocol
    assert "scene_generation_v10_run_mode" not in protocol

    v15_accumulation = json.loads(json.dumps(protocol))
    v15_accumulation["gradient_accumulation_steps"] = 16
    v15_accumulation["scene_generation_cycle_pair_presentations"] = 16
    v15_accumulation["scene_generation_gradient_accumulation_pair_cycle"] = 16
    with pytest.raises(
        ValueError,
        match="hard-failure cached-prefix protocol differs",
    ):
        trainer._scene_hard_failure_protocol_checkpoint_steps(v15_accumulation)


def test_real_smoke_protocol_is_one_fresh_optimizer_update(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    trainer_schedule_binding: Mapping[str, object],
) -> None:
    args = _parse_hard_failure_args(
        monkeypatch,
        tmp_path,
        extra_argv=(
            contract.ONE_PAIR_SMOKE_FLAG,
            "--max-steps",
            "1",
            "--save-total-limit",
            "1",
        ),
    )
    smoke_binding = trainer._scene_state_hard_failure_one_pair_smoke_binding(
        dict(trainer_schedule_binding)
    )
    monkeypatch.setattr(
        trainer,
        "_scene_state_identity_protocol_pairing_summary",
        lambda _manifest: {},
    )
    protocol = trainer.build_training_protocol(
        args,
        Dataset.from_dict({"input_ids": [[1]] for _ in range(32)}),
        effective_training_mode="episode",
        train_samples=32,
        eval_samples=0,
        warmup_steps=0,
        scene_state_identity_pairing_manifest={},
        train_schedule_binding=smoke_binding,
    )

    assert protocol["schema_version"] == 19
    assert protocol["max_steps"] == 1
    assert protocol["gradient_accumulation_steps"] == 1
    assert protocol["save_total_limit"] == 1
    assert protocol["train_sampler_mode"] == contract.ONE_PAIR_SMOKE_SAMPLER_MODE
    assert protocol["scene_generation_hard_failure_run_mode"] == (
        contract.ONE_PAIR_SMOKE_RUN_MODE
    )
    assert protocol["scene_generation_hard_failure_production_eligible"] is False
    assert protocol["train_schedule"]["total_steps"] == 1
    assert protocol["train_schedule"]["checkpoint_steps"] == [1]
    assert protocol["train_schedule"]["optimizer_checkpoint_steps"] == [1]
    assert protocol["train_schedule"]["generation_endpoint_steps"] == [1]
    assert trainer._scene_hard_failure_protocol_checkpoint_steps(protocol) == (1,)


@pytest.mark.parametrize(
    "argv",
    (
        ("--warm-start-from-checkpoint", "/tmp/checkpoint-4"),
        ("--warm-start-mode", "scene_memory_v14_v13_checkpoint4_adapter_only"),
        ("--resume-from-checkpoint", "/tmp/checkpoint-4"),
        ("--resume-checkpoint", "/tmp/checkpoint-4"),
        ("--resume-mode", "optimizer"),
    ),
)
def test_fresh_start_rejects_warm_start_and_resume_flags(
    argv: tuple[str, str],
) -> None:
    with pytest.raises(contract.LaunchContractError, match="fresh_start"):
        contract.validate_fresh_start_arguments(argv)


@pytest.mark.parametrize(
    "lineage",
    (
        {"source_checkpoint": "/tmp/checkpoint-4"},
        {"optimizer_state_imported": True},
        {"scheduler_state_imported": True},
        {"trainer_state_imported": True},
        {"rng_state_imported": True},
    ),
)
def test_fresh_start_rejects_optimizer_and_checkpoint_lineage(
    lineage: dict[str, object],
) -> None:
    with pytest.raises(contract.LaunchContractError, match="fresh_start"):
        contract.validate_fresh_start_arguments((), lineage=lineage)


def test_fresh_start_accepts_only_unrelated_training_arguments() -> None:
    contract.validate_fresh_start_arguments(
        (
            "--model-name-or-path",
            "/ssd/models/gemma",
            "--learning-rate",
            "1e-4",
        ),
        lineage={
            "source_checkpoint": None,
            "optimizer_state_imported": False,
            "scheduler_state_imported": False,
            "trainer_state_imported": False,
            "rng_state_imported": False,
        },
    )


def test_fresh_adapter_topology_is_all42_qo_rank4_semantics2() -> None:
    assert contract.TARGET_LAYERS == tuple(range(42))
    assert contract.RANK == 4
    assert contract.ALPHA == 8
    assert contract.DELTA_HEADS == ("q", "o")
    assert contract.RWKV_MS_SEMANTICS_VERSION == 2
    assert contract.RWKV_MS_NUM_STATES == 4
    assert contract.RWKV_MS_CHUNK_SIZE == 128
    assert contract.STATE_RESET_PER_ROW is True
    assert contract.READ_SIDE_WRITES_ENABLED is False


def _locked_launch_values() -> dict[str, object]:
    return {
        "objective_version": contract.OBJECTIVE_VERSION,
        "max_steps": 64,
        "gradient_accumulation_steps": 1,
        "save_steps": 1,
        "save_total_limit": 64,
        "target_layers": tuple(range(42)),
        "rank": 4,
        "alpha": 8,
        "delta_heads": ("q", "o"),
        "rwkv_ms_num_states": 4,
        "rwkv_ms_semantics_version": 2,
        "rwkv_ms_chunk_size": 128,
        "rwkv_ms_boundary_mode": "fixed_chunk",
        "state_reset_per_row": True,
        "episode_read_write_enabled": False,
        "memory_fusion_mode": "add",
        "memory_fusion_placement": "attention_output",
        "per_device_train_batch_size": 1,
        "memory_kl_weight": 0.0,
        "memory_base_kl_weight": 0.0,
        "validation_split_ratio": 0.0,
        "argv": (),
        "lineage": {
            "source_checkpoint": None,
            "optimizer_state_imported": False,
            "scheduler_state_imported": False,
            "trainer_state_imported": False,
            "rng_state_imported": False,
        },
    }


def test_locked_launch_values_pass_the_live_contract() -> None:
    validated = contract.validate_launch_contract(_locked_launch_values())

    assert validated["status"] == "pass"
    assert validated["run_mode"] == contract.PRODUCTION_RUN_MODE
    assert validated["sampler_mode"] == contract.FIXED_SAMPLER_MODE
    assert validated["source_lock"]["path"] == str(contract.SOURCE_LOCK.resolve())


def test_source_lock_fails_closed_on_self_hash_or_protected_split_drift(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    source_lock = json.loads(contract.SOURCE_LOCK.read_text(encoding="utf-8"))
    copied_lock = tmp_path / contract.SOURCE_LOCK.name
    monkeypatch.setattr(contract, "SOURCE_LOCK", copied_lock)

    invalid_hash = dict(source_lock)
    invalid_hash["lock_sha256"] = "0" * 64
    copied_lock.write_text(json.dumps(invalid_hash), encoding="utf-8")
    with pytest.raises(contract.LaunchContractError, match="lock_sha256_differs"):
        contract.validate_source_lock(copied_lock)

    protected_drift = json.loads(json.dumps(source_lock))
    protected_drift["protected_evaluation"]["official_validation"]["included"] = True
    unsigned = dict(protected_drift)
    unsigned.pop("lock_sha256")
    protected_drift["lock_sha256"] = contract.canonical_sha256(unsigned)
    copied_lock.write_text(json.dumps(protected_drift), encoding="utf-8")
    with pytest.raises(
        contract.LaunchContractError,
        match="protected_evaluation_differs",
    ):
        contract.validate_source_lock(copied_lock)


@pytest.mark.parametrize(
    ("field", "wrong_value"),
    (
        ("target_layers", tuple(range(41))),
        ("rank", 8),
        ("alpha", 4),
        ("delta_heads", ("o",)),
        ("rwkv_ms_num_states", 1),
        ("rwkv_ms_semantics_version", 1),
        ("rwkv_ms_chunk_size", 64),
        ("rwkv_ms_boundary_mode", "sequence_end"),
        ("state_reset_per_row", False),
        ("episode_read_write_enabled", True),
        ("memory_fusion_mode", "gate"),
        ("memory_fusion_placement", "mlp_output"),
        ("per_device_train_batch_size", 2),
        ("memory_kl_weight", 0.1),
        ("memory_base_kl_weight", 0.1),
        ("validation_split_ratio", 0.1),
    ),
)
def test_launch_contract_rejects_topology_or_training_policy_drift(
    field: str,
    wrong_value: object,
) -> None:
    values = _locked_launch_values()
    values[field] = wrong_value

    with pytest.raises(contract.LaunchContractError, match=field):
        contract.validate_launch_contract(values)


@pytest.mark.parametrize(
    "path",
    (
        Path("/ssd/datasets/scene/val.jsonl"),
        Path("/ssd/datasets/scene/validation/rows.jsonl"),
        Path("/ssd/datasets/scene/Hard32/rows.jsonl"),
        Path("/ssd/datasets/scene/test/rows.jsonl"),
        Path("/ssd/datasets/scene/full170.jsonl"),
    ),
)
def test_training_contract_rejects_every_protected_split_path(path: Path) -> None:
    with pytest.raises(contract.LaunchContractError, match="protected"):
        contract.reject_protected_path(path, description="pytest_training_input")


def test_training_contract_accepts_train_only_path() -> None:
    contract.reject_protected_path(
        Path("/ssd/datasets/scene/training/train.jsonl"),
        description="pytest_training_input",
    )


def _git(repo: Path, *arguments: str) -> subprocess.CompletedProcess[str]:
    result = subprocess.run(
        ["git", *arguments],
        cwd=repo,
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    return result


def test_critical_clean_check_ignores_unrelated_worktree_changes(
    tmp_path: Path,
) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init", "--quiet")
    critical = repo / "critical.py"
    unrelated = repo / "notes.txt"
    critical.write_text("critical = True\n", encoding="utf-8")
    unrelated.write_text("original\n", encoding="utf-8")
    _git(repo, "add", "critical.py", "notes.txt")
    _git(
        repo,
        "-c",
        "user.name=Codex",
        "-c",
        "user.email=codex@example.invalid",
        "commit",
        "--quiet",
        "-m",
        "initial",
    )

    unrelated.write_text("user change\n", encoding="utf-8")
    (repo / "untracked.txt").write_text("user file\n", encoding="utf-8")
    contract.validate_critical_worktree(repo, ("critical.py",))

    critical.write_text("critical = False\n", encoding="utf-8")
    with pytest.raises(contract.LaunchContractError, match="critical"):
        contract.validate_critical_worktree(repo, ("critical.py",))


def test_legacy_v15_keeps_its_historical_four_step_contract() -> None:
    assert legacy_v15.TOTAL_PAIR_PRESENTATIONS == 64
    assert legacy_v15.PAIR_PRESENTATIONS_PER_OPTIMIZER_STEP == 16
    assert legacy_v15.GRADIENT_ACCUMULATION_STEPS == 16
    assert legacy_v15.TOTAL_OPTIMIZER_STEPS == 4
    assert legacy_v15.CHECKPOINT_STEPS == (1, 2, 3, 4)
    assert legacy_v15.PRESENTATION_CHECKPOINTS == (16, 32, 48, 64)


def _launcher_environment(
    *,
    run_mode: str = "smoke",
    run_name: str | None = None,
    dry_run: str = "1",
    updates: Mapping[str, str] | None = None,
) -> dict[str, str]:
    environment = os.environ.copy()
    for variable in tuple(environment):
        if variable in _DISTRIBUTED_ENVIRONMENT or variable.startswith(
            _PROTECTED_EVALUATION_ENVIRONMENT_PREFIXES
        ):
            environment.pop(variable)
    environment.update(
        {
            "RUN_MODE": run_mode,
            "RUN_NAME": run_name or f"pytest_{uuid.uuid4().hex}",
            "DRY_RUN": dry_run,
        }
    )
    if updates:
        environment.update(updates)
    return environment


def _run_launcher(
    environment: Mapping[str, str],
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [str(HARD_FAILURE_LAUNCHER)],
        cwd=REPO_ROOT,
        env=dict(environment),
        check=False,
        capture_output=True,
        text=True,
    )


def _dry_run_arguments(
    result: subprocess.CompletedProcess[str],
) -> list[str]:
    assert result.returncode == 0, result.stderr
    command_line = next(
        line.removeprefix("DRY_RUN command:")
        for line in result.stdout.splitlines()
        if line.startswith("DRY_RUN command:")
    )
    return shlex.split(command_line)


def _argument_value(arguments: list[str], flag: str) -> str:
    return arguments[arguments.index(flag) + 1]


def test_hard_failure_launcher_is_executable_and_has_valid_bash_syntax() -> None:
    assert HARD_FAILURE_LAUNCHER.stat().st_mode & stat.S_IXUSR
    result = subprocess.run(
        ["bash", "-n", str(HARD_FAILURE_LAUNCHER)],
        cwd=REPO_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize(
    ("run_mode", "max_steps", "save_total_limit", "smoke"),
    (
        ("smoke", "1", "1", True),
        ("production", "64", "64", False),
    ),
)
def test_hard_failure_launcher_dry_run_emits_fresh_locked_command(
    run_mode: str,
    max_steps: str,
    save_total_limit: str,
    smoke: bool,
) -> None:
    arguments = _dry_run_arguments(
        _run_launcher(_launcher_environment(run_mode=run_mode))
    )

    assert arguments[1:3] == ["-m", "deltamem.train.delta_sft"]
    assert _argument_value(arguments, "--train-file") == str(contract.TRAIN_FILE)
    assert _argument_value(
        arguments,
        "--scene-state-generation-objective-version",
    ) == contract.OBJECTIVE_VERSION
    assert _argument_value(arguments, "--target-layers") == ",".join(
        str(layer) for layer in contract.TARGET_LAYERS
    )
    assert _argument_value(arguments, "--delta-heads") == "q,o"
    assert _argument_value(arguments, "--rank") == "4"
    assert _argument_value(arguments, "--alpha") == "8"
    assert _argument_value(arguments, "--gradient-accumulation-steps") == "1"
    assert _argument_value(arguments, "--max-steps") == max_steps
    assert _argument_value(arguments, "--save-steps") == "1"
    assert _argument_value(arguments, "--save-total-limit") == save_total_limit
    assert _argument_value(arguments, "--validation-split-ratio") == "0"
    assert (contract.ONE_PAIR_SMOKE_FLAG in arguments) is smoke
    assert "--warm-start-from-checkpoint" not in arguments
    assert "--warm-start-mode" not in arguments
    assert "--resume-from-checkpoint" not in arguments
    assert "--resume-checkpoint" not in arguments
    assert "--resume-mode" not in arguments


@pytest.mark.parametrize("run_mode", ("smoke", "production"))
def test_hard_failure_launcher_command_passes_real_parser(
    run_mode: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    arguments = _dry_run_arguments(
        _run_launcher(_launcher_environment(run_mode=run_mode))
    )
    monkeypatch.setattr(sys, "argv", [arguments[2], *arguments[3:]])

    parsed = trainer.parse_args()

    assert parsed.scene_state_generation_objective_version == contract.OBJECTIVE_VERSION
    assert parsed.gradient_accumulation_steps == 1
    assert parsed.max_steps == (1 if run_mode == "smoke" else 64)
    assert parsed.save_total_limit == (1 if run_mode == "smoke" else 64)
    assert parsed.scene_state_hard_failure_one_pair_smoke is (run_mode == "smoke")
    assert parsed.warm_start_from_checkpoint is None
    assert parsed.resume_from_checkpoint is None


@pytest.mark.parametrize(
    ("run_mode", "dry_run", "run_name", "message"),
    (
        ("invalid", "1", "pytest", "RUN_MODE must be smoke or production"),
        ("smoke", "invalid", "pytest", "DRY_RUN must be 0 or 1"),
        ("smoke", "1", "../unsafe", "RUN_NAME contains unsupported characters"),
    ),
)
def test_hard_failure_launcher_rejects_invalid_control_environment(
    run_mode: str,
    dry_run: str,
    run_name: str,
    message: str,
) -> None:
    result = _run_launcher(
        _launcher_environment(
            run_mode=run_mode,
            dry_run=dry_run,
            run_name=run_name,
        )
    )

    assert result.returncode == 2
    assert message in result.stderr


@pytest.mark.parametrize("variable", _DISTRIBUTED_ENVIRONMENT)
def test_hard_failure_launcher_rejects_distributed_environment(
    variable: str,
) -> None:
    result = _run_launcher(
        _launcher_environment(updates={variable: "pytest_forbidden"})
    )

    assert result.returncode == 2
    assert (
        f"distributed_environment_is_forbidden variable={variable}"
        in result.stderr
    )


@pytest.mark.parametrize(
    "variable",
    (
        "HARD32",
        "HARD32_CUSTOM_OVERRIDE",
        "VALIDATION_FILE",
        "VALIDATION_CUSTOM_OVERRIDE",
        "TEST_FILE",
        "TEST_CUSTOM_OVERRIDE",
        "BENCHMARK_PATH",
        "BENCHMARK_CUSTOM_OVERRIDE",
    ),
)
def test_hard_failure_launcher_rejects_protected_evaluation_environment(
    variable: str,
) -> None:
    result = _run_launcher(
        _launcher_environment(updates={variable: "/tmp/forbidden"})
    )

    assert result.returncode == 2
    assert (
        f"protected_evaluation_environment_is_forbidden variable={variable}"
        in result.stderr
    )
