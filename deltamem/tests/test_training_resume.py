from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from datasets import Dataset

import deltamem.train.delta_sft_experimental as experimental_train
from deltamem.core.delta import HFDeltaMemConfig


def test_ddp_training_kwargs_disable_buffer_broadcast(tmp_path: Path) -> None:
    kwargs = experimental_train._build_ddp_training_kwargs(
        distributed=True,
        ddp_backend="nccl",
        local_rank=3,
    )
    assert kwargs == {
        "ddp_find_unused_parameters": False,
        "ddp_broadcast_buffers": False,
        "ddp_backend": "nccl",
        "local_rank": 3,
    }

    single_process_kwargs = experimental_train._build_ddp_training_kwargs(
        distributed=False,
        ddp_backend="nccl",
        local_rank=3,
    )
    training_args = experimental_train.TrainingArguments(
        output_dir=str(tmp_path),
        **single_process_kwargs,
    )

    assert training_args.ddp_broadcast_buffers is False


def _write_complete_checkpoint(path: Path, config: HFDeltaMemConfig) -> None:
    path.mkdir(parents=True)
    config.save_pretrained(path)
    for filename in (
        "delta_mem_adapter.pt",
        "optimizer.pt",
        "scheduler.pt",
        "trainer_state.json",
    ):
        (path / filename).touch()


def _continuation_protocol(
    *,
    max_steps: int = 128,
    num_train_epochs: float = 4.0,
    lr_scheduler_type: str = "constant_with_warmup",
    warmup_steps: int = 8,
) -> dict[str, object]:
    return {
        "schema_version": experimental_train._TRAINING_PROTOCOL_SCHEMA_VERSION,
        "memory_objective_version": experimental_train._MEMORY_OBJECTIVE_VERSION,
        "tokenized_fingerprint": "fixed-dataset",
        "learning_rate": 1e-3,
        "lr_scheduler_type": lr_scheduler_type,
        "warmup_ratio": 0.0625,
        "warmup_steps": warmup_steps,
        "num_train_epochs": num_train_epochs,
        "max_steps": max_steps,
    }


def _write_continuation_checkpoint(
    path: Path,
    *,
    protocol: dict[str, object] | None = None,
    global_step: int = 128,
    effective_max_steps: int = 128,
    epoch: float | None = None,
    include_rng: bool = True,
) -> dict[str, object]:
    _write_complete_checkpoint(path, HFDeltaMemConfig(rank=2))
    active_protocol = _continuation_protocol() if protocol is None else protocol
    (path / experimental_train._TRAINING_PROTOCOL_FILENAME).write_text(
        json.dumps(active_protocol)
    )
    trainer_state: dict[str, object] = {
        "global_step": global_step,
        "max_steps": effective_max_steps,
    }
    if epoch is not None:
        trainer_state["epoch"] = epoch
    (path / "trainer_state.json").write_text(json.dumps(trainer_state))
    if include_rng:
        (path / "rng_state.pth").touch()
    return active_protocol


def _continuation_args(
    checkpoint: Path | str,
    output_dir: Path,
    *,
    max_steps: int = 256,
    num_train_epochs: float = 8.0,
) -> SimpleNamespace:
    return SimpleNamespace(
        resume_mode="extend",
        resume_from_checkpoint=str(checkpoint),
        output_dir=output_dir,
        max_steps=max_steps,
        num_train_epochs=num_train_epochs,
    )


def _objective_source_protocol() -> dict[str, object]:
    return {
        **_continuation_protocol(max_steps=384, num_train_epochs=12.0),
        "memory_loss_mode": "context_dropout_ce",
        "memory_base_kl_weight": 0.0,
        "data_seed": 42,
        "train_samples": 32,
        "eval_samples": 0,
    }


def _objective_pairing_summary() -> dict[str, object]:
    return {
        "pairing_version": experimental_train._CONTENT_CONTRAST_PAIRING_VERSION,
        "pairing_scope": "within_post_split_partition",
        "data_seed": 42,
        "tokenized_fingerprint": "fixed-dataset",
        "manifest_sha256": "a" * 64,
        "splits": {
            "train": {
                "sample_count": 32,
                "rotation": 16,
                "source_fingerprint": "source-fingerprint",
                "paired_fingerprint": "paired-fingerprint",
                "pairs_sha256": "b" * 64,
                "manifest_sha256": "c" * 64,
            }
        },
    }


def _objective_target_protocol() -> dict[str, object]:
    return {
        **_objective_source_protocol(),
        "schema_version": (
            experimental_train._CONTENT_CONTRAST_TRAINING_PROTOCOL_SCHEMA_VERSION
        ),
        "memory_objective_version": (
            experimental_train._CONTENT_CONTRAST_OBJECTIVE_VERSION
        ),
        "memory_loss_mode": "content_contrast_ce",
        "memory_contrast_weight": 0.25,
        "memory_margin": 0.5,
        "memory_kl_weight": 0.0,
        "write_sparsity_weight": 0.0,
        "memory_partition_alignment_weight": 0.0,
        "memory_partition_entropy_weight": 0.0,
        "memory_partition_balance_weight": 0.0,
        "content_contrast_negative_priming_grad": False,
        "content_contrast_pairing": _objective_pairing_summary(),
        "max_steps": 416,
        "num_train_epochs": 13.0,
    }


def _objective_args(checkpoint: Path, output_dir: Path) -> SimpleNamespace:
    return SimpleNamespace(
        resume_mode="objective_ablation",
        resume_from_checkpoint=str(checkpoint),
        output_dir=output_dir,
        max_steps=416,
        num_train_epochs=13.0,
        memory_loss_mode="content_contrast_ce",
        memory_contrast_weight=0.25,
        memory_margin=0.5,
        memory_kl_weight=0.0,
        write_sparsity_weight=0.0,
        memory_partition_alignment_weight=0.0,
        memory_partition_entropy_weight=0.0,
        memory_partition_balance_weight=0.0,
    )


def test_resolve_resume_checkpoint_uses_latest_complete_checkpoint(tmp_path: Path) -> None:
    trainer_output = tmp_path / "trainer"
    older = trainer_output / "checkpoint-100"
    _write_complete_checkpoint(older, HFDeltaMemConfig())
    incomplete = trainer_output / "checkpoint-200"
    incomplete.mkdir()
    (incomplete / "delta_mem_adapter.pt").touch()

    resolved = experimental_train.resolve_resume_checkpoint("latest", trainer_output)

    assert resolved == str(older.resolve())


def test_latest_content_contrast_resume_skips_checkpoint_missing_pairing_manifest(
    tmp_path: Path,
) -> None:
    trainer_output = tmp_path / "trainer"
    older = trainer_output / "checkpoint-100"
    _write_complete_checkpoint(older, HFDeltaMemConfig())
    (older / experimental_train._TRAINING_PROTOCOL_FILENAME).write_text("{}")
    (older / experimental_train._CONTENT_CONTRAST_PAIRING_FILENAME).write_text("{}")
    newer = trainer_output / "checkpoint-200"
    _write_complete_checkpoint(newer, HFDeltaMemConfig())
    (newer / experimental_train._TRAINING_PROTOCOL_FILENAME).write_text("{}")

    resolved = experimental_train.resolve_resume_checkpoint(
        "latest",
        trainer_output,
        require_training_protocol=True,
        require_content_contrast_pairing=True,
    )

    assert resolved == str(older.resolve())


def test_resolve_resume_checkpoint_rejects_incomplete_explicit_path(tmp_path: Path) -> None:
    checkpoint = tmp_path / "checkpoint-100"
    checkpoint.mkdir()
    (checkpoint / "delta_mem_adapter.pt").touch()

    with pytest.raises(FileNotFoundError, match="delta_mem_config.json"):
        experimental_train.resolve_resume_checkpoint(checkpoint, tmp_path / "trainer")


def test_delta_mem_trainer_loads_custom_adapter_checkpoint(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = HFDeltaMemConfig(rank=2)
    checkpoint = tmp_path / "checkpoint-100"
    _write_complete_checkpoint(checkpoint, config)
    model = torch.nn.Linear(2, 2)
    loaded: list[tuple[torch.nn.Module, Path]] = []

    def fake_load_delta_mem_adapter(
        loaded_model: torch.nn.Module,
        input_dir: str | Path,
    ) -> HFDeltaMemConfig:
        loaded.append((loaded_model, Path(input_dir)))
        return config

    monkeypatch.setattr(
        experimental_train,
        "load_delta_mem_adapter",
        fake_load_delta_mem_adapter,
    )
    trainer = object.__new__(experimental_train.DeltaMemTrainer)
    trainer.model = model
    trainer.delta_config = config

    trainer._load_from_checkpoint(str(checkpoint))

    assert loaded == [(model, checkpoint.resolve())]


def test_delta_mem_trainer_rejects_checkpoint_config_mismatch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    checkpoint = tmp_path / "checkpoint-100"
    _write_complete_checkpoint(checkpoint, HFDeltaMemConfig(rank=2))
    monkeypatch.setattr(
        experimental_train,
        "load_delta_mem_adapter",
        lambda *args, **kwargs: pytest.fail("adapter should not load after a config mismatch"),
    )
    trainer = object.__new__(experimental_train.DeltaMemTrainer)
    trainer.model = torch.nn.Linear(2, 2)
    trainer.delta_config = HFDeltaMemConfig(rank=4)

    with pytest.raises(ValueError, match="rank"):
        trainer._load_from_checkpoint(str(checkpoint))


def test_delta_mem_trainer_rejects_rwkv_semantics_mismatch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    checkpoint = tmp_path / "checkpoint-100"
    _write_complete_checkpoint(
        checkpoint,
        HFDeltaMemConfig(memory_backend="rwkv_ms", rwkv_ms_semantics_version=1),
    )
    monkeypatch.setattr(
        experimental_train,
        "load_delta_mem_adapter",
        lambda *args, **kwargs: pytest.fail("adapter should not load across semantics versions"),
    )
    trainer = object.__new__(experimental_train.DeltaMemTrainer)
    trainer.model = torch.nn.Linear(2, 2)
    trainer.delta_config = HFDeltaMemConfig(
        memory_backend="rwkv_ms",
        rwkv_ms_semantics_version=2,
    )

    with pytest.raises(ValueError, match="rwkv_ms_semantics_version"):
        trainer._load_from_checkpoint(str(checkpoint))


def test_split_tokenized_dataset_is_deterministic_and_disjoint() -> None:
    tokenized = Dataset.from_dict({"sample_id": list(range(20))})

    first_train, first_eval = experimental_train.split_tokenized_dataset(
        tokenized,
        validation_split_ratio=0.2,
        data_seed=17,
    )
    second_train, second_eval = experimental_train.split_tokenized_dataset(
        tokenized,
        validation_split_ratio=0.2,
        data_seed=17,
    )

    assert first_eval is not None
    assert second_eval is not None
    assert first_train["sample_id"] == second_train["sample_id"]
    assert first_eval["sample_id"] == second_eval["sample_id"]
    assert len(first_train) == 16
    assert len(first_eval) == 4
    assert set(first_train["sample_id"]).isdisjoint(first_eval["sample_id"])


def test_delta_mem_trainer_loads_custom_best_adapter(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = HFDeltaMemConfig(rank=2)
    checkpoint = tmp_path / "checkpoint-100"
    _write_complete_checkpoint(checkpoint, config)
    model = torch.nn.Linear(2, 2)
    loaded: list[tuple[torch.nn.Module, Path]] = []
    reset: list[torch.nn.Module] = []

    monkeypatch.setattr(
        experimental_train,
        "load_delta_mem_adapter",
        lambda loaded_model, input_dir: loaded.append((loaded_model, Path(input_dir))),
    )
    trainer = object.__new__(experimental_train.DeltaMemTrainer)
    trainer.model = model
    trainer.delta_config = config
    trainer.state = SimpleNamespace(best_model_checkpoint=str(checkpoint))
    trainer.accelerator = SimpleNamespace(unwrap_model=lambda wrapped: wrapped)
    trainer._reset_online_state = lambda reset_model: reset.append(reset_model)

    trainer._load_best_model()

    assert loaded == [(model, checkpoint.resolve())]
    assert reset == [model]


def test_delta_mem_trainer_rejects_resume_protocol_drift(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = HFDeltaMemConfig(rank=2)
    checkpoint = tmp_path / "checkpoint-100"
    _write_complete_checkpoint(checkpoint, config)
    (checkpoint / experimental_train._TRAINING_PROTOCOL_FILENAME).write_text(
        '{"memory_base_kl_weight": 0.1}'
    )
    monkeypatch.setattr(
        experimental_train,
        "load_delta_mem_adapter",
        lambda *args, **kwargs: pytest.fail("adapter should not load after protocol drift"),
    )
    trainer = object.__new__(experimental_train.DeltaMemTrainer)
    trainer.model = torch.nn.Linear(2, 2)
    trainer.delta_config = config
    trainer.training_protocol = {"memory_base_kl_weight": 0.5}

    with pytest.raises(ValueError, match="memory_base_kl_weight"):
        trainer._load_from_checkpoint(str(checkpoint))


def test_delta_mem_trainer_rejects_legacy_memory_objective_protocol(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = HFDeltaMemConfig(rank=2)
    checkpoint = tmp_path / "checkpoint-100"
    _write_complete_checkpoint(checkpoint, config)
    (checkpoint / experimental_train._TRAINING_PROTOCOL_FILENAME).write_text(
        '{"schema_version": 1, "memory_objective_version": "same_read_teacher_v0"}'
    )
    monkeypatch.setattr(
        experimental_train,
        "load_delta_mem_adapter",
        lambda *args, **kwargs: pytest.fail("legacy objective checkpoint should not load"),
    )
    trainer = object.__new__(experimental_train.DeltaMemTrainer)
    trainer.model = torch.nn.Linear(2, 2)
    trainer.delta_config = config
    trainer.training_protocol = {
        "schema_version": experimental_train._TRAINING_PROTOCOL_SCHEMA_VERSION,
        "memory_objective_version": experimental_train._MEMORY_OBJECTIVE_VERSION,
    }

    with pytest.raises(ValueError, match="memory_objective_version|schema_version"):
        trainer._load_from_checkpoint(str(checkpoint))


def test_delta_mem_trainer_requires_exact_content_contrast_pairing_manifest(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = HFDeltaMemConfig(rank=2)
    checkpoint = tmp_path / "checkpoint-100"
    _write_complete_checkpoint(checkpoint, config)
    protocol = {
        "schema_version": experimental_train._CONTENT_CONTRAST_TRAINING_PROTOCOL_SCHEMA_VERSION,
        "memory_objective_version": experimental_train._CONTENT_CONTRAST_OBJECTIVE_VERSION,
        "content_contrast_pairing": {"manifest_sha256": "expected"},
    }
    pairing_manifest = {
        "schema_version": 1,
        "objective_version": experimental_train._CONTENT_CONTRAST_OBJECTIVE_VERSION,
        "manifest_sha256": "expected",
    }
    (checkpoint / experimental_train._TRAINING_PROTOCOL_FILENAME).write_text(
        json.dumps(protocol)
    )
    monkeypatch.setattr(
        experimental_train,
        "load_delta_mem_adapter",
        lambda *args, **kwargs: pytest.fail("adapter should not load before pairing validation"),
    )
    trainer = object.__new__(experimental_train.DeltaMemTrainer)
    trainer.model = torch.nn.Linear(2, 2)
    trainer.delta_config = config
    trainer.training_protocol = protocol
    trainer.content_contrast_pairing_manifest = pairing_manifest

    with pytest.raises(FileNotFoundError, match="missing content_contrast_pairing_manifest"):
        trainer._load_from_checkpoint(str(checkpoint))

    pairing_path = checkpoint / experimental_train._CONTENT_CONTRAST_PAIRING_FILENAME
    pairing_path.write_text(json.dumps({**pairing_manifest, "manifest_sha256": "wrong"}))
    with pytest.raises(ValueError, match="pairing manifest does not match"):
        trainer._load_from_checkpoint(str(checkpoint))


def test_content_contrast_checkpoint_manifest_save_load_roundtrip(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    checkpoint = tmp_path / "checkpoint-100"
    config = HFDeltaMemConfig(rank=2)
    protocol = {
        "schema_version": experimental_train._CONTENT_CONTRAST_TRAINING_PROTOCOL_SCHEMA_VERSION,
        "memory_objective_version": experimental_train._CONTENT_CONTRAST_OBJECTIVE_VERSION,
        "content_contrast_pairing": {"manifest_sha256": "pairing-hash"},
    }
    pairing_manifest = {
        "schema_version": 1,
        "objective_version": experimental_train._CONTENT_CONTRAST_OBJECTIVE_VERSION,
        "manifest_sha256": "pairing-hash",
    }

    def fake_save_adapter(model, output_dir, active_config) -> None:
        del model
        output_path = Path(output_dir)
        active_config.save_pretrained(output_path)
        (output_path / "delta_mem_adapter.pt").touch()

    monkeypatch.setattr(experimental_train, "save_delta_mem_adapter", fake_save_adapter)
    saving_trainer = object.__new__(experimental_train.DeltaMemTrainer)
    saving_trainer.args = SimpleNamespace(output_dir=str(checkpoint))
    saving_trainer.model = torch.nn.Linear(2, 2)
    saving_trainer.delta_config = config
    saving_trainer.training_protocol = protocol
    saving_trainer.content_contrast_pairing_manifest = pairing_manifest
    saving_trainer.accelerator = SimpleNamespace(unwrap_model=lambda wrapped: wrapped)
    saving_trainer.is_world_process_zero = lambda: True

    saving_trainer.save_model(str(checkpoint))

    assert json.loads(
        (checkpoint / experimental_train._TRAINING_PROTOCOL_FILENAME).read_text()
    ) == protocol
    assert json.loads(
        (checkpoint / experimental_train._CONTENT_CONTRAST_PAIRING_FILENAME).read_text()
    ) == pairing_manifest
    for filename in ("optimizer.pt", "scheduler.pt", "trainer_state.json"):
        (checkpoint / filename).touch()

    loaded: list[Path] = []
    monkeypatch.setattr(
        experimental_train,
        "load_delta_mem_adapter",
        lambda model, input_dir: loaded.append(Path(input_dir)),
    )
    loading_trainer = object.__new__(experimental_train.DeltaMemTrainer)
    loading_trainer.model = torch.nn.Linear(2, 2)
    loading_trainer.delta_config = config
    loading_trainer.training_protocol = protocol
    loading_trainer.content_contrast_pairing_manifest = pairing_manifest

    loading_trainer._load_from_checkpoint(str(checkpoint))

    assert loaded == [checkpoint.resolve()]


def test_resume_protocol_extension_is_explicit_and_horizon_only() -> None:
    source = _continuation_protocol()
    target = {**source, "max_steps": 256, "num_train_epochs": 8.0}

    experimental_train.validate_resume_training_protocol(
        source,
        target,
        resume_mode="extend",
    )

    with pytest.raises(ValueError, match="max_steps"):
        experimental_train.validate_resume_training_protocol(
            source,
            target,
            resume_mode="exact",
        )

    with pytest.raises(ValueError, match="learning_rate"):
        experimental_train.validate_resume_training_protocol(
            source,
            {**target, "learning_rate": 2e-3},
            resume_mode="extend",
        )

    with pytest.raises(ValueError, match="target max_steps"):
        experimental_train.validate_resume_training_protocol(
            source,
            source,
            resume_mode="extend",
        )


def test_resume_protocol_rejects_frozen_mlp_checkpointing_mode_drift() -> None:
    source = {
        **_continuation_protocol(),
        "frozen_mlp_activation_checkpointing": False,
    }
    target = {
        **source,
        "frozen_mlp_activation_checkpointing": True,
    }

    with pytest.raises(ValueError, match="frozen_mlp_activation_checkpointing"):
        experimental_train.validate_resume_training_protocol(
            source,
            target,
            resume_mode="exact",
        )


def test_resume_protocol_accepts_legacy_frozen_mlp_checkpointing_key() -> None:
    source = {
        **_continuation_protocol(),
        "frozen_mlp_checkpointing": True,
    }
    target = {
        **_continuation_protocol(),
        "frozen_mlp_activation_checkpointing": True,
    }

    experimental_train.validate_resume_training_protocol(
        source,
        target,
        resume_mode="exact",
    )
    experimental_train.validate_resume_training_protocol(
        _continuation_protocol(),
        {
            **_continuation_protocol(),
            "frozen_mlp_activation_checkpointing": False,
        },
        resume_mode="exact",
    )


def test_resume_protocol_extension_preserves_warmup_and_scheduler() -> None:
    source = _continuation_protocol()

    with pytest.raises(ValueError, match="warmup_steps"):
        experimental_train.validate_resume_training_protocol(
            source,
            {**source, "max_steps": 256, "warmup_steps": 16},
            resume_mode="extend",
        )

    cosine_source = _continuation_protocol(lr_scheduler_type="cosine")
    with pytest.raises(ValueError, match="does not support"):
        experimental_train.validate_resume_training_protocol(
            cosine_source,
            {**cosine_source, "max_steps": 256},
            resume_mode="extend",
        )


def test_resume_protocol_extension_supports_epoch_horizon() -> None:
    source = _continuation_protocol(max_steps=-1, num_train_epochs=4.0)
    target = {**source, "num_train_epochs": 8.0}

    experimental_train.validate_resume_training_protocol(
        source,
        target,
        resume_mode="extend",
    )

    with pytest.raises(ValueError, match="switch from epoch mode"):
        experimental_train.validate_resume_training_protocol(
            source,
            {**target, "max_steps": 256},
            resume_mode="extend",
        )

    with pytest.raises(ValueError, match="greater than the source"):
        experimental_train.validate_resume_training_protocol(
            source,
            source,
            resume_mode="extend",
        )


def test_prepare_training_continuation_records_completed_source(tmp_path: Path) -> None:
    checkpoint = tmp_path / "source" / "trainer" / "checkpoint-128"
    source_protocol = _write_continuation_checkpoint(checkpoint)
    args = _continuation_args(checkpoint, tmp_path / "target")

    manifest = experimental_train.prepare_training_continuation(args, str(checkpoint))

    assert manifest is not None
    assert manifest["source_checkpoint"] == str(checkpoint.resolve())
    assert manifest["source_global_step"] == 128
    assert manifest["target_max_steps"] == 256
    assert manifest["warmup_steps"] == 8
    assert manifest["source_rng_state_files"] == ["rng_state.pth"]
    assert manifest["source_training_protocol_sha256"] == (
        experimental_train._protocol_sha256(source_protocol)
    )


def test_exact_resume_preserves_recorded_warmup_and_continuation_lineage(
    tmp_path: Path,
) -> None:
    checkpoint = tmp_path / "target" / "trainer" / "checkpoint-256"
    _write_continuation_checkpoint(
        checkpoint,
        protocol=_continuation_protocol(max_steps=256, num_train_epochs=8.0),
        global_step=256,
        effective_max_steps=256,
    )
    lineage = {
        "schema_version": experimental_train._CONTINUATION_MANIFEST_SCHEMA_VERSION,
        "mode": "extend",
        "source_global_step": 128,
    }
    (checkpoint / experimental_train._CONTINUATION_MANIFEST_FILENAME).write_text(
        json.dumps(lineage)
    )
    args = SimpleNamespace(resume_mode="exact")

    assert experimental_train.resolve_resume_warmup_steps(16, str(checkpoint)) == 8
    assert experimental_train.resolve_resume_warmup_steps(16, None) == 16
    assert experimental_train.prepare_training_continuation(args, str(checkpoint)) == lineage


def test_prepare_training_continuation_requires_explicit_checkpoint(tmp_path: Path) -> None:
    checkpoint = tmp_path / "source" / "trainer" / "checkpoint-128"
    _write_continuation_checkpoint(checkpoint)
    args = _continuation_args("latest", tmp_path / "target")

    with pytest.raises(ValueError, match="explicit --resume-from-checkpoint"):
        experimental_train.prepare_training_continuation(args, str(checkpoint))


def test_prepare_training_continuation_requires_distinct_output(tmp_path: Path) -> None:
    source_output = tmp_path / "source"
    checkpoint = source_output / "trainer" / "checkpoint-128"
    _write_continuation_checkpoint(checkpoint)
    args = _continuation_args(checkpoint, source_output)

    with pytest.raises(ValueError, match="distinct --output-dir"):
        experimental_train.prepare_training_continuation(args, str(checkpoint))


def test_prepare_training_continuation_requires_rng_state(tmp_path: Path) -> None:
    checkpoint = tmp_path / "source" / "trainer" / "checkpoint-128"
    _write_continuation_checkpoint(checkpoint, include_rng=False)
    args = _continuation_args(checkpoint, tmp_path / "target")

    with pytest.raises(FileNotFoundError, match="missing RNG state"):
        experimental_train.prepare_training_continuation(args, str(checkpoint))


def test_prepare_training_continuation_requires_completed_checkpoint(tmp_path: Path) -> None:
    checkpoint = tmp_path / "source" / "trainer" / "checkpoint-64"
    _write_continuation_checkpoint(
        checkpoint,
        global_step=64,
        effective_max_steps=128,
    )
    args = _continuation_args(checkpoint, tmp_path / "target")

    with pytest.raises(ValueError, match="completed checkpoint"):
        experimental_train.prepare_training_continuation(args, str(checkpoint))


def test_delta_mem_trainer_loads_horizon_extension(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    checkpoint = tmp_path / "source" / "trainer" / "checkpoint-128"
    source_protocol = _write_continuation_checkpoint(checkpoint)
    target_protocol = {**source_protocol, "max_steps": 256, "num_train_epochs": 8.0}
    loaded: list[Path] = []
    monkeypatch.setattr(
        experimental_train,
        "load_delta_mem_adapter",
        lambda model, input_dir: loaded.append(Path(input_dir)),
    )
    trainer = object.__new__(experimental_train.DeltaMemTrainer)
    trainer.model = torch.nn.Linear(2, 2)
    trainer.delta_config = HFDeltaMemConfig(rank=2)
    trainer.training_protocol = target_protocol
    trainer.resume_mode = "extend"

    trainer._load_from_checkpoint(str(checkpoint))

    assert loaded == [checkpoint.resolve()]


def test_delta_mem_trainer_saves_continuation_manifest(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output = tmp_path / "checkpoint-256"
    manifest = {"schema_version": 1, "mode": "extend", "source_global_step": 128}

    def fake_save_adapter(model, output_dir, active_config) -> None:
        del model, active_config
        Path(output_dir).mkdir(parents=True, exist_ok=True)

    monkeypatch.setattr(experimental_train, "save_delta_mem_adapter", fake_save_adapter)
    trainer = object.__new__(experimental_train.DeltaMemTrainer)
    trainer.args = SimpleNamespace(output_dir=str(output))
    trainer.model = torch.nn.Linear(2, 2)
    trainer.delta_config = HFDeltaMemConfig(rank=2)
    trainer.training_protocol = _continuation_protocol(max_steps=256)
    trainer.content_contrast_pairing_manifest = None
    trainer.continuation_manifest = manifest
    trainer.accelerator = SimpleNamespace(unwrap_model=lambda wrapped: wrapped)
    trainer.is_world_process_zero = lambda: True

    trainer.save_model(str(output))

    assert json.loads(
        (output / experimental_train._CONTINUATION_MANIFEST_FILENAME).read_text()
    ) == manifest


def test_resume_protocol_normalizes_legacy_fusion_placement() -> None:
    legacy = _continuation_protocol()
    explicit_legacy = {
        **legacy,
        "memory_fusion_placement": "attention_output",
    }
    experimental_train.validate_resume_training_protocol(
        legacy,
        explicit_legacy,
        resume_mode="exact",
    )

    with pytest.raises(ValueError, match="memory_fusion_placement"):
        experimental_train.validate_resume_training_protocol(
            legacy,
            {**legacy, "memory_fusion_placement": "post_attention_norm"},
            resume_mode="exact",
        )


def test_placement_ablation_protocol_allows_only_placement_and_horizon() -> None:
    source = _continuation_protocol()
    target = {
        **source,
        "max_steps": 160,
        "num_train_epochs": 5.0,
        "memory_fusion_placement": "post_attention_norm",
    }
    experimental_train.validate_resume_training_protocol(
        source,
        target,
        resume_mode="placement_ablation",
    )

    with pytest.raises(ValueError, match="requires memory_fusion_placement to change"):
        experimental_train.validate_resume_training_protocol(
            source,
            {**target, "memory_fusion_placement": "attention_output"},
            resume_mode="placement_ablation",
        )
    with pytest.raises(ValueError, match="learning_rate"):
        experimental_train.validate_resume_training_protocol(
            source,
            {**target, "learning_rate": 2e-3},
            resume_mode="placement_ablation",
        )


def test_placement_ablation_config_allows_only_placement() -> None:
    source = HFDeltaMemConfig(
        rank=2,
        delta_heads=("o",),
        memory_fusion_placement="attention_output",
    )
    target = HFDeltaMemConfig(
        rank=2,
        delta_heads=("o",),
        memory_fusion_placement="post_attention_norm",
    )
    experimental_train.validate_resume_delta_config(
        source,
        target,
        resume_mode="placement_ablation",
    )

    with pytest.raises(ValueError, match="rank"):
        experimental_train.validate_resume_delta_config(
            source,
            HFDeltaMemConfig(
                rank=4,
                delta_heads=("o",),
                memory_fusion_placement="post_attention_norm",
            ),
            resume_mode="placement_ablation",
        )


def test_prepare_placement_ablation_records_strict_lineage(tmp_path: Path) -> None:
    checkpoint = tmp_path / "source" / "trainer" / "checkpoint-128"
    source_protocol = _write_continuation_checkpoint(checkpoint)
    args = SimpleNamespace(
        resume_mode="placement_ablation",
        resume_from_checkpoint=str(checkpoint),
        output_dir=tmp_path / "target",
        max_steps=160,
        num_train_epochs=5.0,
        memory_fusion_placement="post_attention_norm",
    )

    manifest = experimental_train.prepare_training_continuation(args, str(checkpoint))

    assert manifest is not None
    assert manifest["mode"] == "placement_ablation"
    assert manifest["ablation"] == "memory_fusion_placement"
    assert manifest["source_memory_fusion_placement"] == "attention_output"
    assert manifest["target_memory_fusion_placement"] == "post_attention_norm"
    assert manifest["source_global_step"] == 128
    assert manifest["target_max_steps"] == 160
    assert manifest["source_training_protocol_sha256"] == (
        experimental_train._protocol_sha256(source_protocol)
    )


def test_prepare_placement_ablation_requires_fresh_output(tmp_path: Path) -> None:
    checkpoint = tmp_path / "source" / "trainer" / "checkpoint-128"
    _write_continuation_checkpoint(checkpoint)
    output_dir = tmp_path / "target"
    output_dir.mkdir()
    (output_dir / "existing.txt").touch()
    args = SimpleNamespace(
        resume_mode="placement_ablation",
        resume_from_checkpoint=str(checkpoint),
        output_dir=output_dir,
        max_steps=160,
        num_train_epochs=5.0,
        memory_fusion_placement="post_attention_norm",
    )

    with pytest.raises(ValueError, match="fresh, empty"):
        experimental_train.prepare_training_continuation(args, str(checkpoint))


def test_delta_mem_trainer_loads_placement_ablation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    checkpoint = tmp_path / "source" / "trainer" / "checkpoint-128"
    source_protocol = _write_continuation_checkpoint(checkpoint)
    target_protocol = {
        **source_protocol,
        "max_steps": 160,
        "num_train_epochs": 5.0,
        "memory_fusion_placement": "post_attention_norm",
    }
    loaded: list[tuple[Path, tuple[str, ...]]] = []
    monkeypatch.setattr(
        experimental_train,
        "validate_resume_adapter_topology",
        lambda model, source: "topology-sha",
    )
    monkeypatch.setattr(
        experimental_train,
        "load_delta_mem_adapter",
        lambda model, source, *, allowed_config_mismatches=(): loaded.append(
            (Path(source), allowed_config_mismatches)
        ),
    )
    trainer = object.__new__(experimental_train.DeltaMemTrainer)
    trainer.model = torch.nn.Linear(2, 2)
    trainer.delta_config = HFDeltaMemConfig(
        rank=2,
        memory_fusion_placement="post_attention_norm",
    )
    trainer.training_protocol = target_protocol
    trainer.resume_mode = "placement_ablation"
    trainer.continuation_manifest = {"mode": "placement_ablation"}

    trainer._load_from_checkpoint(str(checkpoint))

    assert loaded == [(checkpoint.resolve(), ("memory_fusion_placement",))]
    assert (
        trainer.continuation_manifest["ordered_adapter_parameter_topology_sha256"]
        == "topology-sha"
    )


def test_delta_mem_trainer_saves_ablation_lineage_manifest(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output = tmp_path / "checkpoint-160"
    manifest = {
        "schema_version": experimental_train._ABLATION_LINEAGE_SCHEMA_VERSION,
        "mode": "placement_ablation",
        "source_global_step": 128,
    }

    def fake_save_adapter(model, output_dir, active_config) -> None:
        del model, active_config
        Path(output_dir).mkdir(parents=True, exist_ok=True)

    monkeypatch.setattr(experimental_train, "save_delta_mem_adapter", fake_save_adapter)
    trainer = object.__new__(experimental_train.DeltaMemTrainer)
    trainer.args = SimpleNamespace(output_dir=str(output))
    trainer.model = torch.nn.Linear(2, 2)
    trainer.delta_config = HFDeltaMemConfig(rank=2)
    trainer.training_protocol = _continuation_protocol(max_steps=160)
    trainer.content_contrast_pairing_manifest = None
    trainer.continuation_manifest = manifest
    trainer.accelerator = SimpleNamespace(unwrap_model=lambda wrapped: wrapped)
    trainer.is_world_process_zero = lambda: True

    trainer.save_model(str(output))

    assert json.loads(
        (output / experimental_train._ABLATION_LINEAGE_FILENAME).read_text()
    ) == manifest


def test_objective_ablation_protocol_allows_only_strict_objective_transition() -> None:
    source = _objective_source_protocol()
    target = _objective_target_protocol()

    experimental_train.validate_resume_training_protocol(
        source,
        target,
        resume_mode="objective_ablation",
    )

    with pytest.raises(ValueError, match="learning_rate"):
        experimental_train.validate_resume_training_protocol(
            source,
            {**target, "learning_rate": 2e-3},
            resume_mode="objective_ablation",
        )
    with pytest.raises(ValueError, match="memory_contrast_weight to be positive"):
        experimental_train.validate_resume_training_protocol(
            source,
            {**target, "memory_contrast_weight": 0.0},
            resume_mode="objective_ablation",
        )
    with pytest.raises(ValueError, match="tokenized_fingerprint does not match"):
        invalid_pairing = {
            **_objective_pairing_summary(),
            "tokenized_fingerprint": "different-dataset",
        }
        experimental_train.validate_resume_training_protocol(
            source,
            {**target, "content_contrast_pairing": invalid_pairing},
            resume_mode="objective_ablation",
        )


def test_objective_ablation_config_must_be_exact() -> None:
    source = HFDeltaMemConfig(rank=2, delta_heads=("o",))
    experimental_train.validate_resume_delta_config(
        source,
        HFDeltaMemConfig(rank=2, delta_heads=("o",)),
        resume_mode="objective_ablation",
    )

    with pytest.raises(ValueError, match="rank"):
        experimental_train.validate_resume_delta_config(
            source,
            HFDeltaMemConfig(rank=4, delta_heads=("o",)),
            resume_mode="objective_ablation",
        )


def test_prepare_objective_ablation_records_strict_lineage(tmp_path: Path) -> None:
    checkpoint = tmp_path / "source" / "trainer" / "checkpoint-384"
    source_protocol = _write_continuation_checkpoint(
        checkpoint,
        protocol=_objective_source_protocol(),
        global_step=384,
        effective_max_steps=384,
        epoch=12.0,
    )

    manifest = experimental_train.prepare_training_continuation(
        _objective_args(checkpoint, tmp_path / "target"),
        str(checkpoint),
    )

    assert manifest is not None
    assert manifest["mode"] == "objective_ablation"
    assert manifest["ablation"] == "memory_training_objective"
    assert manifest["source_global_step"] == 384
    assert manifest["source_epoch"] == 12.0
    assert manifest["target_max_steps"] == 416
    assert manifest["source_memory_loss_mode"] == "context_dropout_ce"
    assert manifest["target_memory_loss_mode"] == "content_contrast_ce"
    assert manifest["target_memory_contrast_weight"] == 0.25
    assert manifest["target_memory_margin"] == 0.5
    assert manifest["source_training_protocol_sha256"] == (
        experimental_train._protocol_sha256(source_protocol)
    )


def test_prepare_objective_ablation_requires_epoch_boundary_and_fresh_output(
    tmp_path: Path,
) -> None:
    checkpoint = tmp_path / "source" / "trainer" / "checkpoint-384"
    _write_continuation_checkpoint(
        checkpoint,
        protocol=_objective_source_protocol(),
        global_step=384,
        effective_max_steps=384,
        epoch=11.5,
    )
    args = _objective_args(checkpoint, tmp_path / "target")
    with pytest.raises(ValueError, match="epoch boundary"):
        experimental_train.prepare_training_continuation(args, str(checkpoint))

    (checkpoint / "trainer_state.json").write_text(
        json.dumps({"global_step": 384, "max_steps": 384, "epoch": 12.0})
    )
    args.output_dir.mkdir()
    (args.output_dir / "existing.txt").touch()
    with pytest.raises(ValueError, match="fresh, empty"):
        experimental_train.prepare_training_continuation(args, str(checkpoint))


def test_delta_mem_trainer_loads_objective_ablation_without_source_pairing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    checkpoint = tmp_path / "source" / "trainer" / "checkpoint-384"
    _write_continuation_checkpoint(
        checkpoint,
        protocol=_objective_source_protocol(),
        global_step=384,
        effective_max_steps=384,
        epoch=12.0,
    )
    loaded: list[Path] = []
    monkeypatch.setattr(
        experimental_train,
        "validate_resume_adapter_topology",
        lambda model, source: "objective-topology-sha",
    )
    monkeypatch.setattr(
        experimental_train,
        "load_delta_mem_adapter",
        lambda model, source: loaded.append(Path(source)),
    )
    trainer = object.__new__(experimental_train.DeltaMemTrainer)
    trainer.model = torch.nn.Linear(2, 2)
    trainer.delta_config = HFDeltaMemConfig(rank=2)
    trainer.training_protocol = _objective_target_protocol()
    trainer.content_contrast_pairing_manifest = {"manifest_sha256": "a" * 64}
    trainer.resume_mode = "objective_ablation"
    trainer.continuation_manifest = {"mode": "objective_ablation"}

    trainer._load_from_checkpoint(str(checkpoint))

    assert loaded == [checkpoint.resolve()]
    assert (
        trainer.continuation_manifest["ordered_adapter_parameter_topology_sha256"]
        == "objective-topology-sha"
    )


def test_delta_mem_trainer_saves_objective_ablation_lineage(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output = tmp_path / "checkpoint-416"
    manifest = {
        "schema_version": experimental_train._ABLATION_LINEAGE_SCHEMA_VERSION,
        "mode": "objective_ablation",
        "source_global_step": 384,
    }

    def fake_save_adapter(model, output_dir, active_config) -> None:
        del model, active_config
        Path(output_dir).mkdir(parents=True, exist_ok=True)

    monkeypatch.setattr(experimental_train, "save_delta_mem_adapter", fake_save_adapter)
    trainer = object.__new__(experimental_train.DeltaMemTrainer)
    trainer.args = SimpleNamespace(output_dir=str(output))
    trainer.model = torch.nn.Linear(2, 2)
    trainer.delta_config = HFDeltaMemConfig(rank=2)
    trainer.training_protocol = _objective_target_protocol()
    trainer.content_contrast_pairing_manifest = {"manifest_sha256": "a" * 64}
    trainer.continuation_manifest = manifest
    trainer.accelerator = SimpleNamespace(unwrap_model=lambda wrapped: wrapped)
    trainer.is_world_process_zero = lambda: True

    trainer.save_model(str(output))

    assert json.loads(
        (output / experimental_train._ABLATION_LINEAGE_FILENAME).read_text()
    ) == manifest


def test_exact_resume_accepts_objective_ablation_lineage(tmp_path: Path) -> None:
    checkpoint = tmp_path / "trainer" / "checkpoint-416"
    checkpoint.mkdir(parents=True)
    manifest = {
        "schema_version": experimental_train._ABLATION_LINEAGE_SCHEMA_VERSION,
        "mode": "objective_ablation",
        "source_global_step": 384,
    }
    (checkpoint / experimental_train._ABLATION_LINEAGE_FILENAME).write_text(
        json.dumps(manifest)
    )
    args = SimpleNamespace(resume_mode="exact")

    assert experimental_train.prepare_training_continuation(
        args,
        str(checkpoint),
    ) == manifest
