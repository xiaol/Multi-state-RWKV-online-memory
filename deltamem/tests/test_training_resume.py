from __future__ import annotations

import json
import sys
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


def test_exact_resume_binds_scene_boundary_payload_ce_weight_and_normalizes_legacy_zero() -> None:
    current = {
        **_objective_source_protocol(),
        "scene_boundary_payload_ce_weight": 0.0,
        "scene_boundary_payload_mask_mode": (
            experimental_train._SCENE_BOUNDARY_PAYLOAD_MASK_MODE
        ),
        "scene_boundary_payload_ce_normalization": (
            experimental_train._SCENE_BOUNDARY_PAYLOAD_CE_NORMALIZATION
        ),
    }
    legacy = dict(current)
    legacy.pop("scene_boundary_payload_ce_weight")
    legacy.pop("scene_boundary_payload_mask_mode")
    legacy.pop("scene_boundary_payload_ce_normalization")

    experimental_train.validate_resume_training_protocol(
        legacy,
        current,
        resume_mode="exact",
    )

    weighted = {**current, "scene_boundary_payload_ce_weight": 4.0}
    with pytest.raises(ValueError, match="scene_boundary_payload_ce_weight"):
        experimental_train.validate_resume_training_protocol(
            current,
            weighted,
            resume_mode="exact",
        )

    weighted_without_normalization = dict(weighted)
    weighted_without_normalization.pop("scene_boundary_payload_ce_normalization")
    with pytest.raises(ValueError, match="missing its payload CE normalization"):
        experimental_train.validate_resume_training_protocol(
            weighted_without_normalization,
            weighted,
            resume_mode="exact",
        )


def test_exact_resume_binds_seeded_train_sampler_and_normalizes_legacy_default() -> None:
    legacy = _objective_source_protocol()
    default_protocol = {
        **legacy,
        "train_sampler_seed": None,
        "train_sampler_mode": experimental_train._DEFAULT_TRAIN_SAMPLER_MODE,
    }

    experimental_train.validate_resume_training_protocol(
        legacy,
        default_protocol,
        resume_mode="exact",
    )

    seeded = {
        **default_protocol,
        "train_sampler_seed": 42,
        "train_sampler_mode": experimental_train._SEEDED_TRAIN_SAMPLER_MODE,
    }
    with pytest.raises(ValueError, match="train_sampler_mode, train_sampler_seed"):
        experimental_train.validate_resume_training_protocol(
            default_protocol,
            seeded,
            resume_mode="exact",
        )

    missing_mode = dict(seeded)
    missing_mode.pop("train_sampler_mode")
    with pytest.raises(ValueError, match="missing its train_sampler_mode"):
        experimental_train.validate_resume_training_protocol(
            missing_mode,
            seeded,
            resume_mode="exact",
        )


def _objective_pairing_summary() -> dict[str, object]:
    return {
        "pairing_version": experimental_train._CONTENT_CONTRAST_PAIRING_VERSION,
        "pairing_scope": "within_post_split_partition",
        "target_mode": experimental_train._CONTENT_CONTRAST_TARGET_MODE,
        "target_span_tokens": experimental_train._CONTENT_CONTRAST_TARGET_SPAN_TOKENS,
        "target_token_count": 256,
        "data_seed": 42,
        "tokenized_fingerprint": "fixed-dataset",
        "manifest_sha256": "a" * 64,
        "splits": {
            "train": {
                "sample_count": 32,
                "rotation": 16,
                "target_mode": experimental_train._CONTENT_CONTRAST_TARGET_MODE,
                "target_span_tokens": (
                    experimental_train._CONTENT_CONTRAST_TARGET_SPAN_TOKENS
                ),
                "target_token_count": 256,
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
        "memory_representation_weight": 0.1,
        "memory_representation_margin": 0.1,
        "memory_kl_weight": 0.0,
        "write_sparsity_weight": 0.0,
        "memory_partition_alignment_weight": 0.0,
        "memory_partition_entropy_weight": 0.0,
        "memory_partition_balance_weight": 0.0,
        "content_contrast_negative_priming_grad": True,
        "content_contrast_backward_mode": (
            experimental_train._CONTENT_CONTRAST_BACKWARD_MODE
        ),
        "content_contrast_read_mask_mode": (
            experimental_train._CONTENT_CONTRAST_READ_MASK_MODE
        ),
        "content_contrast_target_mode": (
            experimental_train._CONTENT_CONTRAST_TARGET_MODE
        ),
        "content_contrast_target_span_tokens": (
            experimental_train._CONTENT_CONTRAST_TARGET_SPAN_TOKENS
        ),
        "content_contrast_previous_source_grad": (
            experimental_train._CONTENT_CONTRAST_PREVIOUS_SOURCE_GRAD
        ),
        "content_contrast_representation_mode": (
            experimental_train._CONTENT_CONTRAST_REPRESENTATION_MODE
        ),
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
        memory_representation_weight=0.1,
        memory_representation_margin=0.1,
        memory_kl_weight=0.0,
        write_sparsity_weight=0.0,
        memory_partition_alignment_weight=0.0,
        memory_partition_entropy_weight=0.0,
        memory_partition_balance_weight=0.0,
    )


def _v14_pairing_summary() -> dict[str, object]:
    current = _objective_pairing_summary()
    train_split = dict(current["splits"]["train"])
    train_split.pop("target_mode")
    train_split.pop("target_span_tokens")
    train_split.pop("target_token_count")
    pairing = dict(current)
    pairing.pop("target_mode")
    pairing.pop("target_span_tokens")
    pairing.pop("target_token_count")
    pairing["splits"] = {"train": train_split}
    return pairing


def _v14_protocol() -> dict[str, object]:
    protocol = {
        **_objective_target_protocol(),
        "schema_version": experimental_train._RESIDUAL_HYBRID_W8_SOURCE_SCHEMA_VERSION,
        "memory_objective_version": (
            experimental_train._RESIDUAL_HYBRID_W8_SOURCE_OBJECTIVE_VERSION
        ),
        "content_contrast_representation_mode": (
            experimental_train._RESIDUAL_HYBRID_W8_SOURCE_REPRESENTATION_MODE
        ),
        "content_contrast_pairing": _v14_pairing_summary(),
        "memory_fusion_placement": "attention_output",
        "memory_fusion_residual_scale": 1.0,
        "warmup_ratio": experimental_train._RESIDUAL_HYBRID_W8_TARGET_WARMUP_RATIO,
        "warmup_steps": 8,
        "max_steps": experimental_train._RESIDUAL_HYBRID_W8_SOURCE_GLOBAL_STEP,
        "num_train_epochs": experimental_train._RESIDUAL_HYBRID_W8_SOURCE_EPOCH,
    }
    protocol.pop("content_contrast_target_mode")
    protocol.pop("content_contrast_target_span_tokens")
    protocol.pop("memory_fusion_residual_scale_max", None)
    return protocol


def _synthetic_v14_adapter_state() -> dict[str, torch.Tensor]:
    return {
        f"shared.{index:04d}": torch.tensor([float(index)], dtype=torch.float32)
        for index in range(
            experimental_train._RESIDUAL_HYBRID_W8_SOURCE_ADAPTER_TENSORS
        )
    }


def _residual_gain_names() -> tuple[str, ...]:
    return tuple(
        f"layer.{index}.memory_fusion_residual_gain_raw"
        for index in experimental_train._RESIDUAL_HYBRID_W8_TARGET_LAYERS
    )


def _synthetic_hybrid_target_state(
    source_state: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    target = {name: tensor.clone() for name, tensor in source_state.items()}
    for name in _residual_gain_names():
        target[name] = torch.zeros(1, dtype=torch.float32)
    return target


def _warm_start_args(checkpoint: Path, output_dir: Path) -> SimpleNamespace:
    return SimpleNamespace(
        warm_start_mode=experimental_train._RESIDUAL_HYBRID_W8_WARM_START_MODE,
        warm_start_from_checkpoint=str(checkpoint),
        resume_from_checkpoint=None,
        resume_mode="exact",
        output_dir=output_dir,
        memory_loss_mode="content_contrast_ce",
        memory_contrast_weight=0.25,
        memory_margin=0.5,
        memory_representation_weight=0.1,
        memory_representation_margin=0.1,
        memory_kl_weight=0.0,
        memory_base_kl_weight=0.0,
        write_sparsity_weight=0.0,
        memory_partition_alignment_weight=0.0,
        memory_partition_entropy_weight=0.0,
        memory_partition_balance_weight=0.0,
        lr_scheduler_type="constant_with_warmup",
        warmup_ratio=experimental_train._RESIDUAL_HYBRID_W8_TARGET_WARMUP_RATIO,
        max_steps=experimental_train._RESIDUAL_HYBRID_W8_TARGET_MAX_STEPS,
        num_train_epochs=experimental_train._RESIDUAL_HYBRID_W8_TARGET_EPOCHS,
        memory_fusion_placement=(
            experimental_train._RESIDUAL_HYBRID_W8_TARGET_PLACEMENT
        ),
        memory_fusion_residual_scale=(
            experimental_train._RESIDUAL_HYBRID_W8_TARGET_RESIDUAL_SCALE
        ),
        memory_fusion_residual_scale_max=(
            experimental_train._RESIDUAL_HYBRID_W8_TARGET_RESIDUAL_SCALE_MAX
        ),
        target_layers=",".join(
            str(index) for index in experimental_train._RESIDUAL_HYBRID_W8_TARGET_LAYERS
        ),
    )


def _write_v14_warm_start_checkpoint(path: Path) -> HFDeltaMemConfig:
    path.mkdir(parents=True)
    source_config = HFDeltaMemConfig(
        rank=2,
        delta_heads=("o",),
        memory_fusion_placement="attention_output",
        memory_fusion_residual_scale=1.0,
        target_layers=experimental_train._RESIDUAL_HYBRID_W8_TARGET_LAYERS,
    )
    source_config.save_pretrained(path)
    torch.save(_synthetic_v14_adapter_state(), path / "delta_mem_adapter.pt")
    parameter_ids = list(
        range(experimental_train._RESIDUAL_HYBRID_W8_SOURCE_OPTIMIZER_PARAMETERS)
    )
    torch.save(
        {
            "param_groups": [{"params": parameter_ids}],
            "state": {parameter_id: {} for parameter_id in parameter_ids},
        },
        path / "optimizer.pt",
    )
    torch.save({}, path / "scheduler.pt")
    torch.save({}, path / "rng_state.pth")
    (path / "trainer_state.json").write_text(
        json.dumps(
            {
                "global_step": experimental_train._RESIDUAL_HYBRID_W8_SOURCE_GLOBAL_STEP,
                "max_steps": experimental_train._RESIDUAL_HYBRID_W8_SOURCE_GLOBAL_STEP,
                "epoch": experimental_train._RESIDUAL_HYBRID_W8_SOURCE_EPOCH,
            }
        )
    )
    source_protocol = _v14_protocol()
    (path / experimental_train._TRAINING_PROTOCOL_FILENAME).write_text(
        json.dumps(source_protocol)
    )
    (path / experimental_train._CONTENT_CONTRAST_PAIRING_FILENAME).write_text(
        json.dumps(_v14_pairing_summary())
    )
    return source_config


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
        "content_contrast_backward_mode": (
            experimental_train._CONTENT_CONTRAST_BACKWARD_MODE
        ),
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
        "content_contrast_backward_mode": (
            experimental_train._CONTENT_CONTRAST_BACKWARD_MODE
        ),
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


def test_resume_protocol_normalizes_legacy_fusion_placement_and_scale() -> None:
    legacy = _continuation_protocol()
    explicit_legacy = {
        **legacy,
        "memory_fusion_placement": "attention_output",
        "memory_fusion_residual_scale": 1.0,
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

    with pytest.raises(ValueError, match="memory_fusion_residual_scale"):
        experimental_train.validate_resume_training_protocol(
            legacy,
            {**explicit_legacy, "memory_fusion_residual_scale": 0.875},
            resume_mode="exact",
        )


def test_placement_ablation_protocol_allows_only_fusion_fields_and_horizon() -> None:
    source = _continuation_protocol()
    target = {
        **source,
        "max_steps": 160,
        "num_train_epochs": 5.0,
        "memory_fusion_placement": "normalized_residual_correction",
        "memory_fusion_residual_scale": 0.875,
    }
    experimental_train.validate_resume_training_protocol(
        source,
        target,
        resume_mode="placement_ablation",
    )

    experimental_train.validate_resume_training_protocol(
        source,
        {
            **source,
            "max_steps": 160,
            "num_train_epochs": 5.0,
            "memory_fusion_residual_scale": 0.75,
        },
        resume_mode="placement_ablation",
    )

    with pytest.raises(
        ValueError,
        match="requires memory_fusion_placement or memory_fusion_residual_scale to change",
    ):
        experimental_train.validate_resume_training_protocol(
            source,
            {
                **target,
                "memory_fusion_placement": "attention_output",
                "memory_fusion_residual_scale": 1.0,
            },
            resume_mode="placement_ablation",
        )
    with pytest.raises(ValueError, match="learning_rate"):
        experimental_train.validate_resume_training_protocol(
            source,
            {**target, "learning_rate": 2e-3},
            resume_mode="placement_ablation",
        )


def test_placement_ablation_config_allows_only_fusion_fields() -> None:
    source = HFDeltaMemConfig(
        rank=2,
        delta_heads=("o",),
        memory_fusion_placement="attention_output",
    )
    target = HFDeltaMemConfig(
        rank=2,
        delta_heads=("o",),
        memory_fusion_placement="normalized_residual_correction",
        memory_fusion_residual_scale=0.875,
    )
    experimental_train.validate_resume_delta_config(
        source,
        target,
        resume_mode="placement_ablation",
    )

    experimental_train.validate_resume_delta_config(
        source,
        HFDeltaMemConfig(
            rank=2,
            delta_heads=("o",),
            memory_fusion_residual_scale=0.75,
        ),
        resume_mode="placement_ablation",
    )

    with pytest.raises(
        ValueError,
        match="requires memory_fusion_placement or memory_fusion_residual_scale to change",
    ):
        experimental_train.validate_resume_delta_config(
            source,
            HFDeltaMemConfig(rank=2, delta_heads=("o",)),
            resume_mode="placement_ablation",
        )

    with pytest.raises(ValueError, match="rank"):
        experimental_train.validate_resume_delta_config(
            source,
            HFDeltaMemConfig(
                rank=4,
                delta_heads=("o",),
                memory_fusion_placement="normalized_residual_correction",
                memory_fusion_residual_scale=0.875,
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
        memory_fusion_placement="normalized_residual_correction",
        memory_fusion_residual_scale=0.875,
    )

    manifest = experimental_train.prepare_training_continuation(args, str(checkpoint))

    assert manifest is not None
    assert manifest["mode"] == "placement_ablation"
    assert manifest["ablation"] == "memory_fusion_placement"
    assert manifest["source_memory_fusion_placement"] == "attention_output"
    assert manifest["target_memory_fusion_placement"] == "normalized_residual_correction"
    assert manifest["source_memory_fusion_residual_scale"] == 1.0
    assert manifest["target_memory_fusion_residual_scale"] == 0.875
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
        memory_fusion_placement="normalized_residual_correction",
        memory_fusion_residual_scale=0.875,
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
        "memory_fusion_placement": "normalized_residual_correction",
        "memory_fusion_residual_scale": 0.875,
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
        memory_fusion_placement="normalized_residual_correction",
        memory_fusion_residual_scale=0.875,
    )
    trainer.training_protocol = target_protocol
    trainer.resume_mode = "placement_ablation"
    trainer.continuation_manifest = {"mode": "placement_ablation"}

    trainer._load_from_checkpoint(str(checkpoint))

    assert loaded == [
        (
            checkpoint.resolve(),
            ("memory_fusion_placement", "memory_fusion_residual_scale"),
        )
    ]
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
    with pytest.raises(ValueError, match="content_contrast_backward_mode"):
        experimental_train.validate_resume_training_protocol(
            source,
            {**target, "content_contrast_backward_mode": "joint_graph_v1"},
            resume_mode="objective_ablation",
        )
    with pytest.raises(ValueError, match="content_contrast_representation_mode"):
        experimental_train.validate_resume_training_protocol(
            source,
            {**target, "content_contrast_representation_mode": "wrong_mode"},
            resume_mode="objective_ablation",
        )
    with pytest.raises(ValueError, match="content_contrast_target_mode"):
        experimental_train.validate_resume_training_protocol(
            source,
            {**target, "content_contrast_target_mode": "full_answer_v1"},
            resume_mode="objective_ablation",
        )
    with pytest.raises(ValueError, match="content_contrast_target_span_tokens"):
        experimental_train.validate_resume_training_protocol(
            source,
            {**target, "content_contrast_target_span_tokens": 1},
            resume_mode="objective_ablation",
        )
    with pytest.raises(ValueError, match="memory_representation_margin to be positive"):
        experimental_train.validate_resume_training_protocol(
            source,
            {**target, "memory_representation_margin": 0.0},
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


def test_exact_resume_rejects_legacy_full_answer_content_contrast_protocol() -> None:
    target = _objective_target_protocol()
    legacy = {
        **target,
        "schema_version": 7,
        "memory_objective_version": "content_contrast_ce_v5",
    }
    legacy.pop("content_contrast_target_mode")
    legacy.pop("content_contrast_target_span_tokens")

    with pytest.raises(
        ValueError,
        match="content_contrast_target_mode|memory_objective_version|schema_version",
    ):
        experimental_train.validate_resume_training_protocol(
            legacy,
            target,
            resume_mode="exact",
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
    assert manifest["target_content_contrast_backward_mode"] == (
        experimental_train._CONTENT_CONTRAST_BACKWARD_MODE
    )
    assert manifest["target_content_contrast_read_mask_mode"] == (
        experimental_train._CONTENT_CONTRAST_READ_MASK_MODE
    )
    assert manifest["target_content_contrast_target_mode"] == (
        experimental_train._CONTENT_CONTRAST_TARGET_MODE
    )
    assert manifest["target_content_contrast_target_span_tokens"] == (
        experimental_train._CONTENT_CONTRAST_TARGET_SPAN_TOKENS
    )
    assert manifest["target_content_contrast_previous_source_grad"] is True
    assert manifest["target_content_contrast_representation_mode"] == (
        experimental_train._CONTENT_CONTRAST_REPRESENTATION_MODE
    )
    assert manifest["target_memory_contrast_weight"] == 0.25
    assert manifest["target_memory_margin"] == 0.5
    assert manifest["target_memory_representation_weight"] == 0.1
    assert manifest["target_memory_representation_margin"] == 0.1
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


def test_warm_start_cli_is_mutually_exclusive_and_exposes_hybrid_gain_max(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    checkpoint = tmp_path / "checkpoint-416"
    argv = [
        "delta_sft",
        "--model-path",
        "base-model",
        "--output-dir",
        str(tmp_path / "output"),
        "--warm-start-from-checkpoint",
        str(checkpoint),
        "--warm-start-mode",
        experimental_train._RESIDUAL_HYBRID_W8_WARM_START_MODE,
        "--memory-fusion-placement",
        experimental_train._RESIDUAL_HYBRID_W8_TARGET_PLACEMENT,
        "--memory-fusion-residual-scale",
        str(experimental_train._RESIDUAL_HYBRID_W8_TARGET_RESIDUAL_SCALE),
        "--memory-fusion-residual-scale-max",
        str(experimental_train._RESIDUAL_HYBRID_W8_TARGET_RESIDUAL_SCALE_MAX),
    ]
    monkeypatch.setattr(sys, "argv", argv)

    args = experimental_train.parse_args()

    assert args.warm_start_from_checkpoint == str(checkpoint)
    assert args.memory_fusion_placement == "post_attention_residual_hybrid"
    assert args.memory_fusion_residual_scale == 0.01
    assert args.memory_fusion_residual_scale_max == 0.02

    monkeypatch.setattr(
        sys,
        "argv",
        [
            *argv,
            "--resume-from-checkpoint",
            str(checkpoint),
        ],
    )
    with pytest.raises(SystemExit):
        experimental_train.parse_args()


def test_residual_hybrid_w8_warm_start_args_are_exact(tmp_path: Path) -> None:
    checkpoint = tmp_path / "checkpoint-416"
    args = _warm_start_args(checkpoint, tmp_path / "target")

    experimental_train._validate_residual_hybrid_w8_warm_start_args(args)

    args.memory_fusion_residual_scale_max = 0.05
    with pytest.raises(ValueError, match="memory_fusion_residual_scale_max"):
        experimental_train._validate_residual_hybrid_w8_warm_start_args(args)


def test_prepare_adapter_warm_start_records_non_import_provenance(
    tmp_path: Path,
) -> None:
    checkpoint = tmp_path / "source" / "trainer" / "checkpoint-416"
    _write_v14_warm_start_checkpoint(checkpoint)
    args = _warm_start_args(checkpoint, tmp_path / "target")
    resolved = experimental_train.resolve_adapter_warm_start_checkpoint(checkpoint)

    context = experimental_train.prepare_adapter_warm_start(args, resolved)

    assert context is not None
    assert context.checkpoint == checkpoint.resolve()
    assert context.manifest["source_global_step"] == 416
    assert context.manifest["source_optimizer_parameter_count"] == 1218
    assert context.manifest["source_state_imports"] == {
        "adapter": True,
        "optimizer": False,
        "scheduler": False,
        "trainer_state": False,
        "rng": False,
    }
    assert context.manifest["source_optimizer_imported"] is False
    assert context.manifest["source_scheduler_imported"] is False
    assert context.manifest["source_trainer_state_imported"] is False
    assert context.manifest["source_rng_state_imported"] is False
    assert context.manifest["target_initial_global_step"] == 0
    assert context.manifest["target_max_steps"] == 32
    assert context.manifest["target_warmup_steps"] == 2
    assert context.manifest[
        "source_content_contrast_pairing_manifest_sha256"
    ] == _v14_pairing_summary()["manifest_sha256"]
    assert len(
        context.manifest["source_content_contrast_pairing_file_sha256"]
    ) == 64
    experimental_train.finalize_adapter_warm_start_lineage(
        context,
        target_training_protocol_sha256="d" * 64,
        target_pairing_manifest={"manifest_sha256": "e" * 64},
    )
    assert context.manifest["target_training_protocol_sha256"] == "d" * 64
    assert context.manifest[
        "target_content_contrast_pairing_manifest_sha256"
    ] == "e" * 64
    assert context.manifest["trainer_resume_from_checkpoint"] is None
    assert context.manifest["fresh_optimizer_created"] is True
    assert (
        experimental_train._lineage_manifest_filename(context.manifest)
        == experimental_train._WARM_START_LINEAGE_FILENAME
    )


def test_residual_hybrid_w8_adapter_topology_requires_only_42_new_gains() -> None:
    source_state = _synthetic_v14_adapter_state()
    target_state = _synthetic_hybrid_target_state(source_state)
    gain_names = _residual_gain_names()

    manifest = experimental_train.validate_residual_hybrid_w8_adapter_topology(
        source_state,
        target_state,
        gain_names=gain_names,
        source_optimizer_parameter_count=1218,
        target_trainable_tensor_count=1260,
    )

    assert manifest["source_adapter_tensor_count"] == 1470
    assert manifest["target_adapter_tensor_count"] == 1512
    assert manifest["shared_adapter_tensor_count"] == 1470
    assert manifest["new_residual_gain_tensor_count"] == 42
    assert manifest["target_trainable_tensor_count"] == 1260
    assert manifest["source_adapter_topology_sha256"] == manifest[
        "target_shared_adapter_topology_sha256"
    ]

    missing_gain = dict(target_state)
    missing_gain.pop(gain_names[-1])
    with pytest.raises(ValueError, match="target adapter must contain exactly 1512"):
        experimental_train.validate_residual_hybrid_w8_adapter_topology(
            source_state,
            missing_gain,
            gain_names=gain_names,
            source_optimizer_parameter_count=1218,
            target_trainable_tensor_count=1260,
        )

    shape_mismatch = dict(target_state)
    shape_mismatch[next(iter(source_state))] = torch.zeros(2)
    with pytest.raises(ValueError, match="shape mismatch"):
        experimental_train.validate_residual_hybrid_w8_adapter_topology(
            source_state,
            shape_mismatch,
            gain_names=gain_names,
            source_optimizer_parameter_count=1218,
            target_trainable_tensor_count=1260,
        )

    with pytest.raises(ValueError, match=r"source optimizer count \+ 42"):
        experimental_train.validate_residual_hybrid_w8_adapter_topology(
            source_state,
            target_state,
            gain_names=gain_names,
            source_optimizer_parameter_count=1218,
            target_trainable_tensor_count=1259,
        )


def test_apply_adapter_warm_start_loads_only_shared_weights_and_preserves_gains(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    checkpoint = tmp_path / "source" / "trainer" / "checkpoint-416"
    source_config = _write_v14_warm_start_checkpoint(checkpoint)
    context = experimental_train.prepare_adapter_warm_start(
        _warm_start_args(checkpoint, tmp_path / "target"),
        str(checkpoint),
    )
    assert context is not None
    target_payload = source_config.to_dict()
    target_payload.update(
        {
            "memory_fusion_placement": (
                experimental_train._RESIDUAL_HYBRID_W8_TARGET_PLACEMENT
            ),
            "memory_fusion_residual_scale": (
                experimental_train._RESIDUAL_HYBRID_W8_TARGET_RESIDUAL_SCALE
            ),
            "memory_fusion_residual_scale_max": (
                experimental_train._RESIDUAL_HYBRID_W8_TARGET_RESIDUAL_SCALE_MAX
            ),
        }
    )
    target_config = HFDeltaMemConfig.from_dict(target_payload)
    source_state = _synthetic_v14_adapter_state()
    active_state = {
        name: torch.zeros_like(tensor) for name, tensor in source_state.items()
    }
    for name in _residual_gain_names():
        active_state[name] = torch.tensor(0.01, dtype=torch.bfloat16).float().reshape(1)

    class FakeHybridModule:
        def __init__(self, layer_index: int) -> None:
            self.base = SimpleNamespace(layer_idx=layer_index)
            self.state_name = f"layer.{layer_index}.memory_fusion_residual_gain_raw"
            self.memory_fusion_residual_gain_raw = torch.nn.Parameter(torch.zeros(1))

        def set_memory_fusion_residual_gain(self, gain: float) -> None:
            active_state[self.state_name] = torch.tensor(
                [gain], dtype=torch.float32
            )

        def _resolved_memory_fusion_residual_gain(self, *, device, dtype):
            return active_state[self.state_name].to(device=device, dtype=dtype)[0]

    modules = [
        (f"layer.{index}", FakeHybridModule(index))
        for index in experimental_train._RESIDUAL_HYBRID_W8_TARGET_LAYERS
    ]
    load_calls: list[tuple[Path, bool]] = []

    monkeypatch.setattr(
        experimental_train,
        "get_delta_mem_state_dict",
        lambda model: dict(active_state),
    )
    monkeypatch.setattr(
        experimental_train,
        "iter_delta_mem_modules",
        lambda model: iter(modules),
    )

    def fake_load_adapter(model, source, *, initialize_missing_residual_hybrid_gain=False):
        del model
        load_calls.append((Path(source), initialize_missing_residual_hybrid_gain))
        for name, tensor in source_state.items():
            active_state[name] = tensor.clone()
        return source_config

    monkeypatch.setattr(experimental_train, "load_delta_mem_adapter", fake_load_adapter)
    trainable_names = [
        *list(source_state)[:1218],
        *_residual_gain_names(),
    ]
    manifest = experimental_train.apply_adapter_warm_start(
        object(),
        context,
        target_config,
        trainable_names,
    )

    assert load_calls == [(checkpoint, True)]
    assert manifest["shared_adapter_bit_equality_verified"] is True
    assert manifest["new_residual_gains_preserved_during_load"] is True
    assert manifest["target_effective_residual_gain_initial"] == 0.01
    assert manifest["target_effective_residual_gain_max"] == 0.02
    for name, source_tensor in source_state.items():
        torch.testing.assert_close(active_state[name], source_tensor, rtol=0.0, atol=0.0)
    for name in _residual_gain_names():
        torch.testing.assert_close(
            active_state[name],
            torch.tensor([0.01], dtype=torch.float32),
            rtol=0.0,
            atol=0.0,
        )


def test_residual_hybrid_w8_target_protocol_is_fresh_schema8_w8() -> None:
    source = _v14_protocol()
    target = {
        **_objective_target_protocol(),
        "memory_fusion_placement": (
            experimental_train._RESIDUAL_HYBRID_W8_TARGET_PLACEMENT
        ),
        "memory_fusion_residual_scale": (
            experimental_train._RESIDUAL_HYBRID_W8_TARGET_RESIDUAL_SCALE
        ),
        "memory_fusion_residual_scale_max": (
            experimental_train._RESIDUAL_HYBRID_W8_TARGET_RESIDUAL_SCALE_MAX
        ),
        "warmup_steps": experimental_train._RESIDUAL_HYBRID_W8_TARGET_WARMUP_STEPS,
        "max_steps": experimental_train._RESIDUAL_HYBRID_W8_TARGET_MAX_STEPS,
        "num_train_epochs": experimental_train._RESIDUAL_HYBRID_W8_TARGET_EPOCHS,
    }

    experimental_train.validate_residual_hybrid_w8_target_protocol(source, target)

    with pytest.raises(ValueError, match="warmup_steps"):
        experimental_train.validate_residual_hybrid_w8_target_protocol(
            source,
            {**target, "warmup_steps": 3},
        )


def test_warm_start_uses_fresh_step_zero_optimizer_and_two_step_warmup(
    tmp_path: Path,
) -> None:
    checkpoint = tmp_path / "source" / "trainer" / "checkpoint-416"
    _write_v14_warm_start_checkpoint(checkpoint)
    context = experimental_train.prepare_adapter_warm_start(
        _warm_start_args(checkpoint, tmp_path / "target"),
        str(checkpoint),
    )
    assert context is not None

    assert experimental_train.resolve_trainer_resume_checkpoint(None, context) is None
    with pytest.raises(RuntimeError, match="must not restore Trainer checkpoint state"):
        experimental_train.resolve_trainer_resume_checkpoint(str(checkpoint), context)
    assert experimental_train.compute_warmup_steps(
        train_samples=32,
        per_device_train_batch_size=1,
        world_size=1,
        gradient_accumulation_steps=1,
        num_train_epochs=1.0,
        max_steps=32,
        warmup_ratio=0.0625,
    ) == 2


def test_content_contrast_representation_allows_only_output_and_residual_hybrid() -> None:
    trainer = object.__new__(experimental_train.DeltaMemTrainer)
    trainer.episode_read_write_enabled = False
    trainer.memory_representation_weight = 0.1
    trainer.memory_representation_margin = 0.1
    trainer.memory_kl_weight = 0.0
    trainer.memory_base_kl_weight = 0.0
    trainer.write_sparsity_weight = 0.0
    trainer.memory_partition_alignment_weight = 0.0
    trainer.memory_partition_entropy_weight = 0.0
    trainer.memory_partition_balance_weight = 0.0
    trainer.delta_config = HFDeltaMemConfig(
        delta_heads=("o",),
        memory_fusion_placement="post_attention_residual_hybrid",
        memory_fusion_residual_scale=0.01,
        memory_fusion_residual_scale_max=0.02,
    )

    trainer._validate_content_contrast_runtime()

    trainer.delta_config = HFDeltaMemConfig(
        delta_heads=("o",),
        memory_fusion_placement="post_attention_norm",
    )
    with pytest.raises(ValueError, match="attention_output or post_attention_residual_hybrid"):
        trainer._validate_content_contrast_runtime()


def test_exact_resume_accepts_warm_start_lineage(tmp_path: Path) -> None:
    checkpoint = tmp_path / "trainer" / "checkpoint-16"
    checkpoint.mkdir(parents=True)
    manifest = {
        "schema_version": experimental_train._WARM_START_LINEAGE_SCHEMA_VERSION,
        "mode": experimental_train._RESIDUAL_HYBRID_W8_WARM_START_MODE,
        "source_global_step": 416,
        "source_optimizer_imported": False,
        "trainer_resume_from_checkpoint": None,
        "fresh_optimizer_created": True,
    }
    (checkpoint / experimental_train._WARM_START_LINEAGE_FILENAME).write_text(
        json.dumps(manifest)
    )

    assert experimental_train.prepare_training_continuation(
        SimpleNamespace(resume_mode="exact"),
        str(checkpoint),
    ) == manifest


def test_delta_mem_trainer_saves_finalized_warm_start_lineage(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output = tmp_path / "checkpoint-16"
    manifest = {
        "schema_version": experimental_train._WARM_START_LINEAGE_SCHEMA_VERSION,
        "mode": experimental_train._RESIDUAL_HYBRID_W8_WARM_START_MODE,
        "source_global_step": 416,
        "target_training_protocol_sha256": "d" * 64,
        "target_content_contrast_pairing_manifest_sha256": "e" * 64,
        "trainer_resume_from_checkpoint": None,
        "fresh_optimizer_created": True,
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
    trainer.content_contrast_pairing_manifest = {"manifest_sha256": "e" * 64}
    trainer.continuation_manifest = dict(manifest)
    trainer.accelerator = SimpleNamespace(unwrap_model=lambda wrapped: wrapped)
    trainer.is_world_process_zero = lambda: True

    trainer.save_model(str(output))

    saved = json.loads(
        (output / experimental_train._WARM_START_LINEAGE_FILENAME).read_text()
    )
    assert saved == manifest
    assert saved["trainer_resume_from_checkpoint"] is None
    assert saved["fresh_optimizer_created"] is True
