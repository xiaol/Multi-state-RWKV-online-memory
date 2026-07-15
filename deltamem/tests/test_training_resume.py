from __future__ import annotations

from pathlib import Path

import pytest
import torch

import deltamem.train.delta_sft_experimental as experimental_train
from deltamem.core.delta import HFDeltaMemConfig


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


def test_resolve_resume_checkpoint_uses_latest_complete_checkpoint(tmp_path: Path) -> None:
    trainer_output = tmp_path / "trainer"
    older = trainer_output / "checkpoint-100"
    _write_complete_checkpoint(older, HFDeltaMemConfig())
    incomplete = trainer_output / "checkpoint-200"
    incomplete.mkdir()
    (incomplete / "delta_mem_adapter.pt").touch()

    resolved = experimental_train.resolve_resume_checkpoint("latest", trainer_output)

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
