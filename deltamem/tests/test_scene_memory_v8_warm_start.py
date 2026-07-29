from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import torch
from torch import nn

from experiments.rethinking_rwkv_ms_gemma import scene_memory_v8_warm_start as warm_start
from deltamem.train import delta_sft_experimental as experimental_train


REAL_LOCK_PATH = (
    Path(__file__).resolve().parents[2]
    / "experiments/rethinking_rwkv_ms_gemma"
    / "scene_memory_v8_v7_checkpoint256_lock.json"
)


class FakeAdapterModel(nn.Module):
    def __init__(self, state: dict[str, torch.Tensor]) -> None:
        super().__init__()
        self.adapter_state = {
            name: tensor.detach().cpu().clone() for name, tensor in state.items()
        }


def _json_file(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")


def _build_checkpoint_and_lock(tmp_path: Path) -> tuple[Path, Path]:
    checkpoint = tmp_path / "checkpoint-256"
    checkpoint.mkdir()
    source_state = {
        "layers.0.delta.weight": torch.tensor(
            [[1.0, 2.0], [3.0, 4.0]], dtype=torch.float32
        ),
        "layers.0.delta.bias": torch.tensor([5.0, 6.0], dtype=torch.bfloat16),
    }
    torch.save(source_state, checkpoint / "delta_mem_adapter.pt")

    expected_config = {
        "rank": 4,
        "alpha": 8.0,
        "memory_backend": "rwkv_ms",
        "target_layers": list(range(42)),
        "delta_heads": ["q", "o"],
        "rwkv_ms_semantics_version": 2,
    }
    expected_trainer_state = {
        "global_step": 256,
        "max_steps": 256,
        "epoch": 8.0,
        "num_train_epochs": 8,
    }
    expected_protocol = {
        "schema_version": 10,
        "memory_objective_version": "scene_state_generation_ce_v1",
        "memory_loss_mode": "scene_state_generation_ce",
        "max_steps": 256,
    }
    _json_file(checkpoint / "delta_mem_config.json", expected_config)
    _json_file(checkpoint / "trainer_state.json", expected_trainer_state)
    _json_file(
        checkpoint / "training_protocol.json",
        {
            **expected_protocol,
            "scene_state_identity_pairing": {"manifest_sha256": "pairing"},
        },
    )
    _json_file(
        checkpoint / "scene_state_identity_pairing_manifest.json",
        {"manifest_sha256": "pairing"},
    )
    _json_file(checkpoint / "continuation_manifest.json", {"mode": "extend"})
    torch.save({"state": {1: {}}}, checkpoint / "optimizer.pt")
    torch.save({"last_epoch": 256}, checkpoint / "scheduler.pt")
    torch.save({"cpu": torch.tensor([1])}, checkpoint / "rng_state.pth")

    topology, topology_sha256, tensor_elements = warm_start.ordered_adapter_topology(
        source_state
    )
    del topology
    artifacts = {}
    for filename in warm_start.REQUIRED_SOURCE_ARTIFACTS:
        path = checkpoint / filename
        artifacts[filename] = {
            "bytes": path.stat().st_size,
            "sha256": warm_start.sha256_file(path),
        }
    lock: dict[str, Any] = {
        "schema": warm_start.LOCK_SCHEMA,
        "source_checkpoint": str(checkpoint.resolve()),
        "artifacts": artifacts,
        "adapter_topology": {
            "sha256": topology_sha256,
            "tensor_count": len(source_state),
            "tensor_elements": tensor_elements,
        },
        "expected_delta_config": expected_config,
        "expected_trainer_state": expected_trainer_state,
        "expected_training_protocol": expected_protocol,
        "source_pairing_manifest_sha256": "pairing",
        "source_state_imports": {
            "adapter": True,
            "optimizer": False,
            "scheduler": False,
            "trainer_state": False,
            "rng": False,
            "global_step": False,
        },
        "target_fresh_start": {
            "global_step": 0,
            "optimizer_created_after_adapter_load": True,
            "optimizer_family": "AdamW",
            "optimizer_state": "fresh",
            "scheduler_state": "fresh",
            "trainer_state": "fresh",
            "rng_state": "fresh_from_v8_seed",
        },
    }
    lock["lock_sha256"] = warm_start.canonical_sha256(lock)
    lock_path = tmp_path / "v8_lock.json"
    lock_path.write_text(json.dumps(lock, indent=2), encoding="utf-8")
    return checkpoint, lock_path


def _fresh_start(**overrides: Any) -> warm_start.V8FreshStartContract:
    contract = warm_start.V8FreshStartContract(
        resume_from_checkpoint=None,
        initial_global_step=0,
        optimizer_created=False,
        scheduler_created=False,
        trainer_state_imported=False,
        rng_state_imported=False,
        optim="adamw_torch_fused",
    )
    return replace(contract, **overrides)


def _scene_v8_args(**overrides: Any) -> SimpleNamespace:
    values: dict[str, Any] = {
        "warm_start_mode": warm_start.WARM_START_MODE,
        "resume_from_checkpoint": None,
        "resume_mode": "exact",
        "memory_loss_mode": "scene_state_generation_ce",
        "target_layers": ",".join(str(index) for index in range(42)),
        "delta_heads": "q,o",
        "rank": 4,
        "alpha": 8.0,
        "memory_backend": "rwkv_ms",
        "rwkv_ms_num_states": 4,
        "rwkv_ms_chunk_size": 128,
        "rwkv_ms_semantics_version": 2,
        "output_init": "base_slice_fixed",
        "base_slice_ref_width": 8,
        "online_gain": 0.2,
        "memory_fusion_mode": "add",
        "memory_fusion_placement": "attention_output",
        "memory_fusion_residual_scale": 1.0,
        "memory_fusion_residual_scale_max": 1.0,
        "trainable_delta_scale": True,
        "delta_scale_init": 0.1,
        "delta_scale_max": 0.5,
        "delta_scale_granularity": "head",
        "delta_scale_parameterization": "alpha_over_rank",
        "memory_readout_mode": "delta",
        "memory_write_source": "learned_hidden",
        "memory_write_granularity": "token",
        "optim": "adamw_torch_fused",
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def _patch_fake_adapter_io(
    monkeypatch: pytest.MonkeyPatch,
    *,
    corrupt_after_load: bool = False,
) -> None:
    def get_state(model: FakeAdapterModel) -> dict[str, torch.Tensor]:
        return {
            name: tensor.detach().cpu().clone()
            for name, tensor in model.adapter_state.items()
        }

    def load_state(
        model: FakeAdapterModel,
        state: dict[str, torch.Tensor],
    ) -> None:
        model.adapter_state = {
            name: tensor.detach().cpu().clone() for name, tensor in state.items()
        }
        if corrupt_after_load:
            first_name = next(iter(model.adapter_state))
            model.adapter_state[first_name].flatten()[0] += 1

    monkeypatch.setattr(warm_start, "get_delta_mem_state_dict", get_state)
    monkeypatch.setattr(warm_start, "load_delta_mem_state_dict", load_state)


def test_prepare_pins_every_source_artifact_and_metadata(tmp_path: Path) -> None:
    checkpoint, lock_path = _build_checkpoint_and_lock(tmp_path)

    context = warm_start.prepare_v8_v7_checkpoint256_warm_start(
        checkpoint,
        lock_path=lock_path,
    )

    assert context.checkpoint == checkpoint.resolve()
    assert context.source_trainer_state["global_step"] == 256
    assert context.source_training_protocol["memory_loss_mode"] == (
        "scene_state_generation_ce"
    )
    assert context.source_config["target_layers"] == list(range(42))


def test_prepare_rejects_implicit_wrong_or_drifted_source(tmp_path: Path) -> None:
    checkpoint, lock_path = _build_checkpoint_and_lock(tmp_path)

    with pytest.raises(ValueError, match="explicit"):
        warm_start.prepare_v8_v7_checkpoint256_warm_start(None, lock_path=lock_path)
    with pytest.raises(ValueError, match="implicit"):
        warm_start.prepare_v8_v7_checkpoint256_warm_start(
            "latest", lock_path=lock_path
        )

    other = tmp_path / "other" / "checkpoint-256"
    other.mkdir(parents=True)
    with pytest.raises(ValueError, match="specifically pinned"):
        warm_start.prepare_v8_v7_checkpoint256_warm_start(
            other, lock_path=lock_path
        )

    (checkpoint / "optimizer.pt").write_bytes(b"drift")
    with pytest.raises(ValueError, match="optimizer.pt"):
        warm_start.prepare_v8_v7_checkpoint256_warm_start(
            checkpoint, lock_path=lock_path
        )


def test_ordered_topology_requires_name_shape_and_dtype_equality() -> None:
    source = {
        "first": torch.zeros(2, dtype=torch.float32),
        "second": torch.zeros(3, dtype=torch.bfloat16),
    }
    valid = {name: tensor.clone() for name, tensor in source.items()}
    receipt = warm_start.validate_ordered_adapter_topology(source, valid)
    assert receipt["ordered_parameter_names_equal"] is True
    assert receipt["ordered_shapes_equal"] is True
    assert receipt["ordered_dtypes_equal"] is True

    reversed_order = {
        "second": source["second"].clone(),
        "first": source["first"].clone(),
    }
    with pytest.raises(ValueError, match="ordered adapter parameter names"):
        warm_start.validate_ordered_adapter_topology(source, reversed_order)

    wrong_shape = {**valid, "second": torch.zeros(4, dtype=torch.bfloat16)}
    with pytest.raises(ValueError, match="shape differs"):
        warm_start.validate_ordered_adapter_topology(source, wrong_shape)

    wrong_dtype = {**valid, "second": torch.zeros(3, dtype=torch.float32)}
    with pytest.raises(ValueError, match="dtype differs"):
        warm_start.validate_ordered_adapter_topology(source, wrong_dtype)


def test_apply_imports_only_adapter_and_proves_post_load_bit_identity(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    checkpoint, lock_path = _build_checkpoint_and_lock(tmp_path)
    context = warm_start.prepare_v8_v7_checkpoint256_warm_start(
        checkpoint,
        lock_path=lock_path,
    )
    source = torch.load(
        checkpoint / "delta_mem_adapter.pt",
        map_location="cpu",
        weights_only=True,
    )
    model = FakeAdapterModel({name: torch.zeros_like(tensor) for name, tensor in source.items()})
    _patch_fake_adapter_io(monkeypatch)

    original_torch_load = torch.load
    loaded_paths: list[Path] = []

    def recording_load(path: str | Path, *args: Any, **kwargs: Any) -> Any:
        loaded_paths.append(Path(path).resolve())
        return original_torch_load(path, *args, **kwargs)

    monkeypatch.setattr(warm_start.torch, "load", recording_load)
    receipt = warm_start.apply_v8_v7_checkpoint256_adapter_only_warm_start(
        model,
        context,
        fresh_start=_fresh_start(),
    )

    assert loaded_paths == [(checkpoint / "delta_mem_adapter.pt").resolve()]
    assert receipt["post_load_bit_equal"] is True
    assert receipt["source_state_imports"] == {
        "adapter": True,
        "optimizer": False,
        "scheduler": False,
        "trainer_state": False,
        "rng": False,
        "global_step": False,
    }
    assert receipt["target_fresh_start"]["initial_global_step"] == 0
    assert receipt["target_fresh_start"]["optimizer_state"] == "fresh"
    for name, tensor in source.items():
        assert torch.equal(model.adapter_state[name], tensor)

    receipt_path = tmp_path / "receipt.json"
    assert warm_start.write_v8_warm_start_receipt(receipt, receipt_path) == (
        receipt_path.resolve()
    )
    with pytest.raises(FileExistsError):
        warm_start.write_v8_warm_start_receipt(receipt, receipt_path)


def test_apply_rejects_post_load_bit_drift(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    checkpoint, lock_path = _build_checkpoint_and_lock(tmp_path)
    context = warm_start.prepare_v8_v7_checkpoint256_warm_start(
        checkpoint,
        lock_path=lock_path,
    )
    source = torch.load(
        checkpoint / "delta_mem_adapter.pt",
        map_location="cpu",
        weights_only=True,
    )
    model = FakeAdapterModel({name: torch.zeros_like(tensor) for name, tensor in source.items()})
    _patch_fake_adapter_io(monkeypatch, corrupt_after_load=True)

    with pytest.raises(ValueError, match="not bit-equal"):
        warm_start.apply_v8_v7_checkpoint256_adapter_only_warm_start(
            model,
            context,
            fresh_start=_fresh_start(),
        )


def test_trainer_registers_and_validates_scene_v8_mode() -> None:
    assert warm_start.WARM_START_MODE in experimental_train._WARM_START_MODES
    experimental_train._validate_adapter_warm_start_args(_scene_v8_args())

    with pytest.raises(ValueError, match="target topology differs"):
        experimental_train._validate_adapter_warm_start_args(
            _scene_v8_args(delta_heads="o")
        )
    with pytest.raises(ValueError, match="cannot restore"):
        experimental_train._validate_adapter_warm_start_args(
            _scene_v8_args(resume_from_checkpoint="/tmp/checkpoint-256")
        )


def test_trainer_scene_v8_apply_dispatches_to_adapter_only_loader(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    checkpoint, lock_path = _build_checkpoint_and_lock(tmp_path)
    pinned_context = warm_start.prepare_v8_v7_checkpoint256_warm_start(
        checkpoint,
        lock_path=lock_path,
    )
    source = torch.load(
        checkpoint / "delta_mem_adapter.pt",
        map_location="cpu",
        weights_only=True,
    )
    model = FakeAdapterModel({name: torch.zeros_like(tensor) for name, tensor in source.items()})
    _patch_fake_adapter_io(monkeypatch)
    delta_config = experimental_train.HFDeltaMemConfig(
        rank=4,
        alpha=8.0,
        memory_backend="rwkv_ms",
        target_layers=tuple(range(42)),
        delta_heads=("q", "o"),
        rwkv_ms_num_states=4,
        rwkv_ms_chunk_size=128,
        rwkv_ms_semantics_version=2,
    )
    context = experimental_train.AdapterWarmStartContext(
        checkpoint=checkpoint,
        mode=warm_start.WARM_START_MODE,
        source_protocol=pinned_context.source_training_protocol,
        source_config=delta_config,
        manifest={
            "schema_version": 1,
            "mode": warm_start.WARM_START_MODE,
        },
        scene_v8_context=pinned_context,
        scene_v8_fresh_start=_fresh_start(),
    )

    receipt = experimental_train.apply_adapter_warm_start(
        model,
        context,
        delta_config,
        ["layers.0.delta.weight", "layers.0.delta.bias"],
    )

    assert receipt["mode"] == warm_start.WARM_START_MODE
    assert receipt["post_load_bit_equal"] is True
    assert receipt["target_fresh_start"]["initial_global_step"] == 0
    for name, tensor in source.items():
        assert torch.equal(model.adapter_state[name], tensor)


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"resume_from_checkpoint": "/tmp/checkpoint-256"}, "resume state"),
        ({"initial_global_step": 256}, "global step 0"),
        ({"optimizer_created": True}, "optimizer"),
        ({"scheduler_created": True}, "scheduler"),
        ({"trainer_state_imported": True}, "trainer_state"),
        ({"rng_state_imported": True}, "rng"),
        ({"optim": "sgd"}, "fresh AdamW"),
    ],
)
def test_fresh_start_rejects_all_training_state_imports(
    overrides: dict[str, Any],
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        warm_start.validate_v8_fresh_start_contract(_fresh_start(**overrides))


def test_repository_lock_is_self_consistent() -> None:
    lock = warm_start.load_v8_warm_start_lock(REAL_LOCK_PATH)
    assert lock["adapter_topology"] == {
        "sha256": "a8b72bc0ce82d7ecaafbfb76cdf8500465ab2b63771915840620f28c3cabbc77",
        "tensor_count": 1344,
        "tensor_elements": 2789808,
    }
    assert lock["expected_delta_config"]["target_layers"] == list(range(42))
    assert lock["expected_delta_config"]["delta_heads"] == ["q", "o"]


def test_repository_lock_validates_authoritative_checkpoint_when_present() -> None:
    lock = warm_start.load_v8_warm_start_lock(REAL_LOCK_PATH)
    checkpoint = Path(lock["source_checkpoint"])
    if not checkpoint.is_dir():
        pytest.skip("Pinned V7 checkpoint is not mounted")

    context = warm_start.prepare_v8_v7_checkpoint256_warm_start(
        checkpoint,
        lock_path=REAL_LOCK_PATH,
    )

    assert context.source_trainer_state["global_step"] == 256
    assert context.source_trainer_state["epoch"] == 8.0
    assert context.source_config["target_layers"] == list(range(42))
    assert context.source_config["delta_heads"] == ["q", "o"]


def test_trainer_resolves_and_prepares_authoritative_scene_v8_source_when_present(
    tmp_path: Path,
) -> None:
    lock = warm_start.load_v8_warm_start_lock(REAL_LOCK_PATH)
    checkpoint = Path(lock["source_checkpoint"])
    if not checkpoint.is_dir():
        pytest.skip("Pinned V7 checkpoint is not mounted")

    resolved = experimental_train.resolve_adapter_warm_start_checkpoint(
        checkpoint,
        warm_start_mode=warm_start.WARM_START_MODE,
    )
    args = _scene_v8_args(output_dir=tmp_path / "fresh_v8")
    context = experimental_train.prepare_adapter_warm_start(args, resolved)

    assert context is not None
    assert context.mode == warm_start.WARM_START_MODE
    assert context.checkpoint == checkpoint.resolve()
    assert context.scene_v8_context is not None
    assert context.scene_v8_fresh_start == _fresh_start()
