from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path
from typing import Any

import pytest
import torch
from torch import nn

from experiments.rethinking_rwkv_ms_gemma import scene_memory_v9_warm_start as warm_start


REAL_LOCK_PATH = (
    Path(__file__).resolve().parents[2]
    / "experiments/rethinking_rwkv_ms_gemma"
    / "scene_memory_v9_v8_checkpoint56_lock.json"
)


class FakeAdapterModel(nn.Module):
    def __init__(self, state: dict[str, torch.Tensor]) -> None:
        super().__init__()
        self.adapter_state = {
            name: tensor.detach().cpu().clone() for name, tensor in state.items()
        }


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _self_hashed(payload: dict[str, Any], field: str) -> dict[str, Any]:
    result = dict(payload)
    result[field] = warm_start.canonical_sha256(result)
    return result


def _binding(path: Path) -> dict[str, Any]:
    return {
        "bytes": path.stat().st_size,
        "sha256": warm_start.sha256_file(path),
    }


def _build_checkpoint_chain_and_lock(tmp_path: Path) -> tuple[Path, Path]:
    source_state = {
        "layers.0.delta.weight": torch.tensor(
            [[1.0, 2.0], [3.0, 4.0]], dtype=torch.float32
        ),
        "layers.0.delta.bias": torch.tensor([5.0, 6.0], dtype=torch.bfloat16),
    }
    expected_config = {
        "rank": 4,
        "alpha": 8.0,
        "memory_backend": "rwkv_ms",
        "target_layers": list(range(42)),
        "delta_heads": ["q", "o"],
        "rwkv_ms_semantics_version": 2,
    }
    expected_protocol = {
        "schema_version": 11,
        "memory_objective_version": (
            "scene_state_generation_ce_generated_prefix_unlikelihood_v2"
        ),
        "memory_loss_mode": "scene_state_generation_ce",
        "lr_scheduler_type": "constant_with_warmup",
        "warmup_steps": 4,
    }
    expected_pairing = {
        "schema_version": 2,
        "objective_version": "scene_state_generation_ce_v1",
        "pairing_version": "test_pairing_v1",
    }
    pairing = _self_hashed(expected_pairing, "manifest_sha256")
    root_source_checkpoint = str((tmp_path / "v7" / "checkpoint-256").resolve())
    root_source_lock = {
        "path": str((tmp_path / "v8_source_lock.json").resolve()),
        "lock_sha256": "v8-lock",
    }
    checkpoints: dict[int, Path] = {}
    protocols: dict[int, dict[str, Any]] = {}
    lineages: dict[int, dict[str, Any]] = {}
    root_receipt_sha256 = ""

    for step in warm_start.LINEAGE_STEPS:
        checkpoint = tmp_path / f"block-{step}" / f"checkpoint-{step}"
        checkpoint.mkdir(parents=True)
        checkpoints[step] = checkpoint.resolve()
        protocol = {
            **expected_protocol,
            "max_steps": step,
            "num_train_epochs": 1.0,
            "scene_state_identity_pairing": {
                "manifest_sha256": pairing["manifest_sha256"]
            },
        }
        protocols[step] = protocol
        _write_json(checkpoint / "delta_mem_config.json", expected_config)
        _write_json(
            checkpoint / "trainer_state.json",
            {
                "global_step": step,
                "max_steps": step,
                "num_train_epochs": 1,
                "epoch": step / 152,
            },
        )
        _write_json(checkpoint / "training_protocol.json", protocol)
        _write_json(
            checkpoint / "scene_state_identity_pairing_manifest.json", pairing
        )

        if step == warm_start.LINEAGE_STEPS[0]:
            lineage = {
                "schema": "rwkv_ms_scene_memory_v8_adapter_warm_start_receipt.v1",
                "schema_version": 1,
                "mode": "scene_memory_v8_v7_checkpoint256_adapter_only",
                "source_checkpoint": root_source_checkpoint,
                "source_lock": root_source_lock,
                "source_global_step": 256,
                "source_epoch": 8.0,
                "source_state_imports": warm_start.SOURCE_IMPORT_POLICY,
                "post_load_bit_equal": True,
                "target_fresh_start": {
                    "initial_global_step": 0,
                    "optimizer_created_after_adapter_load": True,
                    "optimizer_state": "fresh",
                    "scheduler_state": "fresh",
                    "trainer_state": "fresh",
                    "rng_state": "fresh_from_v8_seed",
                },
                "trainer_resume_from_checkpoint": None,
                "target_initial_global_step": 0,
                "pre_train_global_step": 0,
                "fresh_optimizer_created": True,
                "fresh_optimizer_state_entries_before_train": 0,
                "fresh_scheduler_created_before_train": False,
                "target_delta_config_sha256": warm_start.canonical_sha256(
                    expected_config
                ),
                "target_training_protocol_sha256": warm_start.canonical_sha256(
                    protocol
                ),
                "target_scene_state_pairing_manifest_sha256": pairing[
                    "manifest_sha256"
                ],
            }
            lineage = _self_hashed(lineage, "receipt_sha256")
            root_receipt_sha256 = lineage["receipt_sha256"]
            lineage_filename = warm_start.WARM_START_LINEAGE_FILENAME
        else:
            previous_step = warm_start.LINEAGE_STEPS[
                warm_start.LINEAGE_STEPS.index(step) - 1
            ]
            previous_checkpoint = checkpoints[previous_step]
            previous_lineage_filename = (
                warm_start.WARM_START_LINEAGE_FILENAME
                if previous_step == warm_start.LINEAGE_STEPS[0]
                else warm_start.CONTINUATION_LINEAGE_FILENAME
            )
            lineage = {
                "schema_version": 1,
                "mode": "extend",
                "source_checkpoint": str(previous_checkpoint),
                "source_global_step": previous_step,
                "source_effective_max_steps": previous_step,
                "source_max_steps": previous_step,
                "target_max_steps": step,
                "source_num_train_epochs": 1.0,
                "target_num_train_epochs": 1.0,
                "lr_scheduler_type": protocol["lr_scheduler_type"],
                "warmup_steps": protocol["warmup_steps"],
                "source_lineage_filename": previous_lineage_filename,
                "source_lineage_file_sha256": warm_start.sha256_file(
                    previous_checkpoint / previous_lineage_filename
                ),
                "root_warm_start_receipt_sha256": root_receipt_sha256,
                "source_training_protocol_sha256": warm_start.canonical_sha256(
                    protocols[previous_step]
                ),
                "target_training_protocol_sha256": warm_start.canonical_sha256(
                    protocol
                ),
            }
            lineage = _self_hashed(lineage, "manifest_sha256")
            lineage_filename = warm_start.CONTINUATION_LINEAGE_FILENAME
        lineages[step] = lineage
        _write_json(checkpoint / lineage_filename, lineage)

    source_checkpoint = checkpoints[56]
    torch.save(source_state, source_checkpoint / "delta_mem_adapter.pt")
    torch.save({"state": {1: {}}}, source_checkpoint / "optimizer.pt")
    torch.save({"last_epoch": 56}, source_checkpoint / "scheduler.pt")
    torch.save({"cpu": torch.tensor([1])}, source_checkpoint / "rng_state.pth")

    _, topology_sha256, tensor_elements = warm_start.ordered_adapter_topology(
        source_state
    )
    artifacts = {
        filename: _binding(source_checkpoint / filename)
        for filename in warm_start.REQUIRED_SOURCE_ARTIFACTS
    }
    continuation_lineage: list[dict[str, Any]] = []
    for step in warm_start.LINEAGE_STEPS:
        checkpoint = checkpoints[step]
        lineage_filename = (
            warm_start.WARM_START_LINEAGE_FILENAME
            if step == warm_start.LINEAGE_STEPS[0]
            else warm_start.CONTINUATION_LINEAGE_FILENAME
        )
        continuation_lineage.append(
            {
                "step": step,
                "checkpoint": str(checkpoint),
                "lineage_filename": lineage_filename,
                "epoch": step / 152,
                "delta_config_canonical_sha256": warm_start.canonical_sha256(
                    expected_config
                ),
                "training_protocol_canonical_sha256": warm_start.canonical_sha256(
                    protocols[step]
                ),
                "artifacts": {
                    filename: _binding(checkpoint / filename)
                    for filename in warm_start._lineage_artifact_names(step)
                },
            }
        )

    lock: dict[str, Any] = {
        "schema": warm_start.LOCK_SCHEMA,
        "source_checkpoint": str(source_checkpoint),
        "artifacts": artifacts,
        "continuation_lineage": continuation_lineage,
        "adapter_topology": {
            "sha256": topology_sha256,
            "tensor_count": len(source_state),
            "tensor_elements": tensor_elements,
        },
        "expected_delta_config": expected_config,
        "expected_training_protocol": expected_protocol,
        "expected_pairing_manifest": expected_pairing,
        "source_pairing_manifest_sha256": pairing["manifest_sha256"],
        "root_warm_start_receipt_sha256": root_receipt_sha256,
        "root_warm_start": {
            "schema": "rwkv_ms_scene_memory_v8_adapter_warm_start_receipt.v1",
            "schema_version": 1,
            "mode": "scene_memory_v8_v7_checkpoint256_adapter_only",
            "source_checkpoint": root_source_checkpoint,
            "source_lock": root_source_lock,
            "source_global_step": 256,
            "source_epoch": 8.0,
        },
        "source_state_imports": warm_start.SOURCE_IMPORT_POLICY,
        "target_fresh_start": warm_start.TARGET_FRESH_START_POLICY,
    }
    lock["lock_sha256"] = warm_start.canonical_sha256(lock)
    lock_path = tmp_path / "v9_lock.json"
    _write_json(lock_path, lock)
    return source_checkpoint, lock_path


def _fresh_start(**overrides: Any) -> warm_start.V9FreshStartContract:
    contract = warm_start.V9FreshStartContract(
        resume_from_checkpoint=None,
        initial_global_step=0,
        optimizer_created=False,
        scheduler_created=False,
        trainer_state_imported=False,
        rng_state_imported=False,
        optim="adamw_torch_fused",
    )
    return replace(contract, **overrides)


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


def test_prepare_pins_checkpoint56_and_complete_continuation_lineage(
    tmp_path: Path,
) -> None:
    checkpoint, lock_path = _build_checkpoint_chain_and_lock(tmp_path)

    context = warm_start.prepare_v9_v8_checkpoint56_warm_start(
        checkpoint,
        lock_path=lock_path,
    )

    assert context.checkpoint == checkpoint
    assert context.source_trainer_state["global_step"] == 56
    assert [entry["step"] for entry in context.continuation_lineage] == [14, 28, 42, 56]
    assert context.source_training_protocol["memory_objective_version"] == (
        "scene_state_generation_ce_generated_prefix_unlikelihood_v2"
    )
    assert context.source_pairing_manifest["objective_version"] == (
        "scene_state_generation_ce_v1"
    )


def test_prepare_rejects_implicit_wrong_or_drifted_source(tmp_path: Path) -> None:
    checkpoint, lock_path = _build_checkpoint_chain_and_lock(tmp_path)

    with pytest.raises(ValueError, match="explicit"):
        warm_start.prepare_v9_v8_checkpoint56_warm_start(None, lock_path=lock_path)
    with pytest.raises(ValueError, match="implicit"):
        warm_start.prepare_v9_v8_checkpoint56_warm_start(
            "latest", lock_path=lock_path
        )

    other = tmp_path / "other" / "checkpoint-56"
    other.mkdir(parents=True)
    with pytest.raises(ValueError, match="specifically pinned"):
        warm_start.prepare_v9_v8_checkpoint56_warm_start(
            other, lock_path=lock_path
        )

    (checkpoint / "optimizer.pt").write_bytes(b"drift")
    with pytest.raises(ValueError, match="optimizer.pt"):
        warm_start.prepare_v9_v8_checkpoint56_warm_start(
            checkpoint, lock_path=lock_path
        )


def test_prepare_rejects_drift_in_any_ancestor_lineage(tmp_path: Path) -> None:
    checkpoint, lock_path = _build_checkpoint_chain_and_lock(tmp_path)
    lock = warm_start.load_v9_warm_start_lock(lock_path)
    ancestor = Path(lock["continuation_lineage"][1]["checkpoint"])
    lineage_path = ancestor / warm_start.CONTINUATION_LINEAGE_FILENAME
    lineage_path.write_text(lineage_path.read_text(encoding="utf-8") + "\n")

    with pytest.raises(ValueError, match="checkpoint-28 artifact"):
        warm_start.prepare_v9_v8_checkpoint56_warm_start(
            checkpoint, lock_path=lock_path
        )


@pytest.mark.parametrize(
    ("lock_field", "wrong_value", "message"),
    [
        (
            "expected_training_protocol",
            {"memory_objective_version": "scene_state_generation_ce_v1"},
            "training protocol",
        ),
        (
            "expected_pairing_manifest",
            {
                "objective_version": (
                    "scene_state_generation_ce_generated_prefix_unlikelihood_v2"
                )
            },
            "pairing manifest",
        ),
    ],
)
def test_prepare_keeps_protocol_and_pairing_objective_versions_distinct(
    tmp_path: Path,
    lock_field: str,
    wrong_value: dict[str, Any],
    message: str,
) -> None:
    checkpoint, lock_path = _build_checkpoint_chain_and_lock(tmp_path)
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    lock[lock_field].update(wrong_value)
    lock.pop("lock_sha256")
    lock["lock_sha256"] = warm_start.canonical_sha256(lock)
    _write_json(lock_path, lock)

    with pytest.raises(ValueError, match=message):
        warm_start.prepare_v9_v8_checkpoint56_warm_start(
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


def test_apply_deserializes_only_adapter_and_proves_bit_identity(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    checkpoint, lock_path = _build_checkpoint_chain_and_lock(tmp_path)
    context = warm_start.prepare_v9_v8_checkpoint56_warm_start(
        checkpoint,
        lock_path=lock_path,
    )
    source = torch.load(
        checkpoint / "delta_mem_adapter.pt",
        map_location="cpu",
        weights_only=True,
    )
    model = FakeAdapterModel(
        {name: torch.zeros_like(tensor) for name, tensor in source.items()}
    )
    _patch_fake_adapter_io(monkeypatch)

    original_torch_load = torch.load
    loaded_paths: list[Path] = []

    def recording_load(path: str | Path, *args: Any, **kwargs: Any) -> Any:
        loaded_paths.append(Path(path).resolve())
        return original_torch_load(path, *args, **kwargs)

    monkeypatch.setattr(warm_start.torch, "load", recording_load)
    receipt = warm_start.apply_v9_v8_checkpoint56_adapter_only_warm_start(
        model,
        context,
        fresh_start=_fresh_start(),
    )

    assert loaded_paths == [(checkpoint / "delta_mem_adapter.pt").resolve()]
    assert receipt["post_load_bit_equal"] is True
    assert receipt["loaded_source_artifacts"] == ["delta_mem_adapter.pt"]
    assert receipt["validated_not_imported_source_artifacts"] == [
        "optimizer.pt",
        "scheduler.pt",
        "rng_state.pth",
        "trainer_state.json",
    ]
    assert receipt["source_state_imports"] == warm_start.SOURCE_IMPORT_POLICY
    assert receipt["target_fresh_start"]["initial_global_step"] == 0
    assert receipt["target_fresh_start"]["optimizer_state"] == "fresh"
    for name, tensor in source.items():
        assert torch.equal(model.adapter_state[name], tensor)

    receipt_path = tmp_path / "receipt.json"
    assert warm_start.write_v9_warm_start_receipt(receipt, receipt_path) == (
        receipt_path.resolve()
    )
    with pytest.raises(FileExistsError):
        warm_start.write_v9_warm_start_receipt(receipt, receipt_path)


def test_apply_rejects_post_load_bit_drift(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    checkpoint, lock_path = _build_checkpoint_chain_and_lock(tmp_path)
    context = warm_start.prepare_v9_v8_checkpoint56_warm_start(
        checkpoint,
        lock_path=lock_path,
    )
    source = torch.load(
        checkpoint / "delta_mem_adapter.pt",
        map_location="cpu",
        weights_only=True,
    )
    model = FakeAdapterModel(
        {name: torch.zeros_like(tensor) for name, tensor in source.items()}
    )
    _patch_fake_adapter_io(monkeypatch, corrupt_after_load=True)

    with pytest.raises(ValueError, match="not bit-equal"):
        warm_start.apply_v9_v8_checkpoint56_adapter_only_warm_start(
            model,
            context,
            fresh_start=_fresh_start(),
        )


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"resume_from_checkpoint": "/tmp/checkpoint-56"}, "resume state"),
        ({"initial_global_step": 56}, "global step 0"),
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
        warm_start.validate_v9_fresh_start_contract(_fresh_start(**overrides))


def test_repository_lock_is_self_consistent() -> None:
    lock = warm_start.load_v9_warm_start_lock(REAL_LOCK_PATH)
    assert lock["adapter_topology"] == {
        "sha256": "a8b72bc0ce82d7ecaafbfb76cdf8500465ab2b63771915840620f28c3cabbc77",
        "tensor_count": 1344,
        "tensor_elements": 2789808,
    }
    assert [entry["step"] for entry in lock["continuation_lineage"]] == [
        14,
        28,
        42,
        56,
    ]
    assert lock["expected_delta_config"]["target_layers"] == list(range(42))
    assert lock["expected_delta_config"]["delta_heads"] == ["q", "o"]
    assert lock["expected_training_protocol"]["memory_objective_version"].endswith(
        "unlikelihood_v2"
    )
    assert lock["expected_pairing_manifest"]["objective_version"] == (
        "scene_state_generation_ce_v1"
    )


def test_repository_lock_validates_authoritative_checkpoint_when_present() -> None:
    lock = warm_start.load_v9_warm_start_lock(REAL_LOCK_PATH)
    checkpoint = Path(lock["source_checkpoint"])
    if not checkpoint.is_dir():
        pytest.skip("Pinned V8 checkpoint-56 is not mounted")

    context = warm_start.prepare_v9_v8_checkpoint56_warm_start(
        checkpoint,
        lock_path=REAL_LOCK_PATH,
    )
    source_state = warm_start._load_pinned_adapter_state(context)
    topology, topology_sha256, tensor_elements = (
        warm_start.ordered_adapter_topology(source_state)
    )

    assert context.source_trainer_state["global_step"] == 56
    assert context.source_config["target_layers"] == list(range(42))
    assert context.source_config["delta_heads"] == ["q", "o"]
    assert len(topology) == context.lock["adapter_topology"]["tensor_count"]
    assert tensor_elements == context.lock["adapter_topology"]["tensor_elements"]
    assert topology_sha256 == context.lock["adapter_topology"]["sha256"]
