from __future__ import annotations

from argparse import Namespace
from contextlib import nullcontext
import json
from pathlib import Path
from types import SimpleNamespace

from datasets import Dataset
import pytest
import torch
import torch.nn.functional as F

import deltamem.train.delta_sft_experimental as experimental_train
from experiments.rethinking_rwkv_ms_gemma import prepare_scene_memory_v8_data as v8


_V8_CHECKPOINT_STEPS = (14, 28, 42, 56, 80, 104, 128, 152)


def _lineage_protocol(step: int) -> dict[str, object]:
    return {
        "max_steps": step,
        "num_train_epochs": 1.0,
        "lr_scheduler_type": "constant_with_warmup",
        "warmup_steps": 4,
        "train_schedule": {
            "schema": experimental_train._SCENE_MEMORY_V8_CURRICULUM_SCHEMA,
            "checkpoint_steps": list(_V8_CHECKPOINT_STEPS),
        },
    }


def _warm_lineage(protocol: dict[str, object]) -> dict[str, object]:
    lock = experimental_train.load_v8_warm_start_lock(
        experimental_train.SCENE_V8_WARM_START_LOCK_PATH
    )
    manifest = {
        "schema": experimental_train.SCENE_V8_WARM_START_RECEIPT_SCHEMA,
        "schema_version": experimental_train._WARM_START_LINEAGE_SCHEMA_VERSION,
        "mode": experimental_train._SCENE_V8_WARM_START_MODE,
        "source_checkpoint": lock["source_checkpoint"],
        "source_lock": {
            "path": str(
                experimental_train.SCENE_V8_WARM_START_LOCK_PATH.resolve()
            ),
            "lock_sha256": lock["lock_sha256"],
        },
        "source_global_step": 256,
        "source_artifacts": lock["artifacts"],
        "source_state_imports": lock["source_state_imports"],
        "post_load_bit_equal": True,
        "target_fresh_start": {
            "initial_global_step": 0,
            "optimizer_implementation": "adamw_torch_fused",
            "optimizer_created_after_adapter_load": True,
            "optimizer_state": "fresh",
            "scheduler_state": "fresh",
            "trainer_state": "fresh",
            "rng_state": "fresh_from_v8_seed",
        },
        "target_training_protocol_sha256": experimental_train._protocol_sha256(
            protocol
        ),
        "trainer_resume_from_checkpoint": None,
        "target_initial_global_step": 0,
        "pre_train_global_step": 0,
        "fresh_optimizer_created": True,
        "fresh_optimizer_class": "torch.optim.adamw.AdamW",
        "fresh_optimizer_state_entries_before_train": 0,
        "fresh_scheduler_created_before_train": False,
    }
    manifest["receipt_sha256"] = experimental_train._canonical_json_sha256(
        manifest
    )
    return manifest


def _write_lineage_checkpoint(
    checkpoint: Path,
    *,
    step: int,
    protocol: dict[str, object],
    lineage: dict[str, object],
) -> None:
    checkpoint.mkdir(parents=True)
    (checkpoint / "trainer_state.json").write_text(
        json.dumps({"global_step": step, "max_steps": step})
    )
    (checkpoint / experimental_train._TRAINING_PROTOCOL_FILENAME).write_text(
        json.dumps(protocol, indent=2, sort_keys=True)
    )
    lineage_filename = experimental_train._lineage_manifest_filename(lineage)
    (checkpoint / lineage_filename).write_text(
        json.dumps(lineage, indent=2, sort_keys=True)
    )


def _base_continuation(
    checkpoint: Path,
    *,
    source_protocol: dict[str, object],
    source_step: int,
    target_step: int,
) -> dict[str, object]:
    return {
        "schema_version": experimental_train._CONTINUATION_MANIFEST_SCHEMA_VERSION,
        "mode": "extend",
        "source_checkpoint": str(checkpoint.resolve()),
        "source_global_step": source_step,
        "source_effective_max_steps": source_step,
        "source_max_steps": source_step,
        "source_num_train_epochs": 1.0,
        "source_training_protocol_sha256": experimental_train._protocol_sha256(
            source_protocol
        ),
        "source_rng_state_files": ["rng_state.pth"],
        "target_max_steps": target_step,
        "target_num_train_epochs": 1.0,
        "lr_scheduler_type": "constant_with_warmup",
        "warmup_steps": 4,
    }


def _v8_args() -> Namespace:
    source_path = v8.DEFAULT_OUTPUT_DIR / "source_manifest.json"
    return Namespace(
        scene_state_source_manifest=source_path,
        expected_scene_state_source_manifest_sha256=(
            experimental_train._sha256_file(source_path)
        ),
        train_file=v8.V7_ROOT / "train32.jsonl",
    )


def _objective_trainer() -> experimental_train.DeltaMemTrainer:
    trainer = object.__new__(experimental_train.DeltaMemTrainer)
    trainer.episode_read_write_enabled = False
    trainer.memory_kl_weight = 0.0
    trainer.memory_base_kl_weight = 0.0
    trainer.memory_representation_weight = 0.0
    trainer.write_sparsity_weight = 0.0
    trainer.memory_partition_alignment_weight = 0.0
    trainer.memory_partition_entropy_weight = 0.0
    trainer.memory_partition_balance_weight = 0.0
    trainer.scene_boundary_payload_ce_weight = 0.0
    trainer.current_gradient_accumulation_steps = 1
    trainer.args = Namespace(optim="adamw_torch")
    trainer.compute_loss_context_manager = nullcontext

    class Accelerator:
        distributed_type = SimpleNamespace(name="NO")

        def __init__(self) -> None:
            self.backward_calls = 0

        def backward(self, loss: torch.Tensor, **kwargs) -> None:
            assert kwargs == {}
            self.backward_calls += 1
            loss.backward()

    trainer.accelerator = Accelerator()
    return trainer


def _model_inputs() -> dict[str, torch.Tensor]:
    return {
        "input_ids": torch.tensor([[7, 8, 0, 0, 0, 0]]),
        "attention_mask": torch.ones(1, 6, dtype=torch.long),
        "labels": torch.tensor([[-100, -100, 0, 0, 0, 0]]),
    }


def _generation_masks() -> dict[str, torch.Tensor]:
    return {
        "target_mask": torch.tensor([[False, False, True, True, True, True]]),
        "content_mask": torch.tensor([[False, False, True, True, True, False]]),
        "schema_mask": torch.tensor([[False, False, True, False, False, False]]),
        "decision_mask": torch.tensor([[False, False, False, True, True, False]]),
        "termination_mask": torch.tensor([[False, False, False, False, False, True]]),
        "pair_target_mask": torch.tensor(
            [[False, False, False, True, False, False]]
        ),
    }


class _SceneModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.parameter = torch.nn.Parameter(torch.tensor(0.2))
        self.active_write_token: int | None = None

    def forward(self, input_ids, attention_mask, **kwargs):
        del attention_mask, kwargs
        coefficient = {10: 1.0, 20: -1.0, None: 0.0}[
            self.active_write_token
        ]
        positions = torch.arange(
            1,
            input_ids.size(1) + 1,
            dtype=self.parameter.dtype,
            device=input_ids.device,
        )
        logits = input_ids.new_zeros(
            (input_ids.size(0), input_ids.size(1), 4),
            dtype=self.parameter.dtype,
        )
        logits[..., 0] = self.parameter * coefficient * positions
        logits[..., 1] = -self.parameter * coefficient * positions
        return {"logits": logits}


def _bind_state_controls(
    trainer: experimental_train.DeltaMemTrainer,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def reset(model: _SceneModel) -> None:
        model.active_write_token = None

    def prime(model: _SceneModel, **kwargs) -> None:
        model.active_write_token = int(kwargs["write_input_ids"][0, 0].item())

    trainer._reset_online_state = reset
    trainer._prime_episode_state = prime
    trainer._capture_live_online_state = lambda active_model: {
        "mock.delta_state": active_model.parameter.reshape(1)
    }
    monkeypatch.setattr(
        experimental_train,
        "set_delta_mem_write_enabled",
        lambda model, enabled: None,
    )
    monkeypatch.setattr(
        experimental_train,
        "set_delta_mem_read_context_mask",
        lambda model, mask: None,
    )


def _branch_kwargs(stratum: str) -> dict[str, torch.Tensor | None]:
    return {
        "write_input_ids": torch.tensor([[10]]),
        "write_attention_mask": torch.ones(1, 1, dtype=torch.long),
        "write_message_ids": None,
        "write_sentence_ids": None,
        "donor_write_input_ids": torch.tensor([[20]]),
        "donor_write_attention_mask": torch.ones(1, 1, dtype=torch.long),
        "donor_write_message_ids": None,
        "donor_write_sentence_ids": None,
        **_generation_masks(),
        "donor_target_token_ids": torch.tensor([1]),
        "target_stratum_codes": torch.tensor(
            [experimental_train._SCENE_STATE_IDENTITY_TARGET_STRATUM_CODES[stratum]]
        ),
    }


def test_v8_curriculum_binding_loads_exact_locked_indices_and_v7_pairs() -> None:
    binding = experimental_train._scene_state_v8_curriculum_binding(_v8_args())
    pairing = experimental_train._scene_state_generation_pairing_binding(_v8_args())
    schedule = [
        json.loads(line)
        for line in (v8.DEFAULT_OUTPUT_DIR / "schedule.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]

    assert binding is not None
    assert binding["total_steps"] == 152
    assert binding["schedule_file_sha256"] == (
        "64fb83996bf7b211505022b94f4fa2e5ee0ab9f1fe87fad0bc53cd536326ea8a"
    )
    assert binding["schedule_entries_sha256"] == (
        "979ca0c2dc253373eed6b4221cd6fa4c37f4a7a6e93173e8ce7f86f811e23df0"
    )
    assert binding["indices"] == tuple(
        entry["train_row_ordinal"] for entry in schedule
    )
    assert len(pairing["entries"]) == 32


def test_v8_locked_training_args_reject_objective_or_horizon_drift() -> None:
    binding = experimental_train._scene_state_v8_curriculum_binding(_v8_args())
    assert binding is not None
    values = {
        "training_mode": "episode",
        "assistant_loss_mode": "final_assistant_only",
        "episode_recent_messages": 0,
        "max_length": 256,
        "max_write_length": 2048,
        "episode_read_write_enabled": False,
        "memory_loss_mode": "scene_state_generation_ce",
        "scene_state_generated_unlikelihood_weight": 0.5,
        "scene_state_generated_unlikelihood_max_wrong_tokens": 4,
        "scene_state_generated_rollout_extra_tokens": 4,
        "scene_state_generated_rollout_max_tokens": 24,
        "scene_boundary_payload_ce_weight": 0.0,
        "memory_dropout_no_memory_prob": 0.0,
        "memory_dropout_state_only_prob": 0.0,
        "memory_base_kl_weight": 0.0,
        "memory_contrast_weight": 0.0,
        "memory_representation_weight": 0.0,
        "memory_kl_weight": 0.0,
        "memory_causal_weight": 0.0,
        "memory_anchor_weight": 0.0,
        "memory_recover_weight": 0.0,
        "write_sparsity_weight": 0.0,
        "memory_partition_alignment_weight": 0.0,
        "memory_partition_entropy_weight": 0.0,
        "memory_partition_balance_weight": 0.0,
        "per_device_train_batch_size": 1,
        "per_device_eval_batch_size": 1,
        "gradient_accumulation_steps": 1,
        "learning_rate": 2e-4,
        "lr_scheduler_type": "constant_with_warmup",
        "warmup_ratio": 0.0,
        "warmup_steps": 4,
        "weight_decay": 0.0,
        "optim": "adamw_torch_fused",
        "num_train_epochs": 1.0,
        "max_steps": 14,
        "logging_steps": 1,
        "save_steps": 14,
        "save_total_limit": 1,
        "validation_split_ratio": 0.0,
        "load_best_model_at_end": False,
        "dataset_num_proc": 1,
        "dataloader_num_workers": 0,
        "frozen_mlp_activation_checkpointing": True,
        "seed": 42,
        "data_seed": 42,
        "train_sampler_seed": None,
        "group_by_length": False,
        "initial_adapter_output_dir": None,
        "prepare_only": False,
        "resume_from_checkpoint": None,
        "warm_start_from_checkpoint": "/pinned/checkpoint-256",
        "warm_start_mode": experimental_train._SCENE_V8_WARM_START_MODE,
        "resume_mode": "exact",
    }

    experimental_train._validate_scene_state_v8_locked_training_args(
        Namespace(**values),
        binding,
    )
    with pytest.raises(ValueError, match="learning_rate"):
        experimental_train._validate_scene_state_v8_locked_training_args(
            Namespace(**{**values, "learning_rate": 5e-4}),
            binding,
        )
    with pytest.raises(ValueError, match="max_steps"):
        experimental_train._validate_scene_state_v8_locked_training_args(
            Namespace(**{**values, "max_steps": 28}),
            binding,
        )


def test_v8_resume_accepts_only_immediate_locked_endpoint() -> None:
    for source_step, target_step in zip(
        _V8_CHECKPOINT_STEPS,
        _V8_CHECKPOINT_STEPS[1:],
    ):
        experimental_train._validate_scene_memory_v8_resume_endpoint(
            source_global_step=source_step,
            target_max_steps=target_step,
            checkpoint_steps=_V8_CHECKPOINT_STEPS,
        )

    with pytest.raises(ValueError, match="immediate next locked endpoint"):
        experimental_train._validate_scene_memory_v8_resume_endpoint(
            source_global_step=14,
            target_max_steps=42,
            checkpoint_steps=_V8_CHECKPOINT_STEPS,
        )
    with pytest.raises(ValueError, match="immediate next locked endpoint"):
        experimental_train._validate_scene_memory_v8_resume_endpoint(
            source_global_step=28,
            target_max_steps=28,
            checkpoint_steps=_V8_CHECKPOINT_STEPS,
        )
    with pytest.raises(ValueError, match="final checkpoint"):
        experimental_train._validate_scene_memory_v8_resume_endpoint(
            source_global_step=152,
            target_max_steps=152,
            checkpoint_steps=_V8_CHECKPOINT_STEPS,
        )


def test_v8_fresh_optimizer_lineage_reaches_checkpoint_and_root_summary(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    protocol = _lineage_protocol(14)
    context = SimpleNamespace(
        mode=experimental_train._SCENE_V8_WARM_START_MODE,
        manifest=_warm_lineage(protocol),
    )
    trainer = object.__new__(experimental_train.DeltaMemTrainer)
    trainer.model = torch.nn.Linear(2, 2)
    trainer.state = SimpleNamespace(global_step=0)
    trainer.optimizer = None
    trainer.lr_scheduler = None
    trainer.continuation_manifest = {"stale": True}

    def create_optimizer() -> None:
        trainer.optimizer = torch.optim.AdamW(trainer.model.parameters(), lr=2e-4)

    trainer.create_optimizer = create_optimizer
    experimental_train.record_scene_v8_fresh_optimizer_lineage(trainer, context)

    assert trainer.continuation_manifest == context.manifest
    assert trainer.continuation_manifest is not context.manifest
    assert trainer.continuation_manifest["pre_train_global_step"] == 0
    assert trainer.continuation_manifest["fresh_optimizer_created"] is True
    assert (
        trainer.continuation_manifest[
            "fresh_optimizer_state_entries_before_train"
        ]
        == 0
    )
    unsigned = dict(trainer.continuation_manifest)
    recorded_receipt = unsigned.pop("receipt_sha256")
    assert recorded_receipt == experimental_train._canonical_json_sha256(unsigned)

    output = tmp_path / "trainer" / "checkpoint-14"

    def fake_save_adapter(model, output_dir, active_config) -> None:
        del model, active_config
        Path(output_dir).mkdir(parents=True, exist_ok=True)

    monkeypatch.setattr(experimental_train, "save_delta_mem_adapter", fake_save_adapter)
    trainer.args = SimpleNamespace(output_dir=str(output))
    trainer.delta_config = experimental_train.HFDeltaMemConfig(rank=4)
    trainer.training_protocol = protocol
    trainer.content_contrast_pairing_manifest = None
    trainer.scene_state_identity_pairing_manifest = None
    trainer.accelerator = SimpleNamespace(unwrap_model=lambda wrapped: wrapped)
    trainer.is_world_process_zero = lambda: True
    trainer.save_model(str(output))
    (output / "trainer_state.json").write_text(
        json.dumps({"global_step": 14, "max_steps": 14})
    )

    saved = json.loads(
        (output / experimental_train._WARM_START_LINEAGE_FILENAME).read_text()
    )
    root_summary = experimental_train._training_lineage_summary(trainer)
    assert saved == trainer.continuation_manifest
    assert root_summary["continuation"] == saved
    assert root_summary["resume_lineage"] == saved
    assert saved["pre_train_global_step"] == 0
    assert saved["fresh_optimizer_created"] is True
    assert saved["fresh_optimizer_state_entries_before_train"] == 0
    experimental_train._scene_memory_v8_checkpoint_lineage(
        output,
        checkpoint_steps=_V8_CHECKPOINT_STEPS,
    )


def test_v8_continuation_generation_save_and_recursive_validation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source_protocol = _lineage_protocol(14)
    source_checkpoint = tmp_path / "run14" / "trainer" / "checkpoint-14"
    source_warm_lineage = _warm_lineage(source_protocol)
    _write_lineage_checkpoint(
        source_checkpoint,
        step=14,
        protocol=source_protocol,
        lineage=source_warm_lineage,
    )
    continuation = experimental_train.prepare_scene_memory_v8_training_continuation(
        _base_continuation(
            source_checkpoint,
            source_protocol=source_protocol,
            source_step=14,
            target_step=28,
        ),
        resume_from_checkpoint=source_checkpoint,
        checkpoint_steps=_V8_CHECKPOINT_STEPS,
    )
    target_protocol = _lineage_protocol(28)
    experimental_train.finalize_scene_memory_v8_training_continuation(
        continuation,
        target_training_protocol=target_protocol,
    )
    experimental_train.validate_scene_memory_v8_active_continuation(
        continuation,
        resume_from_checkpoint=source_checkpoint,
        target_training_protocol=target_protocol,
        checkpoint_steps=_V8_CHECKPOINT_STEPS,
    )

    assert continuation["root_warm_start_receipt_sha256"] == (
        source_warm_lineage["receipt_sha256"]
    )
    assert continuation["source_lineage_filename"] == (
        experimental_train._WARM_START_LINEAGE_FILENAME
    )
    assert continuation["source_lineage_file_sha256"] == (
        experimental_train._sha256_file(
            source_checkpoint / experimental_train._WARM_START_LINEAGE_FILENAME
        )
    )
    assert continuation["source_training_protocol_sha256"] == (
        experimental_train._protocol_sha256(source_protocol)
    )
    assert continuation["target_training_protocol_sha256"] == (
        experimental_train._protocol_sha256(target_protocol)
    )
    unsigned = dict(continuation)
    manifest_sha256 = unsigned.pop("manifest_sha256")
    assert manifest_sha256 == experimental_train._canonical_json_sha256(unsigned)

    target_checkpoint = tmp_path / "run28" / "trainer" / "checkpoint-28"

    def fake_save_adapter(model, output_dir, active_config) -> None:
        del model, active_config
        Path(output_dir).mkdir(parents=True, exist_ok=True)

    monkeypatch.setattr(experimental_train, "save_delta_mem_adapter", fake_save_adapter)
    trainer = object.__new__(experimental_train.DeltaMemTrainer)
    trainer.args = SimpleNamespace(output_dir=str(target_checkpoint))
    trainer.model = torch.nn.Linear(2, 2)
    trainer.delta_config = experimental_train.HFDeltaMemConfig(rank=4)
    trainer.training_protocol = target_protocol
    trainer.content_contrast_pairing_manifest = None
    trainer.scene_state_identity_pairing_manifest = None
    trainer.continuation_manifest = dict(continuation)
    trainer.accelerator = SimpleNamespace(unwrap_model=lambda wrapped: wrapped)
    trainer.is_world_process_zero = lambda: True
    trainer.save_model(str(target_checkpoint))
    (target_checkpoint / "trainer_state.json").write_text(
        json.dumps({"global_step": 28, "max_steps": 28})
    )

    saved = json.loads(
        (
            target_checkpoint / experimental_train._CONTINUATION_MANIFEST_FILENAME
        ).read_text()
    )
    assert saved == continuation
    assert experimental_train._training_lineage_summary(trainer) == {
        "continuation": continuation,
        "resume_lineage": continuation,
    }

    next_continuation = (
        experimental_train.prepare_scene_memory_v8_training_continuation(
            _base_continuation(
                target_checkpoint,
                source_protocol=target_protocol,
                source_step=28,
                target_step=42,
            ),
            resume_from_checkpoint=target_checkpoint,
            checkpoint_steps=_V8_CHECKPOINT_STEPS,
        )
    )
    next_protocol = _lineage_protocol(42)
    experimental_train.finalize_scene_memory_v8_training_continuation(
        next_continuation,
        target_training_protocol=next_protocol,
    )
    experimental_train.validate_scene_memory_v8_active_continuation(
        next_continuation,
        resume_from_checkpoint=target_checkpoint,
        target_training_protocol=next_protocol,
        checkpoint_steps=_V8_CHECKPOINT_STEPS,
    )
    assert next_continuation["root_warm_start_receipt_sha256"] == (
        source_warm_lineage["receipt_sha256"]
    )
    assert next_continuation["source_lineage_filename"] == (
        experimental_train._CONTINUATION_MANIFEST_FILENAME
    )

    source_lineage_path = (
        target_checkpoint / experimental_train._CONTINUATION_MANIFEST_FILENAME
    )
    source_lineage_path.write_text(source_lineage_path.read_text() + "\n")
    with pytest.raises(ValueError, match="active continuation lineage differs"):
        experimental_train.validate_scene_memory_v8_active_continuation(
            next_continuation,
            resume_from_checkpoint=target_checkpoint,
            target_training_protocol=next_protocol,
            checkpoint_steps=_V8_CHECKPOINT_STEPS,
        )


def test_direct_trainer_v8_resume_rejects_skipped_endpoint(
    tmp_path: Path,
) -> None:
    source_protocol = _lineage_protocol(14)
    checkpoint = tmp_path / "run14" / "trainer" / "checkpoint-14"
    _write_lineage_checkpoint(
        checkpoint,
        step=14,
        protocol=source_protocol,
        lineage=_warm_lineage(source_protocol),
    )
    config = experimental_train.HFDeltaMemConfig(rank=4)
    config.save_pretrained(checkpoint)
    for filename in (
        "delta_mem_adapter.pt",
        "optimizer.pt",
        "scheduler.pt",
        "rng_state.pth",
    ):
        (checkpoint / filename).touch()
    source_lineage = experimental_train._scene_memory_v8_checkpoint_lineage(
        checkpoint,
        checkpoint_steps=_V8_CHECKPOINT_STEPS,
    )
    target_protocol = _lineage_protocol(42)
    invalid = _base_continuation(
        checkpoint,
        source_protocol=source_protocol,
        source_step=14,
        target_step=42,
    )
    invalid.update(
        {
            "root_warm_start_receipt_sha256": source_lineage[
                "root_warm_start_receipt_sha256"
            ],
            "source_lineage_filename": source_lineage["lineage_filename"],
            "source_lineage_file_sha256": source_lineage[
                "lineage_file_sha256"
            ],
            "target_training_protocol_sha256": (
                experimental_train._protocol_sha256(target_protocol)
            ),
        }
    )
    invalid["manifest_sha256"] = experimental_train._canonical_json_sha256(
        invalid
    )
    trainer = object.__new__(experimental_train.DeltaMemTrainer)
    trainer.model = torch.nn.Linear(2, 2)
    trainer.delta_config = config
    trainer.training_protocol = target_protocol
    trainer.content_contrast_pairing_manifest = None
    trainer.scene_state_identity_pairing_manifest = None
    trainer.resume_mode = "extend"
    trainer.continuation_manifest = invalid

    with pytest.raises(ValueError, match="immediate next locked endpoint"):
        trainer._load_from_checkpoint(str(checkpoint))


def test_fixed_schedule_sampler_survives_accelerator_dataloader_preparation(
    tmp_path,
) -> None:
    binding = experimental_train._scene_state_v8_curriculum_binding(_v8_args())
    assert binding is not None
    dataset = Dataset.from_dict({"row_index": list(range(32))})
    training_args = experimental_train.TrainingArguments(
        output_dir=str(tmp_path),
        per_device_train_batch_size=1,
        dataloader_pin_memory=False,
        report_to=["none"],
        remove_unused_columns=False,
    )
    trainer = experimental_train.DeltaMemTrainer(
        model=torch.nn.Linear(1, 1),
        args=training_args,
        train_dataset=dataset,
        data_collator=lambda features: {
            "row_index": torch.tensor(
                [feature["row_index"] for feature in features],
                dtype=torch.long,
            )
        },
        train_schedule_indices=binding["indices"],
        train_schedule_binding=binding,
    )

    observed = [
        int(batch["row_index"].item())
        for batch in trainer.get_train_dataloader()
    ]

    assert observed == list(binding["indices"])
    assert len(observed) == 152


def test_fixed_schedule_protocol_allows_horizon_extension_only_with_same_order() -> None:
    binding = experimental_train._scene_state_v8_curriculum_binding(_v8_args())
    assert binding is not None
    schedule = experimental_train._scene_state_v8_curriculum_protocol_summary(binding)
    source = {
        "train_sampler_seed": None,
        "train_sampler_mode": experimental_train._FIXED_TRAIN_SCHEDULE_SAMPLER_MODE,
        "train_schedule": schedule,
        "lr_scheduler_type": "constant_with_warmup",
        "warmup_steps": 4,
        "max_steps": 56,
        "num_train_epochs": 1.0,
    }
    target = {**source, "max_steps": 80}

    experimental_train.validate_resume_training_protocol(
        source,
        target,
        resume_mode="extend",
    )

    drifted_schedule = dict(schedule)
    drifted_schedule["schedule_entries_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="train_schedule"):
        experimental_train.validate_resume_training_protocol(
            source,
            {**target, "train_schedule": drifted_schedule},
            resume_mode="extend",
        )

    with pytest.raises(ValueError, match="seed/schedule"):
        experimental_train.validate_resume_training_protocol(
            source,
            {**target, "train_schedule": None},
            resume_mode="extend",
        )


def test_selected_gold_margin_math_and_gradient_match_full_reference() -> None:
    generator = torch.Generator().manual_seed(20260729)
    base_logits = torch.randn(2, 7, 11, generator=generator, dtype=torch.float64)
    labels = torch.tensor(
        [
            [-100, -100, 3, 4, 5, -100, -100],
            [-100, 2, 1, -100, 7, 8, -100],
        ]
    )
    target_mask = labels.ne(-100)
    selected_logits = base_logits.clone().requires_grad_()
    reference_logits = base_logits.clone().requires_grad_()

    actual = experimental_train.DeltaMemTrainer._scene_state_generation_gold_margins(
        selected_logits,
        labels,
        target_mask,
    )

    shift_mask = target_mask[:, 1:]
    shift_logits = reference_logits[:, :-1].float()
    shift_labels = labels[:, 1:]
    safe_labels = shift_labels.clamp_min(0)
    gold_logits = shift_logits.gather(-1, safe_labels.unsqueeze(-1)).squeeze(-1)
    top_values, top_indices = shift_logits.topk(k=2, dim=-1)
    max_other = torch.where(
        top_indices[..., 0].eq(safe_labels),
        top_values[..., 1],
        top_values[..., 0],
    )
    margins = gold_logits - max_other
    correct = top_indices[..., 0].eq(safe_labels) & shift_mask
    counts = shift_mask.sum(dim=1)
    reference_margin = (margins * shift_mask).sum(dim=1) / counts
    reference_accuracy = correct.sum(dim=1).float() / counts
    reference_first = []
    for row in range(2):
        positions = shift_mask[row].nonzero(as_tuple=False).flatten()
        wrong = (~correct[row].index_select(0, positions)).nonzero(
            as_tuple=False
        ).flatten()
        if wrong.numel() == 0:
            reference_first.append(reference_logits[row].sum() * 0.0)
        else:
            position = int(positions[int(wrong[0].item())].item())
            reference_first.append(
                F.relu(
                    experimental_train._SCENE_STATE_GENERATION_TOP1_MARGIN
                    + max_other[row, position]
                    - gold_logits[row, position]
                )
            )

    torch.testing.assert_close(actual[0], reference_margin, rtol=1e-6, atol=1e-7)
    torch.testing.assert_close(actual[1], reference_accuracy, rtol=0.0, atol=0.0)
    torch.testing.assert_close(
        actual[2],
        torch.stack(reference_first),
        rtol=1e-6,
        atol=1e-7,
    )
    actual_loss = actual[0].sum() + actual[2].sum()
    reference_loss = reference_margin.sum() + torch.stack(reference_first).sum()
    actual_loss.backward()
    reference_loss.backward()
    torch.testing.assert_close(
        selected_logits.grad,
        reference_logits.grad,
        rtol=1e-6,
        atol=1e-7,
    )


def test_generated_wrong_positions_use_first_divergence_and_cap() -> None:
    first, positions = (
        experimental_train.DeltaMemTrainer._scene_state_generated_wrong_positions(
            torch.tensor([1, 9, 8, 7, 6, 5]),
            torch.tensor([1, 2, 3, 4, 6, 0]),
            max_wrong_tokens=3,
        )
    )
    assert first == 1
    assert positions.tolist() == [1, 2, 3]

    exact_first, exact_positions = (
        experimental_train.DeltaMemTrainer._scene_state_generated_wrong_positions(
            torch.tensor([1, 2, 3]),
            torch.tensor([1, 2, 3]),
            max_wrong_tokens=4,
        )
    )
    assert exact_first == 3
    assert exact_positions.numel() == 0


def test_generated_unlikelihood_gradient_lowers_wrong_token_logits() -> None:
    logits = torch.tensor(
        [[2.0, 0.5, -1.0], [0.0, -0.5, 1.5]],
        requires_grad=True,
    )
    wrong_ids = torch.tensor([0, 2])

    loss = experimental_train.DeltaMemTrainer._scene_state_generated_unlikelihood_from_logits(
        logits,
        wrong_ids,
    )
    loss.backward()

    assert loss.item() > 0.0
    assert logits.grad[0, 0].item() > 0.0
    assert logits.grad[1, 2].item() > 0.0
    assert logits.grad[0, 1:].sum().item() < 0.0
    assert logits.grad[1, :2].sum().item() < 0.0


def test_generated_replay_reprime_backpropagates_through_memory_writer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    trainer = _objective_trainer()

    class WriterModel(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.writer = torch.nn.Parameter(torch.tensor(0.2))
            self.reader = torch.nn.Parameter(torch.tensor(0.3))
            self.online_state = None

        def forward(self, input_ids, attention_mask, **kwargs):
            del attention_mask, kwargs
            logits = torch.zeros(
                input_ids.size(0),
                input_ids.size(1),
                3,
                dtype=self.writer.dtype,
                device=input_ids.device,
            )
            logits[..., 0] = self.online_state * self.reader
            return {"logits": logits}

    model = WriterModel()
    trainer._reset_online_state = lambda active_model: setattr(
        active_model,
        "online_state",
        None,
    )

    def prime(active_model, **kwargs):
        active_model.online_state = (
            active_model.writer * kwargs["write_input_ids"][0, 0].float()
        )

    trainer._prime_episode_state = prime
    trainer._scene_state_generated_greedy_rollout = lambda *args, **kwargs: {
        "generated_token_ids": torch.tensor([0, 2]),
        "wrong_positions": torch.tensor([0]),
        "prompt_input_ids": torch.tensor([[7, 8]]),
        "prompt_attention_mask": torch.ones(1, 2, dtype=torch.long),
        "generation_start": 2,
        "first_divergence": 0,
        "exact_through_termination": False,
    }
    monkeypatch.setattr(
        experimental_train,
        "set_delta_mem_write_enabled",
        lambda active_model, enabled: None,
    )
    monkeypatch.setattr(
        experimental_train,
        "set_delta_mem_read_context_mask",
        lambda active_model, mask: None,
    )

    loss, _ = trainer._scene_state_generated_unlikelihood_branch(
        model,
        _model_inputs(),
        online_state_snapshot={"mock.delta_state": torch.ones(1)},
        write_input_ids=torch.tensor([[10]]),
        write_attention_mask=torch.ones(1, 1, dtype=torch.long),
        write_message_ids=None,
        write_sentence_ids=None,
        target_mask=_generation_masks()["target_mask"],
        termination_mask=_generation_masks()["termination_mask"],
    )
    assert loss is not None
    loss.backward()

    assert model.writer.grad is not None and model.writer.grad.abs().item() > 0.0
    assert model.reader.grad is not None and model.reader.grad.abs().item() > 0.0


def test_generated_unlikelihood_adds_one_backward_only_for_value_rows(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    trainer = _objective_trainer()
    trainer.scene_state_generated_unlikelihood_weight = 0.5
    model = _SceneModel()
    _bind_state_controls(trainer, monkeypatch)
    calls = []

    def generated_branch(*args, **kwargs):
        del args, kwargs
        calls.append("value")
        return model.parameter.square(), {
            "scene_generation_generated_unlikelihood_loss": float(
                model.parameter.detach().square().item()
            ),
            "scene_generation_generated_unlikelihood_applied": 1.0,
            "scene_generation_generated_wrong_token_count": 1.0,
            "scene_generation_generated_rollout_token_count": 4.0,
            "scene_generation_generated_first_divergence": 1.0,
            "scene_generation_generated_exact_fraction": 0.0,
        }

    trainer._scene_state_generated_unlikelihood_branch = generated_branch
    _, stats = trainer._scene_state_generation_sequential_backward(
        model,
        _model_inputs(),
        loss_kwargs={},
        gradient_scale=1.0,
        **_branch_kwargs("same_cardinality_value"),
    )

    assert calls == ["value"]
    assert trainer.accelerator.backward_calls == 3
    assert stats["scene_generation_generated_unlikelihood_weighted_loss"] == (
        pytest.approx(0.5 * 0.2**2)
    )

    presence_trainer = _objective_trainer()
    presence_trainer.scene_state_generated_unlikelihood_weight = 0.5
    presence_model = _SceneModel()
    _bind_state_controls(presence_trainer, monkeypatch)
    presence_trainer._scene_state_generated_unlikelihood_branch = generated_branch
    presence_trainer._scene_state_generation_sequential_backward(
        presence_model,
        _model_inputs(),
        loss_kwargs={},
        gradient_scale=1.0,
        **_branch_kwargs("presence"),
    )

    assert calls == ["value"]
    assert presence_trainer.accelerator.backward_calls == 2
