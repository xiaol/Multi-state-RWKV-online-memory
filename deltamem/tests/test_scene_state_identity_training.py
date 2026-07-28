from __future__ import annotations

from argparse import Namespace
from contextlib import nullcontext
import hashlib
import json
from types import SimpleNamespace
import sys

from datasets import Dataset
import pytest
import torch
import torch.nn.functional as F

import deltamem.train.delta_sft_experimental as experimental_train


def _trainer() -> experimental_train.DeltaMemTrainer:
    trainer = object.__new__(experimental_train.DeltaMemTrainer)
    trainer.scene_state_identity_margin = 0.5
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

        @staticmethod
        def backward(loss: torch.Tensor, **kwargs) -> None:
            assert kwargs == {}
            loss.backward()

    trainer.accelerator = Accelerator()
    return trainer


def _scene_inputs() -> dict[str, torch.Tensor]:
    return {
        "input_ids": torch.tensor([[7, 8, 0, 0, 0]]),
        "attention_mask": torch.ones(1, 5, dtype=torch.long),
        "labels": torch.tensor([[-100, -100, 0, 0, 0]]),
    }


def _semantic_mask() -> torch.Tensor:
    return torch.tensor([[False, False, True, True, True]])


def _pair_mask() -> torch.Tensor:
    return torch.tensor([[False, False, True, False, False]])


class _SceneStateModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.parameter = torch.nn.Parameter(torch.tensor(0.2))
        self.active_write_token: int | None = None
        self.read_calls: list[tuple[torch.Tensor, bool, int | None]] = []

    def forward(self, input_ids, attention_mask, labels=None, **kwargs):
        del attention_mask, labels, kwargs
        self.read_calls.append(
            (
                input_ids.detach().clone(),
                torch.is_grad_enabled(),
                self.active_write_token,
            )
        )
        coefficient = {10: 1.0, 20: -0.5, None: 0.25}[
            self.active_write_token
        ]
        position_scale = torch.tensor(
            [0.1, 0.3, 0.7, 1.1, 1.5],
            device=input_ids.device,
            dtype=self.parameter.dtype,
        )
        class_zero = torch.tensor(
            [1.0, 0.0, 0.0],
            device=input_ids.device,
            dtype=self.parameter.dtype,
        )
        logits = (
            self.parameter
            * coefficient
            * position_scale.view(1, -1, 1)
            * class_zero.view(1, 1, -1)
        ).expand(input_ids.size(0), -1, -1)
        full_ce = (self.parameter * coefficient - 0.3).square() + 1.0
        return {"loss": full_ce, "logits": logits}


def _bind_scene_state_controls(
    trainer: experimental_train.DeltaMemTrainer,
    monkeypatch: pytest.MonkeyPatch,
    events: list[tuple[str, object]] | None = None,
) -> None:
    if events is None:
        events = []

    def reset_online_state(model: _SceneStateModel) -> None:
        events.append(("reset", model.active_write_token))
        model.active_write_token = None

    def prime_episode_state(model: _SceneStateModel, **kwargs) -> None:
        token = int(kwargs["write_input_ids"][0, 0].item())
        events.append(("prime", (token, torch.is_grad_enabled())))
        model.active_write_token = token

    trainer._reset_online_state = reset_online_state
    trainer._prime_episode_state = prime_episode_state
    monkeypatch.setattr(
        experimental_train,
        "set_delta_mem_write_enabled",
        lambda model, enabled: events.append(("write", enabled)),
    )
    monkeypatch.setattr(
        experimental_train,
        "set_delta_mem_read_context_mask",
        lambda model, mask: events.append(("read_mask", mask)),
    )


def _scene_branch_kwargs() -> dict[str, torch.Tensor | None]:
    return {
        "write_input_ids": torch.tensor([[10]]),
        "write_attention_mask": torch.ones(1, 1, dtype=torch.long),
        "write_message_ids": None,
        "write_sentence_ids": None,
        "donor_write_input_ids": torch.tensor([[20]]),
        "donor_write_attention_mask": torch.ones(1, 1, dtype=torch.long),
        "donor_write_message_ids": None,
        "donor_write_sentence_ids": None,
        "semantic_mask": _semantic_mask(),
        "pair_target_mask": _pair_mask(),
        "target_stratum_codes": torch.tensor(
            [
                experimental_train._SCENE_STATE_IDENTITY_TARGET_STRATUM_CODES[
                    "same_cardinality_value"
                ]
            ]
        ),
    }


def test_scene_semantic_ce_normalizes_each_row_before_batch_mean() -> None:
    trainer = _trainer()
    logits = torch.tensor(
        [
            [[0.0, 0.0], [2.0, 0.0], [0.0, 0.0], [0.0, 0.0]],
            [[0.0, 0.0], [1.0, 0.0], [3.0, 0.0], [0.0, 0.0]],
        ],
        requires_grad=True,
    )
    labels = torch.tensor([[-100, 0, -100, -100], [-100, 0, 0, -100]])
    attention_mask = torch.ones_like(labels)
    semantic_mask = labels.ne(-100)

    mean_ce, row_ce, token_count = trainer._scene_state_semantic_ce(
        logits,
        labels,
        attention_mask,
        semantic_mask,
    )
    token_ce = F.cross_entropy(
        logits[:, :-1].reshape(-1, 2),
        labels[:, 1:].reshape(-1),
        reduction="none",
        ignore_index=-100,
    ).view(2, 3)
    expected_rows = torch.stack(
        (token_ce[0, 0], token_ce[1, :2].mean())
    )
    assert torch.allclose(row_ce, expected_rows)
    assert torch.allclose(mean_ce, expected_rows.mean())
    assert token_count == 3


def test_scene_identity_hinge_has_correct_per_branch_gradient_direction() -> None:
    trainer = _trainer()
    full = torch.tensor(2.0, requires_grad=True)
    correct_all = torch.tensor([1.0, 2.0], requires_grad=True)
    correct_pair = torch.tensor([1.5, 2.5], requires_grad=True)
    donor_pair = torch.tensor([1.6, 2.6], requires_grad=True)
    zero_all = torch.tensor([1.1, 2.1], requires_grad=True)

    values = trainer._scene_state_identity_objective(
        full,
        correct_all,
        correct_pair,
        donor_pair,
        zero_all,
    )
    values[0].backward()

    assert full.grad.item() == pytest.approx(1.0)
    assert correct_all.grad.tolist() == pytest.approx([0.5, 0.5])
    assert correct_pair.grad.tolist() == pytest.approx([0.5, 0.5])
    assert donor_pair.grad.tolist() == pytest.approx([-0.5, -0.5])
    assert zero_all.grad is None
    assert values[5].item() == pytest.approx(0.1)
    assert values[6].item() == pytest.approx(0.1)
    assert values[7].item() == pytest.approx(0.4)


def test_scene_identity_joint_branches_share_read_and_zero_keeps_adapter_active(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    trainer = _trainer()
    model = _SceneStateModel()
    events: list[tuple[str, object]] = []
    _bind_scene_state_controls(trainer, monkeypatch, events)

    loss, outputs, stats = trainer._compute_scene_state_identity_ce(
        model,
        _scene_inputs(),
        loss_kwargs={},
        **_scene_branch_kwargs(),
    )

    assert [call[2] for call in model.read_calls] == [10, 20, None]
    assert [call[1] for call in model.read_calls] == [True, True, False]
    assert all(
        torch.equal(model.read_calls[0][0], call[0]) for call in model.read_calls[1:]
    )
    assert [event for event in events if event[0] == "prime"] == [
        ("prime", (10, True)),
        ("prime", (20, True)),
    ]
    assert [event for event in events if event[0] == "write"][:3] == [
        ("write", False),
        ("write", False),
        ("write", False),
    ]
    assert stats["scene_state_semantic_token_count"] == 3.0
    assert stats["scene_state_semantic_row_count"] == 1.0
    assert stats["scene_state_target_same_cardinality_value_row_count"] == 1.0
    assert outputs["loss"] is loss
    loss.backward()
    assert model.parameter.grad is not None
    assert torch.isfinite(model.parameter.grad)


def test_scene_identity_sequential_backward_matches_joint_gradient(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    joint_trainer = _trainer()
    joint_model = _SceneStateModel()
    _bind_scene_state_controls(joint_trainer, monkeypatch)
    joint_loss, _, joint_stats = joint_trainer._compute_scene_state_identity_ce(
        joint_model,
        _scene_inputs(),
        loss_kwargs={},
        **_scene_branch_kwargs(),
    )
    joint_loss.backward()
    assert joint_model.parameter.grad is not None
    expected_gradient = joint_model.parameter.grad.detach().clone()

    replay_trainer = _trainer()
    replay_model = _SceneStateModel()
    _bind_scene_state_controls(replay_trainer, monkeypatch)
    replay_loss, replay_stats = (
        replay_trainer._scene_state_identity_sequential_backward(
            replay_model,
            _scene_inputs(),
            loss_kwargs={},
            gradient_scale=1.0,
            **_scene_branch_kwargs(),
        )
    )

    assert replay_model.parameter.grad is not None
    assert replay_loss.item() == pytest.approx(joint_loss.item())
    assert replay_model.parameter.grad.item() == pytest.approx(
        expected_gradient.item(),
        abs=1e-6,
    )
    for key in (
        "scene_state_correct_all_semantic_ce",
        "scene_state_correct_pair_semantic_ce",
        "scene_state_donor_pair_semantic_ce",
        "scene_state_zero_all_semantic_ce",
        "scene_state_donor_pair_gap",
        "scene_state_zero_all_gap",
        "scene_state_donor_margin_loss",
    ):
        assert replay_stats[key] == pytest.approx(joint_stats[key], abs=1e-6)
    assert [call[1] for call in replay_model.read_calls] == [
        False,
        False,
        True,
        True,
    ]
    assert [call[2] for call in replay_model.read_calls] == [
        20,
        None,
        10,
        20,
    ]


def _pairing_row(
    label_token: int,
    write_length: int,
    boundary_count: int = 1,
) -> dict[str, object]:
    write_ids = [label_token] * write_length
    input_ids = [7, 8, label_token, 99]
    labels = [-100, -100, label_token, 99]
    attention = [1] * len(input_ids)
    write_attention = [1] * write_length
    write_message_ids = [0] * write_length
    write_sentence_ids = [0] * write_length
    return {
        "input_ids": input_ids,
        "attention_mask": attention,
        "labels": labels,
        "scene_state_semantic_mask": [False, False, True, True],
        "scene_state_boundary_count": boundary_count,
        "write_input_ids": write_ids,
        "write_attention_mask": write_attention,
        "write_message_ids": write_message_ids,
        "write_sentence_ids": write_sentence_ids,
        "teacher_input_ids": input_ids,
        "teacher_attention_mask": attention,
        "teacher_labels": labels,
        "state_only_write_input_ids": write_ids,
        "state_only_write_attention_mask": write_attention,
        "state_only_write_message_ids": write_message_ids,
        "state_only_write_sentence_ids": write_sentence_ids,
        "state_only_input_ids": input_ids,
        "state_only_attention_mask": attention,
        "state_only_labels": labels,
    }


def test_scene_identity_target_requires_identical_causal_prefixes() -> None:
    source = _pairing_row(10, 2)
    donor = _pairing_row(20, 3)

    target_mask, metadata = (
        experimental_train._select_scene_state_identity_target_with_metadata(
            source,
            donor,
        )
    )
    assert target_mask == [False, False, True, False]
    assert metadata["causal_prefix_token_count"] == 2
    assert metadata["causal_prefix_mode"] == (
        "exact_input_ids_and_attention_before_pair_target_v1"
    )
    assert metadata["target_label_positions"] == [2]
    assert metadata["donor_target_label_positions"] == [2]

    donor["input_ids"][1] = 999
    with pytest.raises(ValueError, match="causal prefixes differ"):
        experimental_train.select_scene_state_identity_target(source, donor)


def test_scene_pairing_is_symmetric_nearest_distinct_and_audited() -> None:
    source = Dataset.from_list(
        [
            _pairing_row(10, 2, 0),
            _pairing_row(20, 3, 1),
            _pairing_row(30, 10, 2),
            _pairing_row(40, 11, 2),
            _pairing_row(50, 20, 1),
            _pairing_row(60, 21, 3),
        ]
    )
    paired, manifest = experimental_train.materialize_scene_state_identity_pairs(
        source,
        split_name="train",
    )
    paired_again, manifest_again = (
        experimental_train.materialize_scene_state_identity_pairs(
            source,
            split_name="train",
        )
    )

    assert paired["scene_state_donor_index"] == [1, 0, 3, 2, 5, 4]
    assert paired_again["scene_state_donor_index"] == [1, 0, 3, 2, 5, 4]
    assert manifest == manifest_again
    assert manifest["pair_count"] == 3
    assert manifest["target_token_count"] == 6
    assert manifest["target_stratum_row_counts"] == {
        "presence": 2,
        "same_cardinality_value": 2,
        "cross_cardinality_value": 2,
    }
    assert manifest["source_boundary_count_histogram"] == {
        "0": 1,
        "1": 2,
        "2": 2,
        "3": 1,
    }
    assert manifest["write_token_count_delta_max"] == 1
    assert manifest["write_token_count_delta_mean"] == pytest.approx(1.0)
    aggregate_manifest = (
        experimental_train.build_scene_state_identity_pairing_manifest(
            tokenized_fingerprint="source-fingerprint",
            tokenized_dataset_sha256="a" * 64,
            data_seed=42,
            train_manifest=manifest,
            eval_manifest=None,
        )
    )
    assert aggregate_manifest["schema_version"] == 2
    assert aggregate_manifest["target_stratum_row_counts"] == manifest[
        "target_stratum_row_counts"
    ]
    assert aggregate_manifest["source_boundary_count_histogram"] == manifest[
        "source_boundary_count_histogram"
    ]
    for index, donor_index in enumerate(paired["scene_state_donor_index"]):
        assert paired[donor_index]["scene_state_donor_index"] == index
        assert paired[index]["scene_state_source_label_sha256"] != paired[index][
            "scene_state_donor_label_sha256"
        ]
        assert sum(paired[index]["scene_state_identity_target_mask"]) == 1
        assert paired[index]["scene_state_identity_target_mask"][2]
        assert paired[index]["scene_state_donor_boundary_count"] == paired[
            donor_index
        ]["scene_state_boundary_count"]


def test_scene_pairing_refines_cardinality_within_nearest_length_budget() -> None:
    source = Dataset.from_list(
        [
            _pairing_row(10, 1, 0),
            _pairing_row(20, 2, 1),
            _pairing_row(30, 2, 1),
            _pairing_row(40, 3, 2),
            _pairing_row(50, 10, 0),
            _pairing_row(60, 12, 2),
        ]
    )

    paired, manifest = experimental_train.materialize_scene_state_identity_pairs(
        source,
        split_name="train",
    )

    assert paired["scene_state_donor_index"] == [3, 2, 1, 0, 5, 4]
    assert manifest["pairing_refinement_applied"] is True
    assert manifest["target_stratum_row_counts"] == {
        "presence": 4,
        "same_cardinality_value": 2,
        "cross_cardinality_value": 0,
    }
    assert manifest["write_token_count_delta_total"] == 4
    assert manifest["write_token_count_delta_max"] == 2
    assert manifest["nearest_baseline_write_token_count_delta_total"] == 4
    assert manifest["nearest_baseline_write_token_count_delta_max"] == 2


class _Tokenizer:
    pad_token_id = 0


def test_scene_pairing_collator_validates_hashes_and_pads_masks() -> None:
    source = Dataset.from_list(
        [_pairing_row(10, 2), _pairing_row(20, 3)]
    )
    paired, _ = experimental_train.materialize_scene_state_identity_pairs(
        source,
        split_name="train",
    )
    features = [paired[index] for index in range(2)]
    batch = experimental_train.EpisodeCausalLMCollator(_Tokenizer())(features)

    assert batch["scene_state_semantic_mask"].dtype == torch.bool
    assert batch["scene_state_identity_target_mask"].dtype == torch.bool
    assert batch["scene_state_identity_target_mask"].sum(dim=1).tolist() == [1, 1]
    assert batch["scene_state_identity_target_stratum"].tolist() == [1, 1]
    assert batch["scene_state_donor_write_input_ids"].shape == (2, 3)

    corrupted = dict(features[0])
    corrupted["scene_state_donor_write_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="donor write"):
        experimental_train.EpisodeCausalLMCollator(_Tokenizer())(
            [corrupted, features[1]]
        )

    corrupted = dict(features[0])
    corrupted["scene_state_donor_boundary_count"] = 2
    with pytest.raises(ValueError, match="target stratum"):
        experimental_train.EpisodeCausalLMCollator(_Tokenizer())(
            [corrupted, features[1]]
        )


def test_scene_semantic_mask_keeps_json_decisions_and_excludes_whitespace(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    content = '{"boundaries": [ 1,\n 2 ]}'
    payload_start = content.index("[")
    payload_end = content.index("]") + 1
    offsets = [
        (0, payload_start),
        (payload_start, payload_start + 1),
        (payload_start + 1, payload_start + 2),
        (payload_start + 2, payload_start + 3),
        (payload_start + 3, payload_start + 5),
        (payload_start + 5, payload_start + 7),
        (payload_end - 1, payload_end),
    ]
    ids = list(range(len(offsets)))
    monkeypatch.setattr(
        experimental_train,
        "_rendered_message_content_span",
        lambda tokenizer, messages, message_index: (content, 0, len(content)),
    )
    monkeypatch.setattr(
        experimental_train,
        "_tokenizer_ids_and_offsets",
        lambda tokenizer, rendered: (ids, offsets),
    )

    mask = experimental_train._scene_boundary_semantic_token_mask(
        object(),
        [{"role": "assistant", "content": content}],
        0,
        ids,
    )

    assert mask == [False, True, False, True, True, True, True]
    assert experimental_train._scene_boundary_payload_metadata(content) == (
        (payload_start, payload_end),
        2,
    )


def _source_manifest(tmp_path) -> tuple[object, object, str]:
    train_file = tmp_path / "train.jsonl"
    train_file.write_text('{"messages": []}\n')
    train_sha = hashlib.sha256(train_file.read_bytes()).hexdigest()
    manifest = {
        "schema": "rwkv_ms_scene_failure_pairs.v1",
        "contract": {
            "episode_contract": {
                "episode_recent_messages": 0,
                "write_phase": "system + user",
                "read_supervision": "system + assistant",
            }
        },
        "partitions": {
            "train": {
                "rows": 1,
                "source_split": "train",
                "data": {"path": str(train_file), "sha256": train_sha},
            }
        },
    }
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, sort_keys=True))
    manifest_sha = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    return train_file, manifest_path, manifest_sha


def test_parse_args_accepts_only_bound_scene_identity_protocol(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    train_file, manifest_path, manifest_sha = _source_manifest(tmp_path)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "delta_sft_experimental.py",
            "--model-path",
            "model",
            "--train-file",
            str(train_file),
            "--output-dir",
            str(tmp_path / "out"),
            "--memory-loss-mode",
            "scene_state_identity_ce",
            "--memory-kl-weight",
            "0",
            "--episode-recent-messages",
            "0",
            "--assistant-loss-mode",
            "final_assistant_only",
            "--scene-state-source-manifest",
            str(manifest_path),
            "--expected-scene-state-source-manifest-sha256",
            manifest_sha,
        ],
    )

    args = experimental_train.parse_args()
    assert args.memory_loss_mode == "scene_state_identity_ce"
    assert args.scene_state_identity_margin == 0.5

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "delta_sft_experimental.py",
            "--model-path",
            "model",
            "--train-file",
            str(train_file),
            "--output-dir",
            str(tmp_path / "out2"),
            "--memory-loss-mode",
            "scene_state_identity_ce",
            "--memory-kl-weight",
            "0",
            "--episode-recent-messages",
            "0",
            "--assistant-loss-mode",
            "final_assistant_only",
        ],
    )
    with pytest.raises(ValueError, match="source manifest"):
        experimental_train.parse_args()
