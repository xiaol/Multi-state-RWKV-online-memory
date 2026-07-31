from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

from datasets import Dataset
import pytest
import torch

import deltamem.train.delta_sft_experimental as experimental_train
from deltamem.tests.test_scene_memory_v10_training import _pair_inputs
from deltamem.tests.test_scene_memory_v9_training import (
    _PairModel,
    _bind_pair_model,
    _masks,
    _trainer,
)


_V15_DATA_ROOT = Path(
    "/run/media/xiaol/B214449214445C0B/delta_mem_data/scene_failure_state/"
    "scene_memory_v15/all32_pair64_v1"
)
_V15_SOURCE_MANIFEST = _V15_DATA_ROOT / "source_manifest.json"
_V15_TRAIN_FILE = Path(
    "/run/media/xiaol/B214449214445C0B/delta_mem_data/scene_failure_state/"
    "scene_memory_v7_fixed_hard32_aligned_train32_v1/train32.jsonl"
)


def _v15_args(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    *,
    smoke: bool,
):
    accumulation_steps = 1 if smoke else 16
    max_steps = 1 if smoke else 4
    argv = [
        "delta-sft",
        "--model-path",
        "model",
        "--output-dir",
        str(tmp_path / ("smoke" if smoke else "production")),
        "--train-file",
        str(_V15_TRAIN_FILE),
        "--warm-start-from-checkpoint",
        "checkpoint-4",
        "--warm-start-mode",
        experimental_train._SCENE_V14_WARM_START_MODE,
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
        "--scene-state-source-manifest",
        str(_V15_SOURCE_MANIFEST),
        "--expected-scene-state-source-manifest-sha256",
        experimental_train._sha256_file(_V15_SOURCE_MANIFEST),
        "--scene-state-generation-objective-version",
        experimental_train._SCENE_STATE_CACHED_PREFIX_IDENTITY_OBJECTIVE_VERSION,
        "--scene-state-generated-prefix-correction-weight",
        "0",
        "--scene-state-generated-unlikelihood-max-wrong-tokens",
        "0",
        "--gradient-accumulation-steps",
        str(accumulation_steps),
        "--learning-rate",
        "1e-4",
        "--lr-scheduler-type",
        "constant",
        "--warmup-ratio",
        "0",
        "--warmup-steps",
        "0",
        "--max-steps",
        str(max_steps),
        "--save-steps",
        "1",
        "--save-total-limit",
        str(max_steps),
        "--no-ignore-data-skip",
    ]
    if smoke:
        argv.append("--scene-state-v15-one-pair-smoke")
    monkeypatch.setattr(
        experimental_train,
        "_validate_scene_v14_warm_start_args",
        lambda _args: None,
    )
    monkeypatch.setattr("sys.argv", argv)
    return experimental_train.parse_args()


def _v15_protocol(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    *,
    smoke: bool,
) -> tuple[dict[str, object], dict[str, object]]:
    args = _v15_args(monkeypatch, tmp_path, smoke=smoke)
    binding = experimental_train._scene_state_v15_curriculum_binding(args)
    assert binding is not None
    if smoke:
        binding = experimental_train._scene_state_v15_one_pair_smoke_binding(binding)
    monkeypatch.setattr(
        experimental_train,
        "_scene_state_identity_protocol_pairing_summary",
        lambda _manifest: {},
    )
    tokenized = Dataset.from_dict({"input_ids": [[1]] for _ in range(32)})
    protocol = experimental_train.build_training_protocol(
        args,
        tokenized,
        effective_training_mode="episode",
        train_samples=32,
        eval_samples=0,
        warmup_steps=0,
        scene_state_identity_pairing_manifest={},
        train_schedule_binding=binding,
    )
    return binding, protocol


def test_cached_identity_objective_is_distinct_from_authoritative_v14() -> None:
    v14 = experimental_train._SCENE_STATE_CACHED_PREFIX_OBJECTIVE_VERSION
    v15 = experimental_train._SCENE_STATE_CACHED_PREFIX_IDENTITY_OBJECTIVE_VERSION

    assert v14 == "scene_state_generation_ce_symmetric_cached_prefix_boundary_v14"
    assert v15 == "scene_state_generation_ce_symmetric_cached_prefix_identity_v15"
    assert v15 != v14
    assert v15 in experimental_train._SCENE_STATE_RECIPROCAL_OBJECTIVE_VERSIONS
    assert v15 in experimental_train._SCENE_STATE_CYCLE_OBJECTIVE_VERSIONS
    assert "selected_pair_and_zero_hinges=telemetry_only" in (
        experimental_train._SCENE_STATE_CACHED_PREFIX_OBJECTIVE_FORMULA
    )
    assert "own_vs_paired_target_logit_hinge(1.0)" in (
        experimental_train._SCENE_STATE_CACHED_PREFIX_IDENTITY_OBJECTIVE_FORMULA
    )


def test_v15_frozen_source_loads_pairing_and_exact_four_cycle_schedule(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    args = _v15_args(monkeypatch, tmp_path, smoke=False)

    pairing = experimental_train._scene_state_generation_pairing_binding(args)
    binding = experimental_train._scene_state_v15_curriculum_binding(args)

    assert pairing["source_identity"]["schema"] == (
        experimental_train._SCENE_MEMORY_V15_SOURCE_SCHEMA
    )
    assert len(pairing["entries"]) == 32
    assert binding is not None
    assert binding["schema"] == experimental_train._SCENE_MEMORY_V15_CURRICULUM_SCHEMA
    assert binding["total_steps"] == 64
    assert binding["checkpoint_steps"] == [16, 32, 48, 64]
    assert binding["pair_indices"] == (
        experimental_train._SCENE_STATE_V15_FOUR_CYCLE_PAIRS
    )
    assert binding["indices"] == tuple(
        low for low, _ in experimental_train._SCENE_STATE_V15_FOUR_CYCLE_PAIRS
    )
    experimental_train._validate_scene_state_v15_four_cycle_schedule(binding)


def test_v15_production_and_smoke_protocols_bind_exact_trainer_contract(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    production_binding, production = _v15_protocol(
        monkeypatch,
        tmp_path,
        smoke=False,
    )

    assert production["schema_version"] == 18
    assert production["memory_objective_version"] == (
        experimental_train._SCENE_STATE_CACHED_PREFIX_IDENTITY_OBJECTIVE_VERSION
    )
    assert production["train_sampler_mode"] == (
        experimental_train._V15_PAIR_TRAIN_SCHEDULE_SAMPLER_MODE
    )
    assert production["scene_generation_v15_run_mode"] == (
        experimental_train._SCENE_STATE_V15_PRODUCTION_RUN_MODE
    )
    assert production["scene_generation_v15_production_eligible"] is True
    assert production["gradient_accumulation_steps"] == 16
    assert production["max_steps"] == 4
    assert production["save_total_limit"] == 4
    assert production["train_schedule"]["total_steps"] == 64
    assert production["train_schedule"]["checkpoint_steps"] == [16, 32, 48, 64]
    assert production["train_schedule"]["optimizer_checkpoint_steps"] == [1, 2, 3, 4]
    assert production["train_schedule"]["microbatch_cycle_size"] == 16
    assert production["scene_generation_objective_formula"] == (
        experimental_train._SCENE_STATE_CACHED_PREFIX_IDENTITY_OBJECTIVE_FORMULA
    )
    assert production["scene_generation_backward_mode"] == (
        experimental_train._SCENE_STATE_CACHED_PREFIX_IDENTITY_BACKWARD_MODE
    )
    assert production["scene_generation_teacher_forced_full_forward_mode"] == (
        "grad_enabled_pair_identity_only_v1"
    )
    assert production["scene_generation_v15_pair_identity_mode"] == (
        experimental_train._SCENE_STATE_CACHED_PREFIX_IDENTITY_MODE
    )
    assert production["scene_generation_v15_pair_identity_margin"] == 1.0
    assert production["scene_generation_v15_pair_identity_optimization_weight"] == 1.0
    assert production["scene_generation_row_objective_audit_filename"] == (
        experimental_train._SCENE_STATE_V15_ROW_AUDIT_FILENAME
    )
    assert production["scene_generation_row_objective_audit_schema"] == (
        experimental_train._SCENE_STATE_V15_ROW_AUDIT_SCHEMA
    )
    assert experimental_train._scene_memory_v10_protocol_checkpoint_steps(
        production
    ) == (1, 2, 3, 4)

    smoke_binding, smoke = _v15_protocol(
        monkeypatch,
        tmp_path,
        smoke=True,
    )
    first_pair = experimental_train._SCENE_STATE_V15_FOUR_CYCLE_PAIRS[0]

    assert production_binding["pair_indices"][0] == first_pair
    assert smoke_binding["source_total_steps"] == 64
    assert smoke_binding["source_checkpoint_steps"] == [16, 32, 48, 64]
    assert smoke_binding["total_steps"] == 1
    assert smoke_binding["checkpoint_steps"] == [1]
    assert smoke_binding["pair_indices"] == (first_pair,)
    assert smoke_binding["indices"] == (first_pair[0],)
    experimental_train._validate_scene_state_v15_one_pair_smoke_schedule(
        smoke_binding
    )
    assert smoke["schema_version"] == 18
    assert smoke["train_sampler_mode"] == (
        experimental_train._V15_ONE_PAIR_SMOKE_SAMPLER_MODE
    )
    assert smoke["scene_generation_v15_run_mode"] == (
        experimental_train._SCENE_STATE_V15_ONE_PAIR_SMOKE_RUN_MODE
    )
    assert smoke["scene_generation_v15_production_eligible"] is False
    assert smoke["gradient_accumulation_steps"] == 1
    assert smoke["max_steps"] == 1
    assert smoke["save_total_limit"] == 1
    assert smoke["train_schedule"]["total_steps"] == 1
    assert smoke["train_schedule"]["checkpoint_steps"] == [1]
    assert smoke["train_schedule"]["optimizer_checkpoint_steps"] == [1]
    assert smoke["train_schedule"]["microbatch_cycle_size"] == 1
    assert experimental_train._scene_memory_v10_protocol_checkpoint_steps(smoke) == (
        1,
    )


def test_v15_save_model_writes_only_v15_row_audit(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    output = tmp_path / "checkpoint-1"
    payload = {
        "schema": experimental_train._SCENE_STATE_V15_ROW_AUDIT_SCHEMA,
        "completed_pair_presentations": 1,
    }
    trainer = object.__new__(experimental_train.DeltaMemTrainer)
    trainer.args = SimpleNamespace(output_dir=str(output))
    trainer.delta_config = object()
    trainer.model = object()
    trainer.accelerator = SimpleNamespace(unwrap_model=lambda model: model)
    trainer.training_protocol = None
    trainer.content_contrast_pairing_manifest = None
    trainer.scene_state_identity_pairing_manifest = None
    trainer.scene_state_generation_objective_version = (
        experimental_train._SCENE_STATE_CACHED_PREFIX_IDENTITY_OBJECTIVE_VERSION
    )
    trainer.is_world_process_zero = lambda: True
    trainer._scene_state_v15_row_audit_payload = lambda: payload
    trainer._scene_state_v14_row_audit_payload = lambda: pytest.fail(
        "V15 save selected the V14 audit payload"
    )
    monkeypatch.setattr(
        experimental_train,
        "save_delta_mem_adapter",
        lambda _model, output_dir, _config: Path(output_dir).mkdir(
            parents=True,
            exist_ok=True,
        ),
    )
    monkeypatch.setattr(
        experimental_train,
        "_training_lineage_summary",
        lambda _trainer: {"continuation": None},
    )

    trainer.save_model(str(output))

    v15_path = output / experimental_train._SCENE_STATE_V15_ROW_AUDIT_FILENAME
    assert json.loads(v15_path.read_text(encoding="utf-8")) == payload
    assert not (
        output / experimental_train._SCENE_STATE_V14_ROW_AUDIT_FILENAME
    ).exists()


def test_pair_identity_hinge_backpropagates_through_own_and_paired_logits() -> None:
    source_pair_logits = torch.tensor([[0.2, 0.8]], requires_grad=True)
    donor_pair_logits = torch.tensor([[0.1, 0.7]], requires_grad=True)

    source = experimental_train.DeltaMemTrainer._scene_state_pair_identity_hinge_metrics(
        source_pair_logits
    )
    donor = experimental_train.DeltaMemTrainer._scene_state_pair_identity_hinge_metrics(
        donor_pair_logits
    )
    loss = 0.5 * (source["hinge_loss"] + donor["hinge_loss"])

    assert source["logit_margin_row"].item() == pytest.approx(-0.6)
    assert donor["logit_margin_row"].item() == pytest.approx(-0.6)
    assert loss.item() == pytest.approx(1.6)
    loss.backward()

    assert source_pair_logits.grad is not None
    assert donor_pair_logits.grad is not None
    assert source_pair_logits.grad.flatten().tolist() == pytest.approx([-0.5, 0.5])
    assert donor_pair_logits.grad.flatten().tolist() == pytest.approx([-0.5, 0.5])


def test_pair_identity_hinge_is_zero_after_the_margin_is_satisfied() -> None:
    pair_logits = torch.tensor([[1.5, 0.2]], requires_grad=True)

    metrics = (
        experimental_train.DeltaMemTrainer._scene_state_pair_identity_hinge_metrics(
            pair_logits
        )
    )

    assert metrics["hinge_loss"].item() == 0.0
    assert metrics["own_beats_paired_row"].item() == 1.0
    assert metrics["margin_satisfied_row"].item() == 1.0
    metrics["hinge_loss"].backward()
    assert pair_logits.grad is not None
    assert pair_logits.grad.tolist() == [[0.0, 0.0]]


def _run_cached_pair(
    monkeypatch: pytest.MonkeyPatch,
    *,
    objective_version: str,
) -> tuple[_PairModel, dict[str, float], int]:
    trainer = _trainer()
    trainer.scene_state_generation_objective_version = objective_version
    trainer.scene_state_generated_unlikelihood_max_wrong_tokens = 0
    trainer.scene_state_generated_rollout_extra_tokens = 0
    trainer.scene_state_generated_rollout_max_tokens = 16
    trainer.scene_state_v14_one_pair_smoke = (
        objective_version == experimental_train._SCENE_STATE_CACHED_PREFIX_OBJECTIVE_VERSION
    )
    trainer.scene_state_v15_one_pair_smoke = (
        objective_version
        == experimental_train._SCENE_STATE_CACHED_PREFIX_IDENTITY_OBJECTIVE_VERSION
    )
    model = _PairModel()
    _bind_pair_model(trainer, monkeypatch)
    masks = _masks()
    source_inputs, donor_inputs = _pair_inputs()

    def exact_rollout(active_model, model_inputs, **kwargs):
        del active_model, kwargs
        suffix = model_inputs["input_ids"][0, 2:].detach()
        return {
            "prompt_input_ids": model_inputs["input_ids"][:, :2].detach(),
            "prompt_attention_mask": torch.ones(1, 2, dtype=torch.long),
            "generated_token_ids": suffix,
            "gold_token_ids": suffix,
            "generation_start": 2,
            "first_divergence": int(suffix.numel()),
            "exact_through_termination": True,
        }

    def cached_logits(active_model, rollout, *, replay_token_ids, **kwargs):
        del rollout, kwargs
        return active_model(
            input_ids=replay_token_ids,
            attention_mask=torch.ones_like(replay_token_ids),
        )["logits"]

    def zero_cached_retention(model_inputs, *, cached_replay_logits, **kwargs):
        del model_inputs, kwargs
        zero_loss = cached_replay_logits.sum() * 0.0
        return zero_loss, {
            "scene_generation_v14_cached_branch_loss": 0.0,
            "scene_generation_v14_cached_exact_retention_hinge": 0.0,
            "scene_generation_v14_cached_ce": 0.0,
            "scene_generation_v14_cached_failed_competitor_hinge": 0.0,
            "scene_generation_v14_cached_replay_token_count": float(
                cached_replay_logits.size(1)
            ),
            "scene_generation_v14_cached_decision_token_count": 2.0,
            "scene_generation_v14_cached_gold_top1_fraction": 1.0,
        }

    trainer._scene_state_generated_greedy_rollout = exact_rollout
    trainer._scene_state_v14_rollout_semantics = lambda *args, **kwargs: (True, None)
    trainer._scene_state_v14_cached_replay_logits = cached_logits
    trainer._scene_state_v14_exact_cached_retention_branch = zero_cached_retention
    trainer._scene_state_v14_record_pair_presentation = lambda *args, **kwargs: None
    trainer._scene_state_v15_record_pair_presentation = lambda *args, **kwargs: None

    _, stats = trainer._scene_state_generation_symmetric_sequential_backward(
        model,
        source_inputs,
        donor_inputs,
        loss_kwargs={},
        source_write_input_ids=torch.tensor([[10]]),
        source_write_attention_mask=torch.ones(1, 1, dtype=torch.long),
        source_write_message_ids=None,
        source_write_sentence_ids=None,
        donor_write_input_ids=torch.tensor([[20]]),
        donor_write_attention_mask=torch.ones(1, 1, dtype=torch.long),
        donor_write_message_ids=None,
        donor_write_sentence_ids=None,
        source_target_mask=masks["target"],
        source_content_mask=masks["content"],
        source_schema_mask=masks["schema"],
        source_decision_mask=masks["decision"],
        source_termination_mask=masks["termination"],
        source_pair_target_mask=masks["pair"],
        donor_target_mask=masks["target"],
        donor_content_mask=masks["content"],
        donor_schema_mask=masks["schema"],
        donor_decision_mask=masks["decision"],
        donor_termination_mask=masks["termination"],
        donor_pair_target_mask=masks["pair"],
        gradient_scale=1.0,
        source_indices=torch.tensor([3]),
        donor_indices=torch.tensor([24]),
        source_row_sha256=torch.full((1, 32), 3, dtype=torch.uint8),
        donor_row_sha256=torch.full((1, 32), 24, dtype=torch.uint8),
    )
    return model, stats, trainer.accelerator.backward_calls


def test_cached_identity_objective_trains_both_sides_without_changing_v14(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    v15_model, v15_stats, v15_backward_calls = _run_cached_pair(
        monkeypatch,
        objective_version=(
            experimental_train._SCENE_STATE_CACHED_PREFIX_IDENTITY_OBJECTIVE_VERSION
        ),
    )

    assert v15_backward_calls == 4
    assert v15_stats["scene_generation_v15_pair_mean_pair_identity_hinge"] == (
        pytest.approx(0.25)
    )
    assert v15_stats["scene_generation_v15_objective_total_loss"] == pytest.approx(
        0.25
    )
    assert v15_model.parameters_by_side.grad is not None
    assert v15_model.parameters_by_side.grad.tolist() == pytest.approx([-1.5, -1.5])

    v14_model, v14_stats, v14_backward_calls = _run_cached_pair(
        monkeypatch,
        objective_version=experimental_train._SCENE_STATE_CACHED_PREFIX_OBJECTIVE_VERSION,
    )

    assert v14_backward_calls == 2
    assert "scene_generation_v15_pair_mean_pair_identity_hinge" not in v14_stats
    assert v14_stats["scene_generation_v14_objective_total_loss"] == 0.0
    assert v14_model.parameters_by_side.grad is not None
    assert v14_model.parameters_by_side.grad.tolist() == [0.0, 0.0]


def _v15_exact_audit_stats() -> dict[str, float]:
    stats: dict[str, float] = {}
    identity_margins = {"source": 0.25, "donor": -0.5}
    for role, identity_margin in identity_margins.items():
        common = f"scene_generation_{role}_"
        cached = "scene_generation_v14_"
        identity = "scene_generation_v15_"
        identity_hinge = max(0.0, 1.0 - identity_margin)
        stats.update(
            {
                f"{common}selected_top_hinge": 0.3,
                f"{common}zero_hinge": 0.4,
                f"{cached}parsed_boundary_exact_{role}": 1.0,
                f"{cached}raw_token_exact_{role}": 1.0,
                f"{cached}first_divergence_{role}": 3.0,
                f"{cached}rollout_token_count_{role}": 3.0,
                f"{cached}cached_branch_kind_code_{role}": 0.0,
                f"{cached}cached_replay_use_cache_{role}": 1.0,
                f"{cached}cached_replay_logits_to_keep_{role}": 1.0,
                f"{cached}cached_replay_token_count_{role}": 3.0,
                f"{cached}cached_replay_selected_cursor_{role}": 1.0,
                f"{cached}cached_decision_token_count_{role}": 2.0,
                f"{cached}cached_selected_decision_ordinal_{role}": 0.0,
                f"{cached}cached_selected_label_position_{role}": 3.0,
                f"{cached}cached_selected_gold_token_id_{role}": 5.0,
                f"{cached}cached_selected_competitor_id_{role}": 6.0,
                f"{cached}cached_competitor_is_actual_greedy_{role}": 0.0,
                f"{cached}cached_replay_top1_matches_actual_{role}": 0.0,
                f"{cached}cached_replay_top1_match_count_{role}": 0.0,
                f"{cached}cached_ce_{role}": 0.0,
                f"{cached}cached_failed_competitor_hinge_{role}": 0.0,
                f"{cached}cached_exact_retention_hinge_{role}": 0.2,
                f"{cached}cached_selected_gold_vs_competitor_margin_{role}": 0.5,
                f"{cached}cached_gold_top1_fraction_{role}": 1.0,
                f"{cached}cached_alignment_kind_code_{role}": -1.0,
                f"{cached}cached_selected_is_termination_{role}": 0.0,
                f"{cached}cached_branch_loss_{role}": 0.2,
                f"{cached}auxiliary_loss_{role}": 0.0,
                f"{cached}auxiliary_telemetry_loss_{role}": 0.7,
                f"{cached}total_side_loss_{role}": 0.2 + identity_hinge,
                f"{identity}pair_identity_hinge_{role}": identity_hinge,
                f"{identity}pair_identity_logit_margin_{role}": identity_margin,
                f"{identity}pair_identity_own_beats_paired_fraction_{role}": float(
                    identity_margin > 0.0
                ),
                f"{identity}pair_identity_margin_satisfied_fraction_{role}": float(
                    identity_margin >= 1.0
                ),
            }
        )
    pair_identity_hinge = 1.125
    pair_total = 0.2 + pair_identity_hinge
    stats.update(
        {
            "scene_generation_v15_pair_mean_cached_branch_loss": 0.2,
            "scene_generation_v15_pair_mean_pair_identity_hinge": (
                pair_identity_hinge
            ),
            "scene_generation_v15_pair_mean_pair_identity_logit_margin": -0.125,
            "scene_generation_v15_pair_mean_pair_identity_own_beats_paired_fraction": 0.5,
            "scene_generation_v15_pair_mean_pair_identity_margin_satisfied_fraction": 0.0,
            "scene_generation_v15_pair_mean_cached_exact_retention_hinge": 0.2,
            "scene_generation_v15_pair_mean_cached_failed_ce": 0.0,
            "scene_generation_v15_pair_mean_cached_failed_competitor_hinge": 0.0,
            "scene_generation_v15_pair_mean_total_side_loss": pair_total,
            "scene_generation_v15_objective_total_loss": pair_total,
            "scene_generation_v15_recomputed_objective_total_loss": pair_total,
        }
    )
    return stats


def test_v15_full_cycle_records_all_16_pairs_and_identity_audit() -> None:
    trainer = object.__new__(experimental_train.DeltaMemTrainer)
    trainer.scene_state_generation_objective_version = (
        experimental_train._SCENE_STATE_CACHED_PREFIX_IDENTITY_OBJECTIVE_VERSION
    )
    trainer.scene_state_v14_one_pair_smoke = False
    trainer.scene_state_v15_one_pair_smoke = False
    trainer._scene_state_cycle_retention_metric_sums = {}
    trainer._scene_state_cycle_retention_metric_presentations = 0
    trainer._scene_state_v15_cycle_pairs = []
    trainer._scene_state_v15_completed_cycles = 0
    trainer._scene_state_v15_row_observations = []
    trainer._scene_state_v15_pair_observations = []
    cycle_pairs = experimental_train._SCENE_STATE_V15_FOUR_CYCLE_PAIRS[:16]
    manifest_pairs: list[dict[str, object]] = [{} for _ in range(32)]
    for source, donor in cycle_pairs:
        manifest_pairs[source] = {
            "source_index": source,
            "donor_index": donor,
            "source_row_sha256": bytes([source] * 32).hex(),
            "donor_row_sha256": bytes([donor] * 32).hex(),
        }
    trainer.scene_state_identity_pairing_manifest = {
        "splits": {"train": {"pairs": manifest_pairs}}
    }
    stats = _v15_exact_audit_stats()
    row_hash = lambda ordinal: torch.full(
        (1, 32), ordinal, dtype=torch.uint8
    )

    averaged: dict[str, float] = {}
    for source, donor in cycle_pairs:
        trainer._scene_state_v15_record_pair_presentation(
            torch.tensor([source]),
            torch.tensor([donor]),
            row_hash(source),
            row_hash(donor),
            stats,
        )
        averaged = trainer._scene_state_cycle_retention_aggregate_memory_stats(
            stats
        )
    payload = trainer._scene_state_v15_row_audit_payload()

    assert averaged["scene_generation_v15_cycle_pair_presentations"] == 16.0
    assert averaged["scene_generation_v15_cycle_index"] == 1.0
    assert trainer._scene_state_v15_completed_cycles == 1
    assert payload["schema"] == experimental_train._SCENE_STATE_V15_ROW_AUDIT_SCHEMA
    assert payload["run_mode"] == experimental_train._SCENE_STATE_V15_PRODUCTION_RUN_MODE
    assert payload["production_eligible"] is True
    assert payload["completed_pair_presentations"] == 16
    assert payload["phases"] == ["cycle1_input"]
    assert [
        (item["source_row_ordinal"], item["donor_row_ordinal"])
        for item in payload["pair_presentations"]
    ] == list(cycle_pairs)
    assert len(payload["rows"]) == 32
    first_row = payload["rows"][0]["cycle1_input"]
    assert first_row["pair_identity_hinge"] == pytest.approx(0.75)
    assert first_row["pair_identity_logit_margin"] == pytest.approx(0.25)
    assert first_row["pair_identity_own_beats_paired_fraction"] == 1.0
    assert first_row["pair_identity_margin_satisfied_fraction"] == 0.0
