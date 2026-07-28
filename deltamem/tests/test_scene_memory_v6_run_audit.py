from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest
import torch

from experiments.rethinking_rwkv_ms_gemma import scene_memory_v6_run_audit as audit


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=True, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def adapter_state(value: float) -> dict[str, torch.Tensor]:
    state: dict[str, torch.Tensor] = {}
    changed = set(audit.REQUIRED_CHANGED_SUFFIXES)
    for layer in range(42):
        prefix = f"model.language_model.layers.{layer}.self_attn."
        for suffix in audit.EXPECTED_ADAPTER_SUFFIXES:
            active_value = value if suffix in changed else 0.0
            if suffix == "delta_scale_raw":
                tensor = torch.tensor(
                    [active_value, 0.0, 0.0, active_value],
                    dtype=torch.float32,
                )
            else:
                tensor = torch.tensor([active_value], dtype=torch.float32)
            state[prefix + suffix] = tensor
    return state


def trainable_names() -> list[str]:
    return [
        name
        for name in adapter_state(0.0)
        if name.rsplit(".self_attn.", 1)[1]
        not in set(audit.EXPECTED_FROZEN_SUFFIXES)
    ]


def valid_change_record() -> dict[str, object]:
    return audit.adapter_change_record(
        adapter_state(0.0),
        adapter_state(1.0),
        trainable_names=trainable_names(),
    )


def step_record(step: int) -> dict[str, object]:
    record: dict[str, object] = {
        "step": step,
        "loss": 2.25,
        "grad_norm": 1.0,
        "learning_rate": 5e-4,
        "delta/scene_state_full_correct_ce": 1.0,
        "delta/scene_state_correct_all_semantic_ce": 1.0,
        "delta/scene_state_correct_pair_semantic_ce": 0.75,
        "delta/scene_state_donor_pair_semantic_ce": 1.0,
        "delta/scene_state_zero_all_semantic_ce": 1.25,
        "delta/scene_state_donor_pair_gap": 0.25,
        "delta/scene_state_zero_all_gap": 0.25,
        "delta/scene_state_donor_margin_loss": 0.25,
        "delta/scene_state_donor_positive_fraction": 1.0,
        "delta/scene_state_zero_positive_fraction": 1.0,
        "delta/scene_state_semantic_token_count": 3.0,
        "delta/scene_state_semantic_row_count": 1.0,
        "delta/scene_state_target_presence_row_count": 1.0,
        "delta/scene_state_target_same_cardinality_value_row_count": 0.0,
        "delta/scene_state_target_cross_cardinality_value_row_count": 0.0,
    }
    return record


def trainer_state(step: int, spec: audit.RunSpec) -> dict[str, object]:
    return {
        "global_step": step,
        "max_steps": spec.max_steps,
        "log_history": [step_record(index) for index in range(1, step + 1)],
    }


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
                "target_mode": audit.OBJECTIVE_PROTOCOL["target_mode"],
                "causal_prefix_mode": audit.SCENE_STATE_IDENTITY_CAUSAL_PREFIX_MODE,
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
    pair_sha = audit.canonical_sha256(pairs)
    counts = {"presence": 24, "same_cardinality_value": 8, "cross_cardinality_value": 0}
    histogram = {"0": 12, "1": 20}
    train: dict[str, object] = {
        "split": "train",
        "pairing_version": audit.SCENE_STATE_IDENTITY_PAIRING_VERSION,
        "pairing_refinement": audit.SCENE_STATE_IDENTITY_PAIRING_REFINEMENT,
        "pairing_refinement_applied": True,
        "target_mode": audit.OBJECTIVE_PROTOCOL["target_mode"],
        "causal_prefix_mode": audit.SCENE_STATE_IDENTITY_CAUSAL_PREFIX_MODE,
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
    train["manifest_sha256"] = audit.canonical_sha256(train)
    manifest: dict[str, object] = {
        "schema_version": 2,
        "objective_version": audit.SCENE_STATE_IDENTITY_OBJECTIVE_VERSION,
        "pairing_version": audit.SCENE_STATE_IDENTITY_PAIRING_VERSION,
        "pairing_refinement": audit.SCENE_STATE_IDENTITY_PAIRING_REFINEMENT,
        "pairing_refinement_applied": True,
        "pairing_scope": "within_post_split_partition",
        "target_mode": audit.OBJECTIVE_PROTOCOL["target_mode"],
        "causal_prefix_mode": audit.SCENE_STATE_IDENTITY_CAUSAL_PREFIX_MODE,
        "semantic_mask_mode": audit.OBJECTIVE_PROTOCOL["semantic_mask_mode"],
        "semantic_loss_normalization": audit.OBJECTIVE_PROTOCOL[
            "semantic_loss_normalization"
        ],
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
    manifest["manifest_sha256"] = audit.canonical_sha256(manifest)
    return manifest


def bind_synthetic_pairing(
    monkeypatch: pytest.MonkeyPatch,
    pairing: dict[str, object],
) -> None:
    train = pairing["splits"]["train"]
    monkeypatch.setattr(audit, "EXPECTED_TARGET_STRATUM_ROW_COUNTS", pairing["target_stratum_row_counts"])
    monkeypatch.setattr(audit, "EXPECTED_SOURCE_BOUNDARY_COUNT_HISTOGRAM", pairing["source_boundary_count_histogram"])
    monkeypatch.setattr(audit, "EXPECTED_WRITE_TOKEN_COUNT_DELTA_MAX", pairing["write_token_count_delta_max"])
    monkeypatch.setattr(audit, "EXPECTED_WRITE_TOKEN_COUNT_DELTA_MEAN", pairing["write_token_count_delta_mean"])
    monkeypatch.setattr(audit, "EXPECTED_WRITE_TOKEN_COUNT_DELTA_TOTAL", pairing["write_token_count_delta_total"])
    monkeypatch.setattr(audit, "EXPECTED_IDENTITY_PAIRS_SHA256", train["pairs_sha256"])
    monkeypatch.setattr(audit, "EXPECTED_CAUSAL_PREFIX_TOKEN_COUNT_HISTOGRAM", {"2": 32})
    monkeypatch.setattr(
        audit,
        "EXPECTED_CAUSAL_PREFIX_SHA256_SET",
        {row["causal_prefix_sha256"] for row in train["pairs"]},
    )


def pairing_protocol_summary(pairing: dict[str, object]) -> dict[str, object]:
    train = pairing["splits"]["train"]
    summary = {
        key: pairing[key]
        for key in (
            "pairing_version",
            "pairing_refinement",
            "pairing_refinement_applied",
            "pairing_scope",
            "target_mode",
            "causal_prefix_mode",
            "semantic_mask_mode",
            "semantic_loss_normalization",
            "target_token_count",
            "target_stratum_row_counts",
            "source_boundary_count_histogram",
            "write_token_count_delta_max",
            "write_token_count_delta_mean",
            "write_token_count_delta_total",
            "nearest_baseline_write_token_count_delta_max",
            "nearest_baseline_write_token_count_delta_total",
            "data_seed",
            "tokenized_fingerprint",
            "tokenized_dataset_sha256",
            "manifest_sha256",
        )
    }
    summary["splits"] = {
        "train": {
            key: train[key]
            for key in (
                "sample_count",
                "pair_count",
                "target_token_count",
                "causal_prefix_mode",
                "target_stratum_row_counts",
                "source_boundary_count_histogram",
                "write_token_count_delta_max",
                "write_token_count_delta_mean",
                "write_token_count_delta_total",
                "nearest_baseline_write_token_count_delta_max",
                "nearest_baseline_write_token_count_delta_total",
                "pairing_refinement_applied",
                "source_fingerprint",
                "paired_fingerprint",
                "pairs_sha256",
                "manifest_sha256",
            )
        }
    }
    return summary


def valid_protocol(pairing: dict[str, object], run_mode: str = "smoke") -> dict[str, object]:
    spec = audit._stage_spec(run_mode)
    dataset_sha = pairing["tokenized_dataset_sha256"]
    return {
        "train_file": str(audit.EXPECTED_TRAIN),
        "tokenized_samples": 32,
        "train_samples": 32,
        "eval_samples": 0,
        "training_mode": "episode",
        "assistant_loss_mode": "final_assistant_only",
        "max_length": 256,
        "max_write_length": 1280,
        "episode_recent_messages": 0,
        "episode_read_write_enabled": False,
        "memory_loss_mode": "scene_state_identity_ce",
        "memory_objective_version": audit.OBJECTIVE_PROTOCOL["objective_version"],
        "scene_state_identity_margin": 0.5,
        "scene_state_margin_mode": "per_row_hinge_relu_v1",
        "scene_state_objective_formula": audit.OBJECTIVE_PROTOCOL["objective_formula"],
        "scene_state_correct_all_semantic_scope": audit.OBJECTIVE_PROTOCOL["correct_all_semantic_scope"],
        "scene_state_pair_semantic_scope": audit.OBJECTIVE_PROTOCOL["pair_semantic_scope"],
        "scene_state_donor_margin_scope": audit.OBJECTIVE_PROTOCOL["donor_margin_scope"],
        "scene_state_zero_diagnostic_scope": audit.OBJECTIVE_PROTOCOL["zero_diagnostic_scope"],
        "scene_state_zero_diagnostic_gradient": False,
        "scene_state_read_time_positions_observable": False,
        "scene_state_pairing_length_control": audit.SCENE_STATE_IDENTITY_PAIRING_LENGTH_CONTROL,
        "scene_state_pairing_refinement": audit.SCENE_STATE_IDENTITY_PAIRING_REFINEMENT,
        "scene_state_identity_target_strata": list(audit.SCENE_STATE_IDENTITY_TARGET_STRATA),
        "scene_state_identity_backward_mode": audit.OBJECTIVE_PROTOCOL["backward_mode"],
        "scene_state_identity_read_protocol": audit.OBJECTIVE_PROTOCOL["read_protocol"],
        "scene_state_identity_zero_protocol": audit.OBJECTIVE_PROTOCOL["zero_protocol"],
        "scene_state_semantic_mask_mode": audit.OBJECTIVE_PROTOCOL["semantic_mask_mode"],
        "scene_state_semantic_loss_normalization": audit.OBJECTIVE_PROTOCOL["semantic_loss_normalization"],
        "scene_state_identity_target_mode": audit.OBJECTIVE_PROTOCOL["target_mode"],
        "scene_state_identity_causal_prefix_mode": audit.SCENE_STATE_IDENTITY_CAUSAL_PREFIX_MODE,
        "scene_state_full_correct_ce_weight": 1.0,
        "scene_state_correct_all_semantic_ce_weight": 1.0,
        "scene_state_donor_margin_weight": 1.0,
        "scene_boundary_payload_ce_weight": 0.0,
        "memory_base_kl_weight": 0.0,
        "memory_kl_weight": 0.0,
        "memory_representation_weight": 0.0,
        "write_sparsity_weight": 0.0,
        "memory_partition_alignment_weight": 0.0,
        "memory_partition_entropy_weight": 0.0,
        "memory_partition_balance_weight": 0.0,
        "memory_dropout_no_memory_prob": 0.0,
        "memory_dropout_state_only_prob": 0.0,
        "validation_split_ratio": 0.0,
        "seed": 42,
        "data_seed": 42,
        "train_sampler_seed": 42,
        "train_sampler_mode": "torch_random_sampler_seed_equals_data_seed_v1",
        "per_device_train_batch_size": 1,
        "gradient_accumulation_steps": 1,
        "learning_rate": 5e-4,
        "lr_scheduler_type": "constant_with_warmup",
        "warmup_steps": spec.warmup_steps,
        "max_steps": spec.max_steps,
        "save_steps": spec.save_steps,
        "frozen_mlp_activation_checkpointing": True,
        "scene_state_source_manifest": {
            "path": str(audit.EXPECTED_PAIR_MANIFEST),
            "file_sha256": audit.EXPECTED_PAIR_MANIFEST_SHA256,
            "train_file": str(audit.EXPECTED_TRAIN),
            "train_file_sha256": audit.EXPECTED_TRAIN_SHA256,
            "train_rows": 32,
            "train_source_split": "train",
        },
        "scene_state_identity_pairing": pairing_protocol_summary(pairing),
        "tokenized_dataset_sha256": dataset_sha,
        "expected_tokenized_dataset_sha256": None,
        "tokenized_cache_identity": {
            "rows": 32,
            "persisted_files": [],
            "ordered_content_sha256": dataset_sha,
        },
    }


def valid_summary(
    run_root: Path,
    protocol: dict[str, object],
    pairing: dict[str, object],
) -> dict[str, object]:
    return {
        "output_dir": str(run_root),
        "resume_from_checkpoint": None,
        "warm_start_from_checkpoint": None,
        "initial_adapter_output_dir": str(run_root / "initial_adapter"),
        "num_replaced_modules": 42,
        "num_trainable_tensors": 42 * len(audit.EXPECTED_TRAINABLE_SUFFIXES),
        "num_checkpointed_frozen_mlps": 42,
        "tokenized_samples": 32,
        "train_samples": 32,
        "eval_samples": 0,
        "training_mode": "episode",
        "assistant_loss_mode": "final_assistant_only",
        "train_sampler_seed": 42,
        "train_sampler_mode": "torch_random_sampler_seed_equals_data_seed_v1",
        "episode_recent_messages": 0,
        "max_write_length": 1280,
        "episode_read_write_enabled": False,
        "memory_loss_mode": "scene_state_identity_ce",
        "memory_objective_version": audit.OBJECTIVE_PROTOCOL["objective_version"],
        "scene_boundary_payload_ce_weight": 0.0,
        "rwkv_ms_semantics_version": 2,
        "seed": 42,
        "data_seed": 42,
        "world_size": 1,
        "local_rank": -1,
        "tokenized_cache": False,
        "tokenized_cache_hit": False,
        "tokenized_cache_dir": None,
        "tokenized_dataset_source": "direct_map",
        "scene_state_identity_margin": 0.5,
        "scene_state_margin_mode": "per_row_hinge_relu_v1",
        "scene_state_identity_backward_mode": audit.OBJECTIVE_PROTOCOL["backward_mode"],
        "scene_state_identity_read_protocol": audit.OBJECTIVE_PROTOCOL["read_protocol"],
        "scene_state_identity_zero_protocol": audit.OBJECTIVE_PROTOCOL["zero_protocol"],
        "scene_state_semantic_mask_mode": audit.OBJECTIVE_PROTOCOL["semantic_mask_mode"],
        "scene_state_semantic_loss_normalization": audit.OBJECTIVE_PROTOCOL["semantic_loss_normalization"],
        "scene_state_identity_target_mode": audit.OBJECTIVE_PROTOCOL["target_mode"],
        "scene_state_full_correct_ce_weight": 1.0,
        "scene_state_correct_all_semantic_ce_weight": 1.0,
        "scene_state_donor_margin_weight": 1.0,
        "training_protocol_sha256": audit.canonical_sha256(protocol),
        "scene_state_identity_pairing_manifest_sha256": pairing["manifest_sha256"],
        "gate_stats": {},
        "output_ratio_stats": {},
    }


def test_adapter_change_requires_all_42_layers_and_preserves_frozen_paths() -> None:
    change = valid_change_record()
    audit.validate_adapter_change_evidence(change)
    assert change["changed_nontrainable_tensor_count"] == 0
    assert set(change["required_changed_layer_coverage"]) == set(
        audit.REQUIRED_CHANGED_SUFFIXES
    )
    assert all(
        count == 42 for count in change["required_changed_layer_coverage"].values()
    )


def test_adapter_change_rejects_changed_frozen_tensor() -> None:
    initial = adapter_state(0.0)
    candidate = adapter_state(1.0)
    frozen = "model.language_model.layers.0.self_attn.memory_q_proj"
    candidate[frozen] = torch.ones_like(candidate[frozen])
    with pytest.raises(audit.AuditError, match="changed frozen adapter tensors"):
        audit.adapter_change_record(initial, candidate, trainable_names=trainable_names())


def test_adapter_change_rejects_missing_layer_coverage() -> None:
    initial = adapter_state(0.0)
    candidate = adapter_state(1.0)
    name = "model.language_model.layers.41.self_attn.beta_bias"
    candidate[name] = initial[name].clone()
    with pytest.raises(audit.AuditError, match="all-layer update coverage"):
        audit.adapter_change_record(initial, candidate, trainable_names=trainable_names())


def test_v2_history_accepts_pair_target_and_zero_diagnostic_metrics() -> None:
    spec = audit.RUN_SPECS["smoke"]
    history = audit.validate_step_history(
        trainer_state(1, spec),
        checkpoint_step=1,
        spec=spec,
    )
    assert history["identity_metrics_finite"] is True
    assert "delta/scene_state_donor_pair_gap" in history["metric_names"]
    assert "delta/scene_state_zero_margin_loss" not in history["metric_names"]


def test_v2_history_rejects_missing_pair_target_metric() -> None:
    spec = audit.RUN_SPECS["smoke"]
    state = trainer_state(1, spec)
    state["log_history"][0].pop("delta/scene_state_correct_pair_semantic_ce")
    with pytest.raises(audit.AuditError, match="history is incomplete"):
        audit.validate_step_history(state, checkpoint_step=1, spec=spec)


def test_v2_history_rejects_nonfinite_zero_diagnostic() -> None:
    spec = audit.RUN_SPECS["smoke"]
    state = trainer_state(1, spec)
    state["log_history"][0]["delta/scene_state_zero_all_gap"] = float("nan")
    with pytest.raises(audit.AuditError, match="must be finite"):
        audit.validate_step_history(state, checkpoint_step=1, spec=spec)


def test_v2_history_rejects_invalid_target_stratum_coverage() -> None:
    spec = audit.RUN_SPECS["smoke"]
    state = trainer_state(1, spec)
    state["log_history"][0][
        "delta/scene_state_target_same_cardinality_value_row_count"
    ] = 1.0
    with pytest.raises(audit.AuditError, match="target-stratum coverage differs"):
        audit.validate_step_history(state, checkpoint_step=1, spec=spec)


def test_pairing_manifest_binds_v2_strata_and_delta_audit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pairing = synthetic_pairing_manifest()
    bind_synthetic_pairing(monkeypatch, pairing)
    path = tmp_path / "scene_state_identity_pairing_manifest.json"
    write_json(path, pairing)
    assert audit.validate_identity_pairing_manifest(path) == pairing


def test_training_protocol_is_v2_and_zero_is_diagnostic_only(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pairing = synthetic_pairing_manifest()
    bind_synthetic_pairing(monkeypatch, pairing)
    protocol = valid_protocol(pairing)
    audit.validate_training_protocol(protocol, run_mode="smoke", pairing=pairing)
    assert protocol["scene_state_zero_diagnostic_gradient"] is False
    assert "scene_state_zero_margin_weight" not in protocol


def test_training_protocol_rejects_obsolete_zero_margin_weight(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pairing = synthetic_pairing_manifest()
    bind_synthetic_pairing(monkeypatch, pairing)
    protocol = valid_protocol(pairing)
    protocol["scene_state_zero_margin_weight"] = 1.0
    with pytest.raises(audit.AuditError, match="obsolete zero-margin weight"):
        audit.validate_training_protocol(protocol, run_mode="smoke", pairing=pairing)


def test_training_summary_locks_fresh_direct_map_tokenization(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pairing = synthetic_pairing_manifest()
    bind_synthetic_pairing(monkeypatch, pairing)
    protocol = valid_protocol(pairing)
    run_root = tmp_path / "run"
    summary = valid_summary(run_root, protocol, pairing)
    path = tmp_path / "training_summary.json"
    write_json(path, summary)
    assert audit.validate_training_summary(
        path,
        run_mode="smoke",
        run_root=run_root,
        protocol=protocol,
        pairing=pairing,
    ) == summary


def test_training_summary_rejects_persisted_cache_source(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pairing = synthetic_pairing_manifest()
    bind_synthetic_pairing(monkeypatch, pairing)
    protocol = valid_protocol(pairing)
    run_root = tmp_path / "run"
    summary = valid_summary(run_root, protocol, pairing)
    summary["tokenized_dataset_source"] = "prepared_cache"
    path = tmp_path / "training_summary.json"
    write_json(path, summary)
    with pytest.raises(audit.AuditError, match="tokenized_dataset_source"):
        audit.validate_training_summary(
            path,
            run_mode="smoke",
            run_root=run_root,
            protocol=protocol,
            pairing=pairing,
        )


def test_existing_checkpoint_receipt_recomputes_history_and_adapter_change(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run_root = tmp_path / "run"
    checkpoint = run_root / "trainer" / "checkpoint-1"
    receipt_path = checkpoint / "checkpoint_receipt.json"
    state_path = checkpoint / "trainer_state.json"
    initial_manifest_path = run_root / "initial_adapter" / "initial_adapter_manifest.json"
    initial_adapter_path = run_root / "initial_adapter" / "delta_mem_adapter.pt"
    candidate_adapter_path = checkpoint / "delta_mem_adapter.pt"
    write_json(state_path, trainer_state(1, audit.RUN_SPECS["smoke"]))
    write_json(initial_manifest_path, {"topology": {"trainable_names": ["tensor"]}})
    initial_adapter_path.write_bytes(b"initial")
    candidate_adapter_path.write_bytes(b"candidate")
    history = audit.validate_step_history(
        trainer_state(1, audit.RUN_SPECS["smoke"]),
        checkpoint_step=1,
        spec=audit.RUN_SPECS["smoke"],
    )
    change = valid_change_record()
    receipt: dict[str, object] = {
        "schema": audit.CHECKPOINT_RECEIPT_SCHEMA,
        "experiment": audit.EXPERIMENT,
        "run_mode": "smoke",
        "run_root": str(run_root),
        "checkpoint_step": 1,
        "checkpoint_dir": str(checkpoint),
        "complete": True,
        "training_summary_required": False,
        "hard32_only": True,
        "full170_authorized": False,
        "test_forbidden": True,
        "auditor": {},
        "launch": {},
        "data_contract": {},
        "source_lock": {},
        "pair_manifest": {},
        "identity_pairing_manifest": {},
        "train_partition": {"rows": 32, "source_split": "train", "row_manifest": {}},
        "hard32_selection": {
            "rows": 32,
            "source_split": "val",
            "test_rows": 0,
            "holdout": {},
            "indices": {},
            "row_manifest": {},
        },
        "initial_adapter": {
            "manifest": {"path": str(initial_manifest_path)},
            "adapter": {"path": str(initial_adapter_path)},
            "config": {},
            "protocol": {},
        },
        "checkpoint_artifacts": {
            "adapter": {"path": str(candidate_adapter_path)},
            "config": {},
            "protocol": {},
            "trainer_state": {"path": str(state_path)},
            "optimizer": {},
            "scheduler": {},
            "rng": [{}],
        },
        "history": history,
        "adapter_change": change,
        "objective": dict(audit.OBJECTIVE_PROTOCOL),
    }
    receipt["receipt_sha256"] = audit.canonical_sha256(receipt)
    write_json(receipt_path, receipt)
    monkeypatch.setattr(audit, "_validate_file_record", lambda *args, **kwargs: None)
    monkeypatch.setattr(audit, "load_finite_adapter", lambda path: {"tensor": torch.tensor([0.0])})
    monkeypatch.setattr(audit, "adapter_change_record", lambda *args, **kwargs: change)
    assert audit.validate_existing_checkpoint_receipt(
        receipt_path,
        expected_run_mode="smoke",
        expected_run_root=run_root,
        expected_step=1,
    ) == receipt

    tampered = copy.deepcopy(receipt)
    tampered["adapter_change"]["maximum_absolute_delta"] = 2.0
    tampered.pop("receipt_sha256")
    tampered["receipt_sha256"] = audit.canonical_sha256(tampered)
    write_json(receipt_path, tampered)
    with pytest.raises(audit.AuditError, match="adapter-change evidence differs"):
        audit.validate_existing_checkpoint_receipt(
            receipt_path,
            expected_run_mode="smoke",
            expected_run_root=run_root,
            expected_step=1,
        )


def test_strict_json_loader_rejects_nonfinite_constants(tmp_path: Path) -> None:
    path = tmp_path / "bad.json"
    path.write_text('{"loss": NaN}\n', encoding="utf-8")
    with pytest.raises(audit.AuditError, match="non-finite JSON constant"):
        audit.load_json_object(path, description="bad evidence")


def test_atomic_receipt_writer_is_exclusive(tmp_path: Path) -> None:
    path = tmp_path / "receipt.json"
    audit.write_json_atomic_exclusive(path, {"complete": True})
    with pytest.raises(audit.AuditError, match="receipt already exists"):
        audit.write_json_atomic_exclusive(path, {"complete": True})
