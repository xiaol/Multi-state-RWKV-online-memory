from __future__ import annotations

from types import SimpleNamespace
import json

import pytest
import torch

from experiments.rethinking_rwkv_ms_gemma import (
    run_natural_memory_native_rwkv_v5_shadow_predictor_recurrent_mechanics as mechanics,
)


def test_signed_protocol_locks_predictor_stage_and_full_mechanics_conditions() -> None:
    protocol, source = mechanics.validate_protocol()

    assert protocol["receipt"]["payload_sha256"] == mechanics.PROTOCOL_PAYLOAD_SHA256
    assert source["receipt"]["payload_sha256"] == mechanics.CROSSFIT_RECEIPT
    assert protocol["stage1_predictor_crossfit"]["feature_positions"].startswith(
        "causal predictors"
    )
    assert protocol["stage1_predictor_crossfit"][
        "heldout_used_for_training_thresholds_or_selection"
    ] is False
    assert protocol["stage2_recurrent_mechanics"]["conditions"] == [
        "correct",
        "donor-shadow",
        "donor-pair",
        "zero",
        "layer-permuted",
        "row-shuffled",
        "norm-random",
        "fixed-correct-gate donor-value",
        "correct-no-feedback",
        "disabled",
    ]
    semantics = protocol["stage2_recurrent_mechanics"]["control_semantics"]
    assert "only the detached shadow" in semantics["row-shuffled"]
    assert "only the detached shadow" in semantics["norm-random"]
    assert "only inside the certified binder" in semantics[
        "fixed-correct-gate donor-value"
    ]
    assert protocol["model_or_adapter_training_authorized"] is False
    assert protocol["generation_authorized"] is False
    assert protocol["protected_splits_opened_by_this_protocol"] == []


def test_predictor_capture_uses_shifted_labels_not_answer_token_positions(
    monkeypatch,
) -> None:
    monkeypatch.setattr(mechanics, "LAYERS", 2)
    monkeypatch.setattr(mechanics, "STATE_DIM", 3)
    labels = torch.tensor([[-100, -100, 20, 21, -100]])
    positions = torch.arange(5, dtype=torch.float32).view(1, 5, 1).expand(-1, -1, 3)
    captured = tuple(
        SimpleNamespace(
            module_name=f"layer-{layer}",
            query_address=positions + 10 * layer,
            recurrent_read=positions + 100 + 10 * layer,
        )
        for layer in range(2)
    )

    query, state = mechanics.predictor_vectors(captured, labels)
    runtime_query, runtime_shadow, masks = mechanics.token_runtime(captured, labels)

    assert mechanics.predictor_mask(labels).tolist() == [[False, True, True, False, False]]
    assert query[:, 0, 0].tolist() == [1.0, 2.0]
    assert state[:, 0, 0].tolist() == [101.0, 102.0]
    assert masks["layer-0"].tolist() == [[False, True, True, False, False]]
    assert torch.equal(runtime_query["layer-0"][:, 3:], torch.zeros(1, 2, 3))
    assert torch.equal(runtime_shadow["layer-0"][:, :1], torch.zeros(1, 1, 3))


def test_predictor_crossfit_reports_token_and_row_heldout_gates(monkeypatch) -> None:
    monkeypatch.setattr(mechanics, "TRAIN_ROWS", 8)
    monkeypatch.setattr(mechanics, "HELDOUT_ROWS", 4)
    monkeypatch.setattr(mechanics, "TRAIN_STEPS", 8)
    monkeypatch.setattr(mechanics, "LAYERS", 2)
    monkeypatch.setattr(mechanics, "STATE_DIM", 4)
    monkeypatch.setattr(mechanics.shadow, "LAYERS", 2)
    monkeypatch.setattr(mechanics.shadow, "STATE_DIM", 4)
    monkeypatch.setattr(mechanics.shadow.source, "LAYERS", 2)
    monkeypatch.setattr(mechanics.shadow.source, "STATE_DIM", 4)
    generator = torch.Generator().manual_seed(31)
    records = []
    for source_index in range(12):
        tokens = 2 + source_index % 2
        query = torch.randn(tokens, 2, 4, generator=generator)
        correct = query + 0.01 * torch.randn(query.shape, generator=generator)
        donor = -query + 0.01 * torch.randn(query.shape, generator=generator)
        records.append(
            {
                "source_index": source_index,
                "split": "train" if source_index < 8 else "heldout",
                "predictor_tokens": tokens,
                "query": query.tolist(),
                "correct": correct.tolist(),
                "matched_donor": donor.tolist(),
                "layer_permuted": correct.roll(1, dims=1).tolist(),
            }
        )

    _, thresholds, result = mechanics.fit_predictor_head(records)

    assert thresholds.shape == (2,)
    assert result["thresholds"]["source"] == "train_only"
    assert result["metrics"]["heldout"]["donor"]["tokens"] == 10
    assert result["metrics"]["heldout"]["donor"]["rows"] == 4
    assert "heldout_donor_token_fraction" in result["checks"]
    assert "heldout_donor_row_fraction" in result["checks"]


def _condition(pass2_differs: bool, disabled_collapse: bool) -> dict:
    passes = []
    for index in range(mechanics.PASSES):
        delta = None
        if index:
            mean = 0.5 ** index
            delta = {"sum": 10.0 * mean, "vectors": 10, "mean": mean}
        parts = {
            name: {"sum": float(index + 1), "tokens": 1, "mean": float(index + 1)}
            for name in ("overall", "first", "later")
        }
        passes.append(
            {
                "pass": index + 1,
                "ce": parts,
                "live_value_delta_from_previous": delta,
                "logits_finite": True,
                "write_enabled_after_read": False,
            }
        )
    return {
        "passes": passes,
        "pass2_differs_pass1": pass2_differs,
        "disabled_exact_collapse": disabled_collapse,
        "tail_contraction_checks": [True, True, True],
        "tail_contracted": True,
    }


def _comparison(changed: float) -> dict:
    return {
        "predictor_logit_changed_fraction_by_row": [changed] * 11,
        "mean_predictor_logit_changed_fraction": changed,
        "rows_at_least_095_changed_fraction": float(changed >= 0.95),
        "ce_delta_by_row": {
            "overall": [0.1] * 11,
            "first": [0.1] * 11,
            "later": [0.1] * 11,
        },
        "ce_positive_row_fraction": {
            "overall": 1.0,
            "first": 1.0,
            "later": 1.0,
        },
    }


def test_stage2_aggregation_enforces_balanced_rows_and_exact_collapse() -> None:
    rows = []
    for rank in range(4):
        conditions = {"correct": _condition(True, False)}
        for name in (
            *mechanics.MATERIAL_CONTROL_CONDITIONS,
            "correct-no-feedback",
            "disabled",
        ):
            condition = _condition(False, name == "disabled")
            condition["comparisons_to_correct"] = {
                "pass1": _comparison(
                    0.0 if name == "correct-no-feedback" else 1.0
                ),
                "pass8": _comparison(1.0),
            }
            conditions[name] = condition
        rows.append(
            {
                "rank": rank,
                "rows": 11,
                "conditions": conditions,
                "state_hashes": {
                    "target_before": "target",
                    "target_after": "target",
                    "donor_before": "donor",
                    "donor_after": "donor",
                },
                "writes": {
                    "target": 1,
                    "donor": 1,
                    "during_reads": 0,
                    "write_disabled_verified_after_every_read": True,
                },
                "checks": {
                    "all_logits_finite": True,
                    "fixed_gate_donor_value_original_fusion_uses_target_state": True,
                },
            }
        )

    result = mechanics.aggregate_stage2(rows)

    assert result["passed"] is True
    assert result["checks"]["exactly_44_heldout_rows_balanced_11_per_rank"] is True
    assert result["checks"]["correct_pass2_differs_pass1"] is True
    assert result["checks"]["disabled_exact_collapse"] is True
    assert result["checks"]["no_feedback_pass1_exact_correct"] is True
    assert result["checks"]["no_feedback_pass8_material"] is True
    assert result["checks"]["disabled_correction_material_pass1_pass8"] is True
    assert result["conditions"]["correct"]["passes"][0]["ce"]["overall"]["tokens"] == 4


def test_main_returns_success_on_non_primary_empty_result(monkeypatch, tmp_path) -> None:
    captured = {}

    def fake_run(**kwargs):
        captured.update(kwargs)
        return {}

    monkeypatch.setattr(mechanics, "run", fake_run)

    assert mechanics.main(
        [
            "--base-model",
            str(tmp_path),
            "--dataset-root",
            str(tmp_path),
            "--output-dir",
            str(tmp_path / "fresh"),
            "--resume-complete-stage1-shards",
        ]
    ) == 0
    assert captured["resume_complete_stage1_shards"] is True


def test_read_write_flag_audit_requires_explicit_false() -> None:
    assert mechanics.reads_are_write_disabled(
        (("a", SimpleNamespace(write_enabled=False)),)
    ) is True
    assert mechanics.reads_are_write_disabled(
        (("a", SimpleNamespace(write_enabled=True)),)
    ) is False


def test_state_hash_detects_mutation_but_clone_is_independent() -> None:
    source = {
        "layer": {
            "delta_state": torch.arange(6, dtype=torch.float32).reshape(1, 2, 3),
            "rwkv_ms_positions": torch.tensor([[1, 2]]),
            "rwkv_ms_previous_source": torch.tensor([[3, 4]]),
        }
    }
    clone = mechanics.clone_state(source)
    before = mechanics.state_sha256(clone)
    source["layer"]["delta_state"].add_(1)

    assert mechanics.state_sha256(clone) == before
    clone["layer"]["delta_state"].add_(1)
    assert mechanics.state_sha256(clone) != before


def test_feature_shard_validation_checks_rank_metadata_shape_and_hashes(
    monkeypatch, tmp_path
) -> None:
    monkeypatch.setattr(mechanics.endpoint, "EVALUATION_ROWS", 1)
    monkeypatch.setattr(mechanics, "LAYERS", 2)
    monkeypatch.setattr(mechanics, "STATE_DIM", 3)
    feature = torch.ones(2, 2, 3).tolist()
    record = {
        "schema": mechanics.FEATURE_SCHEMA,
        "source_index": 0,
        "row_sha256": "target-hash",
        "donor_source_index": 4,
        "donor_row_sha256": "donor-hash",
        "split": "train",
        "predictor_tokens": 2,
        "feature_positions": "labels[:,1:]_shifted_one_token_left",
        "projected_carrier_fixed": True,
        "state_snapshots_detached_and_cloned": True,
        "binder_or_feedback_installed": False,
        "query": feature,
        "correct": feature,
        "matched_donor": feature,
        "layer_permuted": feature,
    }
    for rank in range(4):
        rows = [record] if rank == 0 else []
        (tmp_path / f"stage1-shard-{rank}.jsonl").write_text(
            "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
        )
    split = {
        "rows": [
            {
                "source_index": 0,
                "row_sha256": "target-hash",
                "donor_source_index": 4,
                "donor_row_sha256": "donor-hash",
                "split": "train",
            }
        ]
    }

    rows, _ = mechanics.load_feature_records(tmp_path, split)
    assert len(rows) == 1

    bad = dict(record, donor_row_sha256="wrong")
    (tmp_path / "stage1-shard-0.jsonl").write_text(
        json.dumps(bad) + "\n", encoding="utf-8"
    )
    with pytest.raises(RuntimeError, match="row contract"):
        mechanics.load_feature_records(tmp_path, split)
