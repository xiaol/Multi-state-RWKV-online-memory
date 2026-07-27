from __future__ import annotations

import json

import pytest
import torch

from experiments.rethinking_rwkv_ms_gemma import diagnose_memory_representation as diagnostic


def test_default_output_path_uses_checkpoint_run_root(tmp_path) -> None:
    checkpoint = tmp_path / "run" / "trainer" / "checkpoint-416"
    checkpoint.mkdir(parents=True)

    output = diagnostic.default_output_path(checkpoint)

    assert output == (
        tmp_path
        / "run"
        / "representation_diagnostic"
        / "checkpoint-416_writer_reader_representation.json"
    )


def test_pair_metrics_reports_identical_and_orthogonal_vectors() -> None:
    identical = diagnostic.pair_metrics(
        torch.tensor([3.0, 4.0]),
        torch.tensor([3.0, 4.0]),
    )
    orthogonal = diagnostic.pair_metrics(
        torch.tensor([1.0, 0.0]),
        torch.tensor([0.0, 1.0]),
    )

    assert identical["cosine"] == pytest.approx(1.0)
    assert identical["relative_l2_mean_norm"] == pytest.approx(0.0)
    assert orthogonal["cosine"] == pytest.approx(0.0)
    assert orthogonal["relative_l2_mean_norm"] == pytest.approx(2**0.5)


def test_correction_reference_metrics_reports_tokenwise_ratio_and_cosine() -> None:
    correction = torch.tensor([[3.0, 0.0], [0.0, 4.0]])
    reference = torch.tensor([[6.0, 0.0], [0.0, -8.0]])

    metrics = diagnostic.correction_reference_metrics(correction, reference)

    assert metrics["global_norm_ratio"] == pytest.approx(0.5)
    assert metrics["token_norm_ratio_mean"] == pytest.approx(0.5)
    assert metrics["token_cosine_mean"] == pytest.approx(0.0)
    assert metrics["token_cosine_min"] == pytest.approx(-1.0)
    assert metrics["token_cosine_max"] == pytest.approx(1.0)


def test_causal_read_context_mask_includes_supervised_predictors() -> None:
    labels = torch.tensor(
        [
            [-100, -100, 10, 11, -100],
            [-100, 20, -100, 30, 31],
        ]
    )
    attention_mask = torch.tensor(
        [
            [1, 1, 1, 1, 0],
            [1, 1, 1, 1, 1],
        ]
    )

    mask = diagnostic.causal_read_context_mask(labels, attention_mask)

    assert torch.equal(
        mask,
        torch.tensor(
            [
                [True, True, True, False, False],
                [True, False, True, True, False],
            ]
        ),
    )


def test_replace_module_online_state_replaces_only_requested_layer() -> None:
    base = {
        "layer.0": torch.tensor([0]),
        "layer.0.__rwkv_ms_positions": torch.tensor([1]),
        "layer.0.__rwkv_ms_previous_source": torch.tensor([2]),
        "layer.1": torch.tensor([3]),
    }
    replacement = {
        "layer.0": torch.tensor([10]),
        "layer.0.__rwkv_ms_positions": torch.tensor([11]),
        "layer.0.__rwkv_ms_previous_source": torch.tensor([12]),
    }

    mixed = diagnostic.replace_module_online_state(base, replacement, "layer.0")

    assert mixed["layer.0"].item() == 10
    assert mixed["layer.0.__rwkv_ms_positions"].item() == 11
    assert mixed["layer.0.__rwkv_ms_previous_source"].item() == 12
    assert mixed["layer.1"].item() == 3
    assert base["layer.0"].item() == 0


def test_causal_state_swap_metrics_requires_bidirectional_improvement() -> None:
    metrics = diagnostic.causal_state_swap_metrics(
        correct_ce=1.0,
        donor_ce=1.2,
        donor_with_correct_layer_ce=1.1,
        correct_with_donor_layer_ce=1.05,
    )

    assert metrics["donor_to_correct_ce_gain"] == pytest.approx(0.1)
    assert metrics["correct_to_donor_ce_damage"] == pytest.approx(0.05)
    assert metrics["bidirectional_mean_ce_effect"] == pytest.approx(0.075)
    assert metrics["bidirectional_positive"] is True


def test_pearson_correlation_handles_signal_and_constant_inputs() -> None:
    assert diagnostic.pearson_correlation([1.0, 2.0, 3.0], [2.0, 4.0, 6.0]) == pytest.approx(1.0)
    assert diagnostic.pearson_correlation([1.0, 1.0, 1.0], [2.0, 4.0, 6.0]) is None


def test_load_pairing_donors_prefers_checkpoint_manifest(tmp_path) -> None:
    checkpoint = tmp_path / "checkpoint-416"
    checkpoint.mkdir()
    manifest = {
        "manifest_sha256": "logical-manifest-hash",
        "splits": {
            "train": {
                "manifest_sha256": "logical-split-hash",
                "pairing_version": "post_split_half_rotation_v1",
                "pairs": [
                    {"source_index": 0, "partner_index": 2},
                    {"source_index": 1, "partner_index": 3},
                    {"source_index": 2, "partner_index": 0},
                    {"source_index": 3, "partner_index": 1},
                ],
            }
        },
    }
    (checkpoint / "content_contrast_pairing_manifest.json").write_text(
        json.dumps(manifest),
        encoding="utf-8",
    )
    tokenized = diagnostic.Dataset.from_dict(
        {"write_input_ids": [[index] for index in range(4)]}
    )

    donors, provenance = diagnostic.load_pairing_donors(
        checkpoint,
        split_name="train",
        row_count=4,
        fallback_seed=17,
        tokenized=tokenized,
    )

    assert donors == [2, 3, 0, 1]
    assert provenance["source"] == "checkpoint_pairing_manifest"
    assert provenance["manifest_sha256"] == "logical-manifest-hash"


def test_representation_summary_separates_common_mean_from_content_rank() -> None:
    rows = torch.tensor(
        [
            [10.0, 1.0, 0.0],
            [10.0, -1.0, 0.0],
            [10.0, 0.0, 1.0],
            [10.0, 0.0, -1.0],
        ]
    )

    summary = diagnostic.representation_summary(rows)

    assert summary["mean_vector_l2_norm"] == pytest.approx(10.0)
    assert summary["centered_variation_rms"] == pytest.approx(1.0)
    assert summary["centered_variation_to_mean_norm"] == pytest.approx(0.1)
    assert summary["centered_spectrum"]["numerical_rank"] == 2
    assert summary["centered_spectrum"]["effective_rank"] == pytest.approx(2.0)
    assert summary["uncentered_spectrum"]["top1_energy_fraction"] > 0.99


def test_representation_summary_handles_identical_rows() -> None:
    rows = torch.ones(4, 8)

    summary = diagnostic.representation_summary(rows)

    assert summary["off_diagonal_cosine_mean"] == pytest.approx(1.0)
    assert summary["off_diagonal_relative_l2_mean"] == pytest.approx(0.0)
    assert summary["centered_spectrum"]["effective_rank"] == pytest.approx(0.0)


def test_causal_supervised_features_uses_logit_source_positions() -> None:
    values = torch.tensor(
        [
            [
                [1.0, 10.0],
                [2.0, 20.0],
                [3.0, 30.0],
                [4.0, 40.0],
            ]
        ]
    )
    labels = torch.tensor([[-100, 7, -100, 9]])
    attention_mask = torch.ones_like(labels)

    selected = diagnostic.causal_supervised_features(values, labels, attention_mask)

    assert torch.equal(selected, torch.tensor([[1.0, 10.0], [3.0, 30.0]]))
