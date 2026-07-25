from __future__ import annotations

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
