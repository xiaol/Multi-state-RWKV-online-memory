from __future__ import annotations

from pathlib import Path

import torch

from experiments.rethinking_rwkv_ms_gemma import (
    materialize_natural_memory_native_rwkv_source_cumulative_residual_development as materializer,
)
from experiments.rethinking_rwkv_ms_gemma import (
    run_natural_memory_native_rwkv_source_bound_outer_ffn_development_train as runner,
)


MATERIALIZATION = runner.SCRIPT_DIR / (
    "local_artifacts/"
    "natural_memory_native_rwkv_source_cumulative_residual_development_v1"
)
EXPECTED_SPLIT_SHA256 = (
    "75d31c6a5fe86aace7a1378b8f82fd671fb556a5d601a04cf14a59a195e35797"
)


def test_protocol_and_open_pair_split_are_frozen() -> None:
    protocol = runner.validate_protocol()
    manifest = materializer.load_manifest(MATERIALIZATION / "manifest.json")
    rows = materializer.read_open_development(MATERIALIZATION, manifest)
    train_rows, heldout_rows, split = runner.split_open_rows(rows)

    assert len(train_rows) == runner.TRAIN_ROWS
    assert len(heldout_rows) == runner.HELDOUT_ROWS
    assert runner.canonical_sha256(split) == EXPECTED_SPLIT_SHA256
    assert protocol["split"]["payload_sha256"] == EXPECTED_SPLIT_SHA256
    train_sources = {int(row["source_index"]) for row in train_rows}
    heldout_sources = {int(row["source_index"]) for row in heldout_rows}
    assert train_sources.isdisjoint(heldout_sources)
    assert not ({runner._pair_key(row) for row in train_rows} & {
        runner._pair_key(row) for row in heldout_rows
    })


def test_contrast_loss_rewards_correct_over_donor_and_layer_controls() -> None:
    target = 0
    good_logits = torch.tensor(
        [
            [4.0, 0.0, -1.0],
            [4.0, 0.0, -1.0],
            [0.0, 4.0, -1.0],
            [0.0, -1.0, 4.0],
        ],
        requires_grad=True,
    )
    bad_logits = good_logits.detach().clone()
    bad_logits[2] = good_logits.detach()[1]
    bad_logits[3] = good_logits.detach()[1]
    bad_logits.requires_grad_(True)

    good_loss, good_metrics = runner.training_loss(good_logits, target)
    bad_loss, bad_metrics = runner.training_loss(bad_logits, target)

    assert good_metrics["donor_minus_single"] > runner.DONOR_MARGIN
    assert good_metrics["layer_minus_single"] > runner.LAYER_MARGIN
    assert bad_metrics["donor_minus_single"] == 0.0
    assert bad_metrics["layer_minus_single"] == 0.0
    assert good_loss < bad_loss
    good_loss.backward()
    assert good_logits.grad is not None
    assert bool(torch.isfinite(good_logits.grad).all().item())


def test_checkpoint_dimensions_match_protocol() -> None:
    outer_ffn = runner.SourceBoundOuterFFN(
        state_dim=runner.NATIVE_READ_DIM,
        query_dim=runner.HIDDEN_DIM,
        bottleneck_dim=runner.BOTTLENECK_DIM,
    )
    assert len(tuple(outer_ffn.parameters())) == runner.TRAINABLE_TENSORS
    assert sum(parameter.numel() for parameter in outer_ffn.parameters()) == (
        runner.TRAINABLE_ELEMENTS
    )
