from __future__ import annotations

from types import SimpleNamespace

import torch

from deltamem.core.cumulative_rwkv_residual import SourceBoundMultiAnchorBundleFFN
from experiments.rethinking_rwkv_ms_gemma import (
    materialize_natural_memory_native_rwkv_source_multi_anchor_bundle_development as materializer,
)
from experiments.rethinking_rwkv_ms_gemma import (
    run_natural_memory_native_rwkv_source_multi_anchor_bundle_development_train as runner,
)


def test_protocol_locks_fresh_open_multi_anchor_gate() -> None:
    protocol = runner.validate_protocol()

    assert protocol["receipt"]["payload_sha256"] == (
        runner.PROTOCOL_PAYLOAD_SHA256
    )
    assert protocol["open_development_only"] is True
    assert protocol["protected_mechanics_authorized"] is False
    assert protocol["protected_causal_authorized"] is False
    assert protocol["native_benchmark_authorized"] is False
    assert protocol["split"] == {
        "heldout_pairs": 16,
        "manifest_sha256": materializer.SEALED_MANIFEST_SHA256,
        "payload_sha256": runner.SPLIT_PAYLOAD_SHA256,
        "train_pairs": 24,
    }
    assert protocol["architecture"]["state_value_times_hidden_gate"] is True
    assert protocol["architecture"][
        "one_canonical_source_route_shared_by_all_anchors"
    ] is True


def test_fresh_pair_split_is_pinned_and_disjoint(monkeypatch) -> None:
    manifest = materializer.load_manifest(
        runner.DEFAULT_MATERIALIZATION / "manifest.json"
    )
    rows = materializer.read_open_development(
        runner.DEFAULT_MATERIALIZATION, manifest
    )
    monkeypatch.setattr(runner.base, "SCHEMA", runner.SCHEMA)
    monkeypatch.setattr(runner.base, "SPLIT_SCHEMA", runner.SPLIT_SCHEMA)
    monkeypatch.setattr(runner.base, "SPLIT_SALT", runner.SPLIT_SALT)
    monkeypatch.setattr(runner.base, "TRAIN_PAIRS", runner.TRAIN_PAIRS)
    monkeypatch.setattr(runner.base, "HELDOUT_PAIRS", runner.HELDOUT_PAIRS)
    monkeypatch.setattr(runner.base, "TRAIN_ROWS", runner.TRAIN_ROWS)
    monkeypatch.setattr(runner.base, "HELDOUT_ROWS", runner.HELDOUT_ROWS)

    train_rows, heldout_rows, split = runner.base.split_open_rows(rows)

    assert len(train_rows) == runner.TRAIN_ROWS
    assert len(heldout_rows) == runner.HELDOUT_ROWS
    assert runner.base.canonical_sha256(split) == runner.SPLIT_PAYLOAD_SHA256
    assert set(split["train_sources"]).isdisjoint(split["heldout_sources"])
    assert not ({runner.base._pair_key(row) for row in train_rows} & {
        runner.base._pair_key(row) for row in heldout_rows
    })


def test_bundle_parameter_inventory_and_exact_zero_contract() -> None:
    module = SourceBoundMultiAnchorBundleFFN(
        state_dim=runner.base.NATIVE_READ_DIM,
        hidden_dim=runner.base.HIDDEN_DIM,
        anchor_count=len(runner.base.ANCHORS),
        bottleneck_dim=runner.base.BOTTLENECK_DIM,
    )
    assert sum(parameter.numel() for parameter in module.parameters()) == (
        runner.TRAINABLE_ELEMENTS
    )
    assert len(tuple(module.parameters())) == runner.base.TRAINABLE_TENSORS

    with torch.no_grad():
        module.output_up.weight.fill_(1.0)
        module.query_gate.weight.fill_(1.0)
    direction, _ = module(
        native_reads=torch.zeros(
            1,
            1,
            len(runner.base.ANCHORS),
            runner.base.NATIVE_READ_DIM,
        ),
        hidden_query=torch.randn(1, 1, runner.base.HIDDEN_DIM),
    )
    assert torch.equal(direction, torch.zeros_like(direction))


def test_prompt_latch_shares_correct_source_and_mass_across_controls(
    monkeypatch,
) -> None:
    batch_size = len(runner.base.TRAIN_CONTROLS)
    states = {
        layer: torch.randn(batch_size, 1, 2, 2, 2)
        for layer in runner.base.ANCHORS
    }
    addresses = {
        layer: torch.randn(batch_size, 2, 3) for layer in runner.base.ANCHORS
    }
    occupied = {
        layer: torch.ones(batch_size, 2, dtype=torch.bool)
        for layer in runner.base.ANCHORS
    }
    source_ids = {
        layer: torch.tensor([[10, 20]]).expand(batch_size, -1).clone()
        for layer in runner.base.ANCHORS
    }
    banks = (states, addresses, occupied, source_ids)
    calls = []

    def fake_routed(*args, **kwargs):
        calls.append(kwargs)
        if len(calls) == 1:
            return torch.zeros(batch_size, 3), (
                {
                    "selected_slot": torch.tensor([[0], [1], [1], [1]]),
                    "source_ids": source_ids[runner.base.ANCHORS[-1]],
                    "memory_mass": torch.tensor(
                        [[[0.25]], [[0.5]], [[0.75]], [[1.0]]]
                    ),
                },
            )
        override = kwargs["memory_mass_override"]
        assert torch.equal(override, torch.full((batch_size, 1, 1), 0.25))
        for layer in runner.base.ANCHORS:
            retained = kwargs["banks"][2][layer]
            assert retained[:, 0].all()
            assert not retained[:, 1].any()
        return torch.ones(batch_size, 3), (
            {"memory_mass": override.detach().clone()},
        )

    monkeypatch.setattr(runner, "_ORIGINAL_ROUTED_PREDICTOR_LOGITS", fake_routed)
    monkeypatch.setattr(
        runner.base.screen.retrieval,
        "first_prompt_boundary",
        lambda labels: (3, 2),
    )
    runner._PROMPT_LATCH_AUDITS.clear()

    logits, diagnostics = runner.prompt_latched_routed_predictor_logits(
        object(),
        SimpleNamespace(labels=torch.tensor([[-100, -100, -100, 1]])),
        (),
        {},
        {},
        router=object(),
        banks=banks,
        predictor=3,
    )

    assert torch.equal(logits, torch.ones(batch_size, 3))
    assert torch.equal(
        diagnostics[-1]["prompt_latched_source_ids"],
        torch.full((batch_size,), 10),
    )
    assert runner._PROMPT_LATCH_AUDITS == [
        {
            "effective_mass_exact": True,
            "prompt_mass_shared_across_interventions": True,
            "prompt_source_shared_across_interventions": True,
            "reference_source_id": 10,
        }
    ]


def test_stronger_gate_requires_selection_positive_gain_and_layer_margin() -> None:
    margins = {
        "gain_vs_provider_off": {"mean": 0.02, "positive_fraction": 0.625},
        "donor_both_minus_target": {"mean": 0.03, "positive_fraction": 0.75},
        "layer_both_minus_target": {"mean": 0.01, "positive_fraction": 0.75},
    }
    result = {
        "checks": {"base": True},
        "aggregate": {
            "target_ce_margins": margins,
            "terminal_target_selected_fraction": 0.875,
        },
    }

    passed = runner._strengthen_gate(result, original_view=True)
    failed = runner._strengthen_gate(
        {
            **result,
            "aggregate": {
                **result["aggregate"],
                "terminal_target_selected_fraction": 0.874,
            },
        },
        original_view=True,
    )

    assert passed["passed"] is True
    assert failed["passed"] is False


def test_wrong_prompt_route_trains_only_isolated_target(monkeypatch) -> None:
    metrics = {
        "correct_ce": torch.tensor(8.0),
        "single_ce": torch.tensor(2.0),
        "donor_ce": torch.tensor(7.0),
        "layer_ce": torch.tensor(6.0),
        "donor_minus_single": torch.tensor(5.0),
        "layer_minus_single": torch.tensor(4.0),
        "donor_contrast": torch.tensor(3.0),
        "layer_contrast": torch.tensor(2.0),
    }
    monkeypatch.setattr(
        runner,
        "_ORIGINAL_TRAINING_LOSS",
        lambda logits, target: (torch.tensor(99.0), metrics),
    )
    monkeypatch.setattr(
        runner, "_LAST_TRAIN_CORRECT_ROUTE_SELECTED_TARGET", False
    )

    loss, returned = runner.conditional_training_loss(torch.zeros(4, 3), 0)

    assert loss == runner.base.SINGLE_CE_WEIGHT * metrics["single_ce"]
    assert returned is metrics
