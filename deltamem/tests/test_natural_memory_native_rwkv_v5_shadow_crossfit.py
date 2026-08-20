from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from experiments.rethinking_rwkv_ms_gemma import (
    run_natural_memory_native_rwkv_v5_shadow_crossfit as screen,
)


def test_v5_shadow_protocol_is_signed_and_stops_before_mechanics() -> None:
    protocol, v5_result = screen.validate_protocol()

    assert protocol["receipt"]["payload_sha256"] == screen.PROTOCOL_PAYLOAD_SHA256
    assert protocol["exact_source_loader"]["learned_write_installed"] is False
    assert protocol["exact_source_loader"]["config_overrides"] == []
    assert protocol["shadow_feature_capture"]["binder_or_bridge_installed"] is False
    assert protocol["shadow_feature_capture"]["model_output_changed"] is False
    assert protocol["frozen_inputs"]["signed_v5_source_commit"] == screen.SIGNED_V5_COMMIT
    assert protocol["frozen_inputs"]["signed_v5_delta_impl_sha256"] == (
        screen.SIGNED_V5_DELTA_IMPL_SHA256
    )
    assert protocol["causal_mechanics_authorized"] is False
    assert protocol["training_authorized"] is False
    assert protocol["generation_authorized"] is False
    assert protocol["protected_splits_opened_by_this_protocol"] == []
    assert v5_result["passed"] is True


def test_v5_shadow_crossfit_split_is_donor_component_disjoint() -> None:
    mapping = {
        source: source + 1 if source % 2 == 0 else source - 1
        for source in range(220)
    }

    split, payload = screen.crossfit_split(mapping)

    assert sum(value == "train" for value in split.values()) == 176
    assert sum(value == "heldout" for value in split.values()) == 44
    assert all(split[source] == split[donor] for source, donor in mapping.items())
    assert payload["selection_salt"] == screen.SPLIT_SALT
    assert payload["component_count"] == 110


def test_v5_shadow_state_snapshot_detaches_and_clones(monkeypatch) -> None:
    original = torch.randn(1, 2, 3)
    monkeypatch.setattr(
        screen.causal_train,
        "capture_online_state_references",
        lambda modules: {"layer": {"delta_state": original}},
    )

    captured = screen.clone_online_state((("layer", SimpleNamespace()),))

    clone = captured["layer"]["delta_state"]
    assert clone.data_ptr() != original.data_ptr()
    assert clone.requires_grad is False
    assert torch.equal(clone, original)
    original.add_(1.0)
    assert not torch.equal(clone, original)


def test_v5_shadow_execution_requires_explicit_signed_source(monkeypatch) -> None:
    monkeypatch.setattr(screen, "SIGNED_SOURCE_ROOT", None)
    with pytest.raises(RuntimeError, match=screen.SIGNED_SOURCE_ROOT_ENV):
        screen.validate_execution_source()


def test_v5_shadow_execution_rejects_core_outside_signed_root(
    monkeypatch, tmp_path
) -> None:
    monkeypatch.setattr(screen, "SIGNED_SOURCE_ROOT", tmp_path)
    with pytest.raises(RuntimeError, match="outside the signed source root"):
        screen.validate_execution_source()


def test_v5_shadow_tiny_synthetic_crossfit_contract(monkeypatch) -> None:
    monkeypatch.setattr(screen, "TRAIN_ROWS", 8)
    monkeypatch.setattr(screen, "HELDOUT_ROWS", 4)
    monkeypatch.setattr(screen, "TRAIN_STEPS", 8)
    generator = torch.Generator().manual_seed(19)
    records = []
    for source_index in range(12):
        query = torch.randn(
            screen.LAYERS,
            screen.STATE_DIM,
            generator=generator,
        )
        correct = query + 0.01 * torch.randn(
            query.shape,
            generator=generator,
        )
        donor = -query + 0.01 * torch.randn(
            query.shape,
            generator=generator,
        )
        records.append(
            {
                "source_index": source_index,
                "split": "train" if source_index < 8 else "heldout",
                "query": query.tolist(),
                "correct": correct.tolist(),
                "matched_donor": donor.tolist(),
                "layer_permuted": correct.roll(1, dims=0).tolist(),
            }
        )

    result = screen.train_and_evaluate(records)

    assert result["loss"]["final"] <= result["loss"]["initial"]
    assert result["metrics"]["heldout"]["donor"]["finite"] is True
    assert result["optimizer"]["seed"] == screen.HEAD_SEED
    assert result["weights_saved"] is False
