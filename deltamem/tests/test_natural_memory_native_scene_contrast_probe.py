from __future__ import annotations

from pathlib import Path

import torch

from experiments.rethinking_rwkv_ms_gemma import (
    analyze_natural_memory_native_scene_contrast_probe as analyzer,
)
from experiments.rethinking_rwkv_ms_gemma import (
    run_natural_memory_native_scene_causal as causal,
)
from experiments.rethinking_rwkv_ms_gemma import (
    run_natural_memory_native_scene_contrast_dropout as training,
)
from experiments.rethinking_rwkv_ms_gemma import (
    run_natural_memory_native_scene_contrast_probe as runner,
)
from experiments.rethinking_rwkv_ms_gemma import (
    run_natural_memory_native_scene_contrast_progression as progression,
)
from experiments.rethinking_rwkv_ms_gemma import (
    run_synthetic_compositional_associative_retrieval_v3 as runtime,
)


ARTIFACT_ROOT = Path(
    "experiments/rethinking_rwkv_ms_gemma/local_artifacts"
)


def test_probe_selection_is_locked_to_open_fit_rows() -> None:
    rows = causal.load_rows(ARTIFACT_ROOT / "natural_memory_native_development_v1")

    selected = runner.selected_probe_rows(rows)
    payload = [
        {
            "source_index": int(row["source_index"]),
            "row_sha256": str(row["row_sha256"]),
        }
        for row in selected
    ]

    assert len(selected) == 64
    assert runner.canonical_sha256(payload) == runner.PROBE_PAYLOAD_SHA256
    assert all(int(row["source_index"]) >= 4 for row in selected)


def test_training_result_and_all_three_patches_are_bound() -> None:
    manifests = runner.validate_training_root(
        ARTIFACT_ROOT / "natural_memory_native_scene_contrast_dropout_train_v1"
    )

    assert [manifest["step"] for manifest in manifests] == [8, 16, 32]
    assert all(manifest["parameter_tensors"] == 126 for manifest in manifests)


def test_gate_patch_loader_replaces_only_locked_gate_parameters(tmp_path: Path) -> None:
    class Model(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.layers = torch.nn.ModuleList()
            for _ in range(42):
                layer = torch.nn.Module()
                layer.memory_fusion_hidden_weight = torch.nn.Parameter(torch.zeros(1))
                layer.memory_fusion_read_weight = torch.nn.Parameter(torch.zeros(1))
                layer.memory_fusion_bias = torch.nn.Parameter(torch.zeros(1))
                layer.other = torch.nn.Parameter(torch.tensor([7.0]))
                self.layers.append(layer)

    model = Model()
    state = {
        name: torch.full_like(parameter, 3.0)
        for name, parameter in model.named_parameters()
        if any(name.endswith(f".{family}") for family in training.GATE_FAMILIES)
    }
    digest = runtime._state_dict_sha256(state)
    patch_path = tmp_path / "gate_patch.pt"
    torch.save(
        {
            "schema": training.PATCH_SCHEMA,
            "protocol_payload_sha256": training.PROTOCOL_PAYLOAD_SHA256,
            "source_adapter_files_sha256": training.V9_ADAPTER_FILES_SHA256,
            "step": 8,
            "gate_state_sha256": digest,
            "state_dict": state,
        },
        patch_path,
    )
    manifest = {"step": 8, "gate_state_sha256": digest}

    loaded = runner.load_gate_patch(model, patch_path=patch_path, manifest=manifest)

    assert loaded["gate_state_sha256"] == digest
    assert loaded["runtime_gate_state_sha256"] == digest
    assert loaded["parameter_tensors"] == 126
    assert all(layer.other.item() == 7.0 for layer in model.layers)
    assert all(layer.memory_fusion_bias.item() == 3.0 for layer in model.layers)


def test_candidate_passes_only_with_native_and_causal_gains() -> None:
    indices = tuple(range(20))
    gold = {index: {0} for index in indices}
    correct = {index: {0} for index in indices}
    donor = {index: set() for index in indices}
    zero = {index: set() for index in indices}
    v9 = {index: ({0} if index < 18 else set()) for index in indices}

    result = analyzer.candidate_result(
        step=8,
        predictions={
            "correct_state": correct,
            "matched_donor_state": donor,
            "zero_state": zero,
        },
        v9_predictions=v9,
        gold=gold,
        indices=indices,
    )

    assert result["passed"] is True
    assert result["deltas"]["correct_minus_v9_micro_f1"] > 0.005
    assert result["deltas"]["paired_output_change_fraction_vs_v9"] >= 0.05

    unchanged = analyzer.candidate_result(
        step=16,
        predictions={
            "correct_state": v9,
            "matched_donor_state": donor,
            "zero_state": zero,
        },
        v9_predictions=v9,
        gold=gold,
        indices=indices,
    )
    assert unchanged["passed"] is False
    assert unchanged["gates"]["correct_minus_v9_micro_f1_at_least_0.005"] is False
    assert unchanged["gates"]["paired_output_change_fraction_vs_v9_at_least_0.05"] is False


def test_ranking_prefers_correct_f1_then_causal_margins_then_earlier_step() -> None:
    base = {
        "checkpoint_step": 8,
        "metrics": {"correct_state": {"micro_f1": 0.4}},
        "deltas": {
            "correct_minus_matched_donor_micro_f1": 0.1,
            "correct_minus_zero_micro_f1": 0.2,
        },
    }
    later = {
        **base,
        "checkpoint_step": 16,
        "deltas": {
            "correct_minus_matched_donor_micro_f1": 0.2,
            "correct_minus_zero_micro_f1": 0.2,
        },
    }
    lower_f1 = {
        **base,
        "checkpoint_step": 32,
        "metrics": {"correct_state": {"micro_f1": 0.39}},
    }

    ranked = sorted([lower_f1, base, later], key=analyzer.ranking_key)

    assert [candidate["checkpoint_step"] for candidate in ranked] == [16, 8, 32]


def test_progression_is_authorized_by_signed_checkpoint_16_selection() -> None:
    protocol = progression.validate_protocol()
    selection = progression.validate_selection(
        ARTIFACT_ROOT / "natural_memory_native_scene_contrast_probe_v1/selection.json"
    )

    assert protocol["authorization"]["selected_checkpoint_step"] == 16
    assert selection["selected_checkpoint_step"] == 16
    assert protocol["authorization"]["unused_strength_holdout_authorized"] is False


def test_progression_uses_exactly_the_remaining_open_fit_rows() -> None:
    rows = causal.load_rows(ARTIFACT_ROOT / "natural_memory_native_development_v1")

    remaining = progression.progression_rows(rows)
    payload = [
        {
            "source_index": int(row["source_index"]),
            "row_sha256": str(row["row_sha256"]),
        }
        for row in remaining
    ]

    assert len(remaining) == 220
    assert progression.canonical_sha256(payload) == progression.REMAINING_PAYLOAD_SHA256
