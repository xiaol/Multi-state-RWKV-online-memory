from __future__ import annotations

from pathlib import Path

from experiments.rethinking_rwkv_ms_gemma import (
    analyze_natural_memory_native_multitask_preservation as analyzer,
)
from experiments.rethinking_rwkv_ms_gemma import (
    run_natural_memory_native_multitask_preservation as runner,
)


ARTIFACT_ROOT = Path("experiments/rethinking_rwkv_ms_gemma/local_artifacts")


def test_protocol_keeps_all_protected_splits_closed() -> None:
    protocol = runner.validate_protocol()

    assert protocol["authorization"]["selected_checkpoint_step"] == 16
    assert protocol["authorization"]["publisher_validation_authorized"] is False
    assert protocol["authorization"]["publisher_test_authorized"] is False
    assert protocol["authorization"]["hard32_authorized"] is False
    assert protocol["authorization"]["unused_strength_holdout_authorized"] is False


def test_narrative_rows_are_exact_untouched_remainder() -> None:
    rows = runner.routed.load_rows(ARTIFACT_ROOT / "natural_memory_native_development_v1")

    selected = runner.selected_narrative_rows(rows["narrative"])

    assert len(selected) == 114
    assert [int(row["line_index"]) for row in selected] == list(range(4, 118))


def test_pair_router_changes_only_locked_label_pair() -> None:
    base = {4: {"prediction": {"1": "narration", "2": "dialogue", "3": "narration"}}}
    memory = {4: {"prediction": {"1": "scene_description", "2": "scene_description", "3": "dialogue"}}}

    routed = runner.routed_analysis.routed_narrative_records(base, memory)

    assert routed[4]["prediction"] == {
        "1": "scene_description",
        "2": "dialogue",
        "3": "narration",
    }


def test_preservation_gate_requires_both_narrative_comparators() -> None:
    passing = analyzer.preservation_gates(
        memory_coverage=1.0,
        routed_minus_base=0.01,
        routed_minus_v9=0.0,
        attribution_exact=True,
        scene_progression_passed=True,
    )
    regressed = analyzer.preservation_gates(
        memory_coverage=1.0,
        routed_minus_base=0.01,
        routed_minus_v9=-0.0001,
        attribution_exact=True,
        scene_progression_passed=True,
    )

    assert passing["passed"] is True
    assert regressed["passed"] is False


def test_progression_and_checkpoint_are_bound() -> None:
    progression = runner.validate_progression(
        ARTIFACT_ROOT / "natural_memory_native_scene_contrast_progression_v1/result.json"
    )
    manifest = runner.selected_manifest(
        ARTIFACT_ROOT / "natural_memory_native_scene_contrast_dropout_train_v1"
    )

    assert progression["multitask_preservation_authorized"] is True
    assert manifest["step"] == 16
    assert manifest["gate_state_sha256"] == runner.SELECTED_GATE_STATE_SHA256
