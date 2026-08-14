from __future__ import annotations

from pathlib import Path

from experiments.rethinking_rwkv_ms_gemma import (
    analyze_natural_memory_native_multitask_hybrid as analyzer,
)


ARTIFACT_ROOT = Path("experiments/rethinking_rwkv_ms_gemma/local_artifacts")


def test_hybrid_protocol_is_closed_and_uses_checkpoint_16_only_for_scene() -> None:
    protocol = analyzer.validate_protocol()

    assert "frozen V9" in protocol["decoder"]["narrative"]
    assert protocol["source"]["selected_checkpoint_step"] == 16
    assert protocol["authorization"]["publisher_validation_opened"] is False
    assert protocol["authorization"]["publisher_test_authorized"] is False


def test_hybrid_result_passes_without_reopening_protected_splits(tmp_path: Path) -> None:
    result = analyzer.analyze(
        failed_result=ARTIFACT_ROOT / "natural_memory_native_multitask_preservation_v1/result.json",
        reference_root=ARTIFACT_ROOT / "natural_memory_native_routed_benchmark_v1_r2",
        progression_result=ARTIFACT_ROOT / "natural_memory_native_scene_contrast_progression_v1/result.json",
        output=tmp_path / "result.json",
    )

    assert result["gates"]["passed"] is True
    assert result["fresh_publisher_validation_replication_contract_authorized"] is True
    assert result["publisher_validation_opened"] is False
    assert result["protected_splits_opened"] == []
