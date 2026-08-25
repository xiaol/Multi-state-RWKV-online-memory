from __future__ import annotations

from pathlib import Path

from experiments.rethinking_rwkv_ms_gemma import materialize_natural_memory_native_rwkv_weighted_renewal_bundle_development as materializer
from experiments.rethinking_rwkv_ms_gemma import materialize_natural_memory_native_rwkv_source_multi_anchor_bundle_development as multi_materializer


ROOT = Path(__file__).parents[2]
WEIGHTED_ROOT = ROOT / "experiments/rethinking_rwkv_ms_gemma/local_artifacts/natural_memory_native_rwkv_weighted_renewal_bundle_development_v1"
MULTI_ROOT = ROOT / "experiments/rethinking_rwkv_ms_gemma/local_artifacts/natural_memory_native_rwkv_source_multi_anchor_bundle_development_v1"


def test_weighted_renewal_manifest_is_open_and_component_disjoint() -> None:
    manifest = materializer.load_manifest(WEIGHTED_ROOT / "manifest.json")
    rows = materializer.read_open_development(WEIGHTED_ROOT, manifest)
    audit = manifest["split_contract"]["leakage_audit"]
    assert len(rows) == 80
    assert audit["total_excluded_components"] == 462
    assert audit["remaining_components_before_selection"] == 246
    assert audit["excluded_overlap_component_count"] == 0
    assert manifest["protected_splits_opened"] == []
    assert manifest["access"]["development_is_explicitly_open"] is True
    assert manifest["access"]["protected_reservation_bundle_bytes_read"] is False
    assert manifest["access"]["prior_development_bundle_bytes_read"] is False
    assert manifest["access"]["prior_multi_anchor_bundle_bytes_read"] is False


def test_weighted_renewal_excludes_prior_multi_anchor_components() -> None:
    weighted = materializer.load_manifest(WEIGHTED_ROOT / "manifest.json")
    prior = multi_materializer.load_manifest(MULTI_ROOT / "manifest.json")
    weighted_components = set(
        weighted["split_contract"]["splits"]["development"]["passage_component_ids"]
    )
    prior_components = set(
        prior["split_contract"]["splits"]["development"]["passage_component_ids"]
    )
    assert weighted_components.isdisjoint(prior_components)
    assert weighted["split_contract"]["prior_multi_anchor_development"]["bundle_bytes_read"] is False
