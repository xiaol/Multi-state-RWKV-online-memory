from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from experiments.rethinking_rwkv_ms_gemma import (
    materialize_natural_memory_native_rwkv_source_cumulative_residual as materializer,
)
from experiments.rethinking_rwkv_ms_gemma import (
    rwkv_source_cumulative_residual_split as split,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
ARTIFACT_ROOT = (
    PROJECT_ROOT / "experiments/rethinking_rwkv_ms_gemma/local_artifacts"
)
PARENT_MANIFEST = (
    ARTIFACT_ROOT
    / "natural_memory_native_rwkv_continuous_write_open_fit_v1/manifest.json"
)
CONSUMED_RESULT = (
    ARTIFACT_ROOT
    / "natural_memory_native_rwkv_cumulative_virtual_kv_mechanics_v1/result.json"
)
FRESH_MANIFEST = (
    ARTIFACT_ROOT
    / "natural_memory_native_rwkv_source_cumulative_residual_v1/manifest.json"
)


def test_parent_and_consumed_result_loaders_bind_exact_pinned_receipts() -> None:
    parent = split.load_parent_reservation(PARENT_MANIFEST)
    consumed = split.load_consumed_mechanics(
        CONSUMED_RESULT,
        parent_reservation=parent,
    )

    assert parent.manifest_sha256 == split.PARENT_MANIFEST_SHA256
    assert parent.manifest_receipt == split.PARENT_MANIFEST_RECEIPT
    assert parent.split_receipt == split.PARENT_SPLIT_RECEIPT
    assert len(parent.excluded_component_sources) == 282
    assert {name: len(rows) for name, rows in parent.split_sources.items()} == {
        "fit": 64,
        "retrieval": 32,
        "mechanics": 32,
        "causal": 32,
    }
    assert consumed.result_sha256 == split.CONSUMED_RESULT_SHA256
    assert consumed.result_receipt == split.CONSUMED_RESULT_RECEIPT
    assert consumed.source_indices == parent.split_sources["mechanics"]


def test_fresh_manifest_locks_exact_pairs_and_component_exclusions() -> None:
    manifest = materializer.load_manifest_only(FRESH_MANIFEST)
    contract = manifest["split_contract"]

    assert contract["leakage_audit"] == {
        "source_rows": 1443,
        "passage_component_count": 708,
        "historical_excluded_components": 94,
        "parent_selected_components": 160,
        "total_excluded_components": 254,
        "remaining_components_before_selection": 454,
        "selected_source_rows": 64,
        "selected_passage_components": 64,
        "cross_split_passage_component_count": 0,
    }
    for name in split.SPLIT_NAMES:
        payload = contract["splits"][name]
        assert tuple(tuple(pair) for pair in payload["pairs"]) == split.EXPECTED_PAIRS[name]
        assert (
            payload["source_indices_sha256"]
            == split.EXPECTED_SOURCE_INDICES_SHA256[name]
        )
        assert (
            payload["passage_component_ids_sha256"]
            == split.EXPECTED_COMPONENT_IDS_SHA256[name]
        )
    mechanics_components = set(
        contract["splits"]["mechanics"]["passage_component_ids"]
    )
    causal_components = set(contract["splits"]["causal"]["passage_component_ids"])
    assert len(mechanics_components) == len(causal_components) == 32
    assert not mechanics_components & causal_components


def test_manifest_only_validation_never_requires_bundle_files(tmp_path: Path) -> None:
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_bytes(FRESH_MANIFEST.read_bytes())

    manifest = materializer.load_manifest_only(manifest_path)

    assert manifest["protected_splits_opened"] == []
    assert manifest["first_gate_access"]["default_open_splits"] == []
    assert not (tmp_path / "mechanics.jsonl").exists()
    assert not (tmp_path / "causal.jsonl").exists()


@pytest.mark.parametrize(
    ("name", "kwargs", "message"),
    (
        ("mechanics", {}, "signed authorization"),
        ("causal", {}, "mechanics pass"),
        ("causal", {"allow_mechanics": True}, "mechanics pass"),
    ),
)
def test_protected_bundle_read_fails_before_filesystem_access(
    tmp_path: Path,
    name: str,
    kwargs: dict[str, bool],
    message: str,
) -> None:
    manifest = materializer.load_manifest_only(FRESH_MANIFEST)

    with pytest.raises(PermissionError, match=message):
        materializer.read_authorized_bundle(tmp_path, manifest, name, **kwargs)


def test_parent_manifest_and_consumed_result_tampering_is_rejected() -> None:
    parent_payload = PARENT_MANIFEST.read_bytes()
    parent_manifest = json.loads(parent_payload)
    parent_manifest["split_contract"]["splits"]["causal"]["source_indices"][0] += 1
    with pytest.raises(ValueError, match="receipt differs"):
        split.validate_parent_manifest(
            parent_manifest,
            manifest_sha256=split.PARENT_MANIFEST_SHA256,
        )

    parent = split.load_parent_reservation(PARENT_MANIFEST)
    result = json.loads(CONSUMED_RESULT.read_bytes())
    result["causal_rows_opened"] = 1
    with pytest.raises(ValueError, match="outcome differs"):
        split.validate_consumed_result(
            result,
            result_sha256=split.CONSUMED_RESULT_SHA256,
            parent_reservation=parent,
        )


def test_fresh_manifest_receipt_tampering_is_rejected(tmp_path: Path) -> None:
    manifest = json.loads(FRESH_MANIFEST.read_bytes())
    tampered = copy.deepcopy(manifest)
    tampered["split_contract"]["leakage_audit"]["total_excluded_components"] = 253
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(tampered))

    with pytest.raises(ValueError, match="receipt differs"):
        materializer.load_manifest_only(path)
