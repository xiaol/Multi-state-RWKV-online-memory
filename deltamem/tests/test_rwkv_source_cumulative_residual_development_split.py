from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from experiments.rethinking_rwkv_ms_gemma import (
    materialize_natural_memory_native_rwkv_source_cumulative_residual_development as materializer,
)
from experiments.rethinking_rwkv_ms_gemma import (
    rwkv_source_cumulative_residual_development_split as development_split,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
ARTIFACT_ROOT = (
    PROJECT_ROOT / "experiments/rethinking_rwkv_ms_gemma/local_artifacts"
)
SOURCE_PATH = PROJECT_ROOT / materializer.parent_split.SOURCE_RELATIVE_PATH
TOKENIZER_PATH = Path("/root/X/.cache/hf/gemma-4-E4B-it-a4c2d58")
COMMITTED_ROOT = (
    ARTIFACT_ROOT
    / "natural_memory_native_rwkv_source_cumulative_residual_development_v1"
)


@pytest.fixture(scope="module")
def materialized_root(tmp_path_factory: pytest.TempPathFactory) -> Path:
    output_root = tmp_path_factory.mktemp("source-cumulative-development") / "bundle"
    materializer.materialize(
        source_path=SOURCE_PATH,
        tokenizer_path=TOKENIZER_PATH,
        parent_manifest_path=materializer.DEFAULT_PARENT_MANIFEST,
        protected_manifest_path=materializer.DEFAULT_PROTECTED_MANIFEST,
        output_root=output_root,
    )
    return output_root


def test_open_development_materialization_reproduces_pinned_bytes(
    materialized_root: Path,
) -> None:
    assert (materialized_root / "manifest.json").read_bytes() == (
        COMMITTED_ROOT / "manifest.json"
    ).read_bytes()
    assert (materialized_root / "development.jsonl").read_bytes() == (
        COMMITTED_ROOT / "development.jsonl"
    ).read_bytes()

    manifest = materializer.load_manifest(materialized_root / "manifest.json")
    assert materializer.sha256_bytes(
        (materialized_root / "manifest.json").read_bytes()
    ) == materializer.SEALED_MANIFEST_SHA256
    assert manifest["open_splits"] == ["development"]
    assert manifest["protected_splits_opened"] == []
    assert manifest["access"]["protected_reservation_bundle_bytes_read"] is False


def test_open_development_excludes_every_protected_component() -> None:
    development_manifest = materializer.load_manifest(
        COMMITTED_ROOT / "manifest.json"
    )
    protected_manifest = development_split.load_protected_manifest(
        materializer.DEFAULT_PROTECTED_MANIFEST
    )
    development_components = set(
        development_manifest["split_contract"]["splits"]["development"][
            "passage_component_ids"
        ]
    )
    protected_components = {
        component
        for name in ("mechanics", "causal")
        for component in protected_manifest["split_contract"]["splits"][name][
            "passage_component_ids"
        ]
    }

    assert len(development_components) == 64
    assert len(protected_components) == 64
    assert not development_components & protected_components
    assert development_manifest["split_contract"]["leakage_audit"] == {
        "source_rows": 1443,
        "passage_component_count": 708,
        "historical_excluded_components": 94,
        "parent_selected_components": 160,
        "fresh_protected_components": 64,
        "total_excluded_components": 318,
        "remaining_components_before_selection": 390,
        "selected_source_rows": 64,
        "selected_passage_components": 64,
        "protected_overlap_component_count": 0,
    }


def test_open_development_rows_have_exact_mapping_and_receipts() -> None:
    manifest = materializer.load_manifest(COMMITTED_ROOT / "manifest.json")
    rows = materializer.read_open_development(COMMITTED_ROOT, manifest)
    split_payload = manifest["split_contract"]["splits"]["development"]
    mapping = {int(source): int(donor) for source, donor in split_payload["mapping_pairs"]}

    assert len(rows) == 64
    assert [int(row["source_index"]) for row in rows] == split_payload[
        "source_indices"
    ]
    for row in rows:
        unsigned = dict(row)
        receipt = unsigned.pop("receipt")
        assert int(row["donor_source_index"]) == mapping[int(row["source_index"])]
        assert receipt == materializer._receipt(
            "canonical_bundle_row_without_receipt",
            unsigned,
        )


def test_open_development_rejects_re_receipted_component_tampering() -> None:
    manifest = json.loads((COMMITTED_ROOT / "manifest.json").read_bytes())
    contract = copy.deepcopy(manifest["split_contract"])
    development = contract["splits"]["development"]
    development["passage_component_ids"][0] = "0" * 64
    development["passage_component_ids_sha256"] = materializer.canonical_sha256(
        development["passage_component_ids"]
    )
    unsigned = dict(contract)
    unsigned.pop("receipt")
    contract["receipt"] = development_split._receipt(
        "canonical_split_without_receipt",
        unsigned,
    )

    with pytest.raises(ValueError, match="split binding differs"):
        development_split.validate_split_contract(contract)


def test_open_development_rejects_path_redirection_before_bundle_access(
    tmp_path: Path,
) -> None:
    (tmp_path / "manifest.json").write_bytes(
        (COMMITTED_ROOT / "manifest.json").read_bytes()
    )
    manifest = materializer.load_manifest(tmp_path / "manifest.json")
    redirected = copy.deepcopy(manifest)
    redirected["file_inventory"]["bundles"]["development"]["path"] = (
        "redirected.jsonl"
    )

    with pytest.raises(ValueError, match="manifest differs from sealed bytes"):
        materializer.read_open_development(tmp_path, redirected)
    assert not (tmp_path / "redirected.jsonl").exists()


def test_open_development_materializer_rejects_overwrite(
    materialized_root: Path,
) -> None:
    with pytest.raises(ValueError, match="must be fresh"):
        materializer.materialize(
            source_path=SOURCE_PATH,
            tokenizer_path=TOKENIZER_PATH,
            parent_manifest_path=materializer.DEFAULT_PARENT_MANIFEST,
            protected_manifest_path=materializer.DEFAULT_PROTECTED_MANIFEST,
            output_root=materialized_root,
        )
