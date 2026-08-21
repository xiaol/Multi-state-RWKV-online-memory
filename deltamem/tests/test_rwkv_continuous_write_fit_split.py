from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import pytest

from experiments.rethinking_rwkv_ms_gemma import rwkv_continuous_write_fit_split as split


PROJECT_ROOT = Path(__file__).resolve().parents[2]
PRIOR_MANIFEST = (
    PROJECT_ROOT
    / "experiments/rethinking_rwkv_ms_gemma/local_artifacts/"
    "natural_memory_native_rwkv_bidirectional_sign_open_fit_v1/manifest.json"
)


def _sha256(value: str) -> str:
    return hashlib.sha256(value.encode("ascii")).hexdigest()


def _metadata_rows() -> list[dict[str, object]]:
    return [
        {
            "source_index": source_index,
            "row_sha256": _sha256(f"row:{source_index}"),
            "gold_sha256": _sha256(f"gold:{source_index % 29}"),
            "write_tokens": 128 + source_index % 41,
        }
        for source_index in range(split.SOURCE_ROWS)
    ]


def test_prior_loader_reads_only_pinned_manifest_and_closes_all_components(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original = Path.read_bytes
    paths: list[Path] = []

    def tracked(path: Path) -> bytes:
        paths.append(path)
        if path.suffix == ".jsonl":
            raise AssertionError("sealed bundle bytes were read")
        return original(path)

    monkeypatch.setattr(Path, "read_bytes", tracked)
    prior = split.load_prior_reservation(PRIOR_MANIFEST)

    assert paths == [PRIOR_MANIFEST]
    assert len(prior.source_indices) == split.PRIOR_RESERVED_ROWS == 98
    assert prior.component_closure == prior.source_indices
    assert len(prior.mapping_pairs) == 98
    assert all(
        qualified.startswith(
            f"{split.SOURCE_NAMESPACE}|sha256:{split.SOURCE_SHA256}|source_index:"
        )
        for qualified in prior.qualified_source_ids
    )
    assert prior.payload()["bundle_bytes_read"] is False


def test_split_is_reproducible_component_disjoint_and_excludes_all_prior_rows() -> None:
    prior = split.load_prior_reservation(PRIOR_MANIFEST)
    first = split.build_split(_metadata_rows(), prior)
    second = split.build_split(_metadata_rows(), prior)

    assert first == second
    assert first["source"] == split.SOURCE_BINDING.payload()
    assert first["donor_component_disjoint"] is True
    assert first["plmsc_namespace_consulted"] is False
    assert first["sealed_bundle_bytes_read"] is False
    assert first["protected_splits_opened"] == []
    assert first["prior_reservation"]["reserved_rows"] == 98
    assert first["receipt"]["payload_sha256"] == split.canonical_sha256(
        {key: value for key, value in first.items() if key != "receipt"}
    )

    expected_rows = {"fit": 64, "mechanics": 32, "causal": 32}
    reserved = set(prior.component_closure)
    selected_sets: list[set[int]] = []
    for name in split.SPLIT_NAMES:
        payload = first["splits"][name]
        sources = set(payload["source_indices"])
        selected_sets.append(sources)
        assert len(sources) == expected_rows[name]
        assert not sources & reserved
        mapping = dict(payload["mapping_pairs"])
        assert set(mapping) == sources
        assert all(mapping[mapping[source]] == source for source in sources)
        assert payload["qualified_source_ids_sha256"] == split.canonical_sha256(
            payload["qualified_source_ids"]
        )
        assert payload["qualified_mapping_pairs_sha256"] == split.canonical_sha256(
            payload["qualified_mapping_pairs"]
        )
    assert all(
        not left & right
        for index, left in enumerate(selected_sets)
        for right in selected_sets[index + 1 :]
    )


def test_dataset_qualification_prevents_numeric_cross_dataset_collisions() -> None:
    publisher = split.SOURCE_BINDING.qualified_id(7)
    plmsc = split.DatasetBinding(
        namespace="novel-agent-sft-dataset:publisher-train-derived-development",
        path="v4-scene-boundary-detection/train_derived_development.jsonl",
        sha256="b383625cee07e6a7565142e38bb0b0a4d4a2468b2c91171570115b7b311e1e68",
        rows=360,
    ).qualified_id(7)

    assert publisher != plmsc
    assert publisher.endswith("source_index:7")
    assert plmsc.endswith("source_index:7")


def test_prior_manifest_rejects_rebound_dataset_even_with_same_numeric_indices() -> None:
    manifest_bytes = PRIOR_MANIFEST.read_bytes()
    manifest = json.loads(manifest_bytes)
    rebound = copy.deepcopy(manifest)
    rebound["source"]["namespace"] = "plmsc-development"

    with pytest.raises(ValueError, match="source binding differs"):
        split.validate_prior_manifest(
            rebound,
            manifest_sha256=split.PRIOR_MANIFEST_SHA256,
        )


def test_split_builder_rejects_manually_forged_prior_reservation() -> None:
    prior = split.load_prior_reservation(PRIOR_MANIFEST)
    forged = split.PriorReservation(
        source_indices=prior.source_indices,
        mapping_pairs=prior.mapping_pairs,
        component_closure=prior.component_closure[1:],
        qualified_source_ids=prior.qualified_source_ids[1:],
        manifest_sha256=prior.manifest_sha256,
    )

    with pytest.raises(ValueError, match="reservation differs from its pins"):
        split.build_split(_metadata_rows(), forged)
