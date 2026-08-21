from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import pytest

from experiments.rethinking_rwkv_ms_gemma import prepare_natural_memory_gate
from experiments.rethinking_rwkv_ms_gemma import rwkv_continuous_write_fit_split as split


PROJECT_ROOT = Path(__file__).resolve().parents[2]
PRIOR_MANIFEST = (
    PROJECT_ROOT
    / "experiments/rethinking_rwkv_ms_gemma/local_artifacts/"
    "natural_memory_native_rwkv_bidirectional_sign_open_fit_v1/manifest.json"
)


def _sha256(value: str) -> str:
    return hashlib.sha256(value.encode("ascii")).hexdigest()


def _passage(source_index: int) -> str:
    shared_reserved_shingle = "0123456789abcdefghijklmnopqrstuv"
    shared_open_shingle = "zyxwvutsrqponmlkjihgfedcba987654"
    if source_index == 7:
        return "reserved prefix " + shared_reserved_shingle + " reserved suffix"
    if source_index == 1420:
        return "closure prefix " + shared_reserved_shingle + " closure suffix"
    if source_index == 1400:
        return "open alpha " + shared_open_shingle + " alpha suffix"
    if source_index == 1401:
        return "open beta " + shared_open_shingle + " beta suffix"
    return "passage:" + _sha256(f"passage:{source_index}")


def _metadata_rows() -> list[dict[str, object]]:
    return [
        {
            "source_index": source_index,
            "row_sha256": _sha256(f"row:{source_index}"),
            "gold_sha256": _sha256(f"gold:{source_index % 29}"),
            "write_tokens": 128 + source_index % 41,
            "passage_signature_sha256s": list(
                split.passage_signature_sha256s(_passage(source_index))
            ),
        }
        for source_index in range(split.SOURCE_ROWS)
    ]


def test_prior_loader_reads_only_pinned_manifest_and_closes_all_donor_components(
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
    assert prior.donor_component_closure == prior.source_indices
    assert len(prior.mapping_pairs) == 98


def test_normalized_32_character_shingles_form_true_connected_components() -> None:
    rows = split._validated_metadata_rows(_metadata_rows())
    source_to_component, component_rows = split._passage_components(rows)

    assert source_to_component[7] == source_to_component[1420]
    assert source_to_component[1400] == source_to_component[1401]
    assert set(component_rows[source_to_component[7]]) == {7, 1420}
    assert set(component_rows[source_to_component[1400]]) == {1400, 1401}
    assert split.normalize_passage("Ａ B\nC") == "abc"


def test_passage_signatures_exactly_reuse_scene_segment_leakage_semantics() -> None:
    first = "[P1] shared short segment\n[P2] " + "alpha" * 12
    second = "[P7] shared short segment\n[P8] " + "beta" * 12
    expected = prepare_natural_memory_gate._segments_for_component(
        prepare_natural_memory_gate._extract_segments("scene", first)
    )

    assert set(split.passage_signature_sha256s(first)) == expected
    assert set(split.passage_signature_sha256s(first)) & set(
        split.passage_signature_sha256s(second)
    )
    assert not any("[P1]" in signature for signature in expected)


def test_split_is_reproducible_component_disjoint_and_excludes_prior_closure() -> None:
    prior = split.load_prior_reservation(PRIOR_MANIFEST)
    first = split.build_split(_metadata_rows(), prior)
    second = split.build_split(_metadata_rows(), prior)

    assert first == second
    assert first["source"] == split.SOURCE_BINDING.payload()
    assert first["donor_component_disjoint"] is True
    assert first["passage_component_disjoint"] is True
    assert first["plmsc_namespace_consulted"] is False
    assert first["sealed_bundle_bytes_read"] is False
    assert first["protected_splits_opened"] == []
    assert first["capture_authorization"] == {
        "capture_splits": ["fit", "retrieval"],
        "sealed_inventory_only_splits": ["mechanics", "causal"],
        "captured_rows": 96,
        "sealed_rows": 64,
    }
    assert first["receipt"]["payload_sha256"] == split.canonical_sha256(
        {key: value for key, value in first.items() if key != "receipt"}
    )

    prior_payload = first["prior_reservation"]
    assert prior_payload["manifest_reserved_rows"] == 98
    assert 1420 in prior_payload["excluded_passage_component_sources"]
    excluded = set(prior_payload["excluded_passage_component_sources"])
    expected_rows = {"fit": 64, "retrieval": 32, "mechanics": 32, "causal": 32}
    selected_component_sets: list[set[str]] = []
    all_selected: set[int] = set()
    for name in split.SPLIT_NAMES:
        payload = first["splits"][name]
        sources = set(payload["source_indices"])
        components = set(payload["passage_component_ids"])
        selected_component_sets.append(components)
        all_selected.update(sources)
        assert len(sources) == expected_rows[name]
        assert len(components) == expected_rows[name]
        assert not sources & excluded
        mapping = dict(payload["mapping_pairs"])
        assert set(mapping) == sources
        assert all(mapping[mapping[source]] == source for source in sources)
        assert all(left != right for left, right in payload["donor_component_pairs"])
        assert payload["qualified_source_ids_sha256"] == split.canonical_sha256(
            payload["qualified_source_ids"]
        )
        assert payload["passage_component_ids_sha256"] == split.canonical_sha256(
            payload["passage_component_ids"]
        )
        assert all("|row_sha256:" in identity for identity in payload["qualified_source_ids"])
    assert len(all_selected) == 160
    assert all(
        not left & right
        for index, left in enumerate(selected_component_sets)
        for right in selected_component_sets[index + 1 :]
    )
    assert first["leakage_audit"][
        "cross_split_normalized_32_character_shingle_overlap"
    ] == 0


def test_dataset_qualification_prevents_numeric_cross_dataset_collisions() -> None:
    row_sha256 = _sha256("row:7")
    publisher = split.SOURCE_BINDING.qualified_id(7, row_sha256)
    other = split.DatasetBinding(
        namespace="novel-agent-sft-dataset:publisher-train-derived-development",
        path="v4-scene-boundary-detection/train_derived_development.jsonl",
        sha256="b383625cee07e6a7565142e38bb0b0a4d4a2468b2c91171570115b7b311e1e68",
        rows=360,
    ).qualified_id(7, row_sha256)

    assert publisher != other
    assert "dataset_sha256:" + split.SOURCE_SHA256 in publisher
    assert "source_index:7" in publisher
    assert "row_sha256:" + row_sha256 in publisher


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
        donor_component_closure=prior.donor_component_closure[1:],
        manifest_sha256=prior.manifest_sha256,
    )

    with pytest.raises(ValueError, match="reservation differs from its pins"):
        split.build_split(_metadata_rows(), forged)
