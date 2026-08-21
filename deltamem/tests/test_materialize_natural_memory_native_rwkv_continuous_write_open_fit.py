from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from experiments.rethinking_rwkv_ms_gemma import (
    materialize_natural_memory_native_rwkv_continuous_write_open_fit as materializer,
)
from experiments.rethinking_rwkv_ms_gemma import rwkv_continuous_write_fit_split as split


PROJECT_ROOT = Path(__file__).resolve().parents[2]
PRIOR_MANIFEST = (
    PROJECT_ROOT
    / "experiments/rethinking_rwkv_ms_gemma/local_artifacts/"
    "natural_memory_native_rwkv_bidirectional_sign_open_fit_v1/manifest.json"
)


def _sha256(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _source_rows() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for source_index in range(split.SOURCE_ROWS):
        shared_reserved = "0123456789abcdefghijklmnopqrstuv"
        passage = (
            "reserved " + shared_reserved + " source"
            if source_index == 7
            else "closure " + shared_reserved + " source"
            if source_index == 1420
            else "passage:" + _sha256(f"passage:{source_index}")
        )
        messages = [
            {"role": "system", "content": "detect scene boundaries"},
            {"role": "user", "content": f"[P1] {passage}"},
            {
                "role": "assistant",
                "content": json.dumps({"boundaries": [source_index % 29]}),
            },
        ]
        raw_line = json.dumps(
            {"messages": messages},
            ensure_ascii=False,
            separators=(",", ":"),
        )
        rows.append(
            {
                "source_index": source_index,
                "raw_line": raw_line,
                "row_sha256": _sha256(raw_line),
                "messages": messages,
                "gold": (source_index % 29,),
                "write_tokens": 128 + source_index % 41,
            }
        )
    return rows


def _prepared() -> tuple[
    list[dict[str, object]],
    list[dict[str, object]],
    dict[str, object],
]:
    source_rows = _source_rows()
    metadata = materializer._metadata_rows(source_rows)
    prior = split.load_prior_reservation(PRIOR_MANIFEST)
    contract = split.build_split(metadata, prior)
    return source_rows, metadata, contract


def _materialize(tmp_path: Path) -> Path:
    source_rows, metadata, contract = _prepared()
    root = tmp_path / "continuous-write-open-fit"
    materializer.materialize_prepared(
        source_rows=source_rows,
        metadata_rows=metadata,
        split_contract=contract,
        tokenizer_binding={"fixture": "metadata-only"},
        source_path=PROJECT_ROOT / split.SOURCE_RELATIVE_PATH,
        output_root=root,
    )
    return root


def test_materialization_pins_inventory_qualified_ids_and_all_split_hashes(
    tmp_path: Path,
) -> None:
    root = _materialize(tmp_path)
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))

    assert {path.name for path in root.iterdir()} == set(materializer.FILE_NAMES)
    assert manifest["source"]["namespace"] == split.SOURCE_NAMESPACE
    assert manifest["source"]["sha256"] == split.SOURCE_SHA256
    assert manifest["split_contract"]["prior_reservation"][
        "manifest_reserved_rows"
    ] == 98
    assert manifest["file_inventory"]["exact_names"] == list(
        materializer.FILE_NAMES
    )
    assert manifest["first_gate_access"]["inventory_only_files"] == [
        "mechanics.jsonl",
        "causal.jsonl"
    ]
    for name, expected_rows in materializer.BUNDLE_ROWS.items():
        binding = manifest["file_inventory"]["bundles"][name]
        assert binding["rows"] == expected_rows
        assert len(binding["sha256"]) == 64
        assert len(binding["qualified_source_ids_sha256"]) == 64
        assert len(binding["qualified_mapping_pairs_sha256"]) == 64
        assert len(binding["row_sha256s_sha256"]) == 64


def test_default_validation_never_reads_mechanics_or_causal_bundle_bytes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = _materialize(tmp_path)
    original = Path.read_bytes
    byte_reads: list[str] = []

    def tracked(path: Path) -> bytes:
        byte_reads.append(path.name)
        if path.name in {"mechanics.jsonl", "causal.jsonl"}:
            raise AssertionError("sealed bundle bytes were opened")
        return original(path)

    monkeypatch.setattr(Path, "read_bytes", tracked)
    validated = materializer.validate_materialization(root)

    assert byte_reads == ["manifest.json", "fit.jsonl", "retrieval.jsonl"]
    assert set(validated["groups"]) == {"fit", "retrieval"}
    assert len(validated["groups"]["fit"]) == 64
    assert len(validated["groups"]["retrieval"]) == 32
    assert validated["file_access_audit"] == {
        "byte_read_files": ["manifest.json", "fit.jsonl", "retrieval.jsonl"],
        "inventory_only_files": ["mechanics.jsonl", "causal.jsonl"],
        "mechanics_bytes_read": False,
        "causal_bytes_read": False,
        "exact_inventory_validated": True,
    }


@pytest.mark.parametrize("name", ("mechanics", "causal"))
def test_sealed_bundles_require_independent_explicit_authorization(
    tmp_path: Path,
    name: str,
) -> None:
    root = _materialize(tmp_path)

    with pytest.raises(PermissionError, match="separate authorization"):
        materializer.validate_materialization(root, bundles=(name,))

    validated = materializer.validate_materialization(
        root,
        bundles=(name,),
        allow_mechanics=name == "mechanics",
        allow_causal=name == "causal",
    )
    assert len(validated["groups"][name]) == 32
    assert validated["file_access_audit"][f"{name}_bytes_read"] is True


def test_validator_rejects_any_extra_inventory_file_before_bundle_reads(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = _materialize(tmp_path)
    (root / "unexpected.txt").write_text("unexpected", encoding="utf-8")
    original = Path.read_bytes
    byte_reads: list[str] = []

    def tracked(path: Path) -> bytes:
        byte_reads.append(path.name)
        return original(path)

    monkeypatch.setattr(Path, "read_bytes", tracked)
    with pytest.raises(ValueError, match="file inventory differs"):
        materializer.validate_materialization(root)

    assert byte_reads == ["manifest.json"]
