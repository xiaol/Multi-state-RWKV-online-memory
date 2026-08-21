from __future__ import annotations

import json
from pathlib import Path

import pytest

from experiments.rethinking_rwkv_ms_gemma import (
    materialize_natural_memory_native_rwkv_bidirectional_sign_open_fit as materializer,
)


TOKENIZER_PATH = Path("/root/X/.cache/hf/gemma-4-E4B-it-a4c2d58")
SOURCE_PATH = materializer.PROJECT_ROOT / materializer.SOURCE_RELATIVE_PATH


@pytest.fixture(scope="module")
def materialized_root(tmp_path_factory: pytest.TempPathFactory) -> Path:
    output_root = tmp_path_factory.mktemp("bidirectional-sign-open-fit") / "bundle"
    materializer.materialize(
        source_path=SOURCE_PATH,
        tokenizer_path=TOKENIZER_PATH,
        output_root=output_root,
    )
    return output_root


def test_open_fit_materialization_reproduces_all_signed_locks(
    materialized_root: Path,
) -> None:
    validated = materializer.validate_materialization(materialized_root)
    manifest = validated["manifest"]

    assert manifest["schema"] == materializer.MANIFEST_SCHEMA
    assert manifest["protected_splits_opened"] == []
    assert manifest["source"]["sha256"] == materializer.SOURCE_SHA256
    assert manifest["source"]["rows"] == materializer.SOURCE_ROWS
    assert manifest["source_indices_sha256"] == materializer.EXPECTED_SOURCE_SET_SHA256
    assert manifest["mapping_sha256"] == materializer.EXPECTED_GLOBAL_MAPPING_SHA256
    assert (
        manifest["ordered_components_sha256"]
        == materializer.EXPECTED_ORDERED_COMPONENTS_SHA256
    )
    assert manifest["donor_token_deltas"] == {
        "maximum": 1,
        "mean": pytest.approx(0.12244897959183673),
        "total": 12,
    }
    assert {name: len(rows) for name, rows in validated["groups"].items()} == {
        "development": 64,
        "mechanics": 17,
        "causal": 17,
    }
    assert len(validated["mapping"]) == 98
    assert validated["mapping"][96] == 414
    assert validated["mapping"][414] == 729
    assert validated["mapping"][729] == 96
    assert validated["mapping"][36] == 649
    assert validated["mapping"][649] == 145
    assert validated["mapping"][145] == 36


def test_open_fit_rows_have_exact_shape_and_canonical_receipts(
    materialized_root: Path,
) -> None:
    validated = materializer.validate_materialization(materialized_root)

    for rows in validated["groups"].values():
        for row in rows:
            assert set(row) == {
                "schema",
                "source_index",
                "row_sha256",
                "raw_line",
                "receipt",
            }
            assert row["schema"] == materializer.ROW_SCHEMA
            assert json.loads(row["raw_line"])["messages"][-1]["role"] == "assistant"
            unsigned = dict(row)
            receipt = unsigned.pop("receipt")
            assert receipt == {
                "algorithm": "sha256",
                "payload_scope": "canonical_bundle_row_without_receipt",
                "payload_sha256": materializer.canonical_sha256(unsigned),
            }


def test_development_only_validation_never_opens_protected_bundles(
    materialized_root: Path,
    tmp_path: Path,
) -> None:
    root = tmp_path / "development-only"
    root.mkdir()
    (root / "manifest.json").write_bytes((materialized_root / "manifest.json").read_bytes())
    (root / "development.jsonl").write_bytes(
        (materialized_root / "development.jsonl").read_bytes()
    )

    validated = materializer.validate_materialization(
        root,
        bundles=("development",),
    )

    assert set(validated["groups"]) == {"development"}
    assert len(validated["mapping"]) == 64
    assert not (root / "mechanics.jsonl").exists()
    assert not (root / "causal.jsonl").exists()


def test_open_fit_materialization_rejects_every_overwrite(
    materialized_root: Path,
) -> None:
    with pytest.raises(ValueError, match="must be fresh"):
        materializer.materialize(
            source_path=SOURCE_PATH,
            tokenizer_path=TOKENIZER_PATH,
            output_root=materialized_root,
        )
