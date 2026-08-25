"""Fresh source-and-donor-disjoint split for the narrative native source."""

from __future__ import annotations

import hashlib
import json
from typing import Any, Mapping, Sequence


SCHEMA = "rwkv_ms_narrative_identity_split.v1"
SOURCE_NAMESPACE = "novel-agent-sft-dataset:publisher-train-derived-narrative-v3.2"
SOURCE_RELATIVE_PATH = (
    "experiments/rethinking_rwkv_ms_gemma/local_artifacts/"
    "natural_memory_native_development_v1/v3.2-narrative-type-classification/"
    "train_derived_fit.jsonl"
)
SOURCE_SHA256 = "133aba7801ddb31aaf4ecb6b7d20b32435366434647a5535265053855f69da06"
SOURCE_ROWS = 459
SELECTION_SALT = "rwkv-address-keyed-feedback-narrative-v1:"
PAIR_COUNT = 32
TRAIN_PAIRS = 16
HELDOUT_PAIRS = 16
DEVELOPMENT_ROWS = PAIR_COUNT * 2


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":"))


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def qualified_id(source_index: int, row_sha256: str) -> str:
    return (
        f"{SOURCE_NAMESPACE}|dataset_sha256:{SOURCE_SHA256}|"
        f"source_index:{int(source_index)}|row_sha256:{row_sha256}"
    )


def _row_rank(row: Mapping[str, Any]) -> tuple[str, int]:
    digest = hashlib.sha256(
        (SELECTION_SALT + str(row["row_sha256"])).encode("ascii")
    ).hexdigest()
    return digest, int(row["source_index"])


def _pair_rank(pair: tuple[int, int], rows_by_index: Mapping[int, Mapping[str, Any]]) -> tuple[str, tuple[int, int]]:
    qualified = [
        qualified_id(index, str(rows_by_index[index]["row_sha256"]))
        for index in pair
    ]
    digest = hashlib.sha256(
        (SELECTION_SALT + canonical_json(qualified)).encode("ascii")
    ).hexdigest()
    return digest, pair


def _components(row: Mapping[str, Any]) -> frozenset[str]:
    values = row.get("passage_component_ids")
    if not isinstance(values, list) or not values:
        raise ValueError("Narrative row passage components are missing")
    components = frozenset(str(value) for value in values)
    if len(components) != len(values):
        raise ValueError("Narrative row passage components are duplicated")
    return components


def build_split(metadata_rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    rows = tuple(metadata_rows)
    if len(rows) != SOURCE_ROWS:
        raise ValueError(f"Narrative source row count differs: {len(rows)}")
    rows_by_index = {int(row["source_index"]): row for row in rows}
    if sorted(rows_by_index) != list(range(SOURCE_ROWS)):
        raise ValueError("Narrative source indices are not contiguous")
    row_components = {index: _components(row) for index, row in rows_by_index.items()}
    ordered = sorted(rows, key=_row_rank)
    used_components: set[str] = set()
    pairs: list[tuple[int, int]] = []
    for source_row in ordered:
        source = int(source_row["source_index"])
        source_components = row_components[source]
        if source_components & used_components:
            continue
        candidates = []
        for donor_row in ordered:
            donor = int(donor_row["source_index"])
            donor_components = row_components[donor]
            if donor == source or donor_components & used_components:
                continue
            if source_components & donor_components:
                continue
            if str(donor_row["gold_sha256"]) == str(source_row["gold_sha256"]):
                continue
            candidates.append(donor_row)
        if not candidates:
            continue
        donor_row = min(
            candidates,
            key=lambda row: (
                abs(int(row["write_tokens"]) - int(source_row["write_tokens"])),
                _row_rank(row),
            ),
        )
        donor = int(donor_row["source_index"])
        pairs.append(tuple(sorted((source, donor))))
        used_components.update(source_components)
        used_components.update(row_components[donor])
        if len(pairs) == PAIR_COUNT:
            break
    if len(pairs) != PAIR_COUNT:
        raise ValueError(f"Narrative pair capacity differs: {len(pairs)}")
    pairs = sorted(set(pairs), key=lambda pair: _pair_rank(pair, rows_by_index))
    if len(pairs) != PAIR_COUNT:
        raise ValueError("Narrative pair reservation contains duplicates")
    source_indices = sorted(index for pair in pairs for index in pair)
    mapping_pairs = [[source, donor] for pair in pairs for source, donor in (pair, pair[::-1])]
    selected_components = sorted(
        {component for index in source_indices for component in row_components[index]}
    )
    payload: dict[str, Any] = {
        "schema": SCHEMA,
        "source": {
            "namespace": SOURCE_NAMESPACE,
            "path": SOURCE_RELATIVE_PATH,
            "sha256": SOURCE_SHA256,
            "rows": SOURCE_ROWS,
        },
        "selection": {
            "salt": SELECTION_SALT,
            "pair_count": PAIR_COUNT,
            "train_pairs": TRAIN_PAIRS,
            "heldout_pairs": HELDOUT_PAIRS,
            "split_unit": "reciprocal donor pair with disjoint passage components",
        },
        "splits": {
            "development": {
                "pairs": [list(pair) for pair in pairs],
                "mapping_pairs": mapping_pairs,
                "source_indices": source_indices,
                "passage_component_ids": selected_components,
                "pair_count": PAIR_COUNT,
                "rows": DEVELOPMENT_ROWS,
                "source_indices_sha256": canonical_sha256(source_indices),
                "mapping_pairs_sha256": canonical_sha256(mapping_pairs),
                "passage_component_ids_sha256": canonical_sha256(selected_components),
            }
        },
        "leakage_audit": {
            "source_rows": SOURCE_ROWS,
            "selected_source_rows": DEVELOPMENT_ROWS,
            "selected_pairs": PAIR_COUNT,
            "selected_components": len(selected_components),
            "source_component_disjoint": True,
            "donor_component_disjoint": True,
            "protected_splits_opened": [],
        },
        "open_splits": ["development"],
        "protected_splits_opened": [],
    }
    payload["receipt"] = {
        "algorithm": "sha256",
        "payload_scope": "canonical_split_without_receipt",
        "payload_sha256": canonical_sha256(payload),
    }
    return payload


def validate_split_contract(value: Mapping[str, Any]) -> None:
    unsigned = dict(value)
    receipt = unsigned.pop("receipt", None)
    expected = {
        "algorithm": "sha256",
        "payload_scope": "canonical_split_without_receipt",
        "payload_sha256": canonical_sha256(unsigned),
    }
    if dict(receipt or {}) != expected:
        raise ValueError("Narrative split receipt differs")
    if (
        value.get("schema") != SCHEMA
        or value.get("open_splits") != ["development"]
        or value.get("protected_splits_opened") != []
    ):
        raise ValueError("Narrative split access differs")
    development = value.get("splits", {}).get("development")
    if not isinstance(development, Mapping) or development.get("rows") != DEVELOPMENT_ROWS:
        raise ValueError("Narrative split row count differs")
    if development.get("source_indices_sha256") != canonical_sha256(development.get("source_indices")):
        raise ValueError("Narrative source digest differs")
    if development.get("mapping_pairs_sha256") != canonical_sha256(development.get("mapping_pairs")):
        raise ValueError("Narrative mapping digest differs")
    if development.get("passage_component_ids_sha256") != canonical_sha256(
        development.get("passage_component_ids")
    ):
        raise ValueError("Narrative component digest differs")
