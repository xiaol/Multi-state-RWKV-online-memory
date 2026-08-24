"""Open component-disjoint development split for cumulative RWKV residuals."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Mapping, Sequence

from experiments.rethinking_rwkv_ms_gemma import rwkv_continuous_write_fit_split as parent
from experiments.rethinking_rwkv_ms_gemma import rwkv_source_cumulative_residual_split as protected


SCHEMA = "rwkv_ms_source_cumulative_residual_development_split.v1"
SELECTION_SALT = "rwkv-source-cumulative-residual-open-development-v1:"
PAIR_COUNT = 32
DEVELOPMENT_ROWS = 64

PROTECTED_MANIFEST_SCHEMA = (
    "rwkv_ms_natural_memory_native_rwkv_source_cumulative_residual.v1"
)
PROTECTED_MANIFEST_SHA256 = (
    "5251cc6f4254718620bd6e1328ac41c6fcb9bf837f836d623f874eedf53e9515"
)
PROTECTED_MANIFEST_RECEIPT = (
    "0ef2fb6de7e696dac9881ee223988d3f0e6df531b4957d7830c88599fe60457b"
)
PROTECTED_SPLIT_RECEIPT = (
    "6b22c808c6dc0cf722b74cf45981b3fad93a0a3ccdc3a5b023989487b1d637c6"
)
PROTECTED_COMPONENTS = 64

EXPECTED_PAIRS = (
    (303, 692),
    (808, 1382),
    (157, 844),
    (413, 669),
    (648, 880),
    (630, 1360),
    (547, 616),
    (676, 1081),
    (118, 1038),
    (326, 1066),
    (307, 475),
    (632, 863),
    (281, 1171),
    (375, 782),
    (297, 1061),
    (397, 741),
    (770, 1354),
    (279, 972),
    (376, 565),
    (134, 222),
    (656, 1226),
    (1031, 1296),
    (1229, 1339),
    (796, 828),
    (208, 702),
    (169, 619),
    (388, 1401),
    (56, 653),
    (173, 1159),
    (581, 977),
    (651, 1406),
    (212, 951),
)
EXPECTED_DIGESTS = {
    "source_indices_sha256": (
        "1c084a15ac1fb2c82cd4a720d6bfc31332e9938eb955609045cd0bf97faf1946"
    ),
    "qualified_source_ids_sha256": (
        "31c13d814b4e2e905eade6138ab756c8ac4a02e016ecaba857e67aad629c74eb"
    ),
    "mapping_pairs_sha256": (
        "01a64f48b466be49bc8eee9eabb8fa9534227597a871a17154a49b28b9596566"
    ),
    "qualified_mapping_pairs_sha256": (
        "a45c72e4ddffd7e8a59707b69a26e63142906aa8593fbace377a62ce1c98db44"
    ),
    "passage_component_ids_sha256": (
        "d1d1a2a9a17a77395b003cd137500f438be3e04dacd9fc30f38dd640d16fdf2c"
    ),
    "donor_component_pairs_sha256": (
        "b5edfa1e3e5d6212eaf6d969fcf9be10fa457cd5b5cb64f5fb60951b56019e5b"
    ),
}


def canonical_sha256(value: Any) -> str:
    return parent.canonical_sha256(value)


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _receipt(payload_scope: str, unsigned: Mapping[str, Any]) -> dict[str, str]:
    return {
        "algorithm": "sha256",
        "payload_scope": payload_scope,
        "payload_sha256": canonical_sha256(unsigned),
    }


def _validate_receipt(
    value: Mapping[str, Any],
    *,
    payload_scope: str,
    description: str,
) -> None:
    receipt = value.get("receipt")
    if not isinstance(receipt, Mapping):
        raise ValueError(f"{description} receipt is missing")
    unsigned = dict(value)
    unsigned.pop("receipt")
    if dict(receipt) != _receipt(payload_scope, unsigned):
        raise ValueError(f"{description} receipt differs")


def load_protected_manifest(manifest_path: Path) -> dict[str, Any]:
    from experiments.rethinking_rwkv_ms_gemma import (
        materialize_natural_memory_native_rwkv_source_cumulative_residual as materializer,
    )

    if manifest_path.name != "manifest.json":
        raise ValueError("Protected residual loader accepts only manifest.json")
    payload = manifest_path.read_bytes()
    if sha256_bytes(payload) != PROTECTED_MANIFEST_SHA256:
        raise ValueError("Protected residual manifest file hash differs")
    manifest = materializer.load_manifest_only(manifest_path)
    if (
        manifest.get("schema") != PROTECTED_MANIFEST_SCHEMA
        or manifest.get("receipt", {}).get("payload_sha256")
        != PROTECTED_MANIFEST_RECEIPT
        or manifest.get("split_contract", {}).get("receipt", {}).get(
            "payload_sha256"
        )
        != PROTECTED_SPLIT_RECEIPT
    ):
        raise ValueError("Protected residual manifest binding differs")
    return manifest


def _salted_row_rank(row: Mapping[str, Any]) -> tuple[str, str]:
    qualified = parent._qualified_id(row)
    digest = hashlib.sha256((SELECTION_SALT + qualified).encode("ascii")).hexdigest()
    return digest, qualified


def _pair_rank(
    pair: tuple[int, int],
    rows: Sequence[Mapping[str, Any]],
) -> tuple[str, tuple[str, str]]:
    qualified = tuple(parent._qualified_id(rows[source]) for source in pair)
    digest = hashlib.sha256(
        (SELECTION_SALT + parent.canonical_json(list(qualified))).encode("ascii")
    ).hexdigest()
    return digest, qualified


def _build_pairs(
    rows: Sequence[Mapping[str, Any]],
    source_to_component: Mapping[int, str],
    excluded_components: set[str],
) -> tuple[tuple[int, int], ...]:
    eligible = [
        row
        for row in rows
        if source_to_component[int(row["source_index"])] not in excluded_components
    ]
    ordered = sorted(eligible, key=_salted_row_rank)
    used_components: set[str] = set()
    pairs: list[tuple[int, int]] = []
    for source in ordered:
        source_index = int(source["source_index"])
        source_component = source_to_component[source_index]
        if source_component in used_components:
            continue
        donors = [
            donor
            for donor in eligible
            if source_to_component[int(donor["source_index"])]
            not in used_components | {source_component}
            and donor["gold_sha256"] != source["gold_sha256"]
        ]
        if not donors:
            raise ValueError(
                f"Open residual development source has no donor: {source_index}"
            )
        donor = min(
            donors,
            key=lambda candidate: (
                abs(int(candidate["write_tokens"]) - int(source["write_tokens"])),
                _salted_row_rank(candidate),
            ),
        )
        donor_index = int(donor["source_index"])
        donor_component = source_to_component[donor_index]
        pairs.append(tuple(sorted((source_index, donor_index))))
        used_components.update((source_component, donor_component))
        if len(pairs) == PAIR_COUNT:
            break
    if len(pairs) != PAIR_COUNT or len(used_components) != DEVELOPMENT_ROWS:
        raise ValueError("Open residual development donor-pair capacity differs")
    return tuple(sorted(pairs, key=lambda pair: _pair_rank(pair, rows)))


def _protected_components(manifest: Mapping[str, Any]) -> set[str]:
    split_contract = manifest.get("split_contract")
    splits = split_contract.get("splits") if isinstance(split_contract, Mapping) else None
    if not isinstance(splits, Mapping) or set(splits) != set(protected.SPLIT_NAMES):
        raise ValueError("Protected residual split inventory differs")
    components: set[str] = set()
    for name in protected.SPLIT_NAMES:
        values = splits[name].get("passage_component_ids")
        if (
            not isinstance(values, list)
            or len(values) != 32
            or len(set(values)) != len(values)
            or splits[name].get("passage_component_ids_sha256")
            != canonical_sha256(values)
        ):
            raise ValueError(f"Protected residual {name} components differ")
        components.update(values)
    if len(components) != PROTECTED_COMPONENTS:
        raise ValueError("Protected residual component closure differs")
    return components


def build_split(
    metadata_rows: Sequence[Mapping[str, Any]],
    parent_reservation: protected.ParentReservation,
    protected_manifest: Mapping[str, Any],
) -> dict[str, Any]:
    rows = parent._validated_metadata_rows(metadata_rows)
    if (
        parent_reservation.manifest_sha256 != protected.PARENT_MANIFEST_SHA256
        or parent_reservation.manifest_receipt != protected.PARENT_MANIFEST_RECEIPT
        or parent_reservation.split_receipt != protected.PARENT_SPLIT_RECEIPT
    ):
        raise ValueError("Open residual development parent reservation differs")

    source_to_component, component_rows = parent._passage_components(rows)
    historical_components = {
        source_to_component[source]
        for source in parent_reservation.excluded_component_sources
    }
    parent_components = {
        component
        for values in parent_reservation.split_components.values()
        for component in values
    }
    fresh_protected_components = _protected_components(protected_manifest)
    excluded_components = (
        historical_components | parent_components | fresh_protected_components
    )
    if (
        len(component_rows) != 708
        or len(historical_components) != protected.PARENT_EXCLUDED_COMPONENTS
        or len(parent_components) != protected.PARENT_SELECTED_COMPONENTS
        or len(fresh_protected_components) != PROTECTED_COMPONENTS
        or len(excluded_components) != 318
    ):
        raise ValueError("Open residual development exclusion closure differs")

    pairs = _build_pairs(rows, source_to_component, excluded_components)
    if pairs != EXPECTED_PAIRS:
        raise RuntimeError("Open residual development pair reservation differs")
    development = parent._split_payload(pairs, rows, source_to_component)
    if any(development.get(key) != value for key, value in EXPECTED_DIGESTS.items()):
        raise RuntimeError("Open residual development digest differs")
    selected_components = set(development["passage_component_ids"])
    if (
        selected_components & excluded_components
        or len(selected_components) != DEVELOPMENT_ROWS
    ):
        raise RuntimeError("Open residual development overlaps excluded components")

    payload: dict[str, Any] = {
        "schema": SCHEMA,
        "source": parent.SOURCE_BINDING.payload(),
        "selection": {
            "salt": SELECTION_SALT,
            "inputs": (
                "dataset-qualified source_index, row_sha256, gold_sha256, "
                "write_tokens, and normalized passage signature hashes only"
            ),
            "split_unit": (
                "normalized_passage_32_character_shingle_connected_component"
            ),
            "pair_count": PAIR_COUNT,
            "split_assignment": "all ranked pairs are explicitly open development",
        },
        "parent_reservation": {
            "manifest_schema": protected.PARENT_MANIFEST_SCHEMA,
            "manifest_sha256": protected.PARENT_MANIFEST_SHA256,
            "manifest_receipt": protected.PARENT_MANIFEST_RECEIPT,
            "split_receipt": protected.PARENT_SPLIT_RECEIPT,
            "historical_excluded_components": len(historical_components),
            "parent_selected_components": len(parent_components),
            "all_parent_splits_excluded": list(protected.PARENT_SPLIT_NAMES),
            "parent_bundle_bytes_read": False,
        },
        "protected_reservation": {
            "manifest_schema": PROTECTED_MANIFEST_SCHEMA,
            "manifest_sha256": PROTECTED_MANIFEST_SHA256,
            "manifest_receipt": PROTECTED_MANIFEST_RECEIPT,
            "split_receipt": PROTECTED_SPLIT_RECEIPT,
            "excluded_splits": list(protected.SPLIT_NAMES),
            "excluded_components": len(fresh_protected_components),
            "bundle_bytes_read": False,
        },
        "splits": {"development": development},
        "capture_authorization": {
            "default_open_splits": ["development"],
            "sealed_inventory_only_splits": [],
            "development_is_explicitly_open": True,
            "protected_reservation_bundles_must_remain_unopened": True,
        },
        "leakage_audit": {
            "source_rows": len(rows),
            "passage_component_count": len(component_rows),
            "historical_excluded_components": len(historical_components),
            "parent_selected_components": len(parent_components),
            "fresh_protected_components": len(fresh_protected_components),
            "total_excluded_components": len(excluded_components),
            "remaining_components_before_selection": len(component_rows)
            - len(excluded_components),
            "selected_source_rows": DEVELOPMENT_ROWS,
            "selected_passage_components": len(selected_components),
            "protected_overlap_component_count": len(
                selected_components & fresh_protected_components
            ),
        },
        "donor_component_disjoint": True,
        "passage_component_disjoint": True,
        "protected_bundle_bytes_read": False,
        "open_splits": ["development"],
    }
    payload["receipt"] = _receipt("canonical_split_without_receipt", payload)
    return payload


def validate_split_contract(split_contract: Mapping[str, Any]) -> None:
    _validate_receipt(
        split_contract,
        payload_scope="canonical_split_without_receipt",
        description="Open residual development split",
    )
    if (
        split_contract.get("schema") != SCHEMA
        or split_contract.get("source") != parent.SOURCE_BINDING.payload()
        or split_contract.get("open_splits") != ["development"]
        or split_contract.get("protected_bundle_bytes_read") is not False
        or split_contract.get("donor_component_disjoint") is not True
        or split_contract.get("passage_component_disjoint") is not True
    ):
        raise ValueError("Open residual development split contract differs")
    if split_contract.get("capture_authorization") != {
        "default_open_splits": ["development"],
        "sealed_inventory_only_splits": [],
        "development_is_explicitly_open": True,
        "protected_reservation_bundles_must_remain_unopened": True,
    }:
        raise ValueError("Open residual development authorization differs")
    if split_contract.get("leakage_audit") != {
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
    }:
        raise ValueError("Open residual development leakage audit differs")
    protected_binding = split_contract.get("protected_reservation")
    if protected_binding != {
        "manifest_schema": PROTECTED_MANIFEST_SCHEMA,
        "manifest_sha256": PROTECTED_MANIFEST_SHA256,
        "manifest_receipt": PROTECTED_MANIFEST_RECEIPT,
        "split_receipt": PROTECTED_SPLIT_RECEIPT,
        "excluded_splits": list(protected.SPLIT_NAMES),
        "excluded_components": PROTECTED_COMPONENTS,
        "bundle_bytes_read": False,
    }:
        raise ValueError("Open residual protected reservation binding differs")
    splits = split_contract.get("splits")
    development = splits.get("development") if isinstance(splits, Mapping) else None
    if not isinstance(development, Mapping) or set(splits) != {"development"}:
        raise ValueError("Open residual development split inventory differs")
    if (
        tuple(tuple(pair) for pair in development.get("pairs", ())) != EXPECTED_PAIRS
        or development.get("pair_count") != PAIR_COUNT
        or len(development.get("source_indices", ())) != DEVELOPMENT_ROWS
        or any(development.get(key) != value for key, value in EXPECTED_DIGESTS.items())
    ):
        raise ValueError("Open residual development split binding differs")
    for field, digest_field in (
        ("source_indices", "source_indices_sha256"),
        ("qualified_source_ids", "qualified_source_ids_sha256"),
        ("mapping_pairs", "mapping_pairs_sha256"),
        ("qualified_mapping_pairs", "qualified_mapping_pairs_sha256"),
        ("passage_component_ids", "passage_component_ids_sha256"),
        ("donor_component_pairs", "donor_component_pairs_sha256"),
    ):
        if development.get(digest_field) != canonical_sha256(development.get(field)):
            raise ValueError(f"Open residual development {field} digest differs")
