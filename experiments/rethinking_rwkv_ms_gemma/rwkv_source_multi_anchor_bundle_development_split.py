"""Fresh open component-disjoint split for multi-anchor RWKV bundle training."""

from __future__ import annotations

import hashlib
from typing import Any, Mapping, Sequence

from experiments.rethinking_rwkv_ms_gemma import rwkv_continuous_write_fit_split as parent
from experiments.rethinking_rwkv_ms_gemma import (
    rwkv_source_cumulative_residual_development_split as prior,
)
from experiments.rethinking_rwkv_ms_gemma import (
    rwkv_source_cumulative_residual_split as protected,
)


SCHEMA = "rwkv_ms_source_multi_anchor_bundle_development_split.v1"
SELECTION_SALT = "rwkv-source-multi-anchor-bundle-open-development-v1:"
PAIR_COUNT = 40
DEVELOPMENT_ROWS = 80
PRIOR_DEVELOPMENT_COMPONENTS = prior.DEVELOPMENT_ROWS
EXPECTED_PAIRS = (
    (102, 256),
    (667, 1242),
    (866, 968),
    (550, 1399),
    (1143, 1425),
    (742, 884),
    (425, 895),
    (617, 1442),
    (450, 797),
    (689, 983),
    (468, 907),
    (576, 1172),
    (466, 480),
    (113, 238),
    (995, 1117),
    (412, 889),
    (369, 524),
    (247, 891),
    (573, 663),
    (460, 709),
    (487, 1127),
    (527, 758),
    (1136, 1384),
    (432, 1291),
    (627, 1268),
    (1261, 1320),
    (1255, 1308),
    (1062, 1351),
    (364, 437),
    (763, 991),
    (156, 294),
    (298, 1238),
    (695, 1163),
    (424, 1441),
    (81, 717),
    (542, 827),
    (320, 1318),
    (74, 1379),
    (104, 489),
    (443, 569),
)
EXPECTED_DIGESTS: Mapping[str, str] = {
    "source_indices_sha256": (
        "69ba72ea6bef4f6c2842a3176fb4ba6834dc9ddb5e961e0f3ad6158028f861ec"
    ),
    "qualified_source_ids_sha256": (
        "523019d8354bfa8deddee40346ce20cdaffde20c4c89aad1b90a6292316393a6"
    ),
    "mapping_pairs_sha256": (
        "8c9b6a00309eb2f8d87f17526090b4bf49ccd2196ed9748084de9c75ad403e5e"
    ),
    "qualified_mapping_pairs_sha256": (
        "5b275af58222b460dc65b2fefd1582a91ca38204d53b25416987a9f6de6d4ce3"
    ),
    "passage_component_ids_sha256": (
        "36392058c012b5f901d5d0deabd4c4bb091099afb6ba1e6f6baa4a4b4f3ccabd"
    ),
    "donor_component_pairs_sha256": (
        "ba467e8f45035bf63034e9ae27fac2c3e39bbc663160e19cca16235abee9cc61"
    ),
}


def canonical_sha256(value: Any) -> str:
    return parent.canonical_sha256(value)


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
                f"Multi-anchor development source has no donor: {source_index}"
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
        raise ValueError("Multi-anchor development donor-pair capacity differs")
    return tuple(sorted(pairs, key=lambda pair: _pair_rank(pair, rows)))


def build_split(
    metadata_rows: Sequence[Mapping[str, Any]],
    parent_reservation: protected.ParentReservation,
    protected_manifest: Mapping[str, Any],
    prior_development_manifest: Mapping[str, Any],
) -> dict[str, Any]:
    rows = parent._validated_metadata_rows(metadata_rows)
    prior_contract = prior.build_split(
        rows,
        parent_reservation,
        protected_manifest,
    )
    if (
        prior_development_manifest.get("split_contract") != prior_contract
        or prior_development_manifest.get("access", {}).get(
            "protected_reservation_bundle_bytes_read"
        )
        is not False
    ):
        raise ValueError("Prior open development manifest binding differs")

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
    protected_components = prior._protected_components(protected_manifest)
    prior_components = set(
        prior_contract["splits"]["development"]["passage_component_ids"]
    )
    excluded_components = (
        historical_components
        | parent_components
        | protected_components
        | prior_components
    )
    if (
        len(component_rows) != 708
        or len(historical_components) != protected.PARENT_EXCLUDED_COMPONENTS
        or len(parent_components) != protected.PARENT_SELECTED_COMPONENTS
        or len(protected_components) != prior.PROTECTED_COMPONENTS
        or len(prior_components) != PRIOR_DEVELOPMENT_COMPONENTS
        or len(excluded_components) != 382
    ):
        raise ValueError("Multi-anchor development exclusion closure differs")

    pairs = _build_pairs(rows, source_to_component, excluded_components)
    if pairs != EXPECTED_PAIRS:
        raise RuntimeError("Multi-anchor development pair reservation differs")
    development = parent._split_payload(pairs, rows, source_to_component)
    if any(
        development.get(key) != value for key, value in EXPECTED_DIGESTS.items()
    ):
        raise RuntimeError("Multi-anchor development digest differs")
    selected_components = set(development["passage_component_ids"])
    if (
        selected_components & excluded_components
        or len(selected_components) != DEVELOPMENT_ROWS
    ):
        raise RuntimeError("Multi-anchor development overlaps excluded components")

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
        "parent_reservation": prior_contract["parent_reservation"],
        "protected_reservation": prior_contract["protected_reservation"],
        "prior_open_development": {
            "manifest_schema": prior_development_manifest["schema"],
            "manifest_receipt": prior_development_manifest["receipt"][
                "payload_sha256"
            ],
            "split_receipt": prior_contract["receipt"]["payload_sha256"],
            "excluded_components": len(prior_components),
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
            "protected_components": len(protected_components),
            "prior_development_components": len(prior_components),
            "total_excluded_components": len(excluded_components),
            "remaining_components_before_selection": len(component_rows)
            - len(excluded_components),
            "selected_source_rows": DEVELOPMENT_ROWS,
            "selected_passage_components": len(selected_components),
            "excluded_overlap_component_count": len(
                selected_components & excluded_components
            ),
        },
        "donor_component_disjoint": True,
        "passage_component_disjoint": True,
        "protected_bundle_bytes_read": False,
        "prior_development_bundle_bytes_read": False,
        "open_splits": ["development"],
    }
    payload["receipt"] = _receipt("canonical_split_without_receipt", payload)
    return payload


def validate_split_contract(split_contract: Mapping[str, Any]) -> None:
    _validate_receipt(
        split_contract,
        payload_scope="canonical_split_without_receipt",
        description="Multi-anchor development split",
    )
    if (
        split_contract.get("schema") != SCHEMA
        or split_contract.get("source") != parent.SOURCE_BINDING.payload()
        or split_contract.get("open_splits") != ["development"]
        or split_contract.get("protected_bundle_bytes_read") is not False
        or split_contract.get("prior_development_bundle_bytes_read") is not False
        or split_contract.get("donor_component_disjoint") is not True
        or split_contract.get("passage_component_disjoint") is not True
    ):
        raise ValueError("Multi-anchor development split contract differs")
    if split_contract.get("leakage_audit") != {
        "source_rows": 1443,
        "passage_component_count": 708,
        "historical_excluded_components": 94,
        "parent_selected_components": 160,
        "protected_components": 64,
        "prior_development_components": 64,
        "total_excluded_components": 382,
        "remaining_components_before_selection": 326,
        "selected_source_rows": DEVELOPMENT_ROWS,
        "selected_passage_components": DEVELOPMENT_ROWS,
        "excluded_overlap_component_count": 0,
    }:
        raise ValueError("Multi-anchor development leakage audit differs")
    splits = split_contract.get("splits")
    development = splits.get("development") if isinstance(splits, Mapping) else None
    if not isinstance(development, Mapping) or set(splits) != {"development"}:
        raise ValueError("Multi-anchor development split inventory differs")
    if (
        development.get("pair_count") != PAIR_COUNT
        or len(development.get("source_indices", ())) != DEVELOPMENT_ROWS
        or tuple(tuple(pair) for pair in development["pairs"]) != EXPECTED_PAIRS
        or any(
            development.get(key) != value
            for key, value in EXPECTED_DIGESTS.items()
        )
    ):
        raise ValueError("Multi-anchor development split binding differs")
    for field, digest_field in (
        ("source_indices", "source_indices_sha256"),
        ("qualified_source_ids", "qualified_source_ids_sha256"),
        ("mapping_pairs", "mapping_pairs_sha256"),
        ("qualified_mapping_pairs", "qualified_mapping_pairs_sha256"),
        ("passage_component_ids", "passage_component_ids_sha256"),
        ("donor_component_pairs", "donor_component_pairs_sha256"),
    ):
        if development.get(digest_field) != canonical_sha256(development.get(field)):
            raise ValueError(f"Multi-anchor development {field} digest differs")
