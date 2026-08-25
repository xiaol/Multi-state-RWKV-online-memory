"""Fresh open split for a repaired multi-anchor RWKV value bundle.

This reservation excludes every component used by the historical, parent,
protected, cumulative-development, multi-anchor-development, and weighted
renewal experiments.  Only the explicitly open development bundle may be
materialized from this contract.
"""

from __future__ import annotations

import hashlib
from typing import Any, Mapping, Sequence

from experiments.rethinking_rwkv_ms_gemma import (
    rwkv_continuous_write_fit_split as parent,
)
from experiments.rethinking_rwkv_ms_gemma import (
    rwkv_source_cumulative_residual_development_split as prior,
)
from experiments.rethinking_rwkv_ms_gemma import (
    rwkv_source_cumulative_residual_split as protected,
)
from experiments.rethinking_rwkv_ms_gemma import (
    rwkv_source_multi_anchor_bundle_development_split as multi_anchor,
)
from experiments.rethinking_rwkv_ms_gemma import (
    rwkv_weighted_renewal_bundle_development_split as weighted,
)


SCHEMA = "rwkv_ms_source_multi_anchor_bundle_repair_development_split.v1"
SELECTION_SALT = "rwkv-source-multi-anchor-bundle-open-repair-v1:"
PAIR_COUNT = 40
DEVELOPMENT_ROWS = 80
WEIGHTED_COMPONENTS = 80
EXPECTED_PAIRS = (
    (433, 1387),
    (723, 761),
    (46, 539),
    (910, 919),
    (286, 1300),
    (715, 734),
    (697, 954),
    (403, 1073),
    (840, 1169),
    (293, 650),
    (346, 1042),
    (748, 1380),
    (835, 999),
    (93, 865),
    (52, 740),
    (699, 1013),
    (33, 220),
    (53, 887),
    (224, 526),
    (638, 1030),
    (64, 148),
    (13, 178),
    (520, 1388),
    (647, 1215),
    (532, 1114),
    (848, 1328),
    (924, 1201),
    (295, 1334),
    (970, 1259),
    (136, 496),
    (94, 902),
    (523, 874),
    (1083, 1337),
    (167, 1131),
    (458, 1027),
    (746, 1017),
    (658, 837),
    (275, 333),
    (967, 1277),
    (908, 1129),
)
EXPECTED_DIGESTS: Mapping[str, str] = {
    "source_indices_sha256": (
        "d1998124ec1e0a876eb665b44d1852bf09b42a69e760f7a5e106b676b3d9cd18"
    ),
    "qualified_source_ids_sha256": (
        "3c5ad7719471e2f53a0b6fd0084f02473bdaad08ac6ba9e2e2cb6cef95a0d26c"
    ),
    "mapping_pairs_sha256": (
        "76f29d5561bbf796e6763b9bcd6ba172932094b6b97760b595bf5a8a04d50497"
    ),
    "qualified_mapping_pairs_sha256": (
        "3bbe8387791c9d090fee635e0962792e127693efe0dc5e0c5b9ca01c5409236f"
    ),
    "passage_component_ids_sha256": (
        "0b7c2e5bfe775cefaa5ae8f17990f0c642bcb10debf5cf4daabecf9ba8c4f338"
    ),
    "donor_component_pairs_sha256": (
        "47aef04380e2004c08917e3c754f1b034d4861786bb6e99fa458da5a209b79f3"
    ),
}


def canonical_sha256(value: Any) -> str:
    return parent.canonical_sha256(value)


def _receipt(scope: str, unsigned: Mapping[str, Any]) -> dict[str, str]:
    return {
        "algorithm": "sha256",
        "payload_scope": scope,
        "payload_sha256": canonical_sha256(unsigned),
    }


def _validate_receipt(value: Mapping[str, Any], *, description: str) -> None:
    receipt = value.get("receipt")
    if not isinstance(receipt, Mapping):
        raise ValueError(f"{description} receipt is missing")
    unsigned = dict(value)
    unsigned.pop("receipt")
    if dict(receipt) != _receipt("canonical_split_without_receipt", unsigned):
        raise ValueError(f"{description} receipt differs")


def _salted_row_rank(row: Mapping[str, Any]) -> tuple[str, str]:
    qualified = parent._qualified_id(row)
    digest = hashlib.sha256((SELECTION_SALT + qualified).encode("ascii")).hexdigest()
    return digest, qualified


def _pair_rank(
    pair: tuple[int, int], rows: Sequence[Mapping[str, Any]]
) -> tuple[str, tuple[str, str]]:
    by_index = {int(row["source_index"]): row for row in rows}
    qualified = tuple(parent._qualified_id(by_index[source]) for source in pair)
    digest = hashlib.sha256(
        (SELECTION_SALT + parent.canonical_json(list(qualified))).encode("ascii")
    ).hexdigest()
    return digest, qualified


def _protected_components(manifest: Mapping[str, Any]) -> set[str]:
    split_contract = manifest.get("split_contract")
    splits = split_contract.get("splits") if isinstance(split_contract, Mapping) else None
    if not isinstance(splits, Mapping) or set(splits) != set(protected.SPLIT_NAMES):
        raise ValueError("Protected split inventory differs")
    components: set[str] = set()
    for name in protected.SPLIT_NAMES:
        values = splits[name].get("passage_component_ids")
        if not isinstance(values, list) or len(values) != 32:
            raise ValueError(f"Protected {name} components differ")
        if splits[name].get("passage_component_ids_sha256") != canonical_sha256(values):
            raise ValueError(f"Protected {name} component digest differs")
        components.update(values)
    if len(components) != 64:
        raise ValueError("Protected component closure differs")
    return components


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
            raise ValueError(f"Repair development source has no donor: {source_index}")
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
        raise ValueError("Repair development donor-pair capacity differs")
    return tuple(sorted(pairs, key=lambda pair: _pair_rank(pair, rows)))


def build_split(
    metadata_rows: Sequence[Mapping[str, Any]],
    parent_reservation: protected.ParentReservation,
    protected_manifest: Mapping[str, Any],
    prior_development_manifest: Mapping[str, Any],
    multi_anchor_manifest: Mapping[str, Any],
    weighted_manifest: Mapping[str, Any],
) -> dict[str, Any]:
    rows = parent._validated_metadata_rows(metadata_rows)
    prior_contract = prior.build_split(rows, parent_reservation, protected_manifest)
    if prior_development_manifest.get("split_contract") != prior_contract:
        raise ValueError("Prior cumulative development split differs")
    multi_contract = multi_anchor_manifest.get("split_contract")
    weighted_contract = weighted_manifest.get("split_contract")
    if not isinstance(multi_contract, Mapping) or not isinstance(weighted_contract, Mapping):
        raise ValueError("Prior development split contracts are missing")
    if weighted_contract != weighted.build_split(
        rows, parent_reservation, protected_manifest, prior_development_manifest, multi_anchor_manifest
    ):
        raise ValueError("Prior weighted renewal split differs")

    source_to_component, component_rows = parent._passage_components(rows)
    historical_components = {
        source_to_component[source] for source in parent_reservation.excluded_component_sources
    }
    parent_components = {
        component
        for values in parent_reservation.split_components.values()
        for component in values
    }
    protected_components = _protected_components(protected_manifest)
    prior_components = set(prior_contract["splits"]["development"]["passage_component_ids"])
    multi_components = set(multi_contract["splits"]["development"]["passage_component_ids"])
    weighted_components = set(weighted_contract["splits"]["development"]["passage_component_ids"])
    excluded_components = (
        historical_components
        | parent_components
        | protected_components
        | prior_components
        | multi_components
        | weighted_components
    )
    if (
        len(component_rows) != 708
        or len(historical_components) != 94
        or len(parent_components) != 160
        or len(protected_components) != 64
        or len(prior_components) != 64
        or len(multi_components) != 80
        or len(weighted_components) != WEIGHTED_COMPONENTS
        or len(excluded_components) != 542
        or len(component_rows) - len(excluded_components) != 166
    ):
        raise ValueError("Repair development exclusion closure differs")
    pairs = _build_pairs(rows, source_to_component, excluded_components)
    if pairs != EXPECTED_PAIRS:
        raise RuntimeError("Repair development pair reservation differs")
    development = parent._split_payload(pairs, rows, source_to_component)
    if any(development.get(key) != value for key, value in EXPECTED_DIGESTS.items()):
        raise RuntimeError("Repair development digest differs")
    selected_components = set(development["passage_component_ids"])
    if selected_components & excluded_components or len(selected_components) != DEVELOPMENT_ROWS:
        raise RuntimeError("Repair development overlaps excluded components")
    payload: dict[str, Any] = {
        "schema": SCHEMA,
        "source": parent.SOURCE_BINDING.payload(),
        "selection": {
            "salt": SELECTION_SALT,
            "inputs": "dataset-qualified source_index, row_sha256, gold_sha256, write_tokens, and normalized passage signature hashes only",
            "split_unit": "normalized_passage_32_character_shingle_connected_component",
            "pair_count": PAIR_COUNT,
            "split_assignment": "all ranked pairs are explicitly open development",
        },
        "parent_reservation": {
            "manifest_sha256": protected.PARENT_MANIFEST_SHA256,
            "manifest_receipt": protected.PARENT_MANIFEST_RECEIPT,
            "split_receipt": protected.PARENT_SPLIT_RECEIPT,
            "bundle_bytes_read": False,
        },
        "protected_reservation": {
            "manifest_sha256": prior.PROTECTED_MANIFEST_SHA256,
            "manifest_receipt": prior.PROTECTED_MANIFEST_RECEIPT,
            "split_receipt": prior.PROTECTED_SPLIT_RECEIPT,
            "excluded_components": len(protected_components),
            "bundle_bytes_read": False,
        },
        "prior_cumulative_development": {
            "manifest_receipt": prior_development_manifest["receipt"]["payload_sha256"],
            "split_receipt": prior_contract["receipt"]["payload_sha256"],
            "excluded_components": len(prior_components),
            "bundle_bytes_read": False,
        },
        "prior_multi_anchor_development": {
            "manifest_receipt": multi_anchor_manifest["receipt"]["payload_sha256"],
            "split_receipt": multi_contract["receipt"]["payload_sha256"],
            "excluded_components": len(multi_components),
            "bundle_bytes_read": False,
        },
        "prior_weighted_renewal_development": {
            "manifest_receipt": weighted_manifest["receipt"]["payload_sha256"],
            "split_receipt": weighted_contract["receipt"]["payload_sha256"],
            "excluded_components": len(weighted_components),
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
            "prior_multi_anchor_components": len(multi_components),
            "prior_weighted_components": len(weighted_components),
            "total_excluded_components": len(excluded_components),
            "remaining_components_before_selection": len(component_rows) - len(excluded_components),
            "selected_source_rows": DEVELOPMENT_ROWS,
            "selected_passage_components": len(selected_components),
            "excluded_overlap_component_count": len(selected_components & excluded_components),
        },
        "donor_component_disjoint": True,
        "passage_component_disjoint": True,
        "protected_bundle_bytes_read": False,
        "prior_development_bundle_bytes_read": False,
        "prior_multi_anchor_bundle_bytes_read": False,
        "prior_weighted_bundle_bytes_read": False,
        "open_splits": ["development"],
    }
    payload["receipt"] = _receipt("canonical_split_without_receipt", payload)
    return payload


def validate_split_contract(split_contract: Mapping[str, Any]) -> None:
    _validate_receipt(split_contract, description="Repair development split")
    if split_contract.get("schema") != SCHEMA:
        raise ValueError("Repair development schema differs")
    if split_contract.get("open_splits") != ["development"]:
        raise ValueError("Repair development open split inventory differs")
    for field in (
        "protected_bundle_bytes_read",
        "prior_development_bundle_bytes_read",
        "prior_multi_anchor_bundle_bytes_read",
        "prior_weighted_bundle_bytes_read",
    ):
        if split_contract.get(field) is not False:
            raise ValueError(f"Repair development access flag differs: {field}")
    if split_contract.get("donor_component_disjoint") is not True or split_contract.get("passage_component_disjoint") is not True:
        raise ValueError("Repair development disjointness differs")
    audit = split_contract.get("leakage_audit")
    if not isinstance(audit, Mapping) or audit.get("total_excluded_components") != 542 or audit.get("remaining_components_before_selection") != 166 or audit.get("selected_source_rows") != DEVELOPMENT_ROWS or audit.get("selected_passage_components") != DEVELOPMENT_ROWS or audit.get("excluded_overlap_component_count") != 0:
        raise ValueError("Repair development leakage audit differs")
    development = split_contract.get("splits", {}).get("development")
    if not isinstance(development, Mapping) or development.get("pair_count") != PAIR_COUNT or len(development.get("source_indices", ())) != DEVELOPMENT_ROWS:
        raise ValueError("Repair development pair reservation differs")
    for field, digest_field in (
        ("source_indices", "source_indices_sha256"),
        ("qualified_source_ids", "qualified_source_ids_sha256"),
        ("mapping_pairs", "mapping_pairs_sha256"),
        ("qualified_mapping_pairs", "qualified_mapping_pairs_sha256"),
        ("passage_component_ids", "passage_component_ids_sha256"),
        ("donor_component_pairs", "donor_component_pairs_sha256"),
    ):
        if development.get(digest_field) != canonical_sha256(development.get(field)):
            raise ValueError(f"Repair development {field} digest differs")
