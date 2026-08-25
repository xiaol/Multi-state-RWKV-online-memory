"""Fresh open split for the weighted cumulative-route ablation.

This reservation is independent of the prior multi-anchor bundle result.  It
shares the source and parent firewall, but excludes every component consumed by
the prior cumulative development set and the prior multi-anchor development
set before selecting new reciprocal pairs.
"""

from __future__ import annotations

import hashlib
from typing import Any, Mapping, Sequence

from experiments.rethinking_rwkv_ms_gemma import rwkv_continuous_write_fit_split as parent
from experiments.rethinking_rwkv_ms_gemma import rwkv_source_cumulative_residual_development_split as prior
from experiments.rethinking_rwkv_ms_gemma import rwkv_source_cumulative_residual_split as protected


SCHEMA = "rwkv_ms_source_weighted_renewal_bundle_development_split.v1"
SELECTION_SALT = "rwkv-source-weighted-renewal-bundle-open-pair-split-v1:"
PAIR_COUNT = 40
DEVELOPMENT_ROWS = 80
PRIOR_DEVELOPMENT_COMPONENTS = prior.DEVELOPMENT_ROWS
MULTI_ANCHOR_COMPONENTS = 80


def canonical_sha256(value: Any) -> str:
    return parent.canonical_sha256(value)


def _receipt(payload_scope: str, unsigned: Mapping[str, Any]) -> dict[str, str]:
    return {
        "algorithm": "sha256",
        "payload_scope": payload_scope,
        "payload_sha256": canonical_sha256(unsigned),
    }


def _validate_receipt(value: Mapping[str, Any], *, scope: str, description: str) -> None:
    receipt = value.get("receipt")
    if not isinstance(receipt, Mapping):
        raise ValueError(f"{description} receipt is missing")
    unsigned = dict(value)
    unsigned.pop("receipt")
    if dict(receipt) != _receipt(scope, unsigned):
        raise ValueError(f"{description} receipt differs")


def _salted_row_rank(row: Mapping[str, Any]) -> tuple[str, str]:
    qualified = parent._qualified_id(row)
    digest = hashlib.sha256((SELECTION_SALT + qualified).encode("ascii")).hexdigest()
    return digest, qualified


def _pair_rank(pair: tuple[int, int], rows: Sequence[Mapping[str, Any]]) -> tuple[str, tuple[str, str]]:
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
        row for row in rows
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
            donor for donor in eligible
            if source_to_component[int(donor["source_index"])]
            not in used_components | {source_component}
            and donor["gold_sha256"] != source["gold_sha256"]
        ]
        if not donors:
            raise ValueError(f"Weighted renewal source has no donor: {source_index}")
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
        raise ValueError("Weighted renewal donor-pair capacity differs")
    return tuple(sorted(pairs, key=lambda pair: _pair_rank(pair, rows)))


def _protected_components(manifest: Mapping[str, Any]) -> set[str]:
    split_contract = manifest.get("split_contract")
    splits = split_contract.get("splits") if isinstance(split_contract, Mapping) else None
    if not isinstance(splits, Mapping) or set(splits) != set(protected.SPLIT_NAMES):
        raise ValueError("Protected split inventory differs")
    components: set[str] = set()
    for name in protected.SPLIT_NAMES:
        values = splits[name].get("passage_component_ids")
        if (
            not isinstance(values, list)
            or len(values) != 32
            or len(set(values)) != 32
            or splits[name].get("passage_component_ids_sha256") != canonical_sha256(values)
        ):
            raise ValueError(f"Protected {name} components differ")
        components.update(values)
    if len(components) != 64:
        raise ValueError("Protected component closure differs")
    return components


def build_split(
    metadata_rows: Sequence[Mapping[str, Any]],
    parent_reservation: protected.ParentReservation,
    protected_manifest: Mapping[str, Any],
    prior_development_manifest: Mapping[str, Any],
    multi_anchor_manifest: Mapping[str, Any],
) -> dict[str, Any]:
    rows = parent._validated_metadata_rows(metadata_rows)
    if (
        parent_reservation.manifest_sha256 != protected.PARENT_MANIFEST_SHA256
        or parent_reservation.manifest_receipt != protected.PARENT_MANIFEST_RECEIPT
        or parent_reservation.split_receipt != protected.PARENT_SPLIT_RECEIPT
    ):
        raise ValueError("Parent reservation differs")
    prior_contract = prior.build_split(rows, parent_reservation, protected_manifest)
    if prior_development_manifest.get("split_contract") != prior_contract:
        raise ValueError("Prior cumulative development split differs")
    if prior_development_manifest.get("access", {}).get(
        "protected_reservation_bundle_bytes_read"
    ) is not False:
        raise ValueError("Prior cumulative development access flags differ")
    multi_split = multi_anchor_manifest.get("split_contract")
    if not isinstance(multi_split, Mapping):
        raise ValueError("Prior multi-anchor split is missing")
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
    protected_components = _protected_components(protected_manifest)
    prior_components = set(prior_contract["splits"]["development"]["passage_component_ids"])
    multi_components = set(multi_split["splits"]["development"]["passage_component_ids"])
    excluded_components = (
        historical_components | parent_components | protected_components
        | prior_components | multi_components
    )
    if (
        len(component_rows) != 708
        or len(historical_components) != 94
        or len(parent_components) != 160
        or len(protected_components) != 64
        or len(prior_components) != PRIOR_DEVELOPMENT_COMPONENTS
        or len(multi_components) != MULTI_ANCHOR_COMPONENTS
        or len(excluded_components) != 462
    ):
        raise ValueError("Weighted renewal exclusion closure differs")
    pairs = _build_pairs(rows, source_to_component, excluded_components)
    development = parent._split_payload(pairs, rows, source_to_component)
    selected_components = set(development["passage_component_ids"])
    if selected_components & excluded_components or len(selected_components) != DEVELOPMENT_ROWS:
        raise ValueError("Weighted renewal split overlaps excluded components")
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
            "split_receipt": multi_split["receipt"]["payload_sha256"],
            "excluded_components": len(multi_components),
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
        "open_splits": ["development"],
    }
    payload["receipt"] = _receipt("canonical_split_without_receipt", payload)
    return payload


def validate_split_contract(split_contract: Mapping[str, Any]) -> None:
    _validate_receipt(split_contract, scope="canonical_split_without_receipt", description="Weighted renewal split")
    if split_contract.get("schema") != SCHEMA:
        raise ValueError("Weighted renewal split schema differs")
    if split_contract.get("source") != parent.SOURCE_BINDING.payload():
        raise ValueError("Weighted renewal source differs")
    if split_contract.get("open_splits") != ["development"]:
        raise ValueError("Weighted renewal open split inventory differs")
    if any(split_contract.get(name) is not False for name in (
        "protected_bundle_bytes_read", "prior_development_bundle_bytes_read", "prior_multi_anchor_bundle_bytes_read"
    )):
        raise ValueError("Weighted renewal access flags differ")
    if split_contract.get("donor_component_disjoint") is not True or split_contract.get("passage_component_disjoint") is not True:
        raise ValueError("Weighted renewal disjointness flags differ")
    audit = split_contract.get("leakage_audit")
    if not isinstance(audit, Mapping) or audit.get("total_excluded_components") != 462 or audit.get("selected_source_rows") != 80 or audit.get("selected_passage_components") != 80 or audit.get("excluded_overlap_component_count") != 0:
        raise ValueError("Weighted renewal leakage audit differs")
    splits = split_contract.get("splits")
    development = splits.get("development") if isinstance(splits, Mapping) else None
    if not isinstance(development, Mapping) or set(splits) != {"development"}:
        raise ValueError("Weighted renewal split inventory differs")
    if development.get("pair_count") != PAIR_COUNT or len(development.get("source_indices", ())) != DEVELOPMENT_ROWS:
        raise ValueError("Weighted renewal pair reservation differs")
    for field, digest_field in (
        ("source_indices", "source_indices_sha256"),
        ("qualified_source_ids", "qualified_source_ids_sha256"),
        ("mapping_pairs", "mapping_pairs_sha256"),
        ("qualified_mapping_pairs", "qualified_mapping_pairs_sha256"),
        ("passage_component_ids", "passage_component_ids_sha256"),
        ("donor_component_pairs", "donor_component_pairs_sha256"),
    ):
        if development.get(digest_field) != canonical_sha256(development.get(field)):
            raise ValueError(f"Weighted renewal {field} digest differs")
