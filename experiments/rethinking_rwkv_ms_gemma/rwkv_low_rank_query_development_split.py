"""Fresh open split for low-rank address/query-conditioned RWKV reads.

The split excludes every prior historical, parent, protected, cumulative,
multi-anchor, weighted-renewal, and multi-anchor-repair component.  Only this
route's development bundle is open.
"""

from __future__ import annotations

import hashlib
from typing import Any, Mapping, Sequence

from experiments.rethinking_rwkv_ms_gemma import (
    rwkv_continuous_write_fit_split as parent,
)


SCHEMA = "rwkv_ms_source_low_rank_query_development_split.v1"
SELECTION_SALT = "rwkv-source-low-rank-query-open-v1:"
PAIR_COUNT = 40
DEVELOPMENT_ROWS = 80
EXPECTED_PAIRS = (
    (22, 769), (72, 363), (75, 1147), (122, 885), (133, 1057),
    (205, 1345), (223, 366), (258, 271), (261, 937), (296, 1134),
    (302, 1028), (322, 950), (330, 1256), (339, 1270), (349, 1161),
    (365, 1292), (380, 872), (422, 1262), (440, 767), (453, 1184),
    (459, 593), (469, 1227), (476, 1064), (485, 1297), (530, 935),
    (549, 944), (553, 857), (554, 971), (587, 949), (602, 850),
    (607, 1160), (611, 1063), (614, 855), (714, 718), (771, 773),
    (867, 1164), (911, 920), (987, 1162), (1125, 1246), (1126, 1284),
)


def canonical_sha256(value: Any) -> str:
    return parent.canonical_sha256(value)


def _receipt(unsigned: Mapping[str, Any]) -> dict[str, str]:
    return {
        "algorithm": "sha256",
        "payload_scope": "canonical_split_without_receipt",
        "payload_sha256": canonical_sha256(unsigned),
    }


def _validate_receipt(value: Mapping[str, Any], description: str) -> None:
    receipt = value.get("receipt")
    if not isinstance(receipt, Mapping):
        raise ValueError(f"{description} receipt is missing")
    unsigned = dict(value)
    unsigned.pop("receipt")
    if dict(receipt) != _receipt(unsigned):
        raise ValueError(f"{description} receipt differs")


def _protected_components(manifest: Mapping[str, Any]) -> set[str]:
    split = manifest.get("split_contract")
    splits = split.get("splits") if isinstance(split, Mapping) else None
    if not isinstance(splits, Mapping):
        raise ValueError("Protected split inventory is missing")
    components: set[str] = set()
    for name in ("mechanics", "causal"):
        values = splits[name].get("passage_component_ids")
        if not isinstance(values, list) or len(values) != 32:
            raise ValueError(f"Protected {name} component inventory differs")
        components.update(values)
    if len(components) != 64:
        raise ValueError("Protected component closure differs")
    return components


def _pair_rank(pair: tuple[int, int], rows: Sequence[Mapping[str, Any]]) -> tuple[str, tuple[str, str]]:
    by_index = {int(row["source_index"]): row for row in rows}
    qualified = tuple(parent._qualified_id(by_index[index]) for index in pair)
    digest = hashlib.sha256(
        (SELECTION_SALT + parent.canonical_json(list(qualified))).encode("ascii")
    ).hexdigest()
    return digest, qualified


def build_split(
    metadata_rows: Sequence[Mapping[str, Any]],
    parent_reservation: Any,
    protected_manifest: Mapping[str, Any],
    prior_manifest: Mapping[str, Any],
    multi_manifest: Mapping[str, Any],
    weighted_manifest: Mapping[str, Any],
    repair_manifest: Mapping[str, Any],
) -> dict[str, Any]:
    rows = parent._validated_metadata_rows(metadata_rows)
    source_to_component, component_rows = parent._passage_components(rows)
    historical = {
        source_to_component[source]
        for source in parent_reservation.excluded_component_sources
    }
    parent_components = {
        component
        for values in parent_reservation.split_components.values()
        for component in values
    }
    protected_components = _protected_components(protected_manifest)
    development_manifests = {
        "prior_cumulative": prior_manifest,
        "prior_multi_anchor": multi_manifest,
        "prior_weighted_renewal": weighted_manifest,
        "prior_multi_anchor_repair": repair_manifest,
    }
    development_components = {
        name: set(manifest["split_contract"]["splits"]["development"]["passage_component_ids"])
        for name, manifest in development_manifests.items()
    }
    excluded = (
        historical
        | parent_components
        | protected_components
        | set().union(*development_components.values())
    )
    if (
        len(component_rows) != 708
        or len(historical) != 94
        or len(parent_components) != 160
        or len(protected_components) != 64
        or {name: len(values) for name, values in development_components.items()}
        != {
            "prior_cumulative": 64,
            "prior_multi_anchor": 80,
            "prior_weighted_renewal": 80,
            "prior_multi_anchor_repair": 80,
        }
        or len(excluded) != 622
        or len(component_rows) - len(excluded) != 86
    ):
        raise ValueError("Low-rank query exclusion closure differs")

    eligible = [
        row
        for row in rows
        if source_to_component[int(row["source_index"])] not in excluded
    ]
    if len(eligible) != 106:
        raise ValueError("Low-rank query eligible row count differs")
    pairs = tuple(EXPECTED_PAIRS)
    if len(pairs) != PAIR_COUNT or tuple(sorted(pairs)) != pairs:
        raise ValueError("Low-rank query pair reservation is not canonical")
    selected_sources = {source for pair in pairs for source in pair}
    if len(selected_sources) != DEVELOPMENT_ROWS:
        raise ValueError("Low-rank query source reservation differs")
    if any(source not in {int(row["source_index"]) for row in eligible} for source in selected_sources):
        raise ValueError("Low-rank query pair uses an excluded source")
    payload = parent._split_payload(pairs, rows, source_to_component)
    if len(payload["passage_component_ids"]) != DEVELOPMENT_ROWS:
        raise ValueError("Low-rank query selected component count differs")
    if len(set(payload["passage_component_ids"]) & excluded) != 0:
        raise ValueError("Low-rank query overlaps an excluded component")
    result: dict[str, Any] = {
        "schema": SCHEMA,
        "source": parent.SOURCE_BINDING.payload(),
        "selection": {
            "salt": SELECTION_SALT,
            "split_unit": "normalized_passage_32_character_shingle_connected_component",
            "pair_count": PAIR_COUNT,
            "split_assignment": "all ranked pairs are explicitly open development",
        },
        "prior_reservations": {
            "parent_manifest_sha256": parent_reservation.manifest_sha256,
            "protected_bundle_bytes_read": False,
            "prior_cumulative_manifest_receipt": prior_manifest["receipt"]["payload_sha256"],
            "prior_multi_anchor_manifest_receipt": multi_manifest["receipt"]["payload_sha256"],
            "prior_weighted_manifest_receipt": weighted_manifest["receipt"]["payload_sha256"],
            "prior_multi_anchor_repair_manifest_receipt": repair_manifest["receipt"]["payload_sha256"],
            "prior_bundle_bytes_read": False,
        },
        "splits": {"development": payload},
        "leakage_audit": {
            "source_rows": len(rows),
            "passage_component_count": len(component_rows),
            "historical_excluded_components": len(historical),
            "parent_selected_components": len(parent_components),
            "protected_components": len(protected_components),
            "prior_cumulative_components": len(development_components["prior_cumulative"]),
            "prior_multi_anchor_components": len(development_components["prior_multi_anchor"]),
            "prior_weighted_components": len(development_components["prior_weighted_renewal"]),
            "prior_multi_anchor_repair_components": len(development_components["prior_multi_anchor_repair"]),
            "total_excluded_components": len(excluded),
            "remaining_components_before_selection": len(component_rows) - len(excluded),
            "selected_source_rows": DEVELOPMENT_ROWS,
            "selected_passage_components": len(payload["passage_component_ids"]),
            "excluded_overlap_component_count": 0,
        },
        "donor_component_disjoint": True,
        "passage_component_disjoint": True,
        "protected_bundle_bytes_read": False,
        "prior_development_bundle_bytes_read": False,
        "open_splits": ["development"],
        "protected_splits_opened": [],
    }
    result["receipt"] = _receipt(result)
    return result


def validate_split_contract(value: Mapping[str, Any]) -> None:
    _validate_receipt(value, "Low-rank query split")
    if value.get("schema") != SCHEMA or value.get("open_splits") != ["development"]:
        raise ValueError("Low-rank query split schema/access differs")
    if value.get("protected_bundle_bytes_read") is not False or value.get("protected_splits_opened") != []:
        raise ValueError("Low-rank query protected access differs")
    development = value.get("splits", {}).get("development")
    if not isinstance(development, Mapping) or development.get("pair_count") != PAIR_COUNT:
        raise ValueError("Low-rank query development pair count differs")
    if development.get("source_indices_sha256") != canonical_sha256(development.get("source_indices")):
        raise ValueError("Low-rank query source digest differs")
    if len(development.get("source_indices", ())) != DEVELOPMENT_ROWS:
        raise ValueError("Low-rank query row count differs")
