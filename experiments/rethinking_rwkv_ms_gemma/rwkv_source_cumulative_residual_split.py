"""Fresh component-disjoint mechanics/causal split for cumulative RWKV residuals."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from experiments.rethinking_rwkv_ms_gemma import rwkv_continuous_write_fit_split as parent


SCHEMA = "rwkv_ms_source_cumulative_residual_split.v1"
SELECTION_SALT = "rwkv-cumulative-virtual-kv-fresh-reservation-v1:"
PAIR_COUNT = 32
MECHANICS_PAIRS = 16
CAUSAL_PAIRS = 16
SPLIT_NAMES = ("mechanics", "causal")

PARENT_MANIFEST_SCHEMA = "rwkv_ms_natural_memory_native_rwkv_continuous_write_open_fit.v2"
PARENT_MANIFEST_SHA256 = (
    "c437a7d1f2b850a730fe5b28a08ae32ba02678561bb1265a4eef55bda7f4d468"
)
PARENT_MANIFEST_RECEIPT = (
    "99a878493c3848c96624e2ad658842c99e69769b4a1721b5854ad25af8d0bee2"
)
PARENT_SPLIT_RECEIPT = (
    "d9ad640c208aae6983ce603f5d1918b06ab4ba9e93ed935d9cbfe1ac25f4801a"
)
PARENT_SPLIT_NAMES = ("fit", "retrieval", "mechanics", "causal")
PARENT_EXCLUDED_COMPONENTS = 94
PARENT_SELECTED_COMPONENTS = 160

CONSUMED_RESULT_SCHEMA = (
    "rwkv_ms_natural_memory_native_rwkv_cumulative_virtual_kv_mechanics.v1"
)
CONSUMED_RESULT_STATUS = "cumulative_virtual_kv_mechanics_failed_family_retired"
CONSUMED_RESULT_SHA256 = (
    "23ce8601a84c388b8b1ea0a2bee527c9eb677dd67cf5b80a5f49f44e9deb7d58"
)
CONSUMED_RESULT_RECEIPT = (
    "6dff6c59c7ee03d2a2ae775bc88ebd94877e8c80e06c9808f0b59e7eeb302a27"
)

EXPECTED_PAIRS = {
    "mechanics": (
        (201, 1165),
        (666, 1307),
        (78, 1331),
        (58, 846),
        (179, 922),
        (163, 1422),
        (519, 592),
        (249, 1099),
        (684, 930),
        (350, 1313),
        (267, 1216),
        (152, 1193),
        (798, 1124),
        (354, 1005),
        (190, 421),
        (585, 698),
    ),
    "causal": (
        (272, 618),
        (264, 492),
        (973, 1154),
        (845, 1088),
        (374, 959),
        (622, 979),
        (541, 802),
        (1047, 1086),
        (6, 817),
        (248, 331),
        (27, 864),
        (599, 640),
        (724, 1018),
        (202, 304),
        (396, 465),
        (538, 679),
    ),
}
EXPECTED_SOURCE_INDICES_SHA256 = {
    "mechanics": "03689eb7e9c7d95e953b76dca0ffcf7fedafed8fd4be9dee8db616a4e1a67513",
    "causal": "1a2ac2ed2c762838803d820ddb8b9b4f0b2fb3f452fdba4ad694a1d97b264d8f",
}
EXPECTED_COMPONENT_IDS_SHA256 = {
    "mechanics": "4382ed18086184f9c458f92ad98b757d5974e0390030a507cb3145e097cea4aa",
    "causal": "7c01149606185485ddfa97c7683bdbed3295ac6c9db83a44d94d80fb984c29a3",
}


def canonical_sha256(value: Any) -> str:
    return parent.canonical_sha256(value)


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _validate_receipt(
    value: Mapping[str, Any],
    *,
    payload_scope: str,
    expected_sha256: str,
    description: str,
) -> None:
    receipt = value.get("receipt")
    if not isinstance(receipt, Mapping):
        raise ValueError(f"{description} receipt is missing")
    unsigned = dict(value)
    unsigned.pop("receipt")
    expected = {
        "algorithm": "sha256",
        "payload_scope": payload_scope,
        "payload_sha256": canonical_sha256(unsigned),
    }
    if dict(receipt) != expected or expected["payload_sha256"] != expected_sha256:
        raise ValueError(f"{description} receipt differs")


@dataclass(frozen=True)
class ParentReservation:
    excluded_component_sources: tuple[int, ...]
    split_sources: Mapping[str, tuple[int, ...]]
    split_components: Mapping[str, tuple[str, ...]]
    manifest_sha256: str
    manifest_receipt: str
    split_receipt: str


@dataclass(frozen=True)
class ConsumedMechanics:
    source_indices: tuple[int, ...]
    result_sha256: str
    result_receipt: str


def validate_parent_manifest(
    manifest: Mapping[str, Any],
    *,
    manifest_sha256: str,
) -> ParentReservation:
    if manifest_sha256 != PARENT_MANIFEST_SHA256:
        raise ValueError("Parent continuous-write manifest file hash differs")
    if manifest.get("schema") != PARENT_MANIFEST_SCHEMA:
        raise ValueError("Parent continuous-write manifest schema differs")
    if manifest.get("protected_splits_opened") != []:
        raise ValueError("Parent continuous-write manifest opened protected splits")
    source = manifest.get("source")
    if not isinstance(source, Mapping) or {
        key: source.get(key) for key in ("namespace", "path", "sha256", "rows")
    } != parent.SOURCE_BINDING.payload():
        raise ValueError("Parent continuous-write source binding differs")
    _validate_receipt(
        manifest,
        payload_scope="canonical_manifest_without_receipt",
        expected_sha256=PARENT_MANIFEST_RECEIPT,
        description="Parent continuous-write manifest",
    )
    split_contract = manifest.get("split_contract")
    if not isinstance(split_contract, Mapping):
        raise ValueError("Parent continuous-write split contract is missing")
    if (
        split_contract.get("schema") != parent.SCHEMA
        or split_contract.get("source") != parent.SOURCE_BINDING.payload()
        or split_contract.get("protected_splits_opened") != []
        or split_contract.get("sealed_bundle_bytes_read") is not False
    ):
        raise ValueError("Parent continuous-write split contract differs")
    _validate_receipt(
        split_contract,
        payload_scope="canonical_split_without_receipt",
        expected_sha256=PARENT_SPLIT_RECEIPT,
        description="Parent continuous-write split contract",
    )
    prior = split_contract.get("prior_reservation")
    excluded_sources = (
        prior.get("excluded_passage_component_sources")
        if isinstance(prior, Mapping)
        else None
    )
    if (
        not isinstance(excluded_sources, list)
        or any(isinstance(source, bool) or not isinstance(source, int) for source in excluded_sources)
        or excluded_sources != sorted(set(excluded_sources))
        or prior.get("excluded_passage_component_sources_sha256")
        != canonical_sha256(excluded_sources)
        or prior.get("bundle_bytes_read") is not False
    ):
        raise ValueError("Parent prior component closure differs")

    raw_splits = split_contract.get("splits")
    if not isinstance(raw_splits, Mapping) or set(raw_splits) != set(PARENT_SPLIT_NAMES):
        raise ValueError("Parent split inventory differs")
    split_sources: dict[str, tuple[int, ...]] = {}
    split_components: dict[str, tuple[str, ...]] = {}
    for name in PARENT_SPLIT_NAMES:
        payload = raw_splits[name]
        if not isinstance(payload, Mapping):
            raise ValueError(f"Parent {name} split is invalid")
        sources = payload.get("source_indices")
        components = payload.get("passage_component_ids")
        if (
            not isinstance(sources, list)
            or sources != sorted(set(sources))
            or not isinstance(components, list)
            or len(components) != len(sources)
            or len(set(components)) != len(components)
            or payload.get("source_indices_sha256") != canonical_sha256(sources)
            or payload.get("passage_component_ids_sha256")
            != canonical_sha256(components)
        ):
            raise ValueError(f"Parent {name} split binding differs")
        split_sources[name] = tuple(sources)
        split_components[name] = tuple(components)
    component_sets = [set(split_components[name]) for name in PARENT_SPLIT_NAMES]
    if (
        any(
            left & right
            for index, left in enumerate(component_sets)
            for right in component_sets[index + 1 :]
        )
        or len(set().union(*component_sets)) != PARENT_SELECTED_COMPONENTS
    ):
        raise ValueError("Parent split components overlap or differ")
    return ParentReservation(
        excluded_component_sources=tuple(excluded_sources),
        split_sources=split_sources,
        split_components=split_components,
        manifest_sha256=manifest_sha256,
        manifest_receipt=PARENT_MANIFEST_RECEIPT,
        split_receipt=PARENT_SPLIT_RECEIPT,
    )


def load_parent_reservation(manifest_path: Path) -> ParentReservation:
    if manifest_path.name != "manifest.json":
        raise ValueError("Parent reservation loader accepts only manifest.json")
    payload = manifest_path.read_bytes()
    try:
        manifest = json.loads(payload.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("Parent continuous-write manifest is invalid JSON") from error
    if not isinstance(manifest, Mapping):
        raise ValueError("Parent continuous-write manifest root is not an object")
    return validate_parent_manifest(manifest, manifest_sha256=sha256_bytes(payload))


def validate_consumed_result(
    result: Mapping[str, Any],
    *,
    result_sha256: str,
    parent_reservation: ParentReservation,
) -> ConsumedMechanics:
    if result_sha256 != CONSUMED_RESULT_SHA256:
        raise ValueError("Consumed cumulative mechanics result file hash differs")
    if (
        result.get("schema") != CONSUMED_RESULT_SCHEMA
        or result.get("status") != CONSUMED_RESULT_STATUS
        or result.get("passed") is not False
        or result.get("mechanics_bundle_byte_opens") != 1
        or result.get("mechanics_rows_opened") != 32
        or result.get("causal_rows_opened") != 0
        or result.get("generation_or_native_benchmark_rows_opened") != 0
        or result.get("model_or_adapter_parameters_updated") is not False
        or result.get("full_bandwidth_feedback_installed") is not False
    ):
        raise ValueError("Consumed cumulative mechanics outcome differs")
    _validate_receipt(
        result,
        payload_scope="canonical_result_without_receipt",
        expected_sha256=CONSUMED_RESULT_RECEIPT,
        description="Consumed cumulative mechanics result",
    )
    rows = result.get("rows")
    if not isinstance(rows, list) or len(rows) != 32:
        raise ValueError("Consumed cumulative mechanics rows differ")
    source_indices = tuple(sorted(int(row["source_index"]) for row in rows))
    donor_mapping = {
        int(row["source_index"]): int(row["donor_source_index"])
        for row in rows
    }
    expected_sources = parent_reservation.split_sources["mechanics"]
    if (
        source_indices != expected_sources
        or set(donor_mapping) != set(expected_sources)
        or any(
            donor_mapping.get(donor) != source
            for source, donor in donor_mapping.items()
        )
    ):
        raise ValueError("Consumed mechanics rows do not bind the parent mechanics split")
    return ConsumedMechanics(
        source_indices=source_indices,
        result_sha256=result_sha256,
        result_receipt=CONSUMED_RESULT_RECEIPT,
    )


def load_consumed_mechanics(
    result_path: Path,
    *,
    parent_reservation: ParentReservation,
) -> ConsumedMechanics:
    if result_path.name != "result.json":
        raise ValueError("Consumed mechanics loader accepts only result.json")
    payload = result_path.read_bytes()
    try:
        result = json.loads(payload.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("Consumed cumulative mechanics result is invalid JSON") from error
    if not isinstance(result, Mapping):
        raise ValueError("Consumed cumulative mechanics result root is not an object")
    return validate_consumed_result(
        result,
        result_sha256=sha256_bytes(payload),
        parent_reservation=parent_reservation,
    )


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
            raise ValueError(f"Fresh residual source has no eligible donor: {source_index}")
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
    if len(pairs) != PAIR_COUNT or len(used_components) != 2 * PAIR_COUNT:
        raise ValueError("Fresh residual donor-pair capacity differs")
    return tuple(sorted(pairs, key=lambda pair: _pair_rank(pair, rows)))


def build_split(
    metadata_rows: Sequence[Mapping[str, Any]],
    parent_reservation: ParentReservation,
    consumed_mechanics: ConsumedMechanics,
) -> dict[str, Any]:
    rows = parent._validated_metadata_rows(metadata_rows)
    if (
        parent_reservation.manifest_sha256 != PARENT_MANIFEST_SHA256
        or parent_reservation.manifest_receipt != PARENT_MANIFEST_RECEIPT
        or parent_reservation.split_receipt != PARENT_SPLIT_RECEIPT
        or consumed_mechanics.result_sha256 != CONSUMED_RESULT_SHA256
        or consumed_mechanics.result_receipt != CONSUMED_RESULT_RECEIPT
        or consumed_mechanics.source_indices
        != parent_reservation.split_sources["mechanics"]
    ):
        raise ValueError("Fresh residual reservation inputs differ from their pins")

    source_to_component, component_rows = parent._passage_components(rows)
    prior_components = {
        source_to_component[source]
        for source in parent_reservation.excluded_component_sources
    }
    if len(prior_components) != PARENT_EXCLUDED_COMPONENTS:
        raise ValueError("Historical reservation component closure differs")
    current_components = {
        source_to_component[source]
        for split_sources in parent_reservation.split_sources.values()
        for source in split_sources
    }
    recorded_current_components = {
        component
        for components in parent_reservation.split_components.values()
        for component in components
    }
    if (
        len(current_components) != PARENT_SELECTED_COMPONENTS
        or current_components != recorded_current_components
        or prior_components & current_components
    ):
        raise ValueError("Parent selected component closure differs")
    excluded_components = prior_components | current_components
    if len(excluded_components) != 254 or len(component_rows) != 708:
        raise ValueError("Fresh residual exclusion inventory differs")

    pairs = _build_pairs(rows, source_to_component, excluded_components)
    assignments = {
        "mechanics": pairs[:MECHANICS_PAIRS],
        "causal": pairs[MECHANICS_PAIRS:],
    }
    if assignments != EXPECTED_PAIRS:
        raise RuntimeError("Fresh residual deterministic pair reservation differs")
    splits = {
        name: parent._split_payload(assignments[name], rows, source_to_component)
        for name in SPLIT_NAMES
    }
    for name in SPLIT_NAMES:
        if (
            splits[name]["source_indices_sha256"]
            != EXPECTED_SOURCE_INDICES_SHA256[name]
            or splits[name]["passage_component_ids_sha256"]
            != EXPECTED_COMPONENT_IDS_SHA256[name]
        ):
            raise RuntimeError(f"Fresh residual {name} split digest differs")
    selected_components = {
        component
        for split_payload in splits.values()
        for component in split_payload["passage_component_ids"]
    }
    if selected_components & excluded_components or len(selected_components) != 64:
        raise RuntimeError("Fresh residual split overlaps excluded passage components")

    payload: dict[str, Any] = {
        "schema": SCHEMA,
        "source": parent.SOURCE_BINDING.payload(),
        "selection": {
            "salt": SELECTION_SALT,
            "inputs": (
                "dataset-qualified source_index, row_sha256, gold_sha256, "
                "write_tokens, and normalized passage signature hashes only"
            ),
            "split_unit": "normalized_passage_32_character_shingle_connected_component",
            "pair_count": PAIR_COUNT,
            "split_assignment": {
                "mechanics": f"ranked pairs 0:{MECHANICS_PAIRS}",
                "causal": f"ranked pairs {MECHANICS_PAIRS}:{PAIR_COUNT}",
            },
        },
        "parent_reservation": {
            "manifest_schema": PARENT_MANIFEST_SCHEMA,
            "manifest_sha256": PARENT_MANIFEST_SHA256,
            "manifest_receipt": PARENT_MANIFEST_RECEIPT,
            "split_receipt": PARENT_SPLIT_RECEIPT,
            "historical_excluded_components": len(prior_components),
            "parent_selected_components": len(current_components),
            "all_parent_splits_excluded": list(PARENT_SPLIT_NAMES),
            "parent_bundle_bytes_read": False,
        },
        "consumed_mechanics": {
            "result_schema": CONSUMED_RESULT_SCHEMA,
            "result_status": CONSUMED_RESULT_STATUS,
            "result_sha256": CONSUMED_RESULT_SHA256,
            "result_receipt": CONSUMED_RESULT_RECEIPT,
            "mechanics_rows_opened": 32,
            "causal_rows_opened": 0,
            "result_bytes_read": True,
        },
        "splits": splits,
        "capture_authorization": {
            "default_open_splits": [],
            "sealed_inventory_only_splits": list(SPLIT_NAMES),
            "mechanics_open_requires_signed_protocol": True,
            "causal_open_requires_mechanics_pass_and_separate_signed_protocol": True,
        },
        "leakage_audit": {
            "source_rows": len(rows),
            "passage_component_count": len(component_rows),
            "historical_excluded_components": len(prior_components),
            "parent_selected_components": len(current_components),
            "total_excluded_components": len(excluded_components),
            "remaining_components_before_selection": len(component_rows)
            - len(excluded_components),
            "selected_source_rows": 64,
            "selected_passage_components": 64,
            "cross_split_passage_component_count": 0,
        },
        "donor_component_disjoint": True,
        "passage_component_disjoint": True,
        "sealed_bundle_bytes_read": False,
        "protected_splits_opened": [],
    }
    payload["receipt"] = {
        "algorithm": "sha256",
        "payload_scope": "canonical_split_without_receipt",
        "payload_sha256": canonical_sha256(payload),
    }
    return payload
