"""Leakage-safe metadata split for continuous RWKV write alignment."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from experiments.rethinking_rwkv_ms_gemma import prepare_natural_memory_gate


SCHEMA = "rwkv_ms_continuous_write_fit_split.v2"
SOURCE_NAMESPACE = "novel-agent-sft-dataset:publisher-train-derived-fit"
SOURCE_RELATIVE_PATH = (
    "experiments/rethinking_rwkv_ms_gemma/local_artifacts/"
    "natural_memory_native_development_v1/v4-scene-boundary-detection/"
    "train_derived_fit.jsonl"
)
SOURCE_SHA256 = "8b0552cf1ddd39230896ce1ed6a3842aef94212e70bbc9e76ee8f13c546e6e57"
SOURCE_ROWS = 1443

PRIOR_MANIFEST_SCHEMA = (
    "rwkv_ms_natural_memory_native_rwkv_bidirectional_sign_open_fit.v1"
)
PRIOR_MANIFEST_SHA256 = (
    "fbad372ad50295e9588c10bdeb40807def69e10216fb365acbe38c931aed7773"
)
PRIOR_SOURCE_INDICES_SHA256 = (
    "5b7204d39d54d55126303b8ef14a2498a27fa3f3408531c0f035eea6b300c6d5"
)
PRIOR_MAPPING_SHA256 = (
    "8b7c4893101c470964766f44efa5b48d62a07aff4c410a37d58c89fb216b4745"
)
PRIOR_RESERVED_ROWS = 98

SHINGLE_WIDTH = prepare_natural_memory_gate.SHINGLE_WIDTH
SELECTION_SALT = "rwkv-continuous-write-fit-components-v2:"
PAIR_COUNT = 80
FIT_PAIRS = 32
RETRIEVAL_PAIRS = 16
MECHANICS_PAIRS = 16
CAUSAL_PAIRS = 16
SPLIT_NAMES = ("fit", "retrieval", "mechanics", "causal")
CAPTURE_SPLITS = ("fit", "retrieval")
SEALED_SPLITS = ("mechanics", "causal")


def canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    )


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def normalize_passage(text: str) -> str:
    if not isinstance(text, str):
        raise ValueError("Passage text must be a string")
    return prepare_natural_memory_gate.normalize_passage(text)


def passage_signature_sha256s(text: str) -> tuple[str, ...]:
    if not isinstance(text, str):
        raise ValueError("Passage text must be a string")
    segments = prepare_natural_memory_gate._extract_segments("scene", text)
    signatures = prepare_natural_memory_gate._segments_for_component(segments)
    if not signatures:
        raise ValueError("Normalized passage must not be empty")
    return tuple(sorted(signatures))


@dataclass(frozen=True)
class DatasetBinding:
    namespace: str
    path: str
    sha256: str
    rows: int

    def qualified_id(self, source_index: int, row_sha256: str) -> str:
        if (
            isinstance(source_index, bool)
            or not isinstance(source_index, int)
            or not 0 <= source_index < self.rows
            or not _is_sha256(row_sha256)
        ):
            raise ValueError("Dataset-qualified source identity is invalid")
        return (
            f"{self.namespace}|dataset_sha256:{self.sha256}|"
            f"source_index:{source_index}|row_sha256:{row_sha256}"
        )

    def payload(self) -> dict[str, Any]:
        return {
            "namespace": self.namespace,
            "path": self.path,
            "sha256": self.sha256,
            "rows": self.rows,
        }


SOURCE_BINDING = DatasetBinding(
    namespace=SOURCE_NAMESPACE,
    path=SOURCE_RELATIVE_PATH,
    sha256=SOURCE_SHA256,
    rows=SOURCE_ROWS,
)


@dataclass(frozen=True)
class PriorReservation:
    source_indices: tuple[int, ...]
    mapping_pairs: tuple[tuple[int, int], ...]
    donor_component_closure: tuple[int, ...]
    manifest_sha256: str


def _validate_receipt(manifest: Mapping[str, Any]) -> None:
    receipt = manifest.get("receipt")
    if not isinstance(receipt, Mapping):
        raise ValueError("Prior manifest receipt is missing")
    unsigned = dict(manifest)
    unsigned.pop("receipt")
    expected = {
        "algorithm": "sha256",
        "payload_scope": "canonical_manifest_without_receipt",
        "payload_sha256": canonical_sha256(unsigned),
    }
    if dict(receipt) != expected:
        raise ValueError("Prior manifest receipt differs")


def _donor_component_closure(
    seeds: Sequence[int],
    mapping_pairs: Sequence[tuple[int, int]],
) -> tuple[int, ...]:
    graph: dict[int, set[int]] = {}
    for source, donor in mapping_pairs:
        graph.setdefault(source, set()).add(donor)
        graph.setdefault(donor, set()).add(source)
    closure = set(int(source) for source in seeds)
    frontier = list(closure)
    while frontier:
        source = frontier.pop()
        for donor in graph.get(source, ()):
            if donor not in closure:
                closure.add(donor)
                frontier.append(donor)
    return tuple(sorted(closure))


def validate_prior_manifest(
    manifest: Mapping[str, Any],
    *,
    manifest_sha256: str,
) -> PriorReservation:
    if manifest_sha256 != PRIOR_MANIFEST_SHA256:
        raise ValueError("Prior bidirectional manifest file hash differs")
    if manifest.get("schema") != PRIOR_MANIFEST_SCHEMA:
        raise ValueError("Prior bidirectional manifest schema differs")
    if manifest.get("protected_splits_opened") != []:
        raise ValueError("Prior bidirectional manifest opened protected splits")
    if manifest.get("source") != SOURCE_BINDING.payload():
        raise ValueError("Prior bidirectional manifest source binding differs")
    _validate_receipt(manifest)

    raw_sources = manifest.get("source_indices")
    raw_mapping = manifest.get("mapping_pairs")
    if not isinstance(raw_sources, list) or not isinstance(raw_mapping, list):
        raise ValueError("Prior bidirectional reservation metadata is missing")
    if any(isinstance(source, bool) or not isinstance(source, int) for source in raw_sources):
        raise ValueError("Prior bidirectional source index is invalid")
    source_indices = tuple(sorted(raw_sources))
    if (
        len(source_indices) != PRIOR_RESERVED_ROWS
        or len(set(source_indices)) != PRIOR_RESERVED_ROWS
        or canonical_sha256(list(source_indices)) != PRIOR_SOURCE_INDICES_SHA256
        or manifest.get("source_indices_sha256") != PRIOR_SOURCE_INDICES_SHA256
    ):
        raise ValueError("Prior bidirectional source reservation differs")

    mapping_pairs: list[tuple[int, int]] = []
    for pair in raw_mapping:
        if (
            not isinstance(pair, list)
            or len(pair) != 2
            or any(isinstance(value, bool) or not isinstance(value, int) for value in pair)
        ):
            raise ValueError("Prior bidirectional donor mapping is invalid")
        mapping_pairs.append((pair[0], pair[1]))
    if (
        len(mapping_pairs) != PRIOR_RESERVED_ROWS
        or tuple(source for source, _ in mapping_pairs) != source_indices
        or any(donor not in source_indices for _, donor in mapping_pairs)
        or canonical_sha256([list(pair) for pair in mapping_pairs])
        != PRIOR_MAPPING_SHA256
        or manifest.get("mapping_sha256") != PRIOR_MAPPING_SHA256
    ):
        raise ValueError("Prior bidirectional donor mapping differs")

    raw_splits = manifest.get("splits")
    if not isinstance(raw_splits, Mapping) or set(raw_splits) != {
        "development",
        "mechanics",
        "causal",
    }:
        raise ValueError("Prior bidirectional split metadata differs")
    split_sources: list[set[int]] = []
    for split_name in ("development", "mechanics", "causal"):
        split = raw_splits[split_name]
        if not isinstance(split, Mapping) or not isinstance(
            split.get("source_indices"), list
        ):
            raise ValueError("Prior bidirectional split sources are missing")
        split_sources.append(set(split["source_indices"]))
    if (
        any(
            left & right
            for index, left in enumerate(split_sources)
            for right in split_sources[index + 1 :]
        )
        or set().union(*split_sources) != set(source_indices)
    ):
        raise ValueError("Prior bidirectional split reservations overlap or differ")

    closure = _donor_component_closure(source_indices, mapping_pairs)
    if closure != source_indices:
        raise ValueError("Prior bidirectional donor-component closure differs")
    return PriorReservation(
        source_indices=source_indices,
        mapping_pairs=tuple(mapping_pairs),
        donor_component_closure=closure,
        manifest_sha256=manifest_sha256,
    )


def load_prior_reservation(manifest_path: Path) -> PriorReservation:
    if manifest_path.name != "manifest.json":
        raise ValueError("Prior reservation loader accepts only manifest.json")
    manifest_bytes = manifest_path.read_bytes()
    manifest_sha256 = sha256_bytes(manifest_bytes)
    try:
        manifest = json.loads(manifest_bytes.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("Prior bidirectional manifest is invalid JSON") from error
    if not isinstance(manifest, Mapping):
        raise ValueError("Prior bidirectional manifest root is not an object")
    return validate_prior_manifest(manifest, manifest_sha256=manifest_sha256)


def _validate_prior_reservation(prior: PriorReservation) -> None:
    source_indices = tuple(sorted(prior.source_indices))
    mapping_pairs = tuple(prior.mapping_pairs)
    if (
        prior.manifest_sha256 != PRIOR_MANIFEST_SHA256
        or len(source_indices) != PRIOR_RESERVED_ROWS
        or canonical_sha256(list(source_indices)) != PRIOR_SOURCE_INDICES_SHA256
        or tuple(source for source, _ in mapping_pairs) != source_indices
        or canonical_sha256([list(pair) for pair in mapping_pairs])
        != PRIOR_MAPPING_SHA256
        or prior.donor_component_closure
        != _donor_component_closure(source_indices, mapping_pairs)
        or prior.donor_component_closure != source_indices
    ):
        raise ValueError("Continuous-write prior reservation differs from its pins")


def _validated_metadata_rows(
    rows: Sequence[Mapping[str, Any]],
) -> tuple[dict[str, Any], ...]:
    if len(rows) != SOURCE_ROWS:
        raise ValueError(
            f"Continuous-write metadata requires {SOURCE_ROWS} rows, found {len(rows)}"
        )
    validated: list[dict[str, Any]] = []
    expected_fields = {
        "source_index",
        "row_sha256",
        "gold_sha256",
        "write_tokens",
        "passage_signature_sha256s",
    }
    for expected_index, row in enumerate(rows):
        if not isinstance(row, Mapping) or set(row) != expected_fields:
            raise ValueError("Continuous-write metadata row fields differ")
        if row["source_index"] != expected_index:
            raise ValueError("Continuous-write metadata indices are not contiguous")
        if not _is_sha256(row["row_sha256"]) or not _is_sha256(row["gold_sha256"]):
            raise ValueError("Continuous-write metadata digest is invalid")
        write_tokens = row["write_tokens"]
        if isinstance(write_tokens, bool) or not isinstance(write_tokens, int) or write_tokens < 1:
            raise ValueError("Continuous-write metadata token count is invalid")
        signatures = row["passage_signature_sha256s"]
        if (
            not isinstance(signatures, (list, tuple))
            or not signatures
            or list(signatures) != sorted(set(signatures))
            or any(
                not isinstance(signature, str)
                or signature.split(":", 1)[0]
                not in {"exact", "shingle", "whole", "whole_shingle"}
                or not _is_sha256(signature.split(":", 1)[-1])
                for signature in signatures
            )
            or not any(signature.startswith("exact:") for signature in signatures)
            or not any(signature.startswith("shingle:") for signature in signatures)
        ):
            raise ValueError("Continuous-write passage signatures are invalid")
        validated.append(
            {
                **dict(row),
                "passage_signature_sha256s": tuple(signatures),
            }
        )
    return tuple(validated)


def _qualified_id(row: Mapping[str, Any]) -> str:
    return SOURCE_BINDING.qualified_id(
        int(row["source_index"]),
        str(row["row_sha256"]),
    )


def _passage_components(
    rows: Sequence[Mapping[str, Any]],
) -> tuple[dict[int, str], dict[str, tuple[int, ...]]]:
    parent = list(range(len(rows)))

    def find(source: int) -> int:
        while parent[source] != source:
            parent[source] = parent[parent[source]]
            source = parent[source]
        return source

    def union(left: int, right: int) -> None:
        left_root = find(left)
        right_root = find(right)
        if left_root != right_root:
            smaller, larger = sorted((left_root, right_root))
            parent[larger] = smaller

    owner: dict[str, int] = {}
    for row in rows:
        source = int(row["source_index"])
        for signature in row["passage_signature_sha256s"]:
            prior = owner.setdefault(signature, source)
            union(source, prior)

    by_root: dict[int, list[int]] = {}
    for source in range(len(rows)):
        by_root.setdefault(find(source), []).append(source)
    source_to_component: dict[int, str] = {}
    component_rows: dict[str, tuple[int, ...]] = {}
    for sources in by_root.values():
        ordered = tuple(sorted(sources))
        qualified = [_qualified_id(rows[source]) for source in ordered]
        component_id = "passage_component:" + canonical_sha256(qualified)
        component_rows[component_id] = ordered
        for source in ordered:
            source_to_component[source] = component_id
    if len(source_to_component) != len(rows):
        raise RuntimeError("Continuous-write passage component coverage differs")
    return source_to_component, component_rows


def _salted_row_rank(row: Mapping[str, Any]) -> tuple[str, str]:
    qualified = _qualified_id(row)
    digest = hashlib.sha256((SELECTION_SALT + qualified).encode("ascii")).hexdigest()
    return digest, qualified


def _qualified_pair(
    pair: tuple[int, int],
    rows: Sequence[Mapping[str, Any]],
) -> tuple[str, str]:
    return tuple(_qualified_id(rows[source]) for source in pair)


def _pair_rank(
    pair: tuple[int, int],
    rows: Sequence[Mapping[str, Any]],
) -> tuple[str, tuple[str, str]]:
    qualified = _qualified_pair(pair, rows)
    digest = hashlib.sha256(
        (SELECTION_SALT + canonical_json(list(qualified))).encode("ascii")
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
            raise ValueError(f"Continuous-write source has no eligible donor: {source_index}")
        donor = min(
            donors,
            key=lambda candidate: (
                abs(int(candidate["write_tokens"]) - int(source["write_tokens"])),
                _salted_row_rank(candidate),
            ),
        )
        donor_index = int(donor["source_index"])
        donor_component = source_to_component[donor_index]
        pair = tuple(sorted((source_index, donor_index)))
        pairs.append(pair)
        used_components.update((source_component, donor_component))
        if len(pairs) == PAIR_COUNT:
            break
    if len(pairs) != PAIR_COUNT:
        raise ValueError(f"Expected {PAIR_COUNT} continuous-write pairs, found {len(pairs)}")
    if len(used_components) != 2 * PAIR_COUNT:
        raise RuntimeError("Continuous-write donor pairs reuse passage components")
    return tuple(sorted(pairs, key=lambda pair: _pair_rank(pair, rows)))


def _split_payload(
    pairs: Sequence[tuple[int, int]],
    rows: Sequence[Mapping[str, Any]],
    source_to_component: Mapping[int, str],
) -> dict[str, Any]:
    source_indices = sorted(source for pair in pairs for source in pair)
    mapping: dict[int, int] = {}
    for left, right in pairs:
        mapping[left] = right
        mapping[right] = left
    mapping_pairs = [[source, mapping[source]] for source in source_indices]
    qualified_sources = [_qualified_id(rows[source]) for source in source_indices]
    qualified_mapping = [
        [_qualified_id(rows[source]), _qualified_id(rows[mapping[source]])]
        for source in source_indices
    ]
    component_ids = [source_to_component[source] for source in source_indices]
    donor_component_pairs = [
        [source_to_component[left], source_to_component[right]]
        for left, right in pairs
    ]
    if len(set(component_ids)) != len(component_ids) or any(
        left == right for left, right in donor_component_pairs
    ):
        raise RuntimeError("Continuous-write split reuses a passage component")
    return {
        "pair_count": len(pairs),
        "pairs": [list(pair) for pair in pairs],
        "source_indices": source_indices,
        "source_indices_sha256": canonical_sha256(source_indices),
        "mapping_pairs": mapping_pairs,
        "mapping_pairs_sha256": canonical_sha256(mapping_pairs),
        "qualified_source_ids": qualified_sources,
        "qualified_source_ids_sha256": canonical_sha256(qualified_sources),
        "qualified_mapping_pairs": qualified_mapping,
        "qualified_mapping_pairs_sha256": canonical_sha256(qualified_mapping),
        "passage_component_ids": component_ids,
        "passage_component_ids_sha256": canonical_sha256(component_ids),
        "donor_component_pairs": donor_component_pairs,
        "donor_component_pairs_sha256": canonical_sha256(donor_component_pairs),
    }


def _prior_payload(
    prior: PriorReservation,
    rows: Sequence[Mapping[str, Any]],
    excluded_component_sources: Sequence[int],
) -> dict[str, Any]:
    manifest_qualified = [_qualified_id(rows[source]) for source in prior.source_indices]
    excluded_qualified = [
        _qualified_id(rows[source]) for source in excluded_component_sources
    ]
    return {
        "source": SOURCE_BINDING.payload(),
        "manifest_schema": PRIOR_MANIFEST_SCHEMA,
        "manifest_sha256": prior.manifest_sha256,
        "manifest_source_indices": list(prior.source_indices),
        "manifest_source_indices_sha256": canonical_sha256(list(prior.source_indices)),
        "manifest_mapping_pairs": [list(pair) for pair in prior.mapping_pairs],
        "manifest_mapping_sha256": canonical_sha256(
            [list(pair) for pair in prior.mapping_pairs]
        ),
        "manifest_qualified_source_ids": manifest_qualified,
        "manifest_qualified_source_ids_sha256": canonical_sha256(manifest_qualified),
        "excluded_passage_component_sources": list(excluded_component_sources),
        "excluded_passage_component_sources_sha256": canonical_sha256(
            list(excluded_component_sources)
        ),
        "excluded_qualified_source_ids": excluded_qualified,
        "excluded_qualified_source_ids_sha256": canonical_sha256(excluded_qualified),
        "manifest_reserved_rows": len(prior.source_indices),
        "excluded_component_rows": len(excluded_component_sources),
        "bundle_bytes_read": False,
    }


def build_split(
    metadata_rows: Sequence[Mapping[str, Any]],
    prior: PriorReservation,
) -> dict[str, Any]:
    rows = _validated_metadata_rows(metadata_rows)
    _validate_prior_reservation(prior)
    source_to_component, component_rows = _passage_components(rows)
    excluded_components = {
        source_to_component[source] for source in prior.donor_component_closure
    }
    excluded_component_sources = tuple(
        sorted(
            source
            for component in excluded_components
            for source in component_rows[component]
        )
    )
    if not set(prior.source_indices).issubset(excluded_component_sources):
        raise RuntimeError("Continuous-write prior passage closure lost reserved rows")

    pairs = _build_pairs(rows, source_to_component, excluded_components)
    retrieval = pairs[:RETRIEVAL_PAIRS]
    mechanics = pairs[RETRIEVAL_PAIRS : RETRIEVAL_PAIRS + MECHANICS_PAIRS]
    causal = pairs[
        RETRIEVAL_PAIRS
        + MECHANICS_PAIRS : RETRIEVAL_PAIRS
        + MECHANICS_PAIRS
        + CAUSAL_PAIRS
    ]
    fit = pairs[RETRIEVAL_PAIRS + MECHANICS_PAIRS + CAUSAL_PAIRS :]
    if len(fit) != FIT_PAIRS:
        raise RuntimeError("Continuous-write fit pair count differs")
    assignments = {
        "fit": fit,
        "retrieval": retrieval,
        "mechanics": mechanics,
        "causal": causal,
    }
    splits = {
        name: _split_payload(assignments[name], rows, source_to_component)
        for name in SPLIT_NAMES
    }

    selected_sources = {
        source
        for split_payload in splits.values()
        for source in split_payload["source_indices"]
    }
    selected_components = {
        component
        for split_payload in splits.values()
        for component in split_payload["passage_component_ids"]
    }
    if (
        selected_sources & set(excluded_component_sources)
        or len(selected_sources) != 2 * PAIR_COUNT
        or len(selected_components) != 2 * PAIR_COUNT
    ):
        raise RuntimeError("Continuous-write split overlaps or reuses passage components")
    split_component_sets = [
        set(splits[name]["passage_component_ids"]) for name in SPLIT_NAMES
    ]
    if any(
        left & right
        for index, left in enumerate(split_component_sets)
        for right in split_component_sets[index + 1 :]
    ):
        raise RuntimeError("Continuous-write passage components cross partitions")
    split_signatures = []
    for name in SPLIT_NAMES:
        signatures = {
            signature
            for source in splits[name]["source_indices"]
            for signature in rows[source]["passage_signature_sha256s"]
        }
        split_signatures.append(signatures)
    cross_split_signatures = set().union(
        *(
            left & right
            for index, left in enumerate(split_signatures)
            for right in split_signatures[index + 1 :]
        )
    )
    if cross_split_signatures:
        raise RuntimeError("Normalized passage shingles cross continuous-write partitions")

    payload: dict[str, Any] = {
        "schema": SCHEMA,
        "source": SOURCE_BINDING.payload(),
        "selection": {
            "salt": SELECTION_SALT,
            "inputs": (
                "dataset-qualified source_index, row_sha256, gold_sha256, "
                "write_tokens, and normalized passage signature hashes only"
            ),
            "normalization": "NFKC, remove all whitespace, casefold",
            "shingle_width": SHINGLE_WIDTH,
            "split_unit": "normalized_passage_32_character_shingle_connected_component",
            "pair_count": PAIR_COUNT,
            "donor_pair_rule": (
                "each pair joins two distinct unused passage components; both components "
                "are assigned atomically to one partition"
            ),
            "component_order": (
                "sha256(salt + canonical_json(dataset-qualified donor pair)), then pair"
            ),
            "split_assignment": {
                "retrieval": f"ranked pairs 0:{RETRIEVAL_PAIRS}",
                "mechanics": (
                    f"ranked pairs {RETRIEVAL_PAIRS}:"
                    f"{RETRIEVAL_PAIRS + MECHANICS_PAIRS}"
                ),
                "causal": (
                    f"ranked pairs {RETRIEVAL_PAIRS + MECHANICS_PAIRS}:"
                    f"{RETRIEVAL_PAIRS + MECHANICS_PAIRS + CAUSAL_PAIRS}"
                ),
                "fit": (
                    f"ranked pairs {RETRIEVAL_PAIRS + MECHANICS_PAIRS + CAUSAL_PAIRS}:"
                    f"{PAIR_COUNT}"
                ),
            },
        },
        "prior_reservation": _prior_payload(
            prior,
            rows,
            excluded_component_sources,
        ),
        "splits": splits,
        "capture_authorization": {
            "capture_splits": list(CAPTURE_SPLITS),
            "sealed_inventory_only_splits": list(SEALED_SPLITS),
            "captured_rows": sum(
                len(splits[name]["source_indices"]) for name in CAPTURE_SPLITS
            ),
            "sealed_rows": sum(
                len(splits[name]["source_indices"]) for name in SEALED_SPLITS
            ),
        },
        "leakage_audit": {
            "source_rows": len(rows),
            "passage_component_count": len(component_rows),
            "manifest_reserved_rows": len(prior.source_indices),
            "excluded_components_touching_manifest_sources": len(excluded_components),
            "excluded_component_rows": len(excluded_component_sources),
            "selected_source_rows": len(selected_sources),
            "selected_passage_components": len(selected_components),
            "cross_split_passage_component_count": 0,
            "cross_split_normalized_32_character_shingle_overlap": 0,
        },
        "donor_component_disjoint": True,
        "passage_component_disjoint": True,
        "plmsc_namespace_consulted": False,
        "sealed_bundle_bytes_read": False,
        "protected_splits_opened": [],
    }
    payload["receipt"] = {
        "algorithm": "sha256",
        "payload_scope": "canonical_split_without_receipt",
        "payload_sha256": canonical_sha256(payload),
    }
    return payload
