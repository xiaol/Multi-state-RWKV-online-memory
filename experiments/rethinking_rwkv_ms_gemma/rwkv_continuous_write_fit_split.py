"""Dataset-qualified metadata split for continuous RWKV write alignment."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence


SCHEMA = "rwkv_ms_continuous_write_fit_split.v1"
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

SELECTION_SALT = "rwkv-continuous-write-fit-components-v1:"
PAIR_COUNT = 64
FIT_PAIRS = 32
MECHANICS_PAIRS = 16
CAUSAL_PAIRS = 16
SPLIT_NAMES = ("fit", "mechanics", "causal")


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


@dataclass(frozen=True)
class DatasetBinding:
    namespace: str
    path: str
    sha256: str
    rows: int

    def qualified_id(self, source_index: int) -> str:
        if (
            isinstance(source_index, bool)
            or not isinstance(source_index, int)
            or not 0 <= source_index < self.rows
        ):
            raise ValueError("Dataset-qualified source index is outside the source")
        return (
            f"{self.namespace}|sha256:{self.sha256}|"
            f"source_index:{source_index}"
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
    component_closure: tuple[int, ...]
    qualified_source_ids: tuple[str, ...]
    manifest_sha256: str

    def payload(self) -> dict[str, Any]:
        return {
            "source": SOURCE_BINDING.payload(),
            "manifest_schema": PRIOR_MANIFEST_SCHEMA,
            "manifest_sha256": self.manifest_sha256,
            "source_indices": list(self.source_indices),
            "source_indices_sha256": canonical_sha256(list(self.source_indices)),
            "mapping_pairs": [list(pair) for pair in self.mapping_pairs],
            "mapping_sha256": canonical_sha256(
                [list(pair) for pair in self.mapping_pairs]
            ),
            "component_closure": list(self.component_closure),
            "component_closure_sha256": canonical_sha256(
                list(self.component_closure)
            ),
            "qualified_source_ids": list(self.qualified_source_ids),
            "qualified_source_ids_sha256": canonical_sha256(
                list(self.qualified_source_ids)
            ),
            "reserved_rows": len(self.component_closure),
            "bundle_bytes_read": False,
        }


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


def _component_closure(
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
        any(left & right for index, left in enumerate(split_sources) for right in split_sources[index + 1 :])
        or set().union(*split_sources) != set(source_indices)
    ):
        raise ValueError("Prior bidirectional split reservations overlap or differ")

    closure = _component_closure(source_indices, mapping_pairs)
    if closure != source_indices:
        raise ValueError("Prior bidirectional donor-component closure differs")
    qualified = tuple(SOURCE_BINDING.qualified_id(source) for source in closure)
    return PriorReservation(
        source_indices=source_indices,
        mapping_pairs=tuple(mapping_pairs),
        component_closure=closure,
        qualified_source_ids=qualified,
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
    expected_qualified = tuple(
        SOURCE_BINDING.qualified_id(source) for source in source_indices
    )
    if (
        prior.manifest_sha256 != PRIOR_MANIFEST_SHA256
        or len(source_indices) != PRIOR_RESERVED_ROWS
        or canonical_sha256(list(source_indices)) != PRIOR_SOURCE_INDICES_SHA256
        or tuple(source for source, _ in mapping_pairs) != source_indices
        or canonical_sha256([list(pair) for pair in mapping_pairs])
        != PRIOR_MAPPING_SHA256
        or prior.component_closure
        != _component_closure(source_indices, mapping_pairs)
        or prior.component_closure != source_indices
        or prior.qualified_source_ids != expected_qualified
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
    for expected_index, row in enumerate(rows):
        if not isinstance(row, Mapping) or set(row) != {
            "source_index",
            "row_sha256",
            "gold_sha256",
            "write_tokens",
        }:
            raise ValueError("Continuous-write metadata row fields differ")
        if row["source_index"] != expected_index:
            raise ValueError("Continuous-write metadata indices are not contiguous")
        if not _is_sha256(row["row_sha256"]) or not _is_sha256(row["gold_sha256"]):
            raise ValueError("Continuous-write metadata digest is invalid")
        write_tokens = row["write_tokens"]
        if isinstance(write_tokens, bool) or not isinstance(write_tokens, int) or write_tokens < 1:
            raise ValueError("Continuous-write metadata token count is invalid")
        validated.append(dict(row))
    return tuple(validated)


def _salted_row_rank(row: Mapping[str, Any]) -> tuple[str, int]:
    digest = hashlib.sha256(
        (SELECTION_SALT + str(row["row_sha256"])).encode("ascii")
    ).hexdigest()
    return digest, int(row["source_index"])


def _qualified_pair(pair: tuple[int, int]) -> tuple[str, str]:
    return tuple(SOURCE_BINDING.qualified_id(source) for source in pair)


def _component_rank(pair: tuple[int, int]) -> tuple[str, tuple[str, str]]:
    qualified = _qualified_pair(pair)
    digest = hashlib.sha256(
        (SELECTION_SALT + canonical_json(list(qualified))).encode("ascii")
    ).hexdigest()
    return digest, qualified


def _build_pairs(
    rows: Sequence[Mapping[str, Any]],
    excluded: set[int],
) -> tuple[tuple[int, int], ...]:
    eligible = [row for row in rows if int(row["source_index"]) not in excluded]
    ordered = sorted(eligible, key=_salted_row_rank)
    used: set[int] = set()
    pairs: list[tuple[int, int]] = []
    for source in ordered:
        source_index = int(source["source_index"])
        if source_index in used:
            continue
        donors = [
            donor
            for donor in eligible
            if int(donor["source_index"]) not in used
            and int(donor["source_index"]) != source_index
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
        pair = tuple(sorted((source_index, int(donor["source_index"]))))
        pairs.append(pair)
        used.update(pair)
        if len(pairs) == PAIR_COUNT:
            break
    if len(pairs) != PAIR_COUNT:
        raise ValueError(f"Expected {PAIR_COUNT} continuous-write pairs, found {len(pairs)}")
    return tuple(sorted(pairs, key=_component_rank))


def _split_payload(
    pairs: Sequence[tuple[int, int]],
) -> dict[str, Any]:
    source_indices = sorted(source for pair in pairs for source in pair)
    mapping: dict[int, int] = {}
    for left, right in pairs:
        mapping[left] = right
        mapping[right] = left
    mapping_pairs = [[source, mapping[source]] for source in source_indices]
    qualified_sources = [SOURCE_BINDING.qualified_id(source) for source in source_indices]
    qualified_mapping = [
        [SOURCE_BINDING.qualified_id(source), SOURCE_BINDING.qualified_id(mapping[source])]
        for source in source_indices
    ]
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
    }


def build_split(
    metadata_rows: Sequence[Mapping[str, Any]],
    prior: PriorReservation,
) -> dict[str, Any]:
    rows = _validated_metadata_rows(metadata_rows)
    _validate_prior_reservation(prior)
    excluded = set(prior.component_closure)
    pairs = _build_pairs(rows, excluded)
    mechanics = pairs[:MECHANICS_PAIRS]
    causal = pairs[MECHANICS_PAIRS : MECHANICS_PAIRS + CAUSAL_PAIRS]
    fit = pairs[MECHANICS_PAIRS + CAUSAL_PAIRS :]
    if len(fit) != FIT_PAIRS:
        raise RuntimeError("Continuous-write fit pair count differs")
    assignments = {"fit": fit, "mechanics": mechanics, "causal": causal}
    splits = {name: _split_payload(assignments[name]) for name in SPLIT_NAMES}
    selected = {
        source
        for split in splits.values()
        for source in split["source_indices"]
    }
    if selected & excluded or len(selected) != 2 * PAIR_COUNT:
        raise RuntimeError("Continuous-write split overlaps prior reservations")
    split_sets = [set(splits[name]["source_indices"]) for name in SPLIT_NAMES]
    if any(
        left & right
        for index, left in enumerate(split_sets)
        for right in split_sets[index + 1 :]
    ):
        raise RuntimeError("Continuous-write donor components cross split boundaries")

    payload: dict[str, Any] = {
        "schema": SCHEMA,
        "source": SOURCE_BINDING.payload(),
        "selection": {
            "salt": SELECTION_SALT,
            "inputs": (
                "dataset-qualified source_index, row_sha256, gold_sha256, "
                "and write_tokens only"
            ),
            "pair_count": PAIR_COUNT,
            "component_order": (
                "sha256(salt + canonical_json(dataset-qualified pair)), then pair"
            ),
            "split_assignment": {
                "mechanics": f"ranked pairs 0:{MECHANICS_PAIRS}",
                "causal": (
                    f"ranked pairs {MECHANICS_PAIRS}:"
                    f"{MECHANICS_PAIRS + CAUSAL_PAIRS}"
                ),
                "fit": (
                    f"ranked pairs {MECHANICS_PAIRS + CAUSAL_PAIRS}:"
                    f"{PAIR_COUNT}"
                ),
            },
        },
        "prior_reservation": prior.payload(),
        "splits": splits,
        "donor_component_disjoint": True,
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
