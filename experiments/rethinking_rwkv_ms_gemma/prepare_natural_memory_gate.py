"""Build a passage-disjoint causal memory gate from Novel Agent TRAIN rows.

This source builder deliberately does not know anything about model tensors or
the optimizer.  It emits natural key/value records and the matched state
controls consumed by the projected-KV trainer.  The only permitted corpus
inputs are the three pinned ``train.jsonl`` files; validation and test files
are rejected by path and are never opened.

The builder has two useful properties for the causal experiment:

* passage components, rather than rows, are assigned to splits; and
* a read prompt contains only the natural address/context.  It never contains
  the serialized answer or a memory value.

The output is JSONL plus a deterministic manifest.  No timestamp or random
UUID is written, so rebuilding the same source with the same protocol produces
the same bytes.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import os
import re
import sys
import unicodedata
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


SCHEMA = "novel_natural_causal_memory_gate.v2"
SEALED_LOCK_SCHEMA = "novel_natural_causal_memory_gate.sealed_lock.v2"
HF_MIRROR = "https://hf-mirror.com"
DEFAULT_SOURCE_ROOT = Path(
    os.environ.get(
        "NOVEL_AGENT_TRAIN_ROOT",
        "/root/X/.cache/hf/novel-agent-sft-dataset/training",
    )
)
DEFAULT_MODEL_ID = "google/gemma-4-E4B-it"
DEFAULT_MODEL_REVISION = "a4c2d58be94dda072b918d9db64ee85c8ed34e3f"
DEFAULT_MODEL_PATH = Path("/root/X/.cache/hf/gemma-4-E4B-it-a4c2d58")
DEFAULT_EPISODES_PER_TASK = 32
BUILD_PROFILES = {
    "development": ("train", "development"),
    "sealed_validation": ("sealed_validation",),
}
SHINGLE_WIDTH = 32
MODEL_RUNTIME_ARTIFACTS = (
    "config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "chat_template.jinja",
)
SPLITS = ("train", "development", "sealed_validation")
SPLIT_FRACTIONS = {"train": 0.60, "development": 0.20, "sealed_validation": 0.20}
SPLIT_SEED = 20260806

SOURCE_DEFINITIONS: dict[str, dict[str, Any]] = {
    "attribution": {
        "relative_path": "v3.2-attribution-best-candidate/train.jsonl",
        "sha256": "8c87ca259c2b0c19cc545bee4b32f342448d43826b2936817fd9792ea258216a",
    },
    "narrative": {
        "relative_path": "v3.2-narrative-type-classification/train.jsonl",
        "sha256": "3f7b2ffd2c5a48921a0e114be361f0127ca6e2713b1a94b53b406c5c2f084c0e",
    },
    "scene": {
        "relative_path": "v4-scene-boundary-detection/train.jsonl",
        "sha256": "785fe54c0a4e5c64e33f64f9bc88d64719576407c21eb0d520f9dec5a59b8e22",
    },
}

_SEGMENT_PATTERNS = {
    "attribution": re.compile(
        r"(?ms)(?:^|\n)\s*\[(\d+)\]\s*(.*?)(?=(?:\n\s*\[\d+\]\s*)|\Z)"
    ),
    "narrative": re.compile(
        r"(?ms)(?:^|\n)\s*\[(\d+)\]\s*(.*?)(?=(?:\n\s*\[\d+\]\s*)|\Z)"
    ),
    "scene": re.compile(
        r"(?ms)(?:^|\n)\s*\[P(\d+)\]\s*(.*?)(?=(?:\n\s*\[P\d+\]\s*)|\Z)"
    ),
}
_ANSWER_KEYS = {
    "attribution": frozenset({"best_candidate", "uncertain"}),
    "narrative": frozenset({"labels"}),
    "scene": frozenset({"boundaries"}),
}
_NARRATIVE_TYPES = (
    "dialogue",
    "narration",
    "thought",
    "action",
    "scene_description",
)


def canonical_json(value: Any) -> str:
    """Return the byte-stable JSON representation used in all hashes."""

    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_text(payload: str) -> str:
    return sha256_bytes(payload.encode("utf-8"))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def normalize_passage(text: str) -> str:
    """Normalize only for leakage identity, never for model-visible text."""

    return re.sub(r"\s+", "", unicodedata.normalize("NFKC", text)).casefold()


def rolling_shingles(text: str, width: int = SHINGLE_WIDTH) -> set[str]:
    normalized = normalize_passage(text)
    if not normalized:
        return set()
    if len(normalized) <= width:
        return {normalized}
    return {normalized[index : index + width] for index in range(len(normalized) - width + 1)}


def _forbidden_source_path(path: Path) -> bool:
    """Reject protected splits before opening a candidate source path."""

    parts = {part.casefold() for part in path.parts}
    if parts.intersection({"test", "tests", "val", "validation", "hard32"}):
        return True
    return path.name.casefold() != "train.jsonl"


def require_hf_mirror(endpoint: str | None = None) -> str:
    configured = os.environ.get("HF_ENDPOINT")
    if configured is not None and configured.rstrip("/") != HF_MIRROR:
        raise ValueError(
            f"HF_ENDPOINT must be {HF_MIRROR!r}; environment has {configured!r}"
        )
    chosen = endpoint or configured or HF_MIRROR
    chosen = chosen.rstrip("/")
    if chosen != HF_MIRROR:
        raise ValueError(
            f"This protocol is pinned to HF mirror {HF_MIRROR!r}; got {chosen!r}"
        )
    os.environ["HF_ENDPOINT"] = HF_MIRROR
    return HF_MIRROR


def default_source_paths(root: Path = DEFAULT_SOURCE_ROOT) -> dict[str, Path]:
    return {
        task: root / str(spec["relative_path"])
        for task, spec in SOURCE_DEFINITIONS.items()
    }


def _validate_message_row(row: Any, path: Path, line_number: int) -> tuple[str, str, str]:
    location = f"{path}:{line_number}"
    if not isinstance(row, Mapping):
        raise ValueError(f"Expected JSON object at {location}")
    messages = row.get("messages")
    if not isinstance(messages, list) or len(messages) != 3:
        raise ValueError(f"Expected three messages at {location}")
    roles = tuple(message.get("role") if isinstance(message, Mapping) else None for message in messages)
    if roles != ("system", "user", "assistant"):
        raise ValueError(f"Expected system/user/assistant roles at {location}; got {roles!r}")
    contents: list[str] = []
    for message in messages:
        content = message.get("content")
        if not isinstance(content, str):
            raise ValueError(f"Message content is not text at {location}")
        contents.append(content)
    return contents[0], contents[1], contents[2]


def _parse_json_object(text: str, *, location: str) -> Any:
    candidate = text.strip()
    try:
        return json.loads(candidate)
    except json.JSONDecodeError:
        # A few SFT exports wrap JSON in a markdown fence.  Extract the first
        # balanced object without accepting arbitrary prose as an answer.
        start = candidate.find("{")
        end = candidate.rfind("}")
        if start < 0 or end <= start:
            raise ValueError(f"Assistant response is not JSON at {location}")
        try:
            return json.loads(candidate[start : end + 1])
        except json.JSONDecodeError as exc:
            raise ValueError(f"Assistant response is not JSON at {location}") from exc


def _extract_segments(task: str, user_text: str) -> list[dict[str, Any]]:
    matches = list(_SEGMENT_PATTERNS[task].finditer(user_text))
    segments: list[dict[str, Any]] = []
    for match in matches:
        text = match.group(2).strip()
        if text:
            segments.append({"ordinal": int(match.group(1)), "text": text})
    if not segments:
        segments = [{"ordinal": 1, "text": user_text.strip()}]
    return segments


def _segments_for_component(segments: Sequence[Mapping[str, Any]]) -> set[str]:
    signatures: set[str] = set()
    joined: list[str] = []
    for segment in segments:
        text = str(segment["text"])
        normalized = normalize_passage(text)
        joined.append(normalized)
        if normalized:
            signatures.add("exact:" + sha256_text(normalized))
        for shingle in rolling_shingles(text):
            signatures.add("shingle:" + sha256_text(shingle))
    whole = "".join(joined)
    if whole:
        signatures.add("whole:" + sha256_text(whole))
        for index in range(max(0, len(whole) - SHINGLE_WIDTH + 1)):
            shingle = whole[index : index + SHINGLE_WIDTH]
            if len(shingle) == SHINGLE_WIDTH:
                signatures.add("whole_shingle:" + sha256_text(shingle))
    return signatures


def _canonical_row_id(task: str, row: Mapping[str, Any]) -> str:
    return sha256_text(task + "\n" + canonical_json(row))


def _validate_value(task: str, value: Any, segments: Sequence[Mapping[str, Any]], location: str) -> None:
    if not isinstance(value, Mapping) or set(value) != set(_ANSWER_KEYS[task]):
        raise ValueError(f"Unexpected {task} answer shape at {location}")
    if task == "attribution":
        if not isinstance(value["best_candidate"], str) or not isinstance(value["uncertain"], bool):
            raise ValueError(f"Invalid attribution answer at {location}")
    elif task == "narrative":
        labels = value["labels"]
        if not isinstance(labels, list):
            raise ValueError(f"Invalid narrative labels at {location}")
        ids = [segment["ordinal"] for segment in segments]
        got_ids: list[int] = []
        for label in labels:
            if not isinstance(label, Mapping) or set(label) != {"unit_id", "type"}:
                raise ValueError(f"Invalid narrative label at {location}")
            try:
                unit_id = int(label["unit_id"])
            except (TypeError, ValueError) as exc:
                raise ValueError(f"Invalid narrative unit id at {location}") from exc
            if label["type"] not in _NARRATIVE_TYPES:
                raise ValueError(f"Invalid narrative type at {location}")
            got_ids.append(unit_id)
        if sorted(got_ids) != sorted(ids):
            raise ValueError(f"Narrative labels do not match units at {location}")
    else:
        boundaries = value["boundaries"]
        if not isinstance(boundaries, list) or any(
            not isinstance(boundary, int) or isinstance(boundary, bool) for boundary in boundaries
        ):
            raise ValueError(f"Invalid scene boundaries at {location}")
        maximum = max(0, len(segments) - 1)
        if any(boundary < 1 or boundary > maximum for boundary in boundaries):
            raise ValueError(f"Scene boundary outside [1, {maximum}] at {location}")
        if boundaries != sorted(set(boundaries)):
            raise ValueError(f"Scene boundaries are not sorted and unique at {location}")


@dataclass(frozen=True)
class RawRow:
    task: str
    source_path: str
    source_line: int
    row_ordinal: int
    row_id: str
    system_text: str
    user_text: str
    assistant_text: str
    value: Mapping[str, Any]
    value_json: str
    segments: tuple[Mapping[str, Any], ...]
    signatures: frozenset[str]


@dataclass(frozen=True)
class Item:
    task: str
    source_path: str
    source_line: int
    row_ordinal: int
    row_id: str
    item_id: str
    component_id: str
    target_ordinal: int
    address_text: str
    value: Mapping[str, Any]
    value_json: str
    assistant_text: str
    metadata: Mapping[str, Any]


class _UnionFind:
    def __init__(self, size: int) -> None:
        self.parent = list(range(size))
        self.rank = [0] * size

    def find(self, value: int) -> int:
        parent = self.parent
        while parent[value] != value:
            parent[value] = parent[parent[value]]
            value = parent[value]
        return value

    def union(self, left: int, right: int) -> None:
        root_left = self.find(left)
        root_right = self.find(right)
        if root_left == root_right:
            return
        if self.rank[root_left] < self.rank[root_right]:
            root_left, root_right = root_right, root_left
        self.parent[root_right] = root_left
        if self.rank[root_left] == self.rank[root_right]:
            self.rank[root_left] += 1


def _load_raw_rows(
    source_paths: Mapping[str, Path],
    *,
    enforce_pinned_sources: bool,
) -> tuple[list[RawRow], list[dict[str, Any]]]:
    rows: list[RawRow] = []
    source_stats: list[dict[str, Any]] = []
    for task in ("attribution", "narrative", "scene"):
        requested_path = Path(source_paths[task]).expanduser()
        if _forbidden_source_path(requested_path):
            raise ValueError(f"Only train.jsonl sources are allowed: {requested_path}")
        if requested_path.is_symlink():
            raise ValueError(f"Symbolic-link dataset sources are forbidden: {requested_path}")
        if not requested_path.is_file():
            raise FileNotFoundError(requested_path)
        path = requested_path.resolve(strict=True)
        if _forbidden_source_path(path):
            raise ValueError(f"Resolved source is not a TRAIN file: {path}")
        digest = sha256_file(path)
        expected = SOURCE_DEFINITIONS[task]["sha256"]
        if enforce_pinned_sources and digest != expected:
            raise ValueError(
                f"Pinned {task} TRAIN checksum mismatch: expected {expected}, got {digest}"
            )
        physical_lines = 0
        nonblank = 0
        for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            physical_lines += 1
            if not line.strip():
                continue
            nonblank += 1
            row = json.loads(line)
            system_text, user_text, assistant_text = _validate_message_row(row, path, line_number)
            value = _parse_json_object(assistant_text, location=f"{path}:{line_number}")
            segments = tuple(_extract_segments(task, user_text))
            _validate_value(task, value, segments, f"{path}:{line_number}")
            rows.append(
                RawRow(
                    task=task,
                    source_path=str(path.resolve()),
                    source_line=line_number,
                    row_ordinal=nonblank - 1,
                    row_id=_canonical_row_id(task, row),
                    system_text=system_text,
                    user_text=user_text,
                    assistant_text=assistant_text,
                    value=value,
                    value_json=canonical_json(value),
                    segments=segments,
                    signatures=frozenset(_segments_for_component(segments)),
                )
            )
        source_stats.append(
            {
                "task": task,
                "path": str(path.resolve()),
                "sha256": digest,
                "expected_sha256": expected,
                "physical_lines": physical_lines,
                "nonblank_rows": nonblank,
                "pinned": digest == expected,
            }
        )
    return rows, source_stats


def _component_assignments(
    rows: Sequence[RawRow],
) -> tuple[dict[str, str], dict[str, list[str]], dict[str, Any]]:
    if not rows:
        return {}, {}, {
            "unique_signatures": 0,
            "unique_signatures_by_kind": {},
            "cross_component_signature_overlap_count": 0,
            "signature_components_atomic": True,
        }
    uf = _UnionFind(len(rows))
    first_by_signature: dict[str, int] = {}
    for index, row in enumerate(rows):
        for signature in row.signatures:
            previous = first_by_signature.get(signature)
            if previous is None:
                first_by_signature[signature] = index
            else:
                uf.union(index, previous)
    groups: dict[int, list[str]] = defaultdict(list)
    for index, row in enumerate(rows):
        groups[uf.find(index)].append(row.row_id)
    # Component IDs are independent of input ordering and readable in audits.
    component_rows: dict[str, list[str]] = {}
    row_to_component: dict[str, str] = {}
    for row_ids in groups.values():
        ordered = sorted(row_ids)
        component_id = "pc_" + sha256_text("\n".join(ordered))[:24]
        component_rows[component_id] = ordered
        for row_id in ordered:
            row_to_component[row_id] = component_id
    components_by_signature: dict[str, set[str]] = defaultdict(set)
    signature_kinds: Counter[str] = Counter()
    for row in rows:
        component_id = row_to_component[row.row_id]
        for signature in row.signatures:
            components_by_signature[signature].add(component_id)
    for signature in components_by_signature:
        signature_kinds[signature.split(":", 1)[0]] += 1
    crossing = sum(
        len(components) > 1 for components in components_by_signature.values()
    )
    if crossing:
        raise AssertionError("normalized passage signatures cross components")
    signature_audit = {
        "normalization": "NFKC, whitespace removed, casefolded",
        "shingle_width": SHINGLE_WIDTH,
        "unique_signatures": len(components_by_signature),
        "unique_signatures_by_kind": dict(sorted(signature_kinds.items())),
        "cross_component_signature_overlap_count": crossing,
        "signature_components_atomic": crossing == 0,
    }
    return row_to_component, component_rows, signature_audit


def _context_address(task: str, segments: Sequence[Mapping[str, Any]], target_index: int) -> str:
    if task == "attribution":
        return "\n".join(f"[{int(segment['ordinal'])}] {segment['text']}" for segment in segments)
    radius = 3 if task == "narrative" else 2
    start = max(0, target_index - radius)
    end = min(len(segments), target_index + radius + 1)
    selected = segments[start:end]
    marker = "unit" if task == "narrative" else "paragraph"
    lines = [f"{marker}_context:"]
    for segment in selected:
        prefix = str(int(segment["ordinal"]))
        if task == "scene":
            prefix = "P" + prefix
        lines.append(f"[{prefix}] {segment['text']}")
    lines.append(f"target_{marker}_ordinal: {int(segments[target_index]['ordinal'])}")
    return "\n".join(lines)


def _make_items(rows: Sequence[RawRow], row_to_component: Mapping[str, str]) -> list[Item]:
    items: list[Item] = []
    for row in rows:
        component_id = row_to_component[row.row_id]
        segments = list(row.segments)
        if row.task == "attribution":
            item_id = f"{row.task}:{row.row_id[:20]}"
            items.append(
                Item(
                    task=row.task,
                    source_path=row.source_path,
                    source_line=row.source_line,
                    row_ordinal=row.row_ordinal,
                    row_id=row.row_id,
                    item_id=item_id,
                    component_id=component_id,
                    target_ordinal=1,
                    # Keep the candidate list and natural instructions in the
                    # address.  The assistant response is never included.
                    address_text=row.user_text.strip(),
                    value=row.value,
                    value_json=row.value_json,
                    assistant_text=row.assistant_text,
                    metadata={"item_granularity": "task_row"},
                )
            )
            continue
        if row.task == "narrative":
            labels = {int(label["unit_id"]): label for label in row.value["labels"]}
            for target_index, segment in enumerate(segments):
                unit_id = int(segment["ordinal"])
                label = labels[unit_id]
                value = {"labels": [{"unit_id": str(unit_id), "type": label["type"]}]}
                value_json = canonical_json(value)
                item_id = f"{row.task}:{row.row_id[:20]}:u{unit_id}"
                items.append(
                    Item(
                        task=row.task,
                        source_path=row.source_path,
                        source_line=row.source_line,
                        row_ordinal=row.row_ordinal,
                        row_id=row.row_id,
                        item_id=item_id,
                        component_id=component_id,
                        target_ordinal=unit_id,
                        address_text=_context_address(row.task, segments, target_index),
                        value=value,
                        value_json=value_json,
                        assistant_text=row.assistant_text,
                        metadata={
                            "item_granularity": "unit",
                            "unit_text": str(segment["text"]),
                            "unit_count_in_source_row": len(segments),
                        },
                    )
                )
            continue
        boundaries = set(int(boundary) for boundary in row.value["boundaries"])
        for boundary in range(1, len(segments)):
            local = {"boundaries": [1] if boundary in boundaries else []}
            item_id = f"{row.task}:{row.row_id[:20]}:b{boundary}"
            items.append(
                Item(
                    task=row.task,
                    source_path=row.source_path,
                    source_line=row.source_line,
                    row_ordinal=row.row_ordinal,
                    row_id=row.row_id,
                    item_id=item_id,
                    component_id=component_id,
                    target_ordinal=boundary,
                    address_text=_context_address(row.task, segments, boundary - 1),
                    value=local,
                    value_json=canonical_json(local),
                    assistant_text=row.assistant_text,
                    metadata={
                        "item_granularity": "candidate_boundary",
                        "paragraph_count": len(segments),
                        "global_boundary_ordinal": boundary,
                        "global_boundaries": sorted(boundaries),
                    },
                )
            )
    return items


def load_items(
    source_paths: Mapping[str, Path] | None = None,
    *,
    enforce_pinned_sources: bool = True,
) -> tuple[list[Item], dict[str, Any]]:
    """Load and expand only the three permitted TRAIN sources."""

    paths = dict(source_paths or default_source_paths())
    if set(paths) != set(SOURCE_DEFINITIONS):
        raise ValueError("source_paths must contain exactly attribution, narrative, and scene")
    rows, source_stats = _load_raw_rows(paths, enforce_pinned_sources=enforce_pinned_sources)
    row_to_component, component_rows, signature_audit = _component_assignments(rows)
    items = _make_items(rows, row_to_component)
    # A row-level component must be shared by every unit/boundary expansion.
    for item in items:
        if item.component_id != row_to_component[item.row_id]:
            raise AssertionError("expanded item component differs from source row")
    return items, {
        "sources": source_stats,
        "row_count": len(rows),
        "component_count": len(component_rows),
        "component_rows": {key: value for key, value in sorted(component_rows.items())},
        "signature_audit": signature_audit,
    }


def assign_component_splits(
    component_rows: Mapping[str, Sequence[str]],
    component_task_weights: Mapping[str, Mapping[str, int]] | None = None,
    *,
    seed: int = SPLIT_SEED,
) -> dict[str, str]:
    """Assign whole components to deterministic approximately 60/20/20 splits.

    When task weights are supplied, normalized deficits are balanced per task
    as well as globally.  A component shared by tasks is still a single atomic
    assignment.
    """

    if not component_rows:
        return {}
    ranked = sorted(
        component_rows,
        key=lambda component: sha256_text(f"{seed}:{component}"),
    )
    dimensions = ("__all__", *SOURCE_DEFINITIONS)
    component_weights: dict[str, dict[str, int]] = {}
    for component in ranked:
        task_weights = dict((component_task_weights or {}).get(component, {}))
        component_weights[component] = {
            "__all__": sum(task_weights.values()) or len(component_rows[component]),
            **{task: int(task_weights.get(task, 0)) for task in SOURCE_DEFINITIONS},
        }
    totals = {
        dimension: sum(component_weights[component][dimension] for component in ranked)
        for dimension in dimensions
    }
    targets = {
        split: {
            dimension: totals[dimension] * SPLIT_FRACTIONS[split]
            for dimension in dimensions
        }
        for split in SPLITS
    }
    loads = {
        split: {dimension: 0 for dimension in dimensions} for split in SPLITS
    }
    assignments: dict[str, str] = {}
    for component in ranked:
        weights = component_weights[component]
        # Prefer the split with the largest normalized deficit.  Stable split
        # order resolves exact ties, and no component is ever divided.
        present = [
            dimension
            for dimension in dimensions
            if weights[dimension] > 0 and targets["train"][dimension] > 0
        ]
        choice = max(
            SPLITS,
            key=lambda split: (
                sum(
                    (targets[split][dimension] - loads[split][dimension])
                    / targets[split][dimension]
                    for dimension in present
                )
                / len(present),
                -SPLITS.index(split),
            ),
        )
        assignments[component] = choice
        for dimension in dimensions:
            loads[choice][dimension] += weights[dimension]
    return assignments


def _semantic_factors(item: Item) -> dict[str, Any]:
    """Factor a task value so it can be transferred to another natural key."""

    if item.task == "attribution":
        candidates = _attribution_candidates(item.address_text)
        answer = str(item.value["best_candidate"])
        if answer not in candidates:
            raise ValueError(f"Attribution answer is absent from candidates for {item.item_id}")
        return {
            "candidate_index": candidates.index(answer),
            "uncertain": bool(item.value["uncertain"]),
        }
    if item.task == "narrative":
        return {"type": str(item.value["labels"][0]["type"])}
    return {"is_boundary": bool(item.value["boundaries"])}


def _transfer_domain(item: Item) -> tuple[Any, ...]:
    if item.task == "attribution":
        return (item.task, len(_attribution_candidates(item.address_text)))
    return (item.task,)


def _value_from_factors(item: Item, factors: Mapping[str, Any]) -> tuple[dict[str, Any], str]:
    if item.task == "attribution":
        candidates = _attribution_candidates(item.address_text)
        index = factors.get("candidate_index")
        uncertain = factors.get("uncertain")
        if type(index) is not int or index not in range(len(candidates)) or type(uncertain) is not bool:
            raise ValueError(f"Incompatible attribution factors for {item.item_id}")
        value: dict[str, Any] = {
            "best_candidate": candidates[index],
            "uncertain": uncertain,
        }
    elif item.task == "narrative":
        narrative_type = factors.get("type")
        if narrative_type not in _NARRATIVE_TYPES:
            raise ValueError(f"Incompatible narrative factors for {item.item_id}")
        value = {
            "labels": [
                {"unit_id": str(item.target_ordinal), "type": narrative_type}
            ]
        }
    else:
        is_boundary = factors.get("is_boundary")
        if type(is_boundary) is not bool:
            raise ValueError(f"Incompatible scene factors for {item.item_id}")
        value = {"boundaries": [1] if is_boundary else []}
    return value, canonical_json(value)


def _rewrite_value(
    item: Item,
    *,
    forbidden_value_json: Iterable[str] = (),
) -> tuple[dict[str, Any], str, dict[str, Any]]:
    forbidden = {item.value_json, *forbidden_value_json}
    candidates: list[dict[str, Any]] = []
    if item.task == "attribution":
        for candidate_index in range(len(_attribution_candidates(item.address_text))):
            for uncertain in (False, True):
                candidates.append(
                    {"candidate_index": candidate_index, "uncertain": uncertain}
                )
    elif item.task == "narrative":
        candidates = [{"type": narrative_type} for narrative_type in _NARRATIVE_TYPES]
    else:
        candidates = [{"is_boundary": False}, {"is_boundary": True}]
    ordered = sorted(
        candidates,
        key=lambda factors: sha256_text(
            f"{item.item_id}:target-rewrite:{canonical_json(factors)}"
        ),
    )
    changed: list[tuple[dict[str, Any], str, dict[str, Any]]] = []
    for factors in ordered:
        value, value_json = _value_from_factors(item, factors)
        if value_json != item.value_json:
            changed.append((value, value_json, factors))
    for candidate in changed:
        if candidate[1] not in forbidden:
            return candidate
    if changed:
        return changed[0]
    raise ValueError(f"No label-distinct rewrite for {item.item_id}")


def _attribution_candidates(address_text: str) -> list[str]:
    # Candidate lines are deliberately extracted from the natural address;
    # this is metadata for a valid counterfactual, never inserted into reads.
    candidates: list[str] = []
    in_candidates = False
    for line in address_text.splitlines():
        folded = line.casefold()
        if "候选" in line or "候選" in line or folded.startswith("candidate"):
            in_candidates = True
            continue
        if in_candidates:
            match = re.match(r"\s*-\s*(.+?)\s*$", line)
            if match:
                candidates.append(match.group(1))
            elif line.strip() and not line.lstrip().startswith("-"):
                # Candidate lists are followed by context; stop at the first
                # non-list line so names in prose are not treated as labels.
                if "上下文" in line or folded.startswith("context"):
                    in_candidates = False
    return candidates


def _read_prompt(task: str, address_text: str) -> str:
    return (
        f"natural_task: {task}\n"
        "natural_address:\n"
        f"{address_text}\n"
        "respond_with_the_bound_value:"
    )


def _write_prompt(task: str, address_text: str, value_json: str) -> str:
    return (
        f"natural_task: {task}\n"
        "memory_key:\n"
        f"{address_text}\n"
        "memory_value:\n"
        f"{value_json}\n"
        "end_memory_record"
    )


def _assert_answer_absent(item: Item, read_prompt: str, all_values: Iterable[str]) -> None:
    if item.assistant_text.strip() and item.assistant_text.strip() in read_prompt:
        raise ValueError(f"assistant answer leaked into read prompt for {item.item_id}")
    for value_json in all_values:
        if value_json in read_prompt:
            raise ValueError(f"serialized answer leaked into read prompt for {item.item_id}")
    if "memory_value:" in read_prompt or "end_memory_record" in read_prompt:
        raise ValueError(f"memory format marker leaked into read prompt for {item.item_id}")


def _item_sort_key(item: Item) -> tuple[str, str, int, str]:
    return (item.task, item.component_id, item.target_ordinal, item.item_id)


def _value_swap_plan(
    items: Sequence[Item],
) -> tuple[tuple[int, ...], list[tuple[dict[str, Any], str]]] | None:
    if len(items) != 4:
        raise ValueError("Value swaps require exactly four records")
    identity = tuple(range(4))
    candidates: list[
        tuple[tuple[int, ...], list[tuple[dict[str, Any], str]]]
    ] = []
    for permutation in itertools.permutations(range(4)):
        if permutation == identity or any(
            source_slot == destination_slot
            for destination_slot, source_slot in enumerate(permutation)
        ):
            continue
        transferred: list[tuple[dict[str, Any], str]] = []
        compatible = True
        for destination_slot, source_slot in enumerate(permutation):
            try:
                transferred.append(
                    _value_from_factors(
                        items[destination_slot], _semantic_factors(items[source_slot])
                    )
                )
            except ValueError:
                compatible = False
                break
        if compatible and all(
            value_json != items[slot].value_json
            for slot, (_, value_json) in enumerate(transferred)
        ):
            candidates.append((permutation, transferred))
    if not candidates:
        return None
    return min(
        candidates,
        key=lambda candidate: sha256_text(
            "value-swap:" + ":".join(item.item_id for item in items)
            + ":" + canonical_json(candidate[0])
        ),
    )


def _select_episode_records(target: Item, pool: Sequence[Item]) -> list[Item] | None:
    domain = _transfer_domain(target)
    groups: dict[str, list[Item]] = defaultdict(list)
    for candidate in pool:
        if (
            candidate.component_id == target.component_id
            or _transfer_domain(candidate) != domain
        ):
            continue
        groups[canonical_json(_semantic_factors(candidate))].append(candidate)
    for factor, candidates in groups.items():
        groups[factor] = sorted(
            candidates,
            key=lambda item: sha256_text(
                f"{target.item_id}:episode-record:{factor}:{item.item_id}"
            ),
        )
    factor_order = sorted(
        groups,
        key=lambda factor: sha256_text(f"{target.item_id}:factor:{factor}"),
    )
    target_factor = canonical_json(_semantic_factors(target))
    for factor_choices in itertools.product(factor_order, repeat=3):
        if max(Counter((target_factor, *factor_choices)).values()) > 2:
            continue
        selected = [target]
        used_components = {target.component_id}
        valid = True
        for factor in factor_choices:
            candidate = next(
                (
                    item
                    for item in groups[factor]
                    if item.component_id not in used_components
                ),
                None,
            )
            if candidate is None:
                valid = False
                break
            selected.append(candidate)
            used_components.add(candidate.component_id)
        if valid and _value_swap_plan(selected) is not None:
            return selected
    return None


def _find_donor(
    target: Item,
    pool: Sequence[Item],
    *,
    excluded_components: set[str],
) -> tuple[Item, dict[str, Any], str] | None:
    candidates = sorted(
        (
            item
            for item in pool
            if item.component_id not in excluded_components
            and item.item_id != target.item_id
            and _transfer_domain(item) == _transfer_domain(target)
        ),
        key=lambda item: sha256_text(target.item_id + ":donor:" + item.item_id),
    )
    for candidate in candidates:
        value, value_json = _value_from_factors(target, _semantic_factors(candidate))
        if value_json != target.value_json:
            return candidate, value, value_json
    return None


def _state_record(
    item: Item,
    *,
    slot_id: int,
    value: Mapping[str, Any],
    value_json: str,
    origin: str,
    value_source_item: Item | None = None,
) -> dict[str, Any]:
    key_text = item.address_text
    return {
        "record_id": item.item_id,
        "slot_id": slot_id,
        "component_id": item.component_id,
        "task": item.task,
        "key_text": key_text,
        "value": value,
        "value_json": value_json,
        "write_text": _write_prompt(item.task, key_text, value_json),
        "value_origin": origin,
        "key_source_item_id": item.item_id,
        "key_source_row_id": item.row_id,
        "value_source_item_id": (
            value_source_item.item_id if value_source_item is not None else None
        ),
        "value_source_row_id": (
            value_source_item.row_id if value_source_item is not None else None
        ),
        "source_path": item.source_path,
        "source_line": item.source_line,
        "target_ordinal": item.target_ordinal,
    }


def _record_payload_sha256(records: Sequence[Mapping[str, Any]]) -> str:
    return sha256_text(canonical_json(records))


def _binding_sha256(item: Item, value_json: str) -> str:
    return sha256_text(
        canonical_json(
            {
                "task": item.task,
                "normalized_address": normalize_passage(item.address_text),
                "value_json": value_json,
            }
        )
    )


def _episode(
    split: str,
    task: str,
    records_items: Sequence[Item],
    *,
    pool: Sequence[Item],
    episode_index: int,
    training_binding_hashes: set[str],
) -> dict[str, Any] | None:
    if len(records_items) != 4 or len({item.component_id for item in records_items}) != 4:
        raise ValueError("Every episode must contain four distinct passage components")
    target = records_items[0]
    episode_id = f"{split}:{task}:{target.item_id}:e{episode_index:06d}"
    value_swap = _value_swap_plan(records_items)
    if value_swap is None:
        return None
    value_swap_permutation, swapped_values = value_swap
    excluded_components = {item.component_id for item in records_items}
    donor_plan: list[tuple[Item, dict[str, Any], str]] = []
    for item in records_items:
        donor = _find_donor(
            item,
            pool,
            excluded_components=excluded_components,
        )
        if donor is None:
            return None
        donor_plan.append(donor)
        excluded_components.add(donor[0].component_id)
    correct_records = [
        _state_record(
            item,
            slot_id=slot,
            value=item.value,
            value_json=item.value_json,
            origin="source_binding",
            value_source_item=item,
        )
        for slot, item in enumerate(records_items)
    ]
    donor_records = [
        _state_record(
            item,
            slot_id=slot,
            value=donor_value,
            value_json=donor_value_json,
            origin="donor_semantic_factors",
            value_source_item=donor_item,
        )
        for slot, (item, (donor_item, donor_value, donor_value_json)) in enumerate(
            zip(records_items, donor_plan, strict=True)
        )
    ]
    value_swap_records = [
        _state_record(
            item,
            slot_id=slot,
            value=swapped_value,
            value_json=swapped_value_json,
            origin="in_episode_semantic_factor_swap",
            value_source_item=records_items[value_swap_permutation[slot]],
        )
        for slot, (item, (swapped_value, swapped_value_json)) in enumerate(
            zip(records_items, swapped_values, strict=True)
        )
    ]
    shuffle_candidates = [
        permutation
        for permutation in itertools.permutations(range(4))
        if permutation != tuple(range(4))
    ]
    shuffle_permutation = min(
        shuffle_candidates,
        key=lambda permutation: sha256_text(
            f"{target.item_id}:shuffle:{canonical_json(permutation)}"
        ),
    )
    shuffled_records = [dict(correct_records[slot]) for slot in shuffle_permutation]
    for physical_index, record in enumerate(shuffled_records):
        record["physical_index"] = physical_index
    for records in (correct_records, donor_records, value_swap_records):
        for physical_index, record in enumerate(records):
            record["physical_index"] = physical_index
    state_variants = {
        "correct_state": {
            "records": correct_records,
            "record_payload_sha256": _record_payload_sha256(correct_records),
        },
        "donor_state": {
            "records": donor_records,
            "record_payload_sha256": _record_payload_sha256(donor_records),
        },
        "value_swap": {
            "records": value_swap_records,
            "record_payload_sha256": _record_payload_sha256(value_swap_records),
            "source_slot_by_destination_slot": list(value_swap_permutation),
        },
        "shuffled_slots": {
            "records": shuffled_records,
            "record_payload_sha256": _record_payload_sha256(shuffled_records),
            "physical_order_to_semantic_slot": list(shuffle_permutation),
        },
        "no_state": {
            "records": [],
            "record_payload_sha256": _record_payload_sha256([]),
        },
    }
    queries: list[dict[str, Any]] = []
    query_counterfactual_records: dict[str, dict[str, Any]] = {}
    for target_slot, record in enumerate(correct_records):
        read_prompt = _read_prompt(task, record["key_text"])
        slot_item = records_items[target_slot]
        rewrite_value, rewrite_json, rewrite_factors = _rewrite_value(
            slot_item,
            forbidden_value_json=(
                donor_records[target_slot]["value_json"],
                value_swap_records[target_slot]["value_json"],
            ),
        )
        rewrite_for_query = [dict(value) for value in correct_records]
        rewrite_for_query[target_slot] = _state_record(
            slot_item,
            slot_id=target_slot,
            value=rewrite_value,
            value_json=rewrite_json,
            origin="target_only_semantic_factor_rewrite",
        )
        rewrite_for_query[target_slot]["physical_index"] = target_slot
        rewrite_payload_sha256 = _record_payload_sha256(rewrite_for_query)
        query_counterfactual_records[str(target_slot)] = {
            "base_state": "correct_state",
            "target_slot_rewrite": {
                "replace_slot": target_slot,
                "replacement_record": rewrite_for_query[target_slot],
                "result_record_payload_sha256": rewrite_payload_sha256,
            },
        }
        all_values = {
            state_record["value_json"]
            for state_records in (correct_records, donor_records, value_swap_records)
            for state_record in state_records
        }
        all_values.add(rewrite_json)
        _assert_answer_absent(slot_item, read_prompt, all_values)
        expected = {
            "correct_state": record["value"],
            "donor_state": donor_records[target_slot]["value"],
            "value_swap": value_swap_records[target_slot]["value"],
            "target_slot_rewrite": rewrite_value,
            "shuffled_slots": record["value"],
            "no_state": record["value"],
            "pristine_frozen_base": record["value"],
        }
        binding_sha256_by_condition = {
            "correct_state": _binding_sha256(slot_item, record["value_json"]),
            "donor_state": _binding_sha256(
                slot_item, donor_records[target_slot]["value_json"]
            ),
            "value_swap": _binding_sha256(
                slot_item, value_swap_records[target_slot]["value_json"]
            ),
            "target_slot_rewrite": _binding_sha256(slot_item, rewrite_json),
        }
        absent_from_training = {
            condition: digest not in training_binding_hashes
            for condition, digest in binding_sha256_by_condition.items()
        }
        if split != "train" and not all(absent_from_training.values()):
            raise AssertionError(
                f"Held-out binding overlaps training for {slot_item.item_id}"
            )
        queries.append(
            {
                "query_id": f"{episode_id}:q{target_slot}",
                "query_family": "four_slot_target",
                "shared_correct_runtime_state_group": f"{episode_id}:correct_state",
                "target_slot": target_slot,
                "target_record_id": record["record_id"],
                "address_text": record["key_text"],
                "read_prompt": read_prompt,
                "answer_absent_from_read_prompt": True,
                "gold": record["value"],
                "gold_json": record["value_json"],
                "expected_by_state": expected,
                "record_payload_sha256_by_condition": {
                    **{
                        name: variant["record_payload_sha256"]
                        for name, variant in state_variants.items()
                    },
                    "target_slot_rewrite": rewrite_payload_sha256,
                },
                "binding_sha256_by_condition": binding_sha256_by_condition,
                "binding_absent_from_training": absent_from_training,
                "target_slot_rewrite": {
                    "value": rewrite_value,
                    "value_json": rewrite_json,
                    "semantic_factors": rewrite_factors,
                },
            }
        )
    return {
        "schema": SCHEMA,
        "episode_id": episode_id,
        "split": split,
        "task": task,
        "passage_components": [item.component_id for item in records_items],
        "records": correct_records,
        "state_variants": state_variants,
        # Target-only rewrites are sparse deltas over the shared correct state.
        # The runner must materialize them and hash the resulting tensors.
        "query_counterfactual_records": query_counterfactual_records,
        "queries": queries,
        "controls": {
            "correct_state": "four records, original values",
            "donor_state": "same four keys; all four semantic values supplied by external donor records",
            "value_swap": "same four keys and value factors; factors permuted across all four keys",
            "target_slot_rewrite": "same state except the queried key receives a generated alternate value",
            "shuffled_slots": "same records with physical order permuted",
            "no_state": "empty/reset outer state",
            "pristine_frozen_base": "frozen Gemma base with no memory adapter attached",
        },
        "donor_source_item_ids": [donor[0].item_id for donor in donor_plan],
        "donor_source_component_ids": [
            donor[0].component_id for donor in donor_plan
        ],
        "value_swap_source_slot_by_destination_slot": list(
            value_swap_permutation
        ),
    }


def _build_episodes(
    items: Sequence[Item],
    component_split: Mapping[str, str],
    *,
    output_splits: Sequence[str],
    episodes_per_task: int,
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, Any]]:
    if episodes_per_task <= 0:
        raise ValueError("episodes_per_task must be positive")
    if not output_splits or any(split not in SPLITS for split in output_splits):
        raise ValueError(f"Invalid output splits: {tuple(output_splits)!r}")
    by_split_task: dict[tuple[str, str], list[Item]] = defaultdict(list)
    for item in items:
        split = component_split[item.component_id]
        by_split_task[(split, item.task)].append(item)
    training_binding_hashes = {
        _binding_sha256(item, item.value_json)
        for item in items
        if component_split[item.component_id] == "train"
    }
    output: dict[str, list[dict[str, Any]]] = {split: [] for split in output_splits}
    skipped: dict[str, int] = defaultdict(int)
    by_split_task_counts: dict[str, dict[str, int]] = {
        split: {} for split in output_splits
    }
    for split in output_splits:
        for task in SOURCE_DEFINITIONS:
            pool = sorted(
                by_split_task[(split, task)],
                key=lambda item: sha256_text(
                    f"{split}:{task}:episode-target:{item.item_id}"
                ),
            )
            if len({item.component_id for item in pool}) < 4:
                skipped[f"{split}:{task}:too_few_components"] += len(pool)
                continue
            count = 0
            for target in pool:
                chosen = _select_episode_records(target, pool)
                if chosen is None:
                    skipped[f"{split}:{task}:no_full_value_derangement"] += 1
                    continue
                episode = _episode(
                    split,
                    task,
                    chosen,
                    pool=pool,
                    episode_index=count,
                    training_binding_hashes=training_binding_hashes,
                )
                if episode is None:
                    skipped[f"{split}:{task}:no_compatible_donor_state"] += 1
                    continue
                output[split].append(episode)
                count += 1
                if count == episodes_per_task:
                    break
            if count != episodes_per_task:
                raise ValueError(
                    f"Could build only {count}/{episodes_per_task} episodes for {split}:{task}"
                )
            by_split_task_counts[split][task] = count
    return output, {
        "episodes_by_split_task": by_split_task_counts,
        "training_binding_count": len(training_binding_hashes),
        "heldout_counterfactual_training_binding_overlap_count": 0,
        "skipped": dict(sorted(skipped.items())),
    }


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(canonical_json(row) + "\n")
    return sha256_file(path)


def _model_binding(
    model_id: str,
    revision: str,
    model_path: Path | None,
    endpoint: str,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "model_id": model_id,
        "revision": revision,
        "hf_endpoint": endpoint,
        "local_model_path": str(model_path.expanduser().resolve()) if model_path else None,
    }
    if model_path is not None:
        resolved = model_path.expanduser().resolve()
        if not resolved.is_dir() or resolved.is_symlink():
            raise ValueError(f"Model path is not a regular directory: {resolved}")
        artifact_paths: list[Path] = []
        for name in MODEL_RUNTIME_ARTIFACTS:
            candidate = resolved / name
            if not candidate.is_file() or candidate.is_symlink():
                raise ValueError(f"Required model artifact is invalid: {candidate}")
            artifact_paths.append(candidate)
        optional_generation = resolved / "generation_config.json"
        if optional_generation.is_file() and not optional_generation.is_symlink():
            artifact_paths.append(optional_generation)
        weight_paths = sorted(resolved.glob("*.safetensors"))
        if not weight_paths:
            weight_paths = sorted(resolved.glob("*.bin"))
        if not weight_paths or any(path.is_symlink() for path in weight_paths):
            raise ValueError(f"Model directory has no regular local weight artifacts: {resolved}")
        artifact_paths.extend(weight_paths)
        artifacts = {
            path.name: {"bytes": path.stat().st_size, "sha256": sha256_file(path)}
            for path in sorted(artifact_paths, key=lambda value: value.name)
        }
        payload["local_artifacts"] = artifacts
        payload["weights_bound"] = True
        payload["binding_scope"] = "hf_identity_and_local_runtime_artifacts"
    else:
        payload["weights_bound"] = False
        payload["binding_scope"] = "hf_model_id_and_revision_only"
    payload["binding_sha256"] = sha256_text(canonical_json(payload))
    return payload


def _split_audit(items: Sequence[Item], component_split: Mapping[str, str]) -> dict[str, Any]:
    by_split: dict[str, dict[str, int]] = {
        split: {task: 0 for task in SOURCE_DEFINITIONS} for split in SPLITS
    }
    components_by_split: dict[str, set[str]] = {split: set() for split in SPLITS}
    for item in items:
        split = component_split[item.component_id]
        by_split[split][item.task] += 1
        components_by_split[split].add(item.component_id)
    overlap: dict[str, list[str]] = {}
    for left_index, left in enumerate(SPLITS):
        for right in SPLITS[left_index + 1 :]:
            shared = sorted(components_by_split[left] & components_by_split[right])
            if shared:
                overlap[f"{left}:{right}"] = shared
    if overlap:
        raise AssertionError(f"passage components cross splits: {overlap}")
    task_fractions: dict[str, dict[str, float]] = {}
    task_fraction_abs_error: dict[str, dict[str, float]] = {}
    for task in SOURCE_DEFINITIONS:
        total = sum(by_split[split][task] for split in SPLITS)
        task_fractions[task] = {
            split: by_split[split][task] / total for split in SPLITS
        }
        task_fraction_abs_error[task] = {
            split: abs(task_fractions[task][split] - SPLIT_FRACTIONS[split])
            for split in SPLITS
        }
    return {
        "items_by_split_task": by_split,
        "item_fractions_by_task": task_fractions,
        "item_fraction_abs_error_by_task": task_fraction_abs_error,
        "maximum_item_fraction_abs_error": max(
            error
            for task_errors in task_fraction_abs_error.values()
            for error in task_errors.values()
        ),
        "components_by_split": {split: sorted(values) for split, values in components_by_split.items()},
        "cross_split_component_overlap": overlap,
        "passage_disjoint": not overlap,
        "normalized_signature_cross_split_overlap_count": 0,
        "normalized_units_passage_disjoint": not overlap,
    }


def _is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _validate_sealed_lock_receipt(
    path: Path,
    *,
    benchmark_contract_sha256: str,
) -> dict[str, Any]:
    requested = path.expanduser()
    if requested.is_symlink():
        raise ValueError(f"Sealed lock receipt cannot be a symbolic link: {requested}")
    resolved = requested.resolve(strict=True)
    if not resolved.is_file():
        raise ValueError(f"Sealed lock receipt is not a regular file: {resolved}")
    receipt = json.loads(resolved.read_text(encoding="utf-8"))
    required_digests = (
        "development_manifest_payload_sha256",
        "runner_protocol_sha256",
        "training_configuration_sha256",
        "development_run_receipt_sha256",
        "adapter_files_sha256",
    )
    if (
        not isinstance(receipt, Mapping)
        or receipt.get("schema") != SEALED_LOCK_SCHEMA
        or receipt.get("configuration_frozen") is not True
        or receipt.get("development_gate_passed") is not True
        or receipt.get("benchmark_contract_sha256")
        != benchmark_contract_sha256
        or any(not _is_sha256(receipt.get(field)) for field in required_digests)
    ):
        raise ValueError("Sealed lock receipt does not bind a frozen development protocol")
    payload = dict(receipt)
    return {
        "path": str(resolved),
        "receipt": payload,
        "receipt_sha256": sha256_text(canonical_json(payload)),
    }


def verify_manifest_receipt(manifest: Mapping[str, Any]) -> bool:
    receipt = manifest.get("manifest_receipt")
    if not isinstance(receipt, Mapping):
        return False
    payload = dict(manifest)
    payload.pop("manifest_receipt", None)
    return (
        receipt.get("algorithm") == "sha256"
        and receipt.get("payload_scope") == "canonical_manifest_without_receipt"
        and receipt.get("payload_sha256") == sha256_text(canonical_json(payload))
    )


def protocol_definition() -> dict[str, Any]:
    return {
        "records_per_episode": 4,
        "query_variants_per_episode": 4,
        "state_variants": [
            "correct_state",
            "donor_state",
            "value_swap",
            "target_slot_rewrite",
            "shuffled_slots",
            "no_state",
            "pristine_frozen_base",
        ],
        "donor_rule": "all four target keys retained; every semantic value factor transferred from a distinct external component",
        "value_swap_rule": "derangement of in-episode semantic value factors; every destination value changes",
        "target_rewrite_rule": "only queried slot changes; alternate chosen outside donor and swap values when label cardinality permits",
        "answer_absence_rule": "serialized target and all state values absent from read_prompt",
        "write_scope": "one natural key/value record per write; no cross-record text",
        "read_cache_rule": "read prompt is address only; trainer must preserve outer state and disable writes",
        "source_hash_scope": "record_payload_sha256 hashes JSON records only and is never runtime tensor-state evidence",
        "runtime_state_audit_required": True,
        "pristine_base_rule": "load the frozen Gemma base without attaching the memory adapter",
    }


def build_dataset(
    *,
    output_dir: Path,
    source_paths: Mapping[str, Path] | None = None,
    model_id: str = DEFAULT_MODEL_ID,
    model_revision: str = DEFAULT_MODEL_REVISION,
    model_path: Path | None = None,
    hf_endpoint: str = HF_MIRROR,
    split_seed: int = SPLIT_SEED,
    episodes_per_task: int = DEFAULT_EPISODES_PER_TASK,
    build_profile: str = "development",
    sealed_lock_receipt: Path | None = None,
    enforce_pinned_sources: bool = True,
) -> dict[str, Any]:
    """Build a development or explicitly unlocked sealed dataset package."""

    endpoint = require_hf_mirror(hf_endpoint)
    if episodes_per_task <= 0:
        raise ValueError("episodes_per_task must be positive")
    if build_profile not in BUILD_PROFILES:
        raise ValueError(f"Unknown build profile {build_profile!r}")
    if enforce_pinned_sources and model_path is None:
        raise ValueError("Formal pinned-source builds require --model-path weight binding")
    if build_profile == "development" and sealed_lock_receipt is not None:
        raise ValueError("Development builds must not receive a sealed lock receipt")
    items, source_audit = load_items(
        source_paths,
        enforce_pinned_sources=enforce_pinned_sources,
    )
    # Reconstruct row components from the stable source row IDs.  Item rows
    # are expanded after union-find, so grouping by row ID is exact.
    row_to_component: dict[str, str] = {}
    component_rows: dict[str, set[str]] = defaultdict(set)
    for item in items:
        row_to_component[item.row_id] = item.component_id
        component_rows[item.component_id].add(item.row_id)
    component_task_weights: dict[str, dict[str, int]] = defaultdict(
        lambda: defaultdict(int)
    )
    for item in items:
        component_task_weights[item.component_id][item.task] += 1
    component_split = assign_component_splits(
        {component: sorted(rows) for component, rows in component_rows.items()},
        {
            component: dict(task_counts)
            for component, task_counts in component_task_weights.items()
        },
        seed=split_seed,
    )
    split_audit = _split_audit(items, component_split)
    if enforce_pinned_sources and split_audit["maximum_item_fraction_abs_error"] > 0.03:
        raise ValueError("Pinned corpus split balance exceeds the 0.03 absolute tolerance")
    if not source_audit["signature_audit"]["signature_components_atomic"]:
        raise AssertionError("Normalized passage signatures are not component-atomic")
    model = _model_binding(model_id, model_revision, model_path, endpoint)
    protocol = protocol_definition()
    generator_source_sha256 = sha256_file(Path(__file__).resolve(strict=True))
    benchmark_contract = {
        "schema": SCHEMA,
        "source_sha256": {
            source["task"]: source["sha256"] for source in source_audit["sources"]
        },
        "model_binding_sha256": model["binding_sha256"],
        "split_seed": split_seed,
        "split_fractions": SPLIT_FRACTIONS,
        "component_assignment_sha256": sha256_text(canonical_json(component_split)),
        "generator_source_sha256": generator_source_sha256,
        "protocol_definition_sha256": sha256_text(canonical_json(protocol)),
        "episodes_per_task_per_output_split": episodes_per_task,
        "records_per_episode": 4,
        "query_targets_per_episode": 4,
    }
    benchmark_contract_sha256 = sha256_text(canonical_json(benchmark_contract))
    sealed_lock = None
    if build_profile == "sealed_validation":
        if sealed_lock_receipt is None:
            raise ValueError(
                "Sealed validation cannot be materialized before a frozen lock receipt"
            )
        sealed_lock = _validate_sealed_lock_receipt(
            sealed_lock_receipt,
            benchmark_contract_sha256=benchmark_contract_sha256,
        )
    output_splits = BUILD_PROFILES[build_profile]
    episodes, episode_audit = _build_episodes(
        items,
        component_split,
        output_splits=output_splits,
        episodes_per_task=episodes_per_task,
    )
    output_dir = output_dir.expanduser().resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise ValueError(f"Output directory must be fresh or empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_hashes = {
        split: _write_jsonl(output_dir / f"{split}.jsonl", episodes[split])
        for split in output_splits
    }
    manifest: dict[str, Any] = {
        "schema": SCHEMA,
        "build_profile": build_profile,
        "materialized_splits": list(output_splits),
        "sealed_lock": sealed_lock,
        "hf_endpoint": endpoint,
        "model_binding": model,
        "generator_source_sha256": generator_source_sha256,
        "benchmark_contract": benchmark_contract,
        "benchmark_contract_sha256": benchmark_contract_sha256,
        "source_policy": {
            "allowed_tasks": list(SOURCE_DEFINITIONS),
            "allowed_split": "train",
            "protected_splits_never_opened": ["val", "test", "Hard32"],
            "enforce_pinned_sources": enforce_pinned_sources,
        },
        "sources": source_audit["sources"],
        "row_count": source_audit["row_count"],
        "component_count": source_audit["component_count"],
        "signature_audit": source_audit["signature_audit"],
        "split_policy": {
            "seed": split_seed,
            "fractions": SPLIT_FRACTIONS,
            "assignment": "hash_ranked_task_stratified_weighted_component_v1",
            "component_id": "sha256(sorted_source_row_ids)",
            "shingle_width": SHINGLE_WIDTH,
        },
        "split_audit": split_audit,
        "episode_audit": {
            "episodes_by_split": {
                split: len(episodes[split]) for split in output_splits
            },
            "queries_by_split": {
                split: sum(len(episode["queries"]) for episode in episodes[split])
                for split in output_splits
            },
            **episode_audit,
        },
        "output_sha256": output_hashes,
        "protocol": protocol,
    }
    manifest["manifest_receipt"] = {
        "algorithm": "sha256",
        "payload_scope": "canonical_manifest_without_receipt",
        "payload_sha256": sha256_text(canonical_json(manifest)),
    }
    if not verify_manifest_receipt(manifest):
        raise AssertionError("Manifest self-receipt construction failed")
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest


def _default_cli_sources() -> dict[str, Path]:
    return {
        task: DEFAULT_SOURCE_ROOT / str(spec["relative_path"])
        for task, spec in SOURCE_DEFINITIONS.items()
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    defaults = _default_cli_sources()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--attribution-train", type=Path, default=defaults["attribution"])
    parser.add_argument("--narrative-train", type=Path, default=defaults["narrative"])
    parser.add_argument("--scene-train", type=Path, default=defaults["scene"])
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    parser.add_argument("--model-revision", default=DEFAULT_MODEL_REVISION)
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--hf-endpoint", default=HF_MIRROR)
    parser.add_argument("--split-seed", type=int, default=SPLIT_SEED)
    parser.add_argument(
        "--episodes-per-task",
        type=int,
        default=DEFAULT_EPISODES_PER_TASK,
    )
    parser.add_argument(
        "--build-profile",
        choices=tuple(BUILD_PROFILES),
        default="development",
    )
    parser.add_argument("--sealed-lock-receipt", type=Path)
    parser.add_argument(
        "--allow-unpinned-sources",
        action="store_true",
        help="Permit synthetic/temp TRAIN paths in unit tests; still rejects val/test/Hard32.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    paths = {
        "attribution": args.attribution_train,
        "narrative": args.narrative_train,
        "scene": args.scene_train,
    }
    manifest = build_dataset(
        output_dir=args.output_dir,
        source_paths=paths,
        model_id=args.model_id,
        model_revision=args.model_revision,
        model_path=args.model_path,
        hf_endpoint=args.hf_endpoint,
        split_seed=args.split_seed,
        episodes_per_task=args.episodes_per_task,
        build_profile=args.build_profile,
        sealed_lock_receipt=args.sealed_lock_receipt,
        enforce_pinned_sources=not args.allow_unpinned_sources,
    )
    print(
        json.dumps(
            {
                "output_dir": str(args.output_dir.expanduser().resolve()),
                "schema": manifest["schema"],
                "episodes_by_split": manifest["episode_audit"]["episodes_by_split"],
                "passage_disjoint": manifest["split_audit"]["passage_disjoint"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
