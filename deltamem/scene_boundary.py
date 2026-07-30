"""Small, dependency-free helpers for strict scene-boundary JSON semantics."""

from __future__ import annotations

import json
from typing import Any


def extract_json_with_span(text: str) -> tuple[Any, int, int] | None:
    """Extract the first complete JSON container and its source character span."""

    decoder = json.JSONDecoder()
    for index, character in enumerate(text):
        if character not in "[{":
            continue
        try:
            value, end = decoder.raw_decode(text[index:])
        except json.JSONDecodeError:
            continue
        if isinstance(value, (dict, list)):
            return value, index, index + end
    return None


def extract_json(text: str) -> Any | None:
    """Extract the first complete JSON object or array from model text."""

    stripped = text.strip()
    try:
        return json.loads(stripped)
    except json.JSONDecodeError:
        pass
    extracted = extract_json_with_span(stripped)
    return None if extracted is None else extracted[0]


def literal_boundaries(value: Any) -> dict[tuple[str, Any], Any] | None:
    """Return benchmark-compatible literal boundary identities, or ``None``."""

    if not isinstance(value, dict) or not isinstance(value.get("boundaries"), list):
        return None
    boundaries: dict[tuple[str, Any], Any] = {}
    for item in value["boundaries"]:
        try:
            hash(item)
        except TypeError:
            identity = (
                "json",
                json.dumps(
                    item,
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                ),
            )
        else:
            identity = ("literal", item)
        boundaries.setdefault(identity, item)
    return boundaries


def strict_boundary_exact(prediction: Any, gold: Any) -> bool:
    """Match the benchmark's strict scene exactness at the boundary-set level."""

    predicted_boundaries = literal_boundaries(prediction)
    gold_boundaries = literal_boundaries(gold)
    return bool(
        predicted_boundaries is not None
        and gold_boundaries is not None
        and set(predicted_boundaries) == set(gold_boundaries)
    )


__all__ = (
    "extract_json",
    "extract_json_with_span",
    "literal_boundaries",
    "strict_boundary_exact",
)
