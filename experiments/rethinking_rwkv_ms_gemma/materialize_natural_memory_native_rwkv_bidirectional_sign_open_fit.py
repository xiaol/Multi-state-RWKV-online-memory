#!/usr/bin/env python3
"""Materialize replayable open-TRAIN bundles for the bidirectional-sign gate."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Any, Iterable, Mapping, Sequence

from transformers import AutoTokenizer


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = Path(__file__).resolve().parents[2]
for import_root in (SCRIPT_DIR, PROJECT_ROOT):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

from deltamem.chat_templates import apply_chat_template  # noqa: E402


MANIFEST_SCHEMA = (
    "rwkv_ms_natural_memory_native_rwkv_bidirectional_sign_open_fit.v1"
)
ROW_SCHEMA = (
    "rwkv_ms_natural_memory_native_rwkv_bidirectional_sign_open_fit_row.v1"
)
SOURCE_NAMESPACE = "novel-agent-sft-dataset:publisher-train-derived-fit"
SOURCE_RELATIVE_PATH = (
    "experiments/rethinking_rwkv_ms_gemma/local_artifacts/"
    "natural_memory_native_development_v1/v4-scene-boundary-detection/"
    "train_derived_fit.jsonl"
)
SOURCE_SHA256 = "8b0552cf1ddd39230896ce1ed6a3842aef94212e70bbc9e76ee8f13c546e6e57"
SOURCE_ROWS = 1443

BASE_MODEL_ID = "google/gemma-4-E4B-it"
BASE_MODEL_REVISION = "a4c2d58be94dda072b918d9db64ee85c8ed34e3f"
MODEL_CONFIG_SHA256 = (
    "33b10c02df3c2e8536cf323d29d53262aaa2f4d11dbe19bc729373fbe90295d4"
)
TOKENIZER_FILES = {
    "tokenizer.json": (
        "cc8d3a0ce36466ccc1278bf987df5f71db1719b9ca6b4118264f45cb627bfe0f"
    ),
    "tokenizer_config.json": (
        "90c3a3ba5bf53818383a58e1a776cbcacd2a038d4812eaa373e1522f2d06f3df"
    ),
    "chat_template.jinja": (
        "2f1b4d75d067bae3fe44e676721c7f077d243bc007156cb9c2f8b5836613d082"
    ),
}

SALT = "rwkv-bidirectional-sign-open-fit-components-v2:"
MECHANICS_TRIPLE = (96, 414, 729)
CAUSAL_TRIPLE = (36, 145, 649)
PAIR_COUNT = 46
BUNDLE_NAMES = ("development", "mechanics", "causal")
BUNDLE_ROWS = {"development": 64, "mechanics": 17, "causal": 17}

EXPECTED_SOURCE_SHA256 = {
    "development": (
        "a6049d5956363bff48b881949a8fead4b8b755438b12680822a6dd7021c140a7"
    ),
    "mechanics": (
        "ed24b7b7daf635248e7b3bae8b2c79acc1d62311b97fdff40f9f111eb4f81a47"
    ),
    "causal": (
        "8cf6ee83280c4696caef186d8c7749647d96f26979fe014d15d3dc75224d595b"
    ),
}
EXPECTED_MAPPING_SHA256 = {
    "development": (
        "23413ca2955aa81073f8b9c400f96a11b7953263379b7e0f6a9553605d4a577e"
    ),
    "mechanics": (
        "5c0fbf7583b175b0ebe940ceb408f9f244630be4982cf53d6309e2449e99200d"
    ),
    "causal": (
        "63773af77d764df49d848c75ae9f8ba6a6866639289f186085d689d62f193e4d"
    ),
}
EXPECTED_ORDERED_COMPONENTS_SHA256 = (
    "c6459a19aa38fb491d19d7b5c6d18e6a83ddb3bdc7742559d3e1d2006c3f6a6a"
)
EXPECTED_SOURCE_SET_SHA256 = (
    "5b7204d39d54d55126303b8ef14a2498a27fa3f3408531c0f035eea6b300c6d5"
)
EXPECTED_GLOBAL_MAPPING_SHA256 = (
    "8b7c4893101c470964766f44efa5b48d62a07aff4c410a37d58c89fb216b4745"
)


def canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    )


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


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
    expected = _receipt(payload_scope, unsigned)
    if dict(receipt) != expected:
        raise ValueError(f"{description} receipt differs")


def _strict_gold(content: Any) -> tuple[int, ...]:
    if not isinstance(content, str):
        raise ValueError("Open-fit assistant content is not text")
    value = json.loads(content)
    if not isinstance(value, Mapping) or set(value) != {"boundaries"}:
        raise ValueError("Open-fit assistant gold must contain only boundaries")
    raw_boundaries = value["boundaries"]
    if not isinstance(raw_boundaries, list):
        raise ValueError("Open-fit assistant boundaries are not a list")
    boundaries: list[int] = []
    for boundary in raw_boundaries:
        if isinstance(boundary, bool) or not isinstance(boundary, int):
            raise ValueError("Open-fit assistant boundary is not an integer")
        boundaries.append(boundary)
    if len(set(boundaries)) != len(boundaries):
        raise ValueError("Open-fit assistant boundaries contain duplicates")
    return tuple(sorted(boundaries))


def load_source_rows(path: Path) -> list[dict[str, Any]]:
    if not path.is_file() or sha256_file(path) != SOURCE_SHA256:
        raise ValueError(f"Open-fit source hash differs: {path}")
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        for source_index, line in enumerate(handle):
            raw_line = line[:-1] if line.endswith("\n") else line
            if not raw_line:
                raise ValueError(f"Open-fit source contains a blank row: {source_index}")
            value = json.loads(raw_line)
            if not isinstance(value, Mapping) or set(value) != {"messages"}:
                raise ValueError(f"Open-fit row shape differs: {source_index}")
            messages = value["messages"]
            if (
                not isinstance(messages, list)
                or len(messages) != 3
                or [message.get("role") for message in messages]
                != ["system", "user", "assistant"]
                or any(
                    not isinstance(message, Mapping)
                    or set(message) != {"role", "content"}
                    or not isinstance(message["content"], str)
                    for message in messages
                )
            ):
                raise ValueError(f"Open-fit messages differ: {source_index}")
            rows.append(
                {
                    "source_index": source_index,
                    "raw_line": raw_line,
                    "row_sha256": hashlib.sha256(
                        raw_line.encode("utf-8")
                    ).hexdigest(),
                    "messages": messages,
                    "gold": _strict_gold(messages[-1]["content"]),
                }
            )
    if len(rows) != SOURCE_ROWS:
        raise ValueError(f"Expected {SOURCE_ROWS} open-fit rows, found {len(rows)}")
    return rows


def validate_tokenizer_artifacts(tokenizer_path: Path) -> dict[str, Any]:
    expected = {"config.json": MODEL_CONFIG_SHA256, **TOKENIZER_FILES}
    files: list[dict[str, Any]] = []
    for relative_path, expected_sha256 in expected.items():
        path = tokenizer_path / relative_path
        if not path.is_file():
            raise ValueError(f"Tokenizer artifact is missing: {path}")
        digest = sha256_file(path)
        if digest != expected_sha256:
            raise ValueError(f"Tokenizer artifact hash differs: {path}")
        files.append(
            {
                "relative_path": relative_path,
                "bytes": path.stat().st_size,
                "sha256": digest,
            }
        )
    return {
        "model_id": BASE_MODEL_ID,
        "revision": BASE_MODEL_REVISION,
        "files": files,
    }


def add_write_token_counts(rows: Sequence[dict[str, Any]], tokenizer: Any) -> None:
    for row in rows:
        rendered = apply_chat_template(
            tokenizer,
            row["messages"][:-1],
            tokenize=False,
            add_generation_prompt=False,
        )
        row["write_tokens"] = len(
            tokenizer(rendered, add_special_tokens=False).input_ids
        )


def _salted_row_rank(row: Mapping[str, Any]) -> str:
    return hashlib.sha256(
        (SALT + str(row["row_sha256"])).encode("ascii")
    ).hexdigest()


def _component_rank(component: tuple[int, ...]) -> tuple[str, tuple[int, ...]]:
    digest = hashlib.sha256(
        (SALT + canonical_json(list(component))).encode("ascii")
    ).hexdigest()
    return digest, component


def build_layout(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if len(rows) != SOURCE_ROWS or any("write_tokens" not in row for row in rows):
        raise ValueError("Open-fit layout requires every tokenized source row")
    by_index = {int(row["source_index"]): row for row in rows}
    if sorted(by_index) != list(range(SOURCE_ROWS)):
        raise ValueError("Open-fit source indices are not contiguous")

    reserved = set(MECHANICS_TRIPLE) | set(CAUSAL_TRIPLE)
    if len(reserved) != 6:
        raise ValueError("Open-fit reserved triples overlap")
    for triple, expected_tokens in (
        (MECHANICS_TRIPLE, 380),
        (CAUSAL_TRIPLE, 450),
    ):
        if {int(by_index[index]["write_tokens"]) for index in triple} != {
            expected_tokens
        } or len({by_index[index]["gold"] for index in triple}) != 3:
            raise ValueError(f"Open-fit reserved triple differs: {triple}")

    pairs: list[tuple[int, int]] = []
    used = set(reserved)
    ordered_rows = sorted(
        rows,
        key=lambda row: (_salted_row_rank(row), int(row["source_index"])),
    )
    for source in ordered_rows:
        source_index = int(source["source_index"])
        if source_index in used:
            continue
        donors = [
            donor
            for donor in rows
            if int(donor["source_index"]) not in used
            and donor["gold"] != source["gold"]
        ]
        if not donors:
            raise ValueError(f"Open-fit source has no eligible donor: {source_index}")
        donor = min(
            donors,
            key=lambda candidate: (
                abs(
                    int(candidate["write_tokens"])
                    - int(source["write_tokens"])
                ),
                _salted_row_rank(candidate),
                int(candidate["source_index"]),
            ),
        )
        pair = tuple(sorted((source_index, int(donor["source_index"]))))
        pairs.append(pair)
        used.update(pair)
        if len(pairs) == PAIR_COUNT:
            break
    if len(pairs) != PAIR_COUNT:
        raise ValueError(f"Expected {PAIR_COUNT} open-fit pairs, found {len(pairs)}")

    ranked_pairs = sorted(pairs, key=_component_rank)
    components = {
        "development": ranked_pairs[14:],
        "mechanics": [MECHANICS_TRIPLE, *ranked_pairs[:7]],
        "causal": [CAUSAL_TRIPLE, *ranked_pairs[7:14]],
    }
    source_indices = {
        name: sorted(index for component in group for index in component)
        for name, group in components.items()
    }
    if {name: len(indices) for name, indices in source_indices.items()} != BUNDLE_ROWS:
        raise ValueError("Open-fit bundle row counts differ")
    for name, expected in EXPECTED_SOURCE_SHA256.items():
        if canonical_sha256(source_indices[name]) != expected:
            raise ValueError(f"Open-fit {name} source lock differs")

    mapping: dict[int, int] = {}
    for left, right in ranked_pairs:
        mapping[left] = right
        mapping[right] = left
    mapping.update({96: 414, 414: 729, 729: 96})
    mapping.update({36: 649, 649: 145, 145: 36})
    mapping_pairs = {
        name: [[source, mapping[source]] for source in source_indices[name]]
        for name in BUNDLE_NAMES
    }
    for name, expected in EXPECTED_MAPPING_SHA256.items():
        if canonical_sha256(mapping_pairs[name]) != expected:
            raise ValueError(f"Open-fit {name} mapping lock differs")
    all_sources = sorted(mapping)
    all_mapping_pairs = [[source, mapping[source]] for source in all_sources]
    if canonical_sha256(all_sources) != EXPECTED_SOURCE_SET_SHA256:
        raise ValueError("Open-fit global source lock differs")
    if canonical_sha256(all_mapping_pairs) != EXPECTED_GLOBAL_MAPPING_SHA256:
        raise ValueError("Open-fit global mapping lock differs")

    ordered_components = [
        *components["development"],
        *components["mechanics"],
        *components["causal"],
    ]
    ordered_component_payload = [list(component) for component in ordered_components]
    if (
        canonical_sha256(ordered_component_payload)
        != EXPECTED_ORDERED_COMPONENTS_SHA256
    ):
        raise ValueError("Open-fit ordered-component lock differs")

    absolute_deltas = [
        abs(
            int(by_index[source]["write_tokens"])
            - int(by_index[donor]["write_tokens"])
        )
        for source, donor in all_mapping_pairs
    ]
    if max(absolute_deltas) != 1 or sum(absolute_deltas) != 12:
        raise ValueError("Open-fit donor token deltas differ")
    if any(by_index[source]["gold"] == by_index[donor]["gold"] for source, donor in all_mapping_pairs):
        raise ValueError("Open-fit donor gold must differ")
    split_by_source = {
        source: name for name, indices in source_indices.items() for source in indices
    }
    if any(split_by_source[source] != split_by_source[donor] for source, donor in all_mapping_pairs):
        raise ValueError("Open-fit mapping crosses bundle boundaries")

    return {
        "components": components,
        "ordered_components": ordered_component_payload,
        "source_indices": source_indices,
        "mapping": mapping,
        "mapping_pairs": mapping_pairs,
        "all_sources": all_sources,
        "all_mapping_pairs": all_mapping_pairs,
        "absolute_token_deltas": absolute_deltas,
    }


def _bundle_row(source: Mapping[str, Any]) -> dict[str, Any]:
    unsigned = {
        "schema": ROW_SCHEMA,
        "source_index": int(source["source_index"]),
        "row_sha256": str(source["row_sha256"]),
        "raw_line": str(source["raw_line"]),
    }
    return {
        **unsigned,
        "receipt": _receipt("canonical_bundle_row_without_receipt", unsigned),
    }


def _bundle_bytes(rows: Sequence[Mapping[str, Any]]) -> bytes:
    return (
        "".join(
            json.dumps(
                row,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            )
            + "\n"
            for row in rows
        )
    ).encode("utf-8")


def _recorded_path(path: Path) -> str:
    try:
        return path.resolve().relative_to(PROJECT_ROOT.resolve()).as_posix()
    except ValueError:
        return str(path.resolve())


def materialize(
    *,
    source_path: Path,
    tokenizer_path: Path,
    output_root: Path,
) -> Mapping[str, Any]:
    if output_root.exists():
        raise ValueError(f"Open-fit output must be fresh: {output_root}")
    tokenizer_binding = validate_tokenizer_artifacts(tokenizer_path)
    rows = load_source_rows(source_path)
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_path,
        local_files_only=True,
        trust_remote_code=False,
    )
    add_write_token_counts(rows, tokenizer)
    layout = build_layout(rows)
    by_index = {int(row["source_index"]): row for row in rows}

    bundle_rows = {
        name: [_bundle_row(by_index[index]) for index in layout["source_indices"][name]]
        for name in BUNDLE_NAMES
    }
    bundle_payloads = {
        name: _bundle_bytes(selected_rows)
        for name, selected_rows in bundle_rows.items()
    }
    bundle_bindings = {
        name: {
            "path": f"{name}.jsonl",
            "rows": len(bundle_rows[name]),
            "sha256": hashlib.sha256(bundle_payloads[name]).hexdigest(),
            "payload_sha256": canonical_sha256(bundle_rows[name]),
        }
        for name in BUNDLE_NAMES
    }
    manifest: dict[str, Any] = {
        "schema": MANIFEST_SCHEMA,
        "source": {
            "namespace": SOURCE_NAMESPACE,
            "path": _recorded_path(source_path),
            "sha256": SOURCE_SHA256,
            "rows": SOURCE_ROWS,
        },
        "tokenizer": tokenizer_binding,
        "deterministic_algorithm": {
            "name": "rwkv_bidirectional_sign_open_fit_components_v2",
            "salt": SALT,
            "reserved_triples": [list(MECHANICS_TRIPLE), list(CAUSAL_TRIPLE)],
            "source_order": "sha256(salt + row_sha256), then source_index",
            "pair_count": PAIR_COUNT,
            "donor_order": (
                "absolute write-token delta, sha256(salt + donor row_sha256), "
                "then donor source_index"
            ),
            "donor_gold_constraint": "assistant-content gold must differ",
            "component_order": (
                "sha256(salt + canonical_json(sorted component source indices)), "
                "then component tuple"
            ),
            "split_assignment": {
                "mechanics": "mechanics triple plus ranked pairs 0:7",
                "causal": "causal triple plus ranked pairs 7:14",
                "development": "ranked pairs 14:46",
            },
            "mapping_cycles": {
                "mechanics": [96, 414, 729, 96],
                "causal": [36, 649, 145, 36],
                "pairs": "two-way swap",
            },
        },
        "ordered_components": layout["ordered_components"],
        "ordered_components_sha256": EXPECTED_ORDERED_COMPONENTS_SHA256,
        "source_indices": layout["all_sources"],
        "source_indices_sha256": EXPECTED_SOURCE_SET_SHA256,
        "mapping_pairs": layout["all_mapping_pairs"],
        "mapping_sha256": EXPECTED_GLOBAL_MAPPING_SHA256,
        "donor_token_deltas": {
            "maximum": max(layout["absolute_token_deltas"]),
            "total": sum(layout["absolute_token_deltas"]),
            "mean": sum(layout["absolute_token_deltas"])
            / len(layout["absolute_token_deltas"]),
        },
        "splits": {
            name: {
                "source_indices": layout["source_indices"][name],
                "source_indices_sha256": EXPECTED_SOURCE_SHA256[name],
                "mapping_pairs": layout["mapping_pairs"][name],
                "mapping_sha256": EXPECTED_MAPPING_SHA256[name],
                "bundle": bundle_bindings[name],
            }
            for name in BUNDLE_NAMES
        },
        "protected_splits_opened": [],
    }
    manifest["receipt"] = _receipt(
        "canonical_manifest_without_receipt",
        manifest,
    )

    output_root.mkdir(parents=True, exist_ok=False)
    for name in BUNDLE_NAMES:
        with (output_root / f"{name}.jsonl").open("xb") as handle:
            handle.write(bundle_payloads[name])
    with (output_root / "manifest.json").open("x", encoding="utf-8") as handle:
        handle.write(
            json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
        )
    return manifest


def _validate_manifest_constants(manifest: Mapping[str, Any]) -> None:
    if manifest.get("schema") != MANIFEST_SCHEMA:
        raise ValueError("Open-fit manifest schema differs")
    if manifest.get("protected_splits_opened") != []:
        raise ValueError("Open-fit manifest opens protected splits")
    if manifest.get("ordered_components_sha256") != EXPECTED_ORDERED_COMPONENTS_SHA256:
        raise ValueError("Open-fit manifest component lock differs")
    if manifest.get("source_indices_sha256") != EXPECTED_SOURCE_SET_SHA256:
        raise ValueError("Open-fit manifest source lock differs")
    if manifest.get("mapping_sha256") != EXPECTED_GLOBAL_MAPPING_SHA256:
        raise ValueError("Open-fit manifest mapping lock differs")
    splits = manifest.get("splits")
    if not isinstance(splits, Mapping) or set(splits) != set(BUNDLE_NAMES):
        raise ValueError("Open-fit manifest splits differ")
    for name in BUNDLE_NAMES:
        split = splits[name]
        if (
            not isinstance(split, Mapping)
            or split.get("source_indices_sha256") != EXPECTED_SOURCE_SHA256[name]
            or split.get("mapping_sha256") != EXPECTED_MAPPING_SHA256[name]
        ):
            raise ValueError(f"Open-fit manifest {name} lock differs")
        sources = split.get("source_indices")
        mapping_pairs = split.get("mapping_pairs")
        if (
            not isinstance(sources, list)
            or canonical_sha256(sources) != EXPECTED_SOURCE_SHA256[name]
            or not isinstance(mapping_pairs, list)
            or canonical_sha256(mapping_pairs) != EXPECTED_MAPPING_SHA256[name]
        ):
            raise ValueError(f"Open-fit manifest {name} payload differs")
    if canonical_sha256(manifest.get("source_indices")) != EXPECTED_SOURCE_SET_SHA256:
        raise ValueError("Open-fit manifest global sources differ")
    if canonical_sha256(manifest.get("mapping_pairs")) != EXPECTED_GLOBAL_MAPPING_SHA256:
        raise ValueError("Open-fit manifest global mapping differs")
    if (
        canonical_sha256(manifest.get("ordered_components"))
        != EXPECTED_ORDERED_COMPONENTS_SHA256
    ):
        raise ValueError("Open-fit manifest ordered components differ")


def validate_materialization(
    root: Path,
    *,
    bundles: Iterable[str] = BUNDLE_NAMES,
) -> dict[str, Any]:
    requested = tuple(bundles)
    if len(set(requested)) != len(requested) or any(
        name not in BUNDLE_NAMES for name in requested
    ):
        raise ValueError(f"Invalid open-fit bundle selection: {requested}")
    manifest_path = root / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(manifest, Mapping):
        raise ValueError("Open-fit manifest is not an object")
    _validate_receipt(
        manifest,
        payload_scope="canonical_manifest_without_receipt",
        description="Open-fit manifest",
    )
    _validate_manifest_constants(manifest)

    groups: dict[str, list[dict[str, Any]]] = {}
    requested_sources: set[int] = set()
    for name in requested:
        split = manifest["splits"][name]
        binding = split.get("bundle")
        expected_path = f"{name}.jsonl"
        if (
            not isinstance(binding, Mapping)
            or binding.get("path") != expected_path
            or binding.get("rows") != BUNDLE_ROWS[name]
        ):
            raise ValueError(f"Open-fit {name} bundle binding differs")
        path = root / expected_path
        if sha256_file(path) != binding.get("sha256"):
            raise ValueError(f"Open-fit {name} bundle hash differs")
        selected_rows: list[dict[str, Any]] = []
        with path.open("r", encoding="utf-8", newline="") as handle:
            for line in handle:
                row = json.loads(line)
                if not isinstance(row, Mapping) or set(row) != {
                    "schema",
                    "source_index",
                    "row_sha256",
                    "raw_line",
                    "receipt",
                }:
                    raise ValueError(f"Open-fit {name} row shape differs")
                if row.get("schema") != ROW_SCHEMA:
                    raise ValueError(f"Open-fit {name} row schema differs")
                _validate_receipt(
                    row,
                    payload_scope="canonical_bundle_row_without_receipt",
                    description=f"Open-fit {name} row",
                )
                raw_line = row.get("raw_line")
                if not isinstance(raw_line, str) or hashlib.sha256(
                    raw_line.encode("utf-8")
                ).hexdigest() != row.get("row_sha256"):
                    raise ValueError(f"Open-fit {name} row hash differs")
                selected_rows.append(dict(row))
        sources = [int(row["source_index"]) for row in selected_rows]
        if (
            sources != split["source_indices"]
            or len(selected_rows) != binding["rows"]
            or canonical_sha256(selected_rows) != binding.get("payload_sha256")
        ):
            raise ValueError(f"Open-fit {name} bundle payload differs")
        groups[name] = selected_rows
        requested_sources.update(sources)

    mapping = {
        int(source): int(donor)
        for source, donor in manifest["mapping_pairs"]
        if int(source) in requested_sources
    }
    return {"manifest": dict(manifest), "groups": groups, "mapping": mapping}


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-path",
        type=Path,
        default=PROJECT_ROOT / SOURCE_RELATIVE_PATH,
    )
    parser.add_argument("--tokenizer-path", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    manifest = materialize(
        source_path=args.source_path,
        tokenizer_path=args.tokenizer_path,
        output_root=args.output_root,
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
