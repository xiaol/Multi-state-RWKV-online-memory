#!/usr/bin/env python3
"""Materialize a sealed fresh mechanics/causal reservation for RWKV residuals."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = Path(__file__).resolve().parents[2]
for import_root in (SCRIPT_DIR, PROJECT_ROOT):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

from experiments.rethinking_rwkv_ms_gemma import (  # noqa: E402
    materialize_natural_memory_native_rwkv_bidirectional_sign_open_fit as source_loader,
)
from experiments.rethinking_rwkv_ms_gemma import (  # noqa: E402
    materialize_natural_memory_native_rwkv_continuous_write_open_fit as parent_materializer,
)
from experiments.rethinking_rwkv_ms_gemma import (  # noqa: E402
    rwkv_continuous_write_fit_split as parent_split,
)
from experiments.rethinking_rwkv_ms_gemma import (  # noqa: E402
    rwkv_source_cumulative_residual_split as residual_split,
)


MANIFEST_SCHEMA = "rwkv_ms_natural_memory_native_rwkv_source_cumulative_residual.v1"
ROW_SCHEMA = "rwkv_ms_natural_memory_native_rwkv_source_cumulative_residual_row.v1"
SEALED_MANIFEST_SHA256 = (
    "5251cc6f4254718620bd6e1328ac41c6fcb9bf837f836d623f874eedf53e9515"
)
BUNDLE_NAMES = residual_split.SPLIT_NAMES
BUNDLE_ROWS = {"mechanics": 32, "causal": 32}
FILE_NAMES = ("manifest.json", "mechanics.jsonl", "causal.jsonl")
DEFAULT_PARENT_MANIFEST = (
    PROJECT_ROOT
    / "experiments/rethinking_rwkv_ms_gemma/local_artifacts/"
    "natural_memory_native_rwkv_continuous_write_open_fit_v1/manifest.json"
)
DEFAULT_CONSUMED_RESULT = (
    PROJECT_ROOT
    / "experiments/rethinking_rwkv_ms_gemma/local_artifacts/"
    "natural_memory_native_rwkv_cumulative_virtual_kv_mechanics_v1/result.json"
)


def canonical_sha256(value: Any) -> str:
    return residual_split.canonical_sha256(value)


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


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


def _recorded_path(path: Path) -> str:
    try:
        return path.resolve().relative_to(PROJECT_ROOT.resolve()).as_posix()
    except ValueError:
        return str(path.resolve())


def _bundle_row(
    source: Mapping[str, Any],
    donor: Mapping[str, Any],
    *,
    split_name: str,
) -> dict[str, Any]:
    source_index = int(source["source_index"])
    donor_index = int(donor["source_index"])
    unsigned = {
        "schema": ROW_SCHEMA,
        "split": split_name,
        "source_index": source_index,
        "qualified_source_id": parent_split.SOURCE_BINDING.qualified_id(
            source_index, str(source["row_sha256"])
        ),
        "row_sha256": str(source["row_sha256"]),
        "donor_source_index": donor_index,
        "qualified_donor_source_id": parent_split.SOURCE_BINDING.qualified_id(
            donor_index, str(donor["row_sha256"])
        ),
        "donor_row_sha256": str(donor["row_sha256"]),
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


def _validate_split_contract(split_contract: Mapping[str, Any]) -> None:
    if (
        split_contract.get("schema") != residual_split.SCHEMA
        or split_contract.get("source") != parent_split.SOURCE_BINDING.payload()
        or split_contract.get("protected_splits_opened") != []
        or split_contract.get("sealed_bundle_bytes_read") is not False
    ):
        raise ValueError("Fresh residual split contract differs")
    _validate_receipt(
        split_contract,
        payload_scope="canonical_split_without_receipt",
        description="Fresh residual split contract",
    )
    parent_reservation = split_contract.get("parent_reservation")
    if not isinstance(parent_reservation, Mapping) or parent_reservation != {
        "manifest_schema": residual_split.PARENT_MANIFEST_SCHEMA,
        "manifest_sha256": residual_split.PARENT_MANIFEST_SHA256,
        "manifest_receipt": residual_split.PARENT_MANIFEST_RECEIPT,
        "split_receipt": residual_split.PARENT_SPLIT_RECEIPT,
        "historical_excluded_components": 94,
        "parent_selected_components": 160,
        "all_parent_splits_excluded": list(residual_split.PARENT_SPLIT_NAMES),
        "parent_bundle_bytes_read": False,
    }:
        raise ValueError("Fresh residual parent reservation differs")
    consumed = split_contract.get("consumed_mechanics")
    if not isinstance(consumed, Mapping) or consumed != {
        "result_schema": residual_split.CONSUMED_RESULT_SCHEMA,
        "result_status": residual_split.CONSUMED_RESULT_STATUS,
        "result_sha256": residual_split.CONSUMED_RESULT_SHA256,
        "result_receipt": residual_split.CONSUMED_RESULT_RECEIPT,
        "mechanics_rows_opened": 32,
        "causal_rows_opened": 0,
        "result_bytes_read": True,
    }:
        raise ValueError("Fresh residual consumed-mechanics binding differs")
    splits = split_contract.get("splits")
    if not isinstance(splits, Mapping) or set(splits) != set(BUNDLE_NAMES):
        raise ValueError("Fresh residual split assignments differ")
    selected_components: list[set[str]] = []
    for name in BUNDLE_NAMES:
        payload = splits[name]
        expected_pairs = residual_split.EXPECTED_PAIRS[name]
        expected_sources = sorted(source for pair in expected_pairs for source in pair)
        expected_mapping: dict[int, int] = {}
        for left, right in expected_pairs:
            expected_mapping[left] = right
            expected_mapping[right] = left
        expected_mapping_pairs = [
            [source, expected_mapping[source]] for source in expected_sources
        ]
        qualified_sources = payload.get("qualified_source_ids")
        qualified_mapping = payload.get("qualified_mapping_pairs")
        component_ids = payload.get("passage_component_ids")
        donor_component_pairs = payload.get("donor_component_pairs")
        if (
            not isinstance(payload, Mapping)
            or tuple(tuple(pair) for pair in payload.get("pairs", ()))
            != expected_pairs
            or payload.get("pair_count") != len(expected_pairs)
            or payload.get("source_indices") != expected_sources
            or payload.get("source_indices_sha256")
            != residual_split.EXPECTED_SOURCE_INDICES_SHA256[name]
            or payload.get("source_indices_sha256")
            != canonical_sha256(expected_sources)
            or payload.get("mapping_pairs") != expected_mapping_pairs
            or payload.get("mapping_pairs_sha256")
            != canonical_sha256(expected_mapping_pairs)
            or not isinstance(qualified_sources, list)
            or len(qualified_sources) != len(expected_sources)
            or payload.get("qualified_source_ids_sha256")
            != canonical_sha256(qualified_sources)
            or not isinstance(qualified_mapping, list)
            or len(qualified_mapping) != len(expected_sources)
            or payload.get("qualified_mapping_pairs_sha256")
            != canonical_sha256(qualified_mapping)
            or not isinstance(component_ids, list)
            or len(component_ids) != len(expected_sources)
            or len(set(component_ids)) != len(component_ids)
            or payload.get("passage_component_ids_sha256")
            != residual_split.EXPECTED_COMPONENT_IDS_SHA256[name]
            or payload.get("passage_component_ids_sha256")
            != canonical_sha256(component_ids)
            or not isinstance(donor_component_pairs, list)
            or len(donor_component_pairs) != len(expected_pairs)
            or payload.get("donor_component_pairs_sha256")
            != canonical_sha256(donor_component_pairs)
            or len(expected_sources) != BUNDLE_ROWS[name]
        ):
            raise ValueError(f"Fresh residual {name} split binding differs")
        qualified_by_source = dict(zip(expected_sources, qualified_sources))
        component_by_source = dict(zip(expected_sources, component_ids))
        if qualified_mapping != [
            [qualified_by_source[source], qualified_by_source[expected_mapping[source]]]
            for source in expected_sources
        ] or donor_component_pairs != [
            [component_by_source[left], component_by_source[right]]
            for left, right in expected_pairs
        ]:
            raise ValueError(f"Fresh residual {name} qualified mapping differs")
        selected_components.append(set(component_ids))
    if selected_components[0] & selected_components[1]:
        raise ValueError("Fresh residual split passage components overlap")
    if split_contract.get("capture_authorization") != {
        "default_open_splits": [],
        "sealed_inventory_only_splits": list(BUNDLE_NAMES),
        "mechanics_open_requires_signed_protocol": True,
        "causal_open_requires_mechanics_pass_and_separate_signed_protocol": True,
    }:
        raise ValueError("Fresh residual capture authorization differs")
    if split_contract.get("leakage_audit") != {
        "source_rows": 1443,
        "passage_component_count": 708,
        "historical_excluded_components": 94,
        "parent_selected_components": 160,
        "total_excluded_components": 254,
        "remaining_components_before_selection": 454,
        "selected_source_rows": 64,
        "selected_passage_components": 64,
        "cross_split_passage_component_count": 0,
    }:
        raise ValueError("Fresh residual leakage audit differs")


def materialize_prepared(
    *,
    source_rows: Sequence[Mapping[str, Any]],
    metadata_rows: Sequence[Mapping[str, Any]],
    split_contract: Mapping[str, Any],
    tokenizer_binding: Mapping[str, Any],
    source_path: Path,
    output_root: Path,
) -> Mapping[str, Any]:
    if output_root.exists():
        raise ValueError(f"Fresh residual output must be fresh: {output_root}")
    if list(metadata_rows) != parent_materializer._metadata_rows(source_rows):
        raise ValueError("Fresh residual prepared metadata differs from source rows")
    _validate_split_contract(split_contract)
    by_index = {int(row["source_index"]): row for row in source_rows}

    bundle_payloads: dict[str, bytes] = {}
    bundle_bindings: dict[str, dict[str, Any]] = {}
    for name in BUNDLE_NAMES:
        split_payload = split_contract["splits"][name]
        mapping = {
            int(source): int(donor)
            for source, donor in split_payload["mapping_pairs"]
        }
        rows = [
            _bundle_row(by_index[source], by_index[mapping[source]], split_name=name)
            for source in split_payload["source_indices"]
        ]
        payload = _bundle_bytes(rows)
        bundle_payloads[name] = payload
        bundle_bindings[name] = {
            "path": f"{name}.jsonl",
            "rows": len(rows),
            "bytes": len(payload),
            "sha256": sha256_bytes(payload),
            "payload_sha256": canonical_sha256(rows),
            "source_indices_sha256": split_payload["source_indices_sha256"],
            "qualified_source_ids_sha256": split_payload[
                "qualified_source_ids_sha256"
            ],
            "qualified_mapping_pairs_sha256": split_payload[
                "qualified_mapping_pairs_sha256"
            ],
            "row_sha256s_sha256": canonical_sha256(
                [row["row_sha256"] for row in rows]
            ),
        }

    manifest: dict[str, Any] = {
        "schema": MANIFEST_SCHEMA,
        "source": {
            **parent_split.SOURCE_BINDING.payload(),
            "recorded_path": _recorded_path(source_path),
        },
        "tokenizer": dict(tokenizer_binding),
        "metadata": {
            "rows": len(metadata_rows),
            "payload_sha256": canonical_sha256(list(metadata_rows)),
            "fields": [
                "source_index",
                "row_sha256",
                "gold_sha256",
                "write_tokens",
                "passage_signature_sha256s",
            ],
            "selection_only": True,
        },
        "split_contract": dict(split_contract),
        "file_inventory": {
            "exact_names": list(FILE_NAMES),
            "bundles": bundle_bindings,
        },
        "first_gate_access": {
            "permitted_files": ["manifest.json"],
            "inventory_only_files": ["mechanics.jsonl", "causal.jsonl"],
            "default_open_splits": [],
            "mechanics_open_requires_signed_protocol": True,
            "causal_open_requires_mechanics_pass_and_separate_signed_protocol": True,
        },
        "protected_splits_opened": [],
    }
    manifest["receipt"] = _receipt(
        "canonical_manifest_without_receipt",
        manifest,
    )

    output_root.mkdir(parents=True, exist_ok=False)
    for name in BUNDLE_NAMES:
        (output_root / f"{name}.jsonl").write_bytes(bundle_payloads[name])
    (output_root / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    if {path.name for path in output_root.iterdir()} != set(FILE_NAMES):
        raise RuntimeError("Fresh residual materialization file inventory differs")
    return manifest


def materialize(
    *,
    source_path: Path,
    tokenizer_path: Path,
    parent_manifest_path: Path,
    consumed_result_path: Path,
    output_root: Path,
) -> Mapping[str, Any]:
    if source_path.resolve() != (PROJECT_ROOT / parent_split.SOURCE_RELATIVE_PATH).resolve():
        raise ValueError("Fresh residual materializer requires the pinned FIT source")
    tokenizer_binding = source_loader.validate_tokenizer_artifacts(tokenizer_path)
    source_rows = source_loader.load_source_rows(source_path)
    tokenizer = source_loader.AutoTokenizer.from_pretrained(
        tokenizer_path,
        local_files_only=True,
        trust_remote_code=False,
    )
    source_loader.add_write_token_counts(source_rows, tokenizer)
    metadata_rows = parent_materializer._metadata_rows(source_rows)
    parent_reservation = residual_split.load_parent_reservation(parent_manifest_path)
    consumed_mechanics = residual_split.load_consumed_mechanics(
        consumed_result_path,
        parent_reservation=parent_reservation,
    )
    split_contract = residual_split.build_split(
        metadata_rows,
        parent_reservation,
        consumed_mechanics,
    )
    return materialize_prepared(
        source_rows=source_rows,
        metadata_rows=metadata_rows,
        split_contract=split_contract,
        tokenizer_binding=tokenizer_binding,
        source_path=source_path,
        output_root=output_root,
    )


def load_manifest_only(manifest_path: Path) -> dict[str, Any]:
    if manifest_path.name != "manifest.json":
        raise ValueError("Fresh residual manifest loader accepts only manifest.json")
    payload = manifest_path.read_bytes()
    if sha256_bytes(payload) != SEALED_MANIFEST_SHA256:
        raise ValueError("Fresh residual manifest file hash differs")
    try:
        manifest = json.loads(payload.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("Fresh residual manifest is invalid JSON") from error
    if not isinstance(manifest, Mapping):
        raise ValueError("Fresh residual manifest root is not an object")
    if manifest.get("schema") != MANIFEST_SCHEMA:
        raise ValueError("Fresh residual manifest schema differs")
    source = manifest.get("source")
    if not isinstance(source, Mapping) or {
        key: source.get(key) for key in ("namespace", "path", "sha256", "rows")
    } != parent_split.SOURCE_BINDING.payload():
        raise ValueError("Fresh residual manifest source differs")
    if manifest.get("protected_splits_opened") != []:
        raise ValueError("Fresh residual manifest opened protected data")
    if manifest.get("first_gate_access") != {
        "permitted_files": ["manifest.json"],
        "inventory_only_files": ["mechanics.jsonl", "causal.jsonl"],
        "default_open_splits": [],
        "mechanics_open_requires_signed_protocol": True,
        "causal_open_requires_mechanics_pass_and_separate_signed_protocol": True,
    }:
        raise ValueError("Fresh residual first-gate access contract differs")
    _validate_receipt(
        manifest,
        payload_scope="canonical_manifest_without_receipt",
        description="Fresh residual manifest",
    )
    split_contract = manifest.get("split_contract")
    if not isinstance(split_contract, Mapping):
        raise ValueError("Fresh residual split contract is missing")
    _validate_split_contract(split_contract)
    inventory = manifest.get("file_inventory")
    if (
        not isinstance(inventory, Mapping)
        or inventory.get("exact_names") != list(FILE_NAMES)
        or not isinstance(inventory.get("bundles"), Mapping)
        or set(inventory["bundles"]) != set(BUNDLE_NAMES)
    ):
        raise ValueError("Fresh residual manifest inventory differs")
    for name in BUNDLE_NAMES:
        binding = inventory["bundles"][name]
        if (
            not isinstance(binding, Mapping)
            or binding.get("path") != f"{name}.jsonl"
            or binding.get("rows") != BUNDLE_ROWS[name]
            or not all(
                isinstance(binding.get(field), str)
                and len(binding[field]) == 64
                for field in (
                    "sha256",
                    "payload_sha256",
                    "source_indices_sha256",
                    "qualified_source_ids_sha256",
                    "qualified_mapping_pairs_sha256",
                    "row_sha256s_sha256",
                )
            )
        ):
            raise ValueError(f"Fresh residual {name} inventory binding differs")
    return dict(manifest)


def read_authorized_bundle(
    root: Path,
    manifest: Mapping[str, Any],
    name: str,
    *,
    allow_mechanics: bool = False,
    allow_causal: bool = False,
) -> list[dict[str, Any]]:
    if name not in BUNDLE_NAMES:
        raise ValueError(f"Invalid fresh residual bundle: {name}")
    if name == "mechanics" and not allow_mechanics:
        raise PermissionError("Fresh residual mechanics requires signed authorization")
    if name == "causal":
        del allow_causal
        raise PermissionError(
            "Fresh residual causal requires a dedicated post-mechanics signed loader"
        )
    on_disk_manifest = load_manifest_only(root / "manifest.json")
    if dict(manifest) != on_disk_manifest:
        raise ValueError("Fresh residual authorized manifest differs from sealed bytes")
    binding = on_disk_manifest["file_inventory"]["bundles"][name]
    if binding["path"] != f"{name}.jsonl":
        raise ValueError(f"Fresh residual {name} bundle path differs")
    payload = (root / f"{name}.jsonl").read_bytes()
    if sha256_bytes(payload) != binding["sha256"]:
        raise ValueError(f"Fresh residual {name} file hash differs")
    try:
        lines = payload.decode("utf-8").splitlines()
    except UnicodeDecodeError as error:
        raise ValueError(f"Fresh residual {name} bundle is not UTF-8") from error
    split_payload = on_disk_manifest["split_contract"]["splits"][name]
    mapping = {
        int(source): int(donor)
        for source, donor in split_payload["mapping_pairs"]
    }
    rows: list[dict[str, Any]] = []
    for line in lines:
        row = json.loads(line)
        if not isinstance(row, Mapping):
            raise ValueError(f"Fresh residual {name} row is not an object")
        _validate_receipt(
            row,
            payload_scope="canonical_bundle_row_without_receipt",
            description=f"Fresh residual {name} row",
        )
        source_index = int(row["source_index"])
        donor_index = int(row["donor_source_index"])
        raw_line = row.get("raw_line")
        if (
            row.get("schema") != ROW_SCHEMA
            or row.get("split") != name
            or donor_index != mapping.get(source_index)
            or not isinstance(raw_line, str)
            or hashlib.sha256(raw_line.encode("utf-8")).hexdigest()
            != row.get("row_sha256")
            or row.get("qualified_source_id")
            != parent_split.SOURCE_BINDING.qualified_id(
                source_index, row["row_sha256"]
            )
            or row.get("qualified_donor_source_id")
            != parent_split.SOURCE_BINDING.qualified_id(
                donor_index, row["donor_row_sha256"]
            )
        ):
            raise ValueError(f"Fresh residual {name} row binding differs")
        rows.append(dict(row))
    if (
        len(rows) != BUNDLE_ROWS[name]
        or canonical_sha256(rows) != binding["payload_sha256"]
        or [int(row["source_index"]) for row in rows]
        != split_payload["source_indices"]
    ):
        raise ValueError(f"Fresh residual {name} bundle payload differs")
    row_hashes = {int(row["source_index"]): row["row_sha256"] for row in rows}
    if any(
        row["donor_row_sha256"] != row_hashes[int(row["donor_source_index"])]
        for row in rows
    ):
        raise ValueError(f"Fresh residual {name} donor row hash differs")
    return rows


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-path",
        type=Path,
        default=PROJECT_ROOT / parent_split.SOURCE_RELATIVE_PATH,
    )
    parser.add_argument("--tokenizer-path", type=Path, required=True)
    parser.add_argument(
        "--parent-manifest-path", type=Path, default=DEFAULT_PARENT_MANIFEST
    )
    parser.add_argument(
        "--consumed-result-path", type=Path, default=DEFAULT_CONSUMED_RESULT
    )
    parser.add_argument("--output-root", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    manifest = materialize(
        source_path=args.source_path.expanduser().resolve(strict=True),
        tokenizer_path=args.tokenizer_path.expanduser().resolve(strict=True),
        parent_manifest_path=args.parent_manifest_path.expanduser().resolve(strict=True),
        consumed_result_path=args.consumed_result_path.expanduser().resolve(strict=True),
        output_root=args.output_root.expanduser().resolve(),
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
