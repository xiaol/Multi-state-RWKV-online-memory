#!/usr/bin/env python3
"""Materialize an explicitly open development split for cumulative residuals."""

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
    rwkv_source_cumulative_residual_development_split as development_split,
)
from experiments.rethinking_rwkv_ms_gemma import (  # noqa: E402
    rwkv_source_cumulative_residual_split as protected_split,
)


MANIFEST_SCHEMA = (
    "rwkv_ms_natural_memory_native_rwkv_source_cumulative_residual_development.v1"
)
ROW_SCHEMA = (
    "rwkv_ms_natural_memory_native_rwkv_source_cumulative_residual_development_row.v1"
)
SEALED_MANIFEST_SHA256 = (
    "59b9926fa1023fa39f5616bdfa3f0bf4d1d4549d2f7e948dd628536a1bdb9f38"
)
FILE_NAMES = ("manifest.json", "development.jsonl")
DEVELOPMENT_ROWS = development_split.DEVELOPMENT_ROWS
DEFAULT_PARENT_MANIFEST = (
    PROJECT_ROOT
    / "experiments/rethinking_rwkv_ms_gemma/local_artifacts/"
    "natural_memory_native_rwkv_continuous_write_open_fit_v1/manifest.json"
)
DEFAULT_PROTECTED_MANIFEST = (
    PROJECT_ROOT
    / "experiments/rethinking_rwkv_ms_gemma/local_artifacts/"
    "natural_memory_native_rwkv_source_cumulative_residual_v1/manifest.json"
)


def canonical_sha256(value: Any) -> str:
    return development_split.canonical_sha256(value)


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
) -> dict[str, Any]:
    source_index = int(source["source_index"])
    donor_index = int(donor["source_index"])
    unsigned = {
        "schema": ROW_SCHEMA,
        "split": "development",
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


def materialize_prepared(
    *,
    source_rows: Sequence[Mapping[str, Any]],
    metadata_rows: Sequence[Mapping[str, Any]],
    split_contract: Mapping[str, Any],
    tokenizer_binding: Mapping[str, Any],
    source_path: Path,
    output_root: Path,
) -> dict[str, Any]:
    if output_root.exists():
        raise ValueError(f"Open residual development output must be fresh: {output_root}")
    if list(metadata_rows) != parent_materializer._metadata_rows(source_rows):
        raise ValueError("Open residual development metadata differs from source rows")
    development_split.validate_split_contract(split_contract)
    by_index = {int(row["source_index"]): row for row in source_rows}
    split_payload = split_contract["splits"]["development"]
    mapping = {
        int(source): int(donor)
        for source, donor in split_payload["mapping_pairs"]
    }
    rows = [
        _bundle_row(by_index[source], by_index[mapping[source]])
        for source in split_payload["source_indices"]
    ]
    bundle_payload = _bundle_bytes(rows)
    bundle_binding = {
        "path": "development.jsonl",
        "rows": len(rows),
        "bytes": len(bundle_payload),
        "sha256": sha256_bytes(bundle_payload),
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
            "bundles": {"development": bundle_binding},
        },
        "access": {
            "default_open_splits": ["development"],
            "development_is_explicitly_open": True,
            "protected_reservation_bundle_bytes_read": False,
        },
        "open_splits": ["development"],
        "protected_splits_opened": [],
    }
    manifest["receipt"] = _receipt("canonical_manifest_without_receipt", manifest)

    output_root.mkdir(parents=True, exist_ok=False)
    (output_root / "development.jsonl").write_bytes(bundle_payload)
    (output_root / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    if {path.name for path in output_root.iterdir()} != set(FILE_NAMES):
        raise RuntimeError("Open residual development file inventory differs")
    return manifest


def materialize(
    *,
    source_path: Path,
    tokenizer_path: Path,
    parent_manifest_path: Path,
    protected_manifest_path: Path,
    output_root: Path,
) -> dict[str, Any]:
    expected_source = (PROJECT_ROOT / parent_split.SOURCE_RELATIVE_PATH).resolve()
    if source_path.resolve() != expected_source:
        raise ValueError("Open residual development requires the pinned FIT source")
    tokenizer_binding = source_loader.validate_tokenizer_artifacts(tokenizer_path)
    source_rows = source_loader.load_source_rows(source_path)
    tokenizer = source_loader.AutoTokenizer.from_pretrained(
        tokenizer_path,
        local_files_only=True,
        trust_remote_code=False,
    )
    source_loader.add_write_token_counts(source_rows, tokenizer)
    metadata_rows = parent_materializer._metadata_rows(source_rows)
    parent_reservation = protected_split.load_parent_reservation(
        parent_manifest_path
    )
    protected_manifest = development_split.load_protected_manifest(
        protected_manifest_path
    )
    split_contract = development_split.build_split(
        metadata_rows,
        parent_reservation,
        protected_manifest,
    )
    return materialize_prepared(
        source_rows=source_rows,
        metadata_rows=metadata_rows,
        split_contract=split_contract,
        tokenizer_binding=tokenizer_binding,
        source_path=source_path,
        output_root=output_root,
    )


def load_manifest(manifest_path: Path) -> dict[str, Any]:
    if manifest_path.name != "manifest.json":
        raise ValueError("Open residual development loader accepts only manifest.json")
    payload = manifest_path.read_bytes()
    if sha256_bytes(payload) != SEALED_MANIFEST_SHA256:
        raise ValueError("Open residual development manifest file hash differs")
    try:
        manifest = json.loads(payload.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("Open residual development manifest is invalid JSON") from error
    if not isinstance(manifest, Mapping):
        raise ValueError("Open residual development manifest root is not an object")
    if (
        manifest.get("schema") != MANIFEST_SCHEMA
        or manifest.get("open_splits") != ["development"]
        or manifest.get("protected_splits_opened") != []
        or manifest.get("access")
        != {
            "default_open_splits": ["development"],
            "development_is_explicitly_open": True,
            "protected_reservation_bundle_bytes_read": False,
        }
    ):
        raise ValueError("Open residual development manifest contract differs")
    source = manifest.get("source")
    if not isinstance(source, Mapping) or {
        key: source.get(key) for key in ("namespace", "path", "sha256", "rows")
    } != parent_split.SOURCE_BINDING.payload():
        raise ValueError("Open residual development source binding differs")
    _validate_receipt(
        manifest,
        payload_scope="canonical_manifest_without_receipt",
        description="Open residual development manifest",
    )
    split_contract = manifest.get("split_contract")
    if not isinstance(split_contract, Mapping):
        raise ValueError("Open residual development split contract is missing")
    development_split.validate_split_contract(split_contract)
    inventory = manifest.get("file_inventory")
    bundles = inventory.get("bundles") if isinstance(inventory, Mapping) else None
    binding = bundles.get("development") if isinstance(bundles, Mapping) else None
    if (
        inventory.get("exact_names") != list(FILE_NAMES)
        or set(bundles) != {"development"}
        or not isinstance(binding, Mapping)
        or binding.get("path") != "development.jsonl"
        or binding.get("rows") != DEVELOPMENT_ROWS
        or not all(
            isinstance(binding.get(field), str) and len(binding[field]) == 64
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
        raise ValueError("Open residual development inventory differs")
    return dict(manifest)


def read_open_development(
    root: Path,
    manifest: Mapping[str, Any],
) -> list[dict[str, Any]]:
    on_disk_manifest = load_manifest(root / "manifest.json")
    if dict(manifest) != on_disk_manifest:
        raise ValueError("Open residual development manifest differs from sealed bytes")
    binding = on_disk_manifest["file_inventory"]["bundles"]["development"]
    if binding["path"] != "development.jsonl":
        raise ValueError("Open residual development bundle path differs")
    payload = (root / "development.jsonl").read_bytes()
    if sha256_bytes(payload) != binding["sha256"]:
        raise ValueError("Open residual development file hash differs")
    try:
        lines = payload.decode("utf-8").splitlines()
    except UnicodeDecodeError as error:
        raise ValueError("Open residual development bundle is not UTF-8") from error
    split_payload = on_disk_manifest["split_contract"]["splits"]["development"]
    mapping = {
        int(source): int(donor)
        for source, donor in split_payload["mapping_pairs"]
    }
    rows: list[dict[str, Any]] = []
    for line in lines:
        row = json.loads(line)
        if not isinstance(row, Mapping):
            raise ValueError("Open residual development row is not an object")
        _validate_receipt(
            row,
            payload_scope="canonical_bundle_row_without_receipt",
            description="Open residual development row",
        )
        source_index = int(row["source_index"])
        donor_index = int(row["donor_source_index"])
        raw_line = row.get("raw_line")
        if (
            row.get("schema") != ROW_SCHEMA
            or row.get("split") != "development"
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
            raise ValueError("Open residual development row binding differs")
        rows.append(dict(row))
    if (
        len(rows) != DEVELOPMENT_ROWS
        or canonical_sha256(rows) != binding["payload_sha256"]
        or [int(row["source_index"]) for row in rows]
        != split_payload["source_indices"]
    ):
        raise ValueError("Open residual development bundle payload differs")
    row_hashes = {int(row["source_index"]): row["row_sha256"] for row in rows}
    if any(
        row["donor_row_sha256"] != row_hashes[int(row["donor_source_index"])]
        for row in rows
    ):
        raise ValueError("Open residual development donor row hash differs")
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
        "--parent-manifest-path",
        type=Path,
        default=DEFAULT_PARENT_MANIFEST,
    )
    parser.add_argument(
        "--protected-manifest-path",
        type=Path,
        default=DEFAULT_PROTECTED_MANIFEST,
    )
    parser.add_argument("--output-root", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    manifest = materialize(
        source_path=args.source_path.expanduser().resolve(strict=True),
        tokenizer_path=args.tokenizer_path.expanduser().resolve(strict=True),
        parent_manifest_path=args.parent_manifest_path.expanduser().resolve(strict=True),
        protected_manifest_path=args.protected_manifest_path.expanduser().resolve(
            strict=True
        ),
        output_root=args.output_root.expanduser().resolve(),
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
