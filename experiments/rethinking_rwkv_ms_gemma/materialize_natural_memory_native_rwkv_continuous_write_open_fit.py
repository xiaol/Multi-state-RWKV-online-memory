#!/usr/bin/env python3
"""Materialize dataset-qualified open-FIT bundles for continuous RWKV writes."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Any, Iterable, Mapping, Sequence


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = Path(__file__).resolve().parents[2]
for import_root in (SCRIPT_DIR, PROJECT_ROOT):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

from experiments.rethinking_rwkv_ms_gemma import (  # noqa: E402
    materialize_natural_memory_native_rwkv_bidirectional_sign_open_fit as prior_materializer,
)
from experiments.rethinking_rwkv_ms_gemma import (  # noqa: E402
    rwkv_continuous_write_fit_split as fit_split,
)


MANIFEST_SCHEMA = "rwkv_ms_natural_memory_native_rwkv_continuous_write_open_fit.v2"
ROW_SCHEMA = "rwkv_ms_natural_memory_native_rwkv_continuous_write_open_fit_row.v2"
BUNDLE_NAMES = fit_split.SPLIT_NAMES
BUNDLE_ROWS = {"fit": 64, "retrieval": 32, "mechanics": 32, "causal": 32}
DEFAULT_PRIOR_MANIFEST = (
    PROJECT_ROOT
    / "experiments/rethinking_rwkv_ms_gemma/local_artifacts/"
    "natural_memory_native_rwkv_bidirectional_sign_open_fit_v1/manifest.json"
)
FILE_NAMES = (
    "manifest.json",
    "fit.jsonl",
    "retrieval.jsonl",
    "mechanics.jsonl",
    "causal.jsonl",
)
DEFAULT_OPEN_BUNDLES = fit_split.CAPTURE_SPLITS


def canonical_json(value: Any) -> str:
    return fit_split.canonical_json(value)


def canonical_sha256(value: Any) -> str:
    return fit_split.canonical_sha256(value)


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
    expected = _receipt(payload_scope, unsigned)
    if dict(receipt) != expected:
        raise ValueError(f"{description} receipt differs")


def _recorded_path(path: Path) -> str:
    try:
        return path.resolve().relative_to(PROJECT_ROOT.resolve()).as_posix()
    except ValueError:
        return str(path.resolve())


def _metadata_rows(source_rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    if len(source_rows) != fit_split.SOURCE_ROWS:
        raise ValueError("Continuous-write source row count differs")
    metadata: list[dict[str, Any]] = []
    for expected_index, row in enumerate(source_rows):
        if int(row.get("source_index", -1)) != expected_index:
            raise ValueError("Continuous-write source indices are not contiguous")
        row_sha256 = row.get("row_sha256")
        raw_line = row.get("raw_line")
        messages = row.get("messages")
        gold = row.get("gold")
        write_tokens = row.get("write_tokens")
        if (
            not isinstance(row_sha256, str)
            or not isinstance(raw_line, str)
            or hashlib.sha256(raw_line.encode("utf-8")).hexdigest() != row_sha256
            or not isinstance(messages, list)
            or len(messages) != 3
            or not isinstance(messages[1], Mapping)
            or not isinstance(messages[1].get("content"), str)
            or not isinstance(gold, tuple)
            or isinstance(write_tokens, bool)
            or not isinstance(write_tokens, int)
            or write_tokens < 1
        ):
            raise ValueError(f"Continuous-write source metadata differs: {expected_index}")
        signatures = fit_split.passage_signature_sha256s(messages[1]["content"])
        metadata.append(
            {
                "source_index": expected_index,
                "row_sha256": row_sha256,
                "gold_sha256": canonical_sha256(list(gold)),
                "write_tokens": write_tokens,
                "passage_signature_sha256s": list(signatures),
            }
        )
    return metadata


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
        "qualified_source_id": fit_split.SOURCE_BINDING.qualified_id(
            source_index,
            str(source["row_sha256"]),
        ),
        "row_sha256": str(source["row_sha256"]),
        "donor_source_index": donor_index,
        "qualified_donor_source_id": fit_split.SOURCE_BINDING.qualified_id(
            donor_index,
            str(donor["row_sha256"]),
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
    if split_contract.get("schema") != fit_split.SCHEMA:
        raise ValueError("Continuous-write split contract schema differs")
    if split_contract.get("source") != fit_split.SOURCE_BINDING.payload():
        raise ValueError("Continuous-write split source binding differs")
    if split_contract.get("protected_splits_opened") != []:
        raise ValueError("Continuous-write split contract opened protected data")
    if split_contract.get("sealed_bundle_bytes_read") is not False:
        raise ValueError("Continuous-write split contract read sealed bundle bytes")
    _validate_receipt(
        split_contract,
        payload_scope="canonical_split_without_receipt",
        description="Continuous-write split contract",
    )
    prior = split_contract.get("prior_reservation")
    if (
        not isinstance(prior, Mapping)
        or prior.get("source") != fit_split.SOURCE_BINDING.payload()
        or prior.get("manifest_sha256") != fit_split.PRIOR_MANIFEST_SHA256
        or prior.get("manifest_source_indices_sha256")
        != fit_split.PRIOR_SOURCE_INDICES_SHA256
        or prior.get("manifest_mapping_sha256") != fit_split.PRIOR_MAPPING_SHA256
        or prior.get("manifest_reserved_rows") != fit_split.PRIOR_RESERVED_ROWS
        or prior.get("bundle_bytes_read") is not False
    ):
        raise ValueError("Continuous-write prior reservation binding differs")
    splits = split_contract.get("splits")
    if not isinstance(splits, Mapping) or set(splits) != set(BUNDLE_NAMES):
        raise ValueError("Continuous-write split assignments differ")
    selected: list[set[int]] = []
    for name in BUNDLE_NAMES:
        payload = splits[name]
        pair_count = payload.get("pair_count") if isinstance(payload, Mapping) else None
        if (
            not isinstance(payload, Mapping)
            or isinstance(pair_count, bool)
            or not isinstance(pair_count, int)
            or pair_count * 2 != BUNDLE_ROWS[name]
            or len(payload.get("source_indices", ())) != BUNDLE_ROWS[name]
            or canonical_sha256(payload.get("qualified_source_ids"))
            != payload.get("qualified_source_ids_sha256")
            or canonical_sha256(payload.get("qualified_mapping_pairs"))
            != payload.get("qualified_mapping_pairs_sha256")
            or canonical_sha256(payload.get("passage_component_ids"))
            != payload.get("passage_component_ids_sha256")
        ):
            raise ValueError(f"Continuous-write {name} split binding differs")
        selected.append(set(int(source) for source in payload["source_indices"]))
    if any(
        left & right
        for index, left in enumerate(selected)
        for right in selected[index + 1 :]
    ):
        raise ValueError("Continuous-write split assignments overlap")
    if split_contract.get("capture_authorization") != {
        "capture_splits": ["fit", "retrieval"],
        "sealed_inventory_only_splits": ["mechanics", "causal"],
        "captured_rows": 96,
        "sealed_rows": 64,
    }:
        raise ValueError("Continuous-write capture authorization differs")


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
        raise ValueError(f"Continuous-write output must be fresh: {output_root}")
    if list(metadata_rows) != _metadata_rows(source_rows):
        raise ValueError("Continuous-write prepared metadata differs from source rows")
    _validate_split_contract(split_contract)
    by_index = {int(row["source_index"]): row for row in source_rows}

    bundle_rows: dict[str, list[dict[str, Any]]] = {}
    bundle_payloads: dict[str, bytes] = {}
    bundle_bindings: dict[str, dict[str, Any]] = {}
    for name in BUNDLE_NAMES:
        split_payload = split_contract["splits"][name]
        mapping = {
            int(source): int(donor)
            for source, donor in split_payload["mapping_pairs"]
        }
        rows = [
            _bundle_row(
                by_index[source],
                by_index[mapping[source]],
                split_name=name,
            )
            for source in split_payload["source_indices"]
        ]
        payload = _bundle_bytes(rows)
        bundle_rows[name] = rows
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
            **fit_split.SOURCE_BINDING.payload(),
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
            "permitted_files": ["manifest.json", "fit.jsonl", "retrieval.jsonl"],
            "inventory_only_files": ["mechanics.jsonl", "causal.jsonl"],
            "default_open_bundles": list(DEFAULT_OPEN_BUNDLES),
            "mechanics_open_requires_separate_authorization": True,
            "causal_open_requires_separate_authorization": True,
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
        raise RuntimeError("Continuous-write materialization file inventory differs")
    return manifest


def materialize(
    *,
    source_path: Path,
    tokenizer_path: Path,
    prior_manifest_path: Path,
    output_root: Path,
) -> Mapping[str, Any]:
    if source_path.resolve() != (PROJECT_ROOT / fit_split.SOURCE_RELATIVE_PATH).resolve():
        raise ValueError("Continuous-write materializer requires the pinned FIT source")
    tokenizer_binding = prior_materializer.validate_tokenizer_artifacts(tokenizer_path)
    source_rows = prior_materializer.load_source_rows(source_path)
    tokenizer = prior_materializer.AutoTokenizer.from_pretrained(
        tokenizer_path,
        local_files_only=True,
        trust_remote_code=False,
    )
    prior_materializer.add_write_token_counts(source_rows, tokenizer)
    metadata_rows = _metadata_rows(source_rows)
    prior = fit_split.load_prior_reservation(prior_manifest_path)
    split_contract = fit_split.build_split(metadata_rows, prior)
    return materialize_prepared(
        source_rows=source_rows,
        metadata_rows=metadata_rows,
        split_contract=split_contract,
        tokenizer_binding=tokenizer_binding,
        source_path=source_path,
        output_root=output_root,
    )


def _validate_inventory(root: Path, manifest: Mapping[str, Any]) -> None:
    inventory = manifest.get("file_inventory")
    if not isinstance(inventory, Mapping) or inventory.get("exact_names") != list(
        FILE_NAMES
    ):
        raise ValueError("Continuous-write manifest file inventory differs")
    paths = tuple(root.iterdir())
    if {path.name for path in paths} != set(FILE_NAMES) or any(
        not path.is_file() or path.is_symlink() for path in paths
    ):
        raise ValueError("Continuous-write materialization file inventory differs")
    bundles = inventory.get("bundles")
    if not isinstance(bundles, Mapping) or set(bundles) != set(BUNDLE_NAMES):
        raise ValueError("Continuous-write bundle inventory differs")
    for name in BUNDLE_NAMES:
        binding = bundles[name]
        path = root / f"{name}.jsonl"
        if (
            not isinstance(binding, Mapping)
            or binding.get("path") != path.name
            or binding.get("rows") != BUNDLE_ROWS[name]
            or binding.get("bytes") != path.stat().st_size
        ):
            raise ValueError(f"Continuous-write {name} inventory binding differs")


def _validate_manifest(manifest: Mapping[str, Any]) -> None:
    if manifest.get("schema") != MANIFEST_SCHEMA:
        raise ValueError("Continuous-write manifest schema differs")
    source = manifest.get("source")
    if not isinstance(source, Mapping) or {
        key: source.get(key) for key in ("namespace", "path", "sha256", "rows")
    } != fit_split.SOURCE_BINDING.payload():
        raise ValueError("Continuous-write manifest source differs")
    if manifest.get("protected_splits_opened") != []:
        raise ValueError("Continuous-write manifest opened protected data")
    access = manifest.get("first_gate_access")
    if not isinstance(access, Mapping) or access != {
        "permitted_files": ["manifest.json", "fit.jsonl", "retrieval.jsonl"],
        "inventory_only_files": ["mechanics.jsonl", "causal.jsonl"],
        "default_open_bundles": list(DEFAULT_OPEN_BUNDLES),
        "mechanics_open_requires_separate_authorization": True,
        "causal_open_requires_separate_authorization": True,
    }:
        raise ValueError("Continuous-write first-gate access contract differs")
    _validate_receipt(
        manifest,
        payload_scope="canonical_manifest_without_receipt",
        description="Continuous-write manifest",
    )
    split_contract = manifest.get("split_contract")
    if not isinstance(split_contract, Mapping):
        raise ValueError("Continuous-write manifest split contract is missing")
    _validate_split_contract(split_contract)


def _read_bundle(
    root: Path,
    manifest: Mapping[str, Any],
    name: str,
) -> list[dict[str, Any]]:
    binding = manifest["file_inventory"]["bundles"][name]
    path = root / binding["path"]
    payload = path.read_bytes()
    if sha256_bytes(payload) != binding["sha256"]:
        raise ValueError(f"Continuous-write {name} file hash differs")
    try:
        lines = payload.decode("utf-8").splitlines()
    except UnicodeDecodeError as error:
        raise ValueError(f"Continuous-write {name} bundle is not UTF-8") from error
    rows: list[dict[str, Any]] = []
    split_payload = manifest["split_contract"]["splits"][name]
    mapping = {int(source): int(donor) for source, donor in split_payload["mapping_pairs"]}
    for line in lines:
        row = json.loads(line)
        if not isinstance(row, Mapping) or set(row) != {
            "schema",
            "split",
            "source_index",
            "qualified_source_id",
            "row_sha256",
            "donor_source_index",
            "qualified_donor_source_id",
            "donor_row_sha256",
            "raw_line",
            "receipt",
        }:
            raise ValueError(f"Continuous-write {name} row shape differs")
        _validate_receipt(
            row,
            payload_scope="canonical_bundle_row_without_receipt",
            description=f"Continuous-write {name} row",
        )
        source = int(row["source_index"])
        donor = int(row["donor_source_index"])
        raw_line = row["raw_line"]
        if (
            row["schema"] != ROW_SCHEMA
            or row["split"] != name
            or row["qualified_source_id"]
            != fit_split.SOURCE_BINDING.qualified_id(source, row["row_sha256"])
            or donor != mapping.get(source)
            or row["qualified_donor_source_id"]
            != fit_split.SOURCE_BINDING.qualified_id(donor, row["donor_row_sha256"])
            or not isinstance(raw_line, str)
            or hashlib.sha256(raw_line.encode("utf-8")).hexdigest()
            != row["row_sha256"]
        ):
            raise ValueError(f"Continuous-write {name} row binding differs")
        rows.append(dict(row))
    if (
        len(rows) != binding["rows"]
        or canonical_sha256(rows) != binding["payload_sha256"]
        or [int(row["source_index"]) for row in rows]
        != split_payload["source_indices"]
        or canonical_sha256([row["row_sha256"] for row in rows])
        != binding["row_sha256s_sha256"]
    ):
        raise ValueError(f"Continuous-write {name} bundle payload differs")
    row_hashes = {int(row["source_index"]): row["row_sha256"] for row in rows}
    if any(
        row["donor_row_sha256"] != row_hashes[int(row["donor_source_index"])]
        for row in rows
    ):
        raise ValueError(f"Continuous-write {name} donor row hash differs")
    return rows


def validate_materialization(
    root: Path,
    *,
    bundles: Iterable[str] = DEFAULT_OPEN_BUNDLES,
    allow_mechanics: bool = False,
    allow_causal: bool = False,
) -> dict[str, Any]:
    requested = tuple(bundles)
    if len(set(requested)) != len(requested) or any(
        name not in BUNDLE_NAMES for name in requested
    ):
        raise ValueError(f"Invalid continuous-write bundle selection: {requested}")
    if "mechanics" in requested and not allow_mechanics:
        raise PermissionError(
            "Continuous-write mechanics bundle requires separate authorization"
        )
    if "causal" in requested and not allow_causal:
        raise PermissionError("Continuous-write causal bundle requires separate authorization")
    manifest_path = root / "manifest.json"
    manifest_bytes = manifest_path.read_bytes()
    try:
        manifest = json.loads(manifest_bytes.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("Continuous-write manifest is invalid JSON") from error
    if not isinstance(manifest, Mapping):
        raise ValueError("Continuous-write manifest root is not an object")
    _validate_manifest(manifest)
    _validate_inventory(root, manifest)

    groups = {
        name: _read_bundle(root, manifest, name)
        for name in requested
    }
    read_files = ["manifest.json", *(f"{name}.jsonl" for name in requested)]
    inventory_only = [
        f"{name}.jsonl" for name in BUNDLE_NAMES if name not in requested
    ]
    return {
        "manifest": dict(manifest),
        "groups": groups,
        "file_access_audit": {
            "byte_read_files": read_files,
            "inventory_only_files": inventory_only,
            "mechanics_bytes_read": "mechanics" in requested,
            "causal_bytes_read": "causal" in requested,
            "exact_inventory_validated": True,
        },
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-path",
        type=Path,
        default=PROJECT_ROOT / fit_split.SOURCE_RELATIVE_PATH,
    )
    parser.add_argument("--tokenizer-path", type=Path, required=True)
    parser.add_argument(
        "--prior-manifest-path",
        type=Path,
        default=DEFAULT_PRIOR_MANIFEST,
    )
    parser.add_argument("--output-root", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    manifest = materialize(
        source_path=args.source_path.expanduser().resolve(strict=True),
        tokenizer_path=args.tokenizer_path.expanduser().resolve(strict=True),
        prior_manifest_path=args.prior_manifest_path.expanduser().resolve(strict=True),
        output_root=args.output_root.expanduser().resolve(),
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
