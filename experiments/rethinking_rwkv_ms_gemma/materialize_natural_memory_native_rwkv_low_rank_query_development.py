#!/usr/bin/env python3
"""Materialize the fresh low-rank-query open development bundle."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[1]
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
    materialize_natural_memory_native_rwkv_source_cumulative_residual_development as prior_materializer,
)
from experiments.rethinking_rwkv_ms_gemma import (  # noqa: E402
    materialize_natural_memory_native_rwkv_source_multi_anchor_bundle_development as multi_materializer,
)
from experiments.rethinking_rwkv_ms_gemma import (  # noqa: E402
    materialize_natural_memory_native_rwkv_source_multi_anchor_bundle_repair_development as repair_materializer,
)
from experiments.rethinking_rwkv_ms_gemma import (  # noqa: E402
    materialize_natural_memory_native_rwkv_weighted_renewal_bundle_development as weighted_materializer,
)
from experiments.rethinking_rwkv_ms_gemma import (  # noqa: E402
    rwkv_continuous_write_fit_split as parent_split,
)
from experiments.rethinking_rwkv_ms_gemma import (  # noqa: E402
    rwkv_low_rank_query_development_split as development_split,
)
from experiments.rethinking_rwkv_ms_gemma import (  # noqa: E402
    rwkv_source_cumulative_residual_development_split as prior_split,
)
from experiments.rethinking_rwkv_ms_gemma import (  # noqa: E402
    rwkv_source_cumulative_residual_split as parent_reservation_split,
)


MANIFEST_SCHEMA = "rwkv_ms_natural_memory_native_rwkv_low_rank_query_development.v1"
ROW_SCHEMA = "rwkv_ms_natural_memory_native_rwkv_low_rank_query_development_row.v1"
FILE_NAMES = ("manifest.json", "development.jsonl")
DEVELOPMENT_ROWS = development_split.DEVELOPMENT_ROWS
SEALED_MANIFEST_SHA256 = (
    "f78938e45dbfb508be07c32844315af44e5992817a03b5bba42f909c2265b755"
)

DEFAULT_SOURCE = PROJECT_ROOT / parent_split.SOURCE_RELATIVE_PATH
DEFAULT_TOKENIZER = Path("/root/X/.cache/hf/gemma-4-E4B-it-a4c2d58")
DEFAULT_PARENT_MANIFEST = PROJECT_ROOT / (
    "experiments/rethinking_rwkv_ms_gemma/local_artifacts/"
    "natural_memory_native_rwkv_continuous_write_open_fit_v1/manifest.json"
)
DEFAULT_PROTECTED_MANIFEST = PROJECT_ROOT / (
    "experiments/rethinking_rwkv_ms_gemma/local_artifacts/"
    "natural_memory_native_rwkv_source_cumulative_residual_v1/manifest.json"
)
DEFAULT_PRIOR_MANIFEST = PROJECT_ROOT / (
    "experiments/rethinking_rwkv_ms_gemma/local_artifacts/"
    "natural_memory_native_rwkv_source_cumulative_residual_development_v1/manifest.json"
)
DEFAULT_MULTI_MANIFEST = PROJECT_ROOT / (
    "experiments/rethinking_rwkv_ms_gemma/local_artifacts/"
    "natural_memory_native_rwkv_source_multi_anchor_bundle_development_v1/manifest.json"
)
DEFAULT_WEIGHTED_MANIFEST = PROJECT_ROOT / (
    "experiments/rethinking_rwkv_ms_gemma/local_artifacts/"
    "natural_memory_native_rwkv_weighted_renewal_bundle_development_v1/manifest.json"
)
DEFAULT_REPAIR_MANIFEST = PROJECT_ROOT / (
    "experiments/rethinking_rwkv_ms_gemma/local_artifacts/"
    "natural_memory_native_rwkv_source_multi_anchor_bundle_repair_development_v1/manifest.json"
)


def canonical_sha256(value: Any) -> str:
    return development_split.canonical_sha256(value)


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _receipt(scope: str, value: Mapping[str, Any]) -> dict[str, str]:
    return {
        "algorithm": "sha256",
        "payload_scope": scope,
        "payload_sha256": canonical_sha256(value),
    }


def _recorded_path(path: Path) -> str:
    try:
        return path.resolve().relative_to(PROJECT_ROOT.resolve()).as_posix()
    except ValueError:
        return str(path.resolve())


def _bundle_row(source: Mapping[str, Any], donor: Mapping[str, Any]) -> dict[str, Any]:
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
    return "".join(
        json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        + "\n"
        for row in rows
    ).encode("utf-8")


def materialize(
    *,
    source_path: Path,
    tokenizer_path: Path,
    parent_manifest_path: Path,
    protected_manifest_path: Path,
    prior_manifest_path: Path,
    multi_manifest_path: Path,
    weighted_manifest_path: Path,
    repair_manifest_path: Path,
    output_root: Path,
) -> dict[str, Any]:
    if output_root.exists():
        raise ValueError(f"Low-rank query output must be fresh: {output_root}")
    source_path = source_path.expanduser().resolve(strict=True)
    if source_path != DEFAULT_SOURCE.resolve():
        raise ValueError("Low-rank query requires the pinned FIT source")
    tokenizer_binding = source_loader.validate_tokenizer_artifacts(tokenizer_path)
    source_rows = source_loader.load_source_rows(source_path)
    tokenizer = source_loader.AutoTokenizer.from_pretrained(
        tokenizer_path, local_files_only=True, trust_remote_code=False
    )
    source_loader.add_write_token_counts(source_rows, tokenizer)
    metadata_rows = parent_materializer._metadata_rows(source_rows)

    parent_reservation = parent_reservation_split.load_parent_reservation(
        parent_manifest_path
    )
    protected_manifest = prior_split.load_protected_manifest(protected_manifest_path)
    prior_manifest = prior_materializer.load_manifest(prior_manifest_path)
    multi_manifest = multi_materializer.load_manifest(multi_manifest_path)
    weighted_manifest = weighted_materializer.load_manifest(weighted_manifest_path)
    repair_manifest = repair_materializer.load_manifest(repair_manifest_path)
    split_contract = development_split.build_split(
        metadata_rows,
        parent_reservation,
        protected_manifest,
        prior_manifest,
        multi_manifest,
        weighted_manifest,
        repair_manifest,
    )
    by_index = {int(row["source_index"]): row for row in source_rows}
    split_payload = split_contract["splits"]["development"]
    mapping = {int(source): int(donor) for source, donor in split_payload["mapping_pairs"]}
    bundle_rows = [
        _bundle_row(by_index[source], by_index[mapping[source]])
        for source in split_payload["source_indices"]
    ]
    bundle_payload = _bundle_bytes(bundle_rows)
    manifest: dict[str, Any] = {
        "schema": MANIFEST_SCHEMA,
        "source": {
            **parent_split.SOURCE_BINDING.payload(),
            "recorded_path": _recorded_path(source_path),
        },
        "tokenizer": dict(tokenizer_binding),
        "metadata": {
            "rows": len(metadata_rows),
            "payload_sha256": canonical_sha256(parent_materializer._metadata_rows(source_rows)),
            "fields": ["source_index", "row_sha256", "gold_sha256", "write_tokens", "passage_signature_sha256s"],
            "selection_only": True,
        },
        "split_contract": split_contract,
        "prior_reservations": {
            "parent_manifest_sha256": parent_reservation.manifest_sha256,
            "protected_bundle_bytes_read": False,
            "prior_manifests_bundle_bytes_read": False,
            "prior_cumulative_manifest_receipt": prior_manifest["receipt"]["payload_sha256"],
            "prior_multi_anchor_manifest_receipt": multi_manifest["receipt"]["payload_sha256"],
            "prior_weighted_manifest_receipt": weighted_manifest["receipt"]["payload_sha256"],
            "prior_multi_anchor_repair_manifest_receipt": repair_manifest["receipt"]["payload_sha256"],
        },
        "file_inventory": {
            "exact_names": list(FILE_NAMES),
            "bundles": {
                "development": {
                    "path": "development.jsonl",
                    "rows": len(bundle_rows),
                    "bytes": len(bundle_payload),
                    "sha256": sha256_bytes(bundle_payload),
                    "payload_sha256": canonical_sha256(bundle_rows),
                    "source_indices_sha256": split_payload["source_indices_sha256"],
                    "qualified_source_ids_sha256": split_payload["qualified_source_ids_sha256"],
                    "qualified_mapping_pairs_sha256": split_payload["qualified_mapping_pairs_sha256"],
                    "row_sha256s_sha256": canonical_sha256([row["row_sha256"] for row in bundle_rows]),
                }
            },
        },
        "access": {
            "default_open_splits": ["development"],
            "development_is_explicitly_open": True,
            "protected_reservation_bundles_must_remain_unopened": True,
            "prior_development_bundle_bytes_read": False,
            "prior_multi_anchor_bundle_bytes_read": False,
            "prior_weighted_bundle_bytes_read": False,
            "prior_multi_anchor_repair_bundle_bytes_read": False,
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
    return manifest


def load_manifest(manifest_path: Path, *, expected_sha256: str | None = None) -> dict[str, Any]:
    payload = manifest_path.read_bytes()
    digest = sha256_bytes(payload)
    if expected_sha256 is not None and digest != expected_sha256:
        raise ValueError("Low-rank query manifest file hash differs")
    manifest = json.loads(payload.decode("utf-8"))
    if not isinstance(manifest, Mapping) or manifest.get("schema") != MANIFEST_SCHEMA:
        raise ValueError("Low-rank query manifest schema differs")
    receipt = manifest.get("receipt")
    if not isinstance(receipt, Mapping):
        raise ValueError("Low-rank query manifest receipt is missing")
    unsigned = dict(manifest)
    unsigned.pop("receipt")
    if dict(receipt) != _receipt("canonical_manifest_without_receipt", unsigned):
        raise ValueError("Low-rank query manifest receipt differs")
    development_split.validate_split_contract(manifest["split_contract"])
    if manifest.get("open_splits") != ["development"] or manifest.get("protected_splits_opened") != []:
        raise ValueError("Low-rank query manifest access differs")
    return dict(manifest)


def read_open_development(
    root: Path, manifest: Mapping[str, Any]
) -> list[dict[str, Any]]:
    manifest_on_disk = load_manifest(root / "manifest.json")
    if dict(manifest_on_disk) != dict(manifest):
        raise ValueError("Low-rank query manifest differs from disk")
    binding = manifest["file_inventory"]["bundles"]["development"]
    payload = (root / "development.jsonl").read_bytes()
    if sha256_bytes(payload) != binding["sha256"]:
        raise ValueError("Low-rank query development bundle hash differs")
    split_payload = manifest["split_contract"]["splits"]["development"]
    mapping = {
        int(source): int(donor)
        for source, donor in split_payload["mapping_pairs"]
    }
    rows: list[dict[str, Any]] = []
    for line in payload.decode("utf-8").splitlines():
        row = json.loads(line)
        if not isinstance(row, Mapping):
            raise ValueError("Low-rank query development row is not an object")
        receipt = row.get("receipt")
        unsigned = dict(row)
        unsigned.pop("receipt", None)
        if dict(receipt or {}) != _receipt(
            "canonical_bundle_row_without_receipt", unsigned
        ):
            raise ValueError("Low-rank query development row receipt differs")
        source = int(row["source_index"])
        donor = int(row["donor_source_index"])
        raw_line = str(row["raw_line"])
        if (
            row.get("schema") != ROW_SCHEMA
            or row.get("split") != "development"
            or mapping.get(source) != donor
            or hashlib.sha256(raw_line.encode("utf-8")).hexdigest()
            != row.get("row_sha256")
        ):
            raise ValueError("Low-rank query development row binding differs")
        rows.append(dict(row))
    if (
        len(rows) != DEVELOPMENT_ROWS
        or canonical_sha256(rows) != binding["payload_sha256"]
        or [int(row["source_index"]) for row in rows]
        != split_payload["source_indices"]
    ):
        raise ValueError("Low-rank query development payload differs")
    return rows


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-path", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--tokenizer-path", type=Path, default=DEFAULT_TOKENIZER)
    parser.add_argument("--parent-manifest-path", type=Path, default=DEFAULT_PARENT_MANIFEST)
    parser.add_argument("--protected-manifest-path", type=Path, default=DEFAULT_PROTECTED_MANIFEST)
    parser.add_argument("--prior-manifest-path", type=Path, default=DEFAULT_PRIOR_MANIFEST)
    parser.add_argument("--multi-manifest-path", type=Path, default=DEFAULT_MULTI_MANIFEST)
    parser.add_argument("--weighted-manifest-path", type=Path, default=DEFAULT_WEIGHTED_MANIFEST)
    parser.add_argument("--repair-manifest-path", type=Path, default=DEFAULT_REPAIR_MANIFEST)
    parser.add_argument("--output-root", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    manifest = materialize(
        source_path=args.source_path,
        tokenizer_path=args.tokenizer_path,
        parent_manifest_path=args.parent_manifest_path,
        protected_manifest_path=args.protected_manifest_path,
        prior_manifest_path=args.prior_manifest_path,
        multi_manifest_path=args.multi_manifest_path,
        weighted_manifest_path=args.weighted_manifest_path,
        repair_manifest_path=args.repair_manifest_path,
        output_root=args.output_root,
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
