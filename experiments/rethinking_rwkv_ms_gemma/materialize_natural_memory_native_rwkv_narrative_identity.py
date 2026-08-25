"""Materialize a fresh open narrative identity bundle."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[1]
for import_root in (SCRIPT_DIR, PROJECT_ROOT):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

from transformers import AutoTokenizer

from deltamem.chat_templates import apply_chat_template
from experiments.rethinking_rwkv_ms_gemma import prepare_natural_memory_gate
from experiments.rethinking_rwkv_ms_gemma import (
    rwkv_narrative_identity_split as split,
)


MANIFEST_SCHEMA = "rwkv_ms_narrative_identity_manifest.v1"
ROW_SCHEMA = "rwkv_ms_narrative_identity_row.v1"
FILE_NAMES = ("manifest.json", "development.jsonl")
HF_ENDPOINT = "https://hf-mirror.com"
DEFAULT_SOURCE = PROJECT_ROOT / split.SOURCE_RELATIVE_PATH
DEFAULT_TOKENIZER = Path("/root/X/.cache/hf/gemma-4-E4B-it-a4c2d58")
DEFAULT_OUTPUT = SCRIPT_DIR / (
    "local_artifacts/natural_memory_native_rwkv_narrative_identity_v1"
)


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_sha256(value: Any) -> str:
    return split.canonical_sha256(value)


def receipt(scope: str, value: Mapping[str, Any]) -> dict[str, str]:
    return {
        "algorithm": "sha256",
        "payload_scope": scope,
        "payload_sha256": canonical_sha256(value),
    }


def _load_source_rows(source_path: Path) -> list[dict[str, Any]]:
    if sha256_file(source_path) != split.SOURCE_SHA256:
        raise ValueError("Narrative source hash differs")
    rows: list[dict[str, Any]] = []
    with source_path.open("r", encoding="utf-8", newline="") as handle:
        for source_index, line in enumerate(handle):
            raw_line = line[:-1] if line.endswith("\n") else line
            value = json.loads(raw_line)
            if not isinstance(value, Mapping) or set(value) != {"messages"}:
                raise ValueError(f"Narrative row shape differs: {source_index}")
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
                raise ValueError(f"Narrative message shape differs: {source_index}")
            rows.append(
                {
                    "source_index": source_index,
                    "raw_line": raw_line,
                    "row_sha256": sha256_bytes(raw_line.encode("utf-8")),
                    "messages": messages,
                    "gold_sha256": sha256_bytes(
                        messages[-1]["content"].encode("utf-8")
                    ),
                }
            )
    if len(rows) != split.SOURCE_ROWS:
        raise ValueError(f"Narrative source rows differ: {len(rows)}")
    return rows


def _metadata_rows(rows: Sequence[Mapping[str, Any]], tokenizer: Any) -> list[dict[str, Any]]:
    metadata: list[dict[str, Any]] = []
    for row in rows:
        messages = row["messages"]
        rendered = apply_chat_template(
            tokenizer,
            messages[:-1],
            tokenize=False,
            add_generation_prompt=False,
        )
        write_tokens = len(tokenizer(rendered, add_special_tokens=False).input_ids)
        segments = prepare_natural_memory_gate._extract_segments(
            "narrative", messages[1]["content"]
        )
        components = prepare_natural_memory_gate._segments_for_component(segments)
        metadata.append(
            {
                "source_index": int(row["source_index"]),
                "row_sha256": str(row["row_sha256"]),
                "gold_sha256": str(row["gold_sha256"]),
                "write_tokens": int(write_tokens),
                "passage_component_ids": list(components),
            }
        )
    return metadata


def _bundle_row(source: Mapping[str, Any], donor: Mapping[str, Any]) -> dict[str, Any]:
    unsigned = {
        "schema": ROW_SCHEMA,
        "split": "development",
        "source_index": int(source["source_index"]),
        "qualified_source_id": split.qualified_id(
            int(source["source_index"]), str(source["row_sha256"])
        ),
        "row_sha256": str(source["row_sha256"]),
        "donor_source_index": int(donor["source_index"]),
        "qualified_donor_source_id": split.qualified_id(
            int(donor["source_index"]), str(donor["row_sha256"])
        ),
        "donor_row_sha256": str(donor["row_sha256"]),
        "raw_line": str(source["raw_line"]),
    }
    return {**unsigned, "receipt": receipt("canonical_row_without_receipt", unsigned)}


def materialize(
    *,
    source_path: Path,
    tokenizer_path: Path,
    output_root: Path,
) -> dict[str, Any]:
    if os.environ.get("HF_ENDPOINT") != HF_ENDPOINT:
        raise ValueError(f"HF_ENDPOINT must be exactly {HF_ENDPOINT}")
    source_path = source_path.expanduser().resolve(strict=True)
    if output_root.exists():
        raise ValueError(f"Narrative output must be fresh: {output_root}")
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_path.expanduser().resolve(strict=True),
        local_files_only=True,
        trust_remote_code=False,
    )
    source_rows = _load_source_rows(source_path)
    metadata_rows = _metadata_rows(source_rows, tokenizer)
    split_contract = split.build_split(metadata_rows)
    selected = split_contract["splits"]["development"]["source_indices"]
    by_index = {int(row["source_index"]): row for row in source_rows}
    mapping = {
        int(source): int(donor)
        for source, donor in split_contract["splits"]["development"]["mapping_pairs"]
    }
    rows = [_bundle_row(by_index[index], by_index[mapping[index]]) for index in selected]
    payload = "".join(
        json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        + "\n"
        for row in rows
    ).encode("utf-8")
    manifest: dict[str, Any] = {
        "schema": MANIFEST_SCHEMA,
        "source": {
            "namespace": split.SOURCE_NAMESPACE,
            "path": split.SOURCE_RELATIVE_PATH,
            "sha256": split.SOURCE_SHA256,
            "rows": split.SOURCE_ROWS,
            "recorded_path": str(source_path.relative_to(PROJECT_ROOT)),
        },
        "tokenizer": {
            "path": str(tokenizer_path.expanduser().resolve()),
            "config_sha256": sha256_file(tokenizer_path / "config.json"),
            "tokenizer_sha256": sha256_file(tokenizer_path / "tokenizer.json"),
        },
        "metadata": {
            "rows": len(metadata_rows),
            "payload_sha256": canonical_sha256(metadata_rows),
            "selection_only": True,
        },
        "split_contract": split_contract,
        "file_inventory": {
            "exact_names": list(FILE_NAMES),
            "bundles": {
                "development": {
                    "path": "development.jsonl",
                    "rows": len(rows),
                    "bytes": len(payload),
                    "sha256": sha256_bytes(payload),
                    "payload_sha256": canonical_sha256(rows),
                }
            },
        },
        "access": {
            "default_open_splits": ["development"],
            "development_is_explicitly_open": True,
            "protected_splits_opened": [],
        },
        "open_splits": ["development"],
        "protected_splits_opened": [],
    }
    manifest["receipt"] = receipt("canonical_manifest_without_receipt", manifest)
    output_root.mkdir(parents=True, exist_ok=False)
    (output_root / "development.jsonl").write_bytes(payload)
    (output_root / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest


def load_manifest(path: Path) -> dict[str, Any]:
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(manifest, Mapping) or manifest.get("schema") != MANIFEST_SCHEMA:
        raise ValueError("Narrative manifest schema differs")
    unsigned = dict(manifest)
    if dict(unsigned.pop("receipt", {})) != receipt(
        "canonical_manifest_without_receipt", unsigned
    ):
        raise ValueError("Narrative manifest receipt differs")
    split.validate_split_contract(manifest["split_contract"])
    if manifest.get("open_splits") != ["development"] or manifest.get(
        "protected_splits_opened"
    ) != []:
        raise ValueError("Narrative manifest access differs")
    return dict(manifest)


def read_open_development(root: Path, manifest: Mapping[str, Any]) -> list[dict[str, Any]]:
    on_disk = load_manifest(root / "manifest.json")
    if dict(on_disk) != dict(manifest):
        raise ValueError("Narrative manifest differs from disk")
    binding = manifest["file_inventory"]["bundles"]["development"]
    payload = (root / binding["path"]).read_bytes()
    if sha256_bytes(payload) != binding["sha256"]:
        raise ValueError("Narrative development payload hash differs")
    mapping = {
        int(source): int(donor)
        for source, donor in manifest["split_contract"]["splits"]["development"][
            "mapping_pairs"
        ]
    }
    rows: list[dict[str, Any]] = []
    for line in payload.decode("utf-8").splitlines():
        row = json.loads(line)
        unsigned = dict(row)
        if dict(unsigned.pop("receipt", {})) != receipt(
            "canonical_row_without_receipt", unsigned
        ):
            raise ValueError("Narrative row receipt differs")
        source = int(row["source_index"])
        donor = int(row["donor_source_index"])
        if (
            row.get("schema") != ROW_SCHEMA
            or row.get("split") != "development"
            or mapping.get(source) != donor
            or sha256_bytes(str(row["raw_line"]).encode("utf-8"))
            != row.get("row_sha256")
            or row.get("qualified_source_id") != split.qualified_id(source, row["row_sha256"])
            or row.get("qualified_donor_source_id")
            != split.qualified_id(donor, row["donor_row_sha256"])
        ):
            raise ValueError("Narrative row binding differs")
        rows.append(dict(row))
    if len(rows) != split.DEVELOPMENT_ROWS or canonical_sha256(rows) != binding["payload_sha256"]:
        raise ValueError("Narrative development row inventory differs")
    return rows


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-path", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--tokenizer-path", type=Path, default=DEFAULT_TOKENIZER)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
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
