#!/usr/bin/env python3
"""Open the committed final rows after a passing development gate."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence


SCRIPT_DIR = Path(__file__).resolve().parent
SPLIT_ROOT = SCRIPT_DIR / "local_artifacts/natural_memory_native_recurrent_routed_posttrain_split_v1"
FINAL_ROWS_PER_TASK = 64
TASKS = ("attribution", "narrative", "scene")
MANIFEST_RECEIPT = "05314bfcaa3f4c6febe860f33bf7867af8d57a80e9e1b9020b1cc318bceebc96"
FINAL_COMMITMENT_RECEIPT = "c8c106a00e1379e26bbae5b774f0fe831de2c2527c3195a1e42881a33b2b2fae"
OPEN_RECEIPT = "159cf93c913715f0c90e03ca659bf3bd4f1deb9d3e12c64f923d7b5b71340ad8"


def canonical_sha256(value: Any) -> str:
    payload = json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_signed(path: Path, receipt: str) -> Mapping[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    signed = value.get("receipt")
    unsigned = dict(value)
    unsigned.pop("receipt", None)
    if not isinstance(signed, Mapping) or signed.get("payload_sha256") != receipt:
        raise ValueError(f"Receipt differs: {path}")
    if canonical_sha256(unsigned) != receipt:
        raise ValueError(f"Signed payload differs: {path}")
    return value


def write_fresh(path: Path, value: Mapping[str, Any]) -> None:
    if path.exists():
        raise ValueError(f"Final output must be fresh: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=True, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--development-result", type=Path, required=True)
    parser.add_argument("--development-receipt", required=True)
    parser.add_argument("--protocol-receipt", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)

    manifest = validate_signed(SPLIT_ROOT / "manifest.json", MANIFEST_RECEIPT)
    commitment = validate_signed(SPLIT_ROOT / "final_commitment.json", FINAL_COMMITMENT_RECEIPT)
    open_receipt = validate_signed(SPLIT_ROOT / "open_split_receipt.json", OPEN_RECEIPT)
    development = validate_signed(args.development_result.expanduser().resolve(strict=True), args.development_receipt)
    if (
        development.get("status") != "development_passed_final_evaluation_authorized"
        or development.get("passed") is not True
        or development.get("final_rows_opened") is not False
        or manifest.get("final_commitment_payload_sha256") != FINAL_COMMITMENT_RECEIPT
        or open_receipt.get("final_files_written") != []
        or commitment.get("semantic_content_opened_during_commitment") is not False
    ):
        raise ValueError("Final opening requires a passing sealed development result")

    output_dir = args.output_dir.expanduser().resolve()
    files: dict[str, Any] = {}
    for task in TASKS:
        source_path = Path(manifest["tasks"][task]["source_file"]).expanduser().resolve(strict=True)
        source_lines = [line for line in source_path.read_text(encoding="utf-8").splitlines() if line]
        rows = manifest["tasks"][task]["splits"]["final"]["rows"]
        if len(rows) != FINAL_ROWS_PER_TASK:
            raise ValueError(f"Final row count differs for {task}")
        selected_lines = [source_lines[int(row["source_ordinal"])] for row in rows]
        for row, raw_line in zip(rows, selected_lines):
            if hashlib.sha256(raw_line.encode("utf-8")).hexdigest() != row["row_sha256"]:
                raise ValueError(f"Final row hash differs for {task}:{row['source_ordinal']}")
        path = output_dir / "final" / task / "final.jsonl"
        payload = "\n".join(selected_lines) + "\n"
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists():
            raise ValueError(f"Final output must be fresh: {path}")
        path.write_text(payload, encoding="utf-8")
        files[str(path.relative_to(output_dir))] = {
            "rows": len(selected_lines),
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
            "row_payload_sha256": canonical_sha256(rows),
            "source_file": str(source_path),
        }

    opening = {
        "schema": "rwkv_ms_natural_memory_native_recurrent_routed_final_opening.v1",
        "manifest_receipt": MANIFEST_RECEIPT,
        "final_commitment_receipt": FINAL_COMMITMENT_RECEIPT,
        "open_split_receipt": OPEN_RECEIPT,
        "development_result": str(args.development_result.expanduser().resolve()),
        "development_result_receipt": args.development_receipt,
        "protocol_payload_sha256": args.protocol_receipt,
        "files": files,
        "rows_per_task": FINAL_ROWS_PER_TASK,
        "final_rows_opened": True,
        "publisher_validation_opened": False,
        "publisher_test_opened": False,
    }
    opening["receipt"] = {
        "algorithm": "sha256",
        "payload_scope": "canonical_opening_without_receipt",
        "payload_sha256": canonical_sha256(opening),
    }
    write_fresh(output_dir / "final_opening.json", opening)
    print(json.dumps(opening["receipt"], sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
