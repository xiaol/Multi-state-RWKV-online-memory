#!/usr/bin/env python3
"""Run the signed four-A100 bidirectional-sign development retry once."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence


HF_ENDPOINT = "https://hf-mirror.com"
SIGNED_SOURCE_ENV = "RWKV_V5_EXACT_SOURCE_ROOT"
if os.environ.get("HF_ENDPOINT") != HF_ENDPOINT:
    raise RuntimeError(f"HF_ENDPOINT must be explicitly set to {HF_ENDPOINT}")
if not os.environ.get(SIGNED_SOURCE_ENV):
    raise RuntimeError(f"{SIGNED_SOURCE_ENV} must be explicitly set")

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[1]
PROTOCOL = (
    SCRIPT_DIR
    / "natural_memory_native_rwkv_bidirectional_sign_development_gate_protocol_v2.json"
)
CORE = SCRIPT_DIR / "rwkv_bidirectional_sign_development_gate_core.py"
SCHEMA = "rwkv_ms_bidirectional_sign_development_gate.v2"
EXPECTED_PROTOCOL_FILE_SHA256 = (
    "01a6b87e39b8ce4ec3e3a0443df21ede876d5748bdef75746f734b5f3ef22147"
)
EXPECTED_PROTOCOL_PAYLOAD_SHA256 = (
    "8805ea053bf111b0be2317bb15426eea0f25c6fd5727de58bfdfb7e2cab10530"
)
EXPECTED_CORE_SHA256 = (
    "1518efab7bc074b992cb5ae4f01b66d51494f9dd82bb5db9618d426fd166cde0"
)


def _canonical_sha256(value: Any) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _require_hash_pin(name: str, value: str) -> None:
    if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
        raise RuntimeError(f"{name} must be a lowercase SHA-256 hex digest")
    if value == "0" * 64:
        raise RuntimeError(f"{name} is an unset launcher hash pin")


def validate_launcher_contract(protocol_path: Path) -> Mapping[str, Any]:
    """Validate signed bytes and the pre-import protocol trust boundary."""
    protocol_path = protocol_path.resolve()
    if protocol_path != PROTOCOL.resolve():
        raise ValueError("Bidirectional development protocol path differs")
    _require_hash_pin(
        "EXPECTED_PROTOCOL_FILE_SHA256", EXPECTED_PROTOCOL_FILE_SHA256
    )
    _require_hash_pin(
        "EXPECTED_PROTOCOL_PAYLOAD_SHA256", EXPECTED_PROTOCOL_PAYLOAD_SHA256
    )
    _require_hash_pin("EXPECTED_CORE_SHA256", EXPECTED_CORE_SHA256)
    if not protocol_path.is_file():
        raise FileNotFoundError(f"Bidirectional development protocol is missing: {protocol_path}")
    if _sha256_file(protocol_path) != EXPECTED_PROTOCOL_FILE_SHA256:
        raise ValueError("Bidirectional development protocol file hash differs")
    if not CORE.is_file() or _sha256_file(CORE) != EXPECTED_CORE_SHA256:
        raise ValueError("Bidirectional development gate core hash differs")

    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    if not isinstance(protocol, dict):
        raise ValueError("Bidirectional development protocol is not an object")
    receipt = protocol.get("receipt")
    if not isinstance(receipt, dict) or set(receipt) != {
        "algorithm",
        "payload_scope",
        "payload_sha256",
    }:
        raise ValueError("Bidirectional development protocol receipt schema differs")
    unsigned = dict(protocol)
    unsigned.pop("receipt")
    payload_sha256 = _canonical_sha256(unsigned)
    if (
        protocol.get("schema") != SCHEMA
        or receipt.get("algorithm") != "sha256"
        or receipt.get("payload_scope") != "canonical_protocol_without_receipt"
        or payload_sha256 != EXPECTED_PROTOCOL_PAYLOAD_SHA256
        or receipt.get("payload_sha256") != EXPECTED_PROTOCOL_PAYLOAD_SHA256
    ):
        raise ValueError("Bidirectional development protocol receipt differs")

    manifests = protocol.get("manifests")
    if not isinstance(manifests, dict) or not isinstance(manifests.get("files"), list):
        raise ValueError("Bidirectional development protocol manifest is missing")
    launcher_relative = Path(__file__).resolve().relative_to(PROJECT_ROOT).as_posix()
    for row in manifests["files"]:
        if not isinstance(row, dict):
            raise ValueError("Bidirectional development manifest row is not an object")
        if set(row) != {"role", "scope", "path", "sha256"}:
            raise ValueError("Bidirectional development manifest row schema differs")
        row_path = Path(str(row.get("path"))).as_posix()
        if (
            row_path == launcher_relative
            or Path(row_path).name == Path(__file__).name
            or row.get("role") == "launcher"
        ):
            raise ValueError("Development launcher must not be in protocol DAG")
    return protocol


for import_root in (SCRIPT_DIR, PROJECT_ROOT):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))


def main(argv: Sequence[str] | None = None) -> int:
    validate_launcher_contract(PROTOCOL)
    from experiments.rethinking_rwkv_ms_gemma import (
        rwkv_bidirectional_sign_development_gate_core as gate,
    )

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-model", type=Path, required=True)
    parser.add_argument("--open-fit-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    gate.run(
        protocol_path=PROTOCOL.resolve(strict=True),
        launcher_path=Path(__file__).resolve(strict=True),
        base_model=args.base_model.expanduser().resolve(strict=True),
        open_fit_root=args.open_fit_root.expanduser().resolve(strict=True),
        output_dir=args.output_dir.expanduser().resolve(),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
