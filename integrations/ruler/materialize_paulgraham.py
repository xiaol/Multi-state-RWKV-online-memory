#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Materialize RULER's PaulGrahamEssays.json from an upstream essay checkout."
    )
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--output-file", type=Path, required=True)
    parser.add_argument("--manifest-file", type=Path)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def git_revision(path: Path) -> str | None:
    result = subprocess.run(
        ("git", "-C", str(path), "rev-parse", "HEAD"),
        check=False,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip() if result.returncode == 0 else None


def atomic_write(path: Path, payload: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(payload, encoding="utf-8")
    temporary.replace(path)


def main() -> None:
    args = parse_args()
    source_dir = args.source_dir.resolve()
    files = sorted(source_dir.glob("*.txt"))
    if not files:
        raise FileNotFoundError(f"No .txt essays found in {source_dir}")
    if args.output_file.exists() and not args.overwrite:
        raise FileExistsError(f"Output already exists: {args.output_file}")

    records = []
    text_parts = []
    for path in files:
        payload = path.read_bytes()
        text_parts.append(payload.decode("utf-8"))
        records.append(
            {
                "name": path.name,
                "bytes": len(payload),
                "sha256": sha256_bytes(payload),
            }
        )
    combined_text = "\n\n".join(text_parts)
    atomic_write(
        args.output_file,
        json.dumps({"text": combined_text}, ensure_ascii=False) + "\n",
    )

    manifest_path = args.manifest_file or args.output_file.with_suffix(".manifest.json")
    manifest = {
        "source_dir": str(source_dir),
        "source_revision": git_revision(source_dir),
        "essay_count": len(records),
        "combined_characters": len(combined_text),
        "source_files": records,
        "output_file": str(args.output_file.resolve()),
        "output_sha256": sha256_file(args.output_file),
    }
    atomic_write(manifest_path, json.dumps(manifest, indent=2) + "\n")
    print("RULER_CORPUS=" + json.dumps(manifest, sort_keys=True))


if __name__ == "__main__":
    main()
