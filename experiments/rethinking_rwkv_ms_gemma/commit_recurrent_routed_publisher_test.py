#!/usr/bin/env python3
"""Download and hash publisher test files without parsing their contents."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any, Mapping

from huggingface_hub import HfApi, hf_hub_download


SCRIPT_DIR = Path(__file__).resolve().parent
OUTPUT_ROOT = (
    SCRIPT_DIR
    / "local_artifacts/recurrent_routed_gated_query_value_publisher_test_commit_v1"
)
CACHE_ROOT = Path("/root/x/.cache/huggingface")
HF_ENDPOINT = "https://hf-mirror.com"
REPO_ID = "mikuhhn1239/novel-agent-sft-dataset"
REVISION = "5d3040d21f51b3ce90b9396b058e552c47f43cd5"
TRAINING_RESULT = (
    SCRIPT_DIR
    / "local_artifacts/recurrent_routed_gated_query_value_train20_v1/result.json"
)
TRAINING_RECEIPT = (
    "5596ade12c16b9ce5e0e6a1e27e4bc5ef860b0f788f595343f653f4c27a52d1a"
)
DEVELOPMENT_RESULT = (
    SCRIPT_DIR
    / "local_artifacts/recurrent_routed_gated_query_value_development_v1/result.json"
)
DEVELOPMENT_RECEIPT = (
    "5492272781cd46ebf0df6056e3b62936d7d1c2cef1fff4e80ad1623eba365123"
)
TASK_FILES = {
    "attribution": "training/v3.2-attribution-best-candidate/test.jsonl",
    "narrative": "training/v3.2-narrative-type-classification/test.jsonl",
    "scene": "training/v4-scene-boundary-detection/test.jsonl",
}


def canonical_sha256(value: Any) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_result(path: Path, receipt: str, status: str) -> Mapping[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    signature = value.pop("receipt")
    if (
        signature.get("payload_sha256") != receipt
        or canonical_sha256(value) != receipt
        or value.get("status") != status
        or value.get("passed") is not True
        or value.get("publisher_test_opened") is not False
    ):
        raise ValueError(f"Publisher-test authorization differs: {path}")
    return value


def main() -> int:
    if os.environ.get("HF_ENDPOINT") != HF_ENDPOINT:
        raise ValueError(f"HF_ENDPOINT must be exactly {HF_ENDPOINT}")
    validate_result(
        TRAINING_RESULT,
        TRAINING_RECEIPT,
        "gated_query_value_training_complete_development_evaluation_authorized",
    )
    validate_result(
        DEVELOPMENT_RESULT,
        DEVELOPMENT_RECEIPT,
        "development_passed_final_evaluation_authorized",
    )
    info = HfApi(endpoint=HF_ENDPOINT).dataset_info(REPO_ID, revision=REVISION)
    if info.sha != REVISION:
        raise ValueError("Publisher-test dataset revision differs")
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=False)
    files = {}
    for task, filename in TASK_FILES.items():
        path = Path(
            hf_hub_download(
                repo_id=REPO_ID,
                filename=filename,
                repo_type="dataset",
                revision=REVISION,
                cache_dir=CACHE_ROOT,
            )
        ).resolve(strict=True)
        files[task] = {
            "repo_filename": filename,
            "cache_path": str(path),
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
    commitment = {
        "schema": "rwkv_ms_recurrent_routed_publisher_test_commitment.v1",
        "repo_id": REPO_ID,
        "repo_type": "dataset",
        "revision": REVISION,
        "hf_endpoint": HF_ENDPOINT,
        "training_result_receipt": TRAINING_RECEIPT,
        "development_result_receipt": DEVELOPMENT_RECEIPT,
        "files": files,
        "semantic_content_parsed_before_commitment": False,
        "rows_counted_before_commitment": False,
        "publisher_validation_opened": False,
        "publisher_test_opened": False,
    }
    commitment["receipt"] = {
        "algorithm": "sha256",
        "payload_scope": "canonical_commitment_without_receipt",
        "payload_sha256": canonical_sha256(commitment),
    }
    (OUTPUT_ROOT / "commitment.json").write_text(
        json.dumps(commitment, ensure_ascii=True, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "output_root": str(OUTPUT_ROOT),
                "commitment_receipt": commitment["receipt"]["payload_sha256"],
                "files": {
                    task: {
                        "bytes": value["bytes"],
                        "sha256": value["sha256"],
                    }
                    for task, value in files.items()
                },
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
