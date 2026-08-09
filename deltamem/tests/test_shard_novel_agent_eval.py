from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

from experiments.rethinking_rwkv_ms_gemma import shard_novel_agent_eval as shard


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload) + "\n", encoding="utf-8")


def append_jsonl(path: Path, payload: object) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload) + "\n")


def generated_row(index: int, row_hash: str) -> dict[str, object]:
    return {
        "fingerprint": "fingerprint",
        "key": f"task:{index}",
        "condition": "normal",
        "task": "task",
        "status": "ok",
        "row_sha256": row_hash,
        "line_index": index,
        "raw_generation": "{}",
        "parsed_json": {},
        "score": {},
        "gold": {},
    }


def test_prepare_and_merge_preserve_official_crlf_row_hashes(tmp_path: Path) -> None:
    dataset_path = tmp_path / "val.jsonl"
    dataset_path.write_bytes(b'{"row":0}\r\n{"row":1}\r\n')
    expected_hashes = [
        hashlib.sha256(b'{"row":0}').hexdigest(),
        hashlib.sha256(b'{"row":1}').hexdigest(),
    ]
    main_eval_dir = tmp_path / "main"
    manifest = {
        "fingerprint": "fingerprint",
        "fingerprint_payload": {
            "conditions": ["base", "normal"],
            "datasets": {
                "task": {
                    "path": str(dataset_path),
                    "selected_rows": 2,
                }
            },
            "device": "cuda:1",
        },
    }
    write_json(main_eval_dir / "manifest.json", manifest)
    shard_dir = tmp_path / "shard"
    shard.prepare(
        SimpleNamespace(
            main_eval_dir=main_eval_dir,
            shard_dir=shard_dir,
            task="task",
            target_condition="normal",
            owned_indices="1",
            physical_gpu=3,
            replace=False,
        )
    )

    placeholders = shard.read_jsonl(shard_dir / "normal.jsonl")
    assert placeholders == [
        {
            "fingerprint": "fingerprint",
            "key": "task:0",
            "condition": "normal",
            "status": "ok",
            "row_sha256": expected_hashes[0],
            shard.PLACEHOLDER_FIELD: True,
        }
    ]
    append_jsonl(shard_dir / "normal.jsonl", generated_row(1, expected_hashes[1]))
    shard.write_jsonl_atomic(
        main_eval_dir / "normal.jsonl",
        [generated_row(0, expected_hashes[0])],
    )

    shard.merge(
        SimpleNamespace(
            main_eval_dir=main_eval_dir,
            target_condition="normal",
            shard_dir=[shard_dir],
            require_complete=True,
        )
    )

    merged = shard.read_jsonl(main_eval_dir / "normal.jsonl")
    assert [row["line_index"] for row in merged] == [0, 1]
    assert shard.read_json(main_eval_dir / "distributed_shard_merge.json")[
        "complete"
    ]
