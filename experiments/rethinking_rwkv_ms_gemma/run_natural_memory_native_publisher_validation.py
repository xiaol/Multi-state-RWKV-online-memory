#!/usr/bin/env python3
"""Run one append-only shard of the locked publisher native validation."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

import torch


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = Path(__file__).resolve().parents[2]
for import_root in (SCRIPT_DIR, PROJECT_ROOT):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

from common import load_model_and_tokenizer  # noqa: E402
from experiments.rethinking_rwkv_ms_gemma import (  # noqa: E402
    analyze_novel_agent_eval as recovery,
)
from experiments.rethinking_rwkv_ms_gemma import (  # noqa: E402
    diagnose_native_attribution_candidate_likelihood as likelihood,
)
from experiments.rethinking_rwkv_ms_gemma import run_novel_agent_eval as evaluator  # noqa: E402


SCHEMA = "rwkv_ms_natural_memory_native_publisher_validation_shard.v1"
INPUT_SCHEMA = "rwkv_ms_natural_memory_native_publisher_validation_input.v1"
PROTOCOL = SCRIPT_DIR / "natural_memory_native_publisher_validation_protocol_v1.json"
PROTOCOL_PAYLOAD_SHA256 = (
    "58a7fd37973c1ab1fcc11fea5e6dc039b1c777e2eef12628b43d7be1eaa2c9c1"
)
MEMORY_ADAPTER_SHA256 = (
    "b063940a9be0712f830a992e9114055e4488297b0842245e6e26563b303545a9"
)
BASE_CONFIG_SHA256 = (
    "33b10c02df3c2e8536cf323d29d53262aaa2f4d11dbe19bc729373fbe90295d4"
)
TASKS = {
    "attribution": {
        "path": "v3.2-attribution-best-candidate/val.jsonl",
        "sha256": "01373fa09c89568a47927f19088f78bbf62cd4b7eb66803100c06e8a3f2152a4",
        "source_rows": 30,
        "excluded_source_indices": (0,),
        "allowed_shard_counts": (1,),
        "conditions": ("base",),
        "max_new_tokens": None,
    },
    "narrative": {
        "path": "v3.2-narrative-type-classification/val.jsonl",
        "sha256": "4a1d27c543e715a069e6b3c8253021b0f86fe1690aac116da09eba43896e1b6a",
        "source_rows": 39,
        "excluded_source_indices": (),
        "allowed_shard_counts": (4,),
        "conditions": ("base", "memory"),
        "max_new_tokens": 1024,
    },
    "scene": {
        "path": "v4-scene-boundary-detection/val.jsonl",
        "sha256": "61e94bcc536a124b07aef2c38ba285d7073d94a223866b58ddc7e5e1f509d513",
        "source_rows": 170,
        "excluded_source_indices": (),
        "allowed_shard_counts": (2,),
        "conditions": ("base", "memory"),
        "max_new_tokens": 128,
    },
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


def validate_protocol() -> Mapping[str, Any]:
    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    receipt = protocol.get("receipt")
    if not isinstance(receipt, Mapping):
        raise ValueError("Publisher validation protocol receipt is missing")
    unsigned = dict(protocol)
    unsigned.pop("receipt")
    digest = canonical_sha256(unsigned)
    if digest != PROTOCOL_PAYLOAD_SHA256 or receipt.get("payload_sha256") != digest:
        raise ValueError("Publisher validation protocol hash differs")
    return protocol


def validate_task_contract(
    task: str,
    *,
    shard_index: int,
    shard_count: int,
    conditions: Sequence[str],
) -> None:
    if task not in TASKS:
        raise ValueError(f"Unknown publisher validation task: {task}")
    spec = TASKS[task]
    if shard_count not in spec["allowed_shard_counts"]:
        raise ValueError(f"Publisher validation {task} shard count differs")
    if not 0 <= shard_index < shard_count:
        raise ValueError("Publisher validation shard index is invalid")
    if tuple(conditions) != tuple(spec["conditions"]):
        raise ValueError(f"Publisher validation {task} conditions differ")


def load_rows(dataset_root: Path, task: str) -> list[dict[str, Any]]:
    spec = TASKS[task]
    path = dataset_root / str(spec["path"])
    if sha256_file(path) != spec["sha256"]:
        raise ValueError(f"Publisher validation {task} dataset hash differs")
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            if not raw_line.strip():
                continue
            row = json.loads(raw_line)
            messages = row.get("messages")
            if not isinstance(messages, list) or len(messages) < 3:
                raise ValueError(f"Invalid publisher validation row: {path}")
            gold = recovery.extract_json(str(messages[-1].get("content", "")))
            if not isinstance(gold, Mapping):
                raise ValueError(f"Invalid publisher validation gold: {path}")
            rows.append(
                {
                    "source_index": len(rows),
                    "messages": messages[:-1],
                    "gold": dict(gold),
                    "row_sha256": hashlib.sha256(
                        raw_line.rstrip("\n").encode("utf-8")
                    ).hexdigest(),
                }
            )
    if len(rows) != int(spec["source_rows"]):
        raise ValueError(
            f"Expected {spec['source_rows']} publisher {task} rows, found {len(rows)}"
        )
    return rows


def selected_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    task: str,
    shard_index: int,
    shard_count: int,
) -> list[Mapping[str, Any]]:
    excluded = set(TASKS[task]["excluded_source_indices"])
    return [
        row
        for row in rows
        if int(row["source_index"]) not in excluded
        and int(row["source_index"]) % shard_count == shard_index
    ]


def input_binding(
    *,
    base_model: Path,
    memory_dir: Path,
    dataset_root: Path,
    task: str,
    shard_index: int,
    shard_count: int,
    conditions: Sequence[str],
    dtype: str,
    attn_implementation: str,
) -> Mapping[str, Any]:
    if sha256_file(base_model / "config.json") != BASE_CONFIG_SHA256:
        raise ValueError("Publisher validation base config differs")
    if sha256_file(memory_dir / "delta_mem_adapter.pt") != MEMORY_ADAPTER_SHA256:
        raise ValueError("Publisher validation memory adapter differs")
    dataset_path = dataset_root / str(TASKS[task]["path"])
    if sha256_file(dataset_path) != TASKS[task]["sha256"]:
        raise ValueError(f"Publisher validation {task} dataset hash differs")
    return {
        "schema": INPUT_SCHEMA,
        "protocol_payload_sha256": PROTOCOL_PAYLOAD_SHA256,
        "base_model": str(base_model),
        "base_model_revision": "a4c2d58be94dda072b918d9db64ee85c8ed34e3f",
        "base_config_sha256": BASE_CONFIG_SHA256,
        "memory_dir": str(memory_dir),
        "memory_adapter_sha256": MEMORY_ADAPTER_SHA256,
        "memory_config_sha256": sha256_file(memory_dir / "delta_mem_config.json"),
        "dataset_root": str(dataset_root),
        "dataset_file": str(dataset_path),
        "dataset_sha256": TASKS[task]["sha256"],
        "task": task,
        "shard_index": shard_index,
        "shard_count": shard_count,
        "conditions": list(conditions),
        "dtype": dtype,
        "attn_implementation": attn_implementation,
        "runner_sha256": sha256_file(Path(__file__)),
        "likelihood_runner_sha256": sha256_file(
            SCRIPT_DIR / "diagnose_native_attribution_candidate_likelihood.py"
        ),
        "generation_runner_sha256": sha256_file(
            SCRIPT_DIR / "run_novel_agent_eval.py"
        ),
    }


def write_or_validate_binding(path: Path, binding: Mapping[str, Any]) -> None:
    if path.is_file():
        if json.loads(path.read_text(encoding="utf-8")) != binding:
            raise ValueError(f"Publisher validation input binding differs: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(binding, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def read_completed(path: Path) -> dict[int, Mapping[str, Any]]:
    if not path.is_file():
        return {}
    records: dict[int, Mapping[str, Any]] = {}
    with path.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            if not raw_line.strip():
                continue
            record = json.loads(raw_line)
            index = int(record["source_index"])
            if index in records:
                raise ValueError(f"Duplicate publisher validation record: {path}:{index}")
            records[index] = record
    return records


def append_record(path: Path, record: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def validate_resume(
    existing: Mapping[int, Mapping[str, Any]],
    expected: Sequence[Mapping[str, Any]],
    *,
    task: str,
    condition: str,
    shard_index: int,
    shard_count: int,
) -> None:
    expected_by_index = {int(row["source_index"]): row for row in expected}
    if not set(existing) <= set(expected_by_index):
        raise ValueError(f"Unexpected publisher validation indices: {task}:{condition}")
    for index, record in existing.items():
        row = expected_by_index[index]
        required = {
            "schema": SCHEMA,
            "protocol_payload_sha256": PROTOCOL_PAYLOAD_SHA256,
            "task": task,
            "condition": condition,
            "shard_index": shard_index,
            "shard_count": shard_count,
            "source_index": index,
            "row_sha256": row["row_sha256"],
        }
        if any(record.get(key) != value for key, value in required.items()):
            raise ValueError(f"Resumed publisher record differs: {task}:{condition}:{index}")


def common_record_fields(
    row: Mapping[str, Any],
    *,
    task: str,
    condition: str,
    shard_index: int,
    shard_count: int,
) -> dict[str, Any]:
    return {
        "schema": SCHEMA,
        "protocol_payload_sha256": PROTOCOL_PAYLOAD_SHA256,
        "task": task,
        "condition": condition,
        "shard_index": shard_index,
        "shard_count": shard_count,
        "source_index": row["source_index"],
        "row_sha256": row["row_sha256"],
    }


def attribution_record(
    model,
    tokenizer,
    row: Mapping[str, Any],
    *,
    device: str,
    shard_index: int,
    shard_count: int,
) -> Mapping[str, Any]:
    messages = list(row["messages"])
    candidates = recovery.parse_candidates(str(messages[-1].get("content", "")))
    if not candidates:
        raise ValueError("Publisher attribution row has no candidates")
    fields = common_record_fields(
        row,
        task="attribution",
        condition="base",
        shard_index=shard_index,
        shard_count=shard_count,
    )
    if len(candidates) == 1:
        return {
            **fields,
            "selected": candidates[0],
            "candidate_scores": [],
            "deterministic_singleton_candidate": True,
        }
    scores = [
        likelihood.continuation_nll(
            model,
            tokenizer,
            messages=messages,
            candidate=candidate,
            device=device,
            use_online_memory=False,
        )
        for candidate in candidates
    ]
    return {
        **fields,
        "selected": min(scores, key=lambda item: float(item["nll_mean"]))[
            "candidate"
        ],
        "candidate_scores": scores,
    }


def generation_record(
    model,
    tokenizer,
    row: Mapping[str, Any],
    *,
    task: str,
    condition: str,
    device: str,
    shard_index: int,
    shard_count: int,
) -> Mapping[str, Any]:
    result = evaluator.generate_one(
        model=model,
        tokenizer=tokenizer,
        messages=list(row["messages"]),
        max_new_tokens=int(TASKS[task]["max_new_tokens"]),
        device=device,
        online_memory_protocol=(
            "write_then_read" if condition == "memory" else "legacy_write_only"
        ),
    )
    parsed = result["parsed_json"]
    if task == "narrative":
        prediction: Any = recovery.recover_narrative(parsed)
    elif task == "scene":
        recovered = recovery.recover_scene(parsed)
        prediction = None if recovered is None else sorted(recovered)
    else:
        raise ValueError(f"Unsupported publisher generation task: {task}")
    return {
        **common_record_fields(
            row,
            task=task,
            condition=condition,
            shard_index=shard_index,
            shard_count=shard_count,
        ),
        "prediction": prediction,
        "raw_generation": result["raw_generation"],
        "input_tokens": result["input_tokens"],
        "output_tokens": result["output_tokens"],
        "hit_max_new_tokens": result["hit_max_new_tokens"],
        "elapsed_seconds": result["elapsed_seconds"],
        "peak_cuda_memory_bytes": result["peak_cuda_memory_bytes"],
    }


def run_condition(
    *,
    condition: str,
    model,
    tokenizer,
    rows: Sequence[Mapping[str, Any]],
    task: str,
    output_dir: Path,
    shard_index: int,
    shard_count: int,
    device: str,
) -> None:
    path = output_dir / f"{condition}.jsonl"
    existing = read_completed(path)
    validate_resume(
        existing,
        rows,
        task=task,
        condition=condition,
        shard_index=shard_index,
        shard_count=shard_count,
    )
    for ordinal, row in enumerate(rows, start=1):
        index = int(row["source_index"])
        if index in existing:
            continue
        if task == "attribution":
            record = attribution_record(
                model,
                tokenizer,
                row,
                device=device,
                shard_index=shard_index,
                shard_count=shard_count,
            )
        else:
            record = generation_record(
                model,
                tokenizer,
                row,
                task=task,
                condition=condition,
                device=device,
                shard_index=shard_index,
                shard_count=shard_count,
            )
        append_record(path, record)
        print(
            f"PUBLISHER_VALIDATION_PROGRESS task={task} condition={condition} "
            f"shard={shard_index}/{shard_count} row={index} "
            f"ordinal={ordinal}/{len(rows)}",
            flush=True,
        )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-model", type=Path, required=True)
    parser.add_argument("--memory-dir", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--task", choices=tuple(TASKS), required=True)
    parser.add_argument("--shard-index", type=int, required=True)
    parser.add_argument("--shard-count", type=int, required=True)
    parser.add_argument("--conditions", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--attn-implementation", default="sdpa")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    validate_protocol()
    conditions = tuple(
        condition.strip() for condition in args.conditions.split(",") if condition.strip()
    )
    validate_task_contract(
        args.task,
        shard_index=args.shard_index,
        shard_count=args.shard_count,
        conditions=conditions,
    )
    base_model = args.base_model.expanduser().resolve(strict=True)
    memory_dir = args.memory_dir.expanduser().resolve(strict=True)
    dataset_root = args.dataset_root.expanduser().resolve(strict=True)
    output_root = args.output_root.expanduser().resolve()
    output_dir = output_root / args.task / f"shard-{args.shard_index}-of-{args.shard_count}"
    binding = input_binding(
        base_model=base_model,
        memory_dir=memory_dir,
        dataset_root=dataset_root,
        task=args.task,
        shard_index=args.shard_index,
        shard_count=args.shard_count,
        conditions=conditions,
        dtype=args.dtype,
        attn_implementation=args.attn_implementation,
    )
    write_or_validate_binding(output_dir / "input_binding.json", binding)
    task_rows = selected_rows(
        load_rows(dataset_root, args.task),
        task=args.task,
        shard_index=args.shard_index,
        shard_count=args.shard_count,
    )
    for condition in conditions:
        load_kwargs = {
            "base_model": str(base_model),
            "device": args.device,
            "dtype": args.dtype,
            "attn_implementation": args.attn_implementation,
        }
        if condition == "memory":
            load_kwargs["memory_dir"] = str(memory_dir)
        model, tokenizer = load_model_and_tokenizer(**load_kwargs)
        run_condition(
            condition=condition,
            model=model,
            tokenizer=tokenizer,
            rows=task_rows,
            task=args.task,
            output_dir=output_dir,
            shard_index=args.shard_index,
            shard_count=args.shard_count,
            device=args.device,
        )
        del model, tokenizer
        gc.collect()
        torch.cuda.empty_cache()
    print(
        f"PUBLISHER_VALIDATION_SHARD_COMPLETE task={args.task} "
        f"shard={args.shard_index}/{args.shard_count}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
