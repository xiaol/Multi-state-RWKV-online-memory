#!/usr/bin/env python3
"""Run one resumable shard of the routed native development benchmark."""

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


SCHEMA = "rwkv_ms_natural_memory_native_routed_shard.v1"
PROTOCOL = SCRIPT_DIR / "natural_memory_native_routed_decoder_protocol_v1.json"
PROTOCOL_PAYLOAD_SHA256 = (
    "d55d7e785315a6dedc7b4710ef452eb0dffd05ce6a4decec941feec290e50b3a"
)
MEMORY_ADAPTER_SHA256 = (
    "b063940a9be0712f830a992e9114055e4488297b0842245e6e26563b303545a9"
)
NATIVE_DATASET_RECEIPT_SHA256 = (
    "7f1056c33009a30d63179b49e9f95fe1c9fb4438b434d2ad3a22cd22039704e4"
)
TASKS = {
    "attribution": {
        "path": "v3.2-attribution-best-candidate/train_derived_development.jsonl",
        "expected_rows": 93,
        "max_new_tokens": 1024,
    },
    "narrative": {
        "path": "v3.2-narrative-type-classification/train_derived_development.jsonl",
        "expected_rows": 118,
        "max_new_tokens": 1024,
    },
    "scene": {
        "path": "v4-scene-boundary-detection/train_derived_development.jsonl",
        "expected_rows": 361,
        "max_new_tokens": 128,
    },
}
SELECTION_ROWS = 4


def canonical_sha256(value: Any) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def validate_protocol() -> Mapping[str, Any]:
    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    receipt = protocol.get("receipt")
    if not isinstance(receipt, Mapping):
        raise ValueError("Routed decoder protocol receipt is missing")
    unsigned = dict(protocol)
    unsigned.pop("receipt")
    digest = canonical_sha256(unsigned)
    if (
        digest != PROTOCOL_PAYLOAD_SHA256
        or receipt.get("payload_sha256") != digest
    ):
        raise ValueError("Routed decoder protocol hash differs")
    return protocol


def input_binding(
    *,
    base_model: Path,
    memory_dir: Path,
    dataset_root: Path,
    shard_index: int,
    world_size: int,
    dtype: str,
    attn_implementation: str,
) -> Mapping[str, Any]:
    dataset_manifest_path = dataset_root / "manifest.json"
    dataset_manifest = json.loads(dataset_manifest_path.read_text(encoding="utf-8"))
    if (
        dataset_manifest.get("receipt", {}).get("payload_sha256")
        != NATIVE_DATASET_RECEIPT_SHA256
    ):
        raise ValueError("Native development dataset receipt differs")
    adapter_path = memory_dir / "delta_mem_adapter.pt"
    if evaluator.sha256_file(adapter_path) != MEMORY_ADAPTER_SHA256:
        raise ValueError("Routed benchmark memory adapter differs")
    return {
        "schema": "rwkv_ms_natural_memory_native_routed_input.v1",
        "protocol_payload_sha256": PROTOCOL_PAYLOAD_SHA256,
        "base_model": str(base_model),
        "base_config_sha256": evaluator.sha256_file(base_model / "config.json"),
        "memory_dir": str(memory_dir),
        "memory_adapter_sha256": MEMORY_ADAPTER_SHA256,
        "memory_config_sha256": evaluator.sha256_file(
            memory_dir / "delta_mem_config.json"
        ),
        "dataset_root": str(dataset_root),
        "dataset_manifest_sha256": evaluator.sha256_file(dataset_manifest_path),
        "dataset_receipt_payload_sha256": NATIVE_DATASET_RECEIPT_SHA256,
        "shard_index": shard_index,
        "world_size": world_size,
        "dtype": dtype,
        "attn_implementation": attn_implementation,
        "runner_sha256": evaluator.sha256_file(Path(__file__)),
        "likelihood_runner_sha256": evaluator.sha256_file(
            SCRIPT_DIR / "diagnose_native_attribution_candidate_likelihood.py"
        ),
    }


def write_or_validate_binding(path: Path, binding: Mapping[str, Any]) -> None:
    if path.is_file():
        if json.loads(path.read_text(encoding="utf-8")) != binding:
            raise ValueError(f"Routed shard input binding differs: {path}")
        return
    path.write_text(
        json.dumps(binding, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def load_rows(dataset_root: Path) -> dict[str, list[dict[str, Any]]]:
    rows_by_task: dict[str, list[dict[str, Any]]] = {}
    for task, task_spec in TASKS.items():
        path = dataset_root / str(task_spec["path"])
        rows: list[dict[str, Any]] = []
        with path.open("r", encoding="utf-8") as handle:
            for raw_line in handle:
                if not raw_line.strip():
                    continue
                row = json.loads(raw_line)
                messages = row.get("messages")
                if not isinstance(messages, list) or len(messages) < 3:
                    raise ValueError(f"Invalid publisher row in {path}")
                gold = recovery.extract_json(str(messages[-1].get("content", "")))
                if gold is None:
                    raise ValueError(f"Invalid publisher gold in {path}")
                rows.append(
                    {
                        "line_index": len(rows),
                        "messages": messages[:-1],
                        "gold": gold,
                        "row_sha256": hashlib.sha256(
                            raw_line.rstrip("\n").encode("utf-8")
                        ).hexdigest(),
                    }
                )
        if len(rows) != int(task_spec["expected_rows"]):
            raise ValueError(
                f"Expected {task_spec['expected_rows']} {task} rows, found {len(rows)}"
            )
        rows_by_task[task] = rows
    return rows_by_task


def selected_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    shard_index: int,
    world_size: int,
) -> list[Mapping[str, Any]]:
    return [
        row
        for row in rows
        if int(row["line_index"]) >= SELECTION_ROWS
        and int(row["line_index"]) % world_size == shard_index
    ]


def read_completed(path: Path) -> dict[int, Mapping[str, Any]]:
    if not path.is_file():
        return {}
    records: dict[int, Mapping[str, Any]] = {}
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        if not raw_line.strip():
            continue
        record = json.loads(raw_line)
        index = int(record["line_index"])
        if index in records:
            raise ValueError(f"Duplicate routed benchmark record {path}:{index}")
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
) -> None:
    expected_by_index = {int(row["line_index"]): row for row in expected}
    if not set(existing) <= set(expected_by_index):
        raise ValueError(f"Unexpected resumed indices for {task}:{condition}")
    for index, record in existing.items():
        row = expected_by_index[index]
        required = {
            "schema": SCHEMA,
            "task": task,
            "condition": condition,
            "shard_index": shard_index,
            "line_index": index,
            "row_sha256": row["row_sha256"],
            "protocol_payload_sha256": PROTOCOL_PAYLOAD_SHA256,
        }
        if any(record.get(key) != value for key, value in required.items()):
            raise ValueError(f"Resumed routed record differs: {task}:{condition}:{index}")


def attribution_record(
    model,
    tokenizer,
    row: Mapping[str, Any],
    *,
    condition: str,
    device: str,
    shard_index: int,
) -> Mapping[str, Any]:
    messages = list(row["messages"])
    candidates = recovery.parse_candidates(str(messages[-1].get("content", "")))
    if not candidates:
        raise ValueError("Attribution row has no candidates")
    if len(candidates) == 1:
        return {
            "schema": SCHEMA,
            "protocol_payload_sha256": PROTOCOL_PAYLOAD_SHA256,
            "task": "attribution",
            "condition": condition,
            "shard_index": shard_index,
            "line_index": row["line_index"],
            "row_sha256": row["row_sha256"],
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
            use_online_memory=condition == "memory",
        )
        for candidate in candidates
    ]
    selected = min(scores, key=lambda item: float(item["nll_mean"]))["candidate"]
    return {
        "schema": SCHEMA,
        "protocol_payload_sha256": PROTOCOL_PAYLOAD_SHA256,
        "task": "attribution",
        "condition": condition,
        "shard_index": shard_index,
        "line_index": row["line_index"],
        "row_sha256": row["row_sha256"],
        "selected": selected,
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
    parsed_json = result["parsed_json"]
    if task == "narrative":
        prediction: Any = recovery.recover_narrative(parsed_json)
    elif task == "scene":
        recovered = recovery.recover_scene(parsed_json)
        prediction = None if recovered is None else sorted(recovered)
    else:
        raise ValueError(f"Unsupported generation task: {task}")
    return {
        "schema": SCHEMA,
        "protocol_payload_sha256": PROTOCOL_PAYLOAD_SHA256,
        "task": task,
        "condition": condition,
        "shard_index": shard_index,
        "line_index": row["line_index"],
        "row_sha256": row["row_sha256"],
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
    rows_by_task: Mapping[str, Sequence[Mapping[str, Any]]],
    output_dir: Path,
    shard_index: int,
    world_size: int,
    device: str,
) -> None:
    tasks = ("attribution", "narrative") if condition == "memory" else tuple(TASKS)
    for task in tasks:
        task_rows = selected_rows(
            rows_by_task[task],
            shard_index=shard_index,
            world_size=world_size,
        )
        path = output_dir / f"{task}.{condition}.jsonl"
        existing = read_completed(path)
        validate_resume(
            existing,
            task_rows,
            task=task,
            condition=condition,
            shard_index=shard_index,
        )
        for ordinal, row in enumerate(task_rows, start=1):
            index = int(row["line_index"])
            if index in existing:
                continue
            if task == "attribution":
                record = attribution_record(
                    model,
                    tokenizer,
                    row,
                    condition=condition,
                    device=device,
                    shard_index=shard_index,
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
                )
            append_record(path, record)
            print(
                f"ROUTED_PROGRESS shard={shard_index} condition={condition} "
                f"task={task} row={index} ordinal={ordinal}/{len(task_rows)}",
                flush=True,
            )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-model", type=Path, required=True)
    parser.add_argument("--memory-dir", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--shard-index", type=int, required=True)
    parser.add_argument("--world-size", type=int, default=4)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--attn-implementation", default="sdpa")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    validate_protocol()
    if args.world_size != 4 or not 0 <= args.shard_index < args.world_size:
        raise ValueError("Routed benchmark requires four valid shards")
    base_model = args.base_model.expanduser().resolve(strict=True)
    memory_dir = args.memory_dir.expanduser().resolve(strict=True)
    dataset_root = args.dataset_root.expanduser().resolve(strict=True)
    output_root = args.output_root.expanduser().resolve()
    output_dir = output_root / f"shard-{args.shard_index}"
    output_dir.mkdir(parents=True, exist_ok=True)
    binding = input_binding(
        base_model=base_model,
        memory_dir=memory_dir,
        dataset_root=dataset_root,
        shard_index=args.shard_index,
        world_size=args.world_size,
        dtype=args.dtype,
        attn_implementation=args.attn_implementation,
    )
    write_or_validate_binding(output_dir / "input_binding.json", binding)
    rows_by_task = load_rows(dataset_root)

    base_model_instance, tokenizer = load_model_and_tokenizer(
        base_model=str(base_model),
        device=args.device,
        dtype=args.dtype,
        attn_implementation=args.attn_implementation,
    )
    run_condition(
        condition="base",
        model=base_model_instance,
        tokenizer=tokenizer,
        rows_by_task=rows_by_task,
        output_dir=output_dir,
        shard_index=args.shard_index,
        world_size=args.world_size,
        device=args.device,
    )
    del base_model_instance, tokenizer
    gc.collect()
    torch.cuda.empty_cache()

    memory_model, tokenizer = load_model_and_tokenizer(
        base_model=str(base_model),
        memory_dir=str(memory_dir),
        device=args.device,
        dtype=args.dtype,
        attn_implementation=args.attn_implementation,
    )
    run_condition(
        condition="memory",
        model=memory_model,
        tokenizer=tokenizer,
        rows_by_task=rows_by_task,
        output_dir=output_dir,
        shard_index=args.shard_index,
        world_size=args.world_size,
        device=args.device,
    )
    print(f"ROUTED_SHARD_COMPLETE shard={args.shard_index}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
