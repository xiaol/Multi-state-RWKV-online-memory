#!/usr/bin/env python3
"""Generate one fresh four-GPU shard of the locked hybrid validation."""

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
from experiments.rethinking_rwkv_ms_gemma import (  # noqa: E402
    run_natural_memory_native_scene_contrast_dropout as training,
)
from experiments.rethinking_rwkv_ms_gemma import (  # noqa: E402
    run_natural_memory_native_scene_contrast_probe as probe,
)


SCHEMA = "rwkv_ms_natural_memory_native_hybrid_publisher_validation_shard.v1"
INPUT_SCHEMA = "rwkv_ms_natural_memory_native_hybrid_publisher_validation_input.v1"
PROTOCOL = SCRIPT_DIR / "natural_memory_native_hybrid_publisher_validation_protocol_v1.json"
PROTOCOL_PAYLOAD_SHA256 = "1d8435431748b95ce3a36f8bc2217ce61b55bb2a594e517299d6f9d5e995ae0c"
WORLD_SIZE = 4
BASE_CONFIG_SHA256 = "33b10c02df3c2e8536cf323d29d53262aaa2f4d11dbe19bc729373fbe90295d4"
V9_ADAPTER_FILES_SHA256 = "1ceda7e288e832df14ace3fb9b4c5db0edc4395945ecbd34c76363f0d0f9e6fb"
V9_ADAPTER_WEIGHTS_SHA256 = "b063940a9be0712f830a992e9114055e4488297b0842245e6e26563b303545a9"
HYBRID_RESULT_RECEIPT_SHA256 = "07809e382c2bfba7ed20e85261371ea7adfb2c346243b15d3873cd88eac8fc0f"
HYBRID_RESULT_FILE_SHA256 = "6565eaff4621a4002017e6b3a31c692d42d946f57894928688461151544d3ecb"
SELECTED_STEP = 16
SELECTED_MANIFEST_SHA256 = "00c603cd02a5dcc3bd325ee6daadd47349f6700f257deee2185a3991f1644b7e"
SELECTED_MANIFEST_RECEIPT_SHA256 = "9da055dc39cb74c54acc6201e32bef095d599273d1f65c2f200550cca97ef891"
SELECTED_GATE_STATE_SHA256 = "5b0670683046e9701c24171a5c9d8cfc58e1078f25b72806662732260bab7d4f"
SELECTED_PATCH_SHA256 = "c7bd4f6a396c06404cd884f3f6ff92ad5d891621fe295a3e91e9b33d73d4834a"
TASKS = {
    "attribution": {
        "path": "v3.2-attribution-best-candidate/val.jsonl",
        "sha256": "01373fa09c89568a47927f19088f78bbf62cd4b7eb66803100c06e8a3f2152a4",
        "source_rows": 30,
        "excluded_source_indices": (0,),
        "conditions": ("base",),
        "max_new_tokens": None,
    },
    "narrative": {
        "path": "v3.2-narrative-type-classification/val.jsonl",
        "sha256": "4a1d27c543e715a069e6b3c8253021b0f86fe1690aac116da09eba43896e1b6a",
        "source_rows": 39,
        "excluded_source_indices": (),
        "conditions": ("base", "v9_memory"),
        "max_new_tokens": 1024,
    },
    "scene": {
        "path": "v4-scene-boundary-detection/val.jsonl",
        "sha256": "61e94bcc536a124b07aef2c38ba285d7073d94a223866b58ddc7e5e1f509d513",
        "source_rows": 170,
        "excluded_source_indices": (),
        "conditions": ("base", "v9_memory", "checkpoint16_memory"),
        "max_new_tokens": 128,
    },
}
BASE_PHASE = (
    ("attribution", "base"),
    ("narrative", "base"),
    ("scene", "base"),
)
V9_PHASE = (
    ("narrative", "v9_memory"),
    ("scene", "v9_memory"),
)
CHECKPOINT_PHASE = (("scene", "checkpoint16_memory"),)


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
    value = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    receipt = value.get("receipt")
    if not isinstance(receipt, Mapping):
        raise ValueError("Hybrid publisher-validation protocol receipt is missing")
    unsigned = dict(value)
    unsigned.pop("receipt")
    digest = canonical_sha256(unsigned)
    if digest != PROTOCOL_PAYLOAD_SHA256 or receipt.get("payload_sha256") != digest:
        raise ValueError("Hybrid publisher-validation protocol hash differs")
    return value


def validate_hybrid_result(path: Path) -> Mapping[str, Any]:
    value = probe.validate_signed_json(path, description="Hybrid authorization result")
    if (
        sha256_file(path) != HYBRID_RESULT_FILE_SHA256
        or value["receipt"].get("payload_sha256") != HYBRID_RESULT_RECEIPT_SHA256
        or value.get("fresh_publisher_validation_replication_contract_authorized") is not True
        or value.get("publisher_validation_opened") is not False
        or value.get("protected_splits_opened") != []
    ):
        raise ValueError("Hybrid authorization result binding differs")
    return value


def selected_manifest(training_root: Path) -> Mapping[str, Any]:
    manifests = probe.validate_training_root(training_root)
    manifest = next(item for item in manifests if int(item["step"]) == SELECTED_STEP)
    manifest_path = training_root / f"checkpoint-{SELECTED_STEP}" / "manifest.json"
    if (
        sha256_file(manifest_path) != SELECTED_MANIFEST_SHA256
        or manifest.get("receipt", {}).get("payload_sha256") != SELECTED_MANIFEST_RECEIPT_SHA256
        or manifest.get("gate_state_sha256") != SELECTED_GATE_STATE_SHA256
        or manifest.get("patch_file", {}).get("sha256") != SELECTED_PATCH_SHA256
    ):
        raise ValueError("Checkpoint-16 manifest binding differs")
    return manifest


def validate_worker_index(worker_index: int) -> None:
    if not 0 <= worker_index < WORLD_SIZE:
        raise ValueError("Hybrid publisher validation requires a valid four-way worker")


def validate_condition(task: str, condition: str) -> None:
    if task not in TASKS or condition not in TASKS[task]["conditions"]:
        raise ValueError(f"Invalid hybrid validation condition: {task}:{condition}")


def load_rows(dataset_root: Path, task: str) -> list[dict[str, Any]]:
    spec = TASKS[task]
    path = dataset_root / str(spec["path"])
    if sha256_file(path) != spec["sha256"]:
        raise ValueError(f"Hybrid publisher-validation {task} dataset hash differs")
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            if not raw_line.strip():
                continue
            row = json.loads(raw_line)
            messages = row.get("messages")
            if not isinstance(messages, list) or len(messages) < 3:
                raise ValueError(f"Invalid hybrid validation row: {path}")
            rows.append(
                {
                    "source_index": len(rows),
                    "messages": messages[:-1],
                    "row_sha256": hashlib.sha256(
                        raw_line.rstrip("\n").encode("utf-8")
                    ).hexdigest(),
                }
            )
    if len(rows) != int(spec["source_rows"]):
        raise ValueError(
            f"Expected {spec['source_rows']} hybrid {task} rows, found {len(rows)}"
        )
    return rows


def selected_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    task: str,
    worker_index: int,
) -> list[Mapping[str, Any]]:
    validate_worker_index(worker_index)
    excluded = set(TASKS[task]["excluded_source_indices"])
    return [
        row
        for row in rows
        if int(row["source_index"]) not in excluded
        and int(row["source_index"]) % WORLD_SIZE == worker_index
    ]


def input_binding(
    *,
    base_model: Path,
    memory_dir: Path,
    dataset_root: Path,
    hybrid_result: Path,
    training_root: Path,
    manifest: Mapping[str, Any],
    worker_index: int,
    dtype: str,
    attn_implementation: str,
) -> Mapping[str, Any]:
    if sha256_file(base_model / "config.json") != BASE_CONFIG_SHA256:
        raise ValueError("Hybrid validation base config differs")
    adapter_files = training.gate.snapshot_directory_files(memory_dir)
    if training.gate._sha256_json(adapter_files) != V9_ADAPTER_FILES_SHA256:
        raise ValueError("Hybrid validation V9 adapter differs")
    if sha256_file(memory_dir / "delta_mem_adapter.pt") != V9_ADAPTER_WEIGHTS_SHA256:
        raise ValueError("Hybrid validation V9 adapter weights differ")
    validate_hybrid_result(hybrid_result)
    dataset_files = {
        task: {
            "path": str(dataset_root / str(spec["path"])),
            "sha256": sha256_file(dataset_root / str(spec["path"])),
        }
        for task, spec in TASKS.items()
    }
    if any(dataset_files[task]["sha256"] != spec["sha256"] for task, spec in TASKS.items()):
        raise ValueError("Hybrid validation dataset binding differs")
    return {
        "schema": INPUT_SCHEMA,
        "protocol_payload_sha256": PROTOCOL_PAYLOAD_SHA256,
        "base_model": str(base_model),
        "base_model_revision": "a4c2d58be94dda072b918d9db64ee85c8ed34e3f",
        "base_config_sha256": BASE_CONFIG_SHA256,
        "memory_dir": str(memory_dir),
        "v9_adapter_files_sha256": V9_ADAPTER_FILES_SHA256,
        "v9_adapter_weights_sha256": V9_ADAPTER_WEIGHTS_SHA256,
        "dataset_root": str(dataset_root),
        "dataset_files": dataset_files,
        "hybrid_result": str(hybrid_result),
        "hybrid_result_sha256": HYBRID_RESULT_FILE_SHA256,
        "hybrid_result_receipt_sha256": HYBRID_RESULT_RECEIPT_SHA256,
        "training_root": str(training_root),
        "checkpoint_step": SELECTED_STEP,
        "checkpoint_manifest_sha256": SELECTED_MANIFEST_SHA256,
        "checkpoint_manifest_receipt_sha256": SELECTED_MANIFEST_RECEIPT_SHA256,
        "checkpoint_gate_state_sha256": manifest["gate_state_sha256"],
        "checkpoint_patch_sha256": manifest["patch_file"]["sha256"],
        "worker_index": worker_index,
        "world_size": WORLD_SIZE,
        "task_conditions": {
            task: list(spec["conditions"]) for task, spec in TASKS.items()
        },
        "dtype": dtype,
        "attn_implementation": attn_implementation,
        "runner_sha256": sha256_file(Path(__file__)),
        "likelihood_runner_sha256": sha256_file(
            SCRIPT_DIR / "diagnose_native_attribution_candidate_likelihood.py"
        ),
        "generation_runner_sha256": sha256_file(SCRIPT_DIR / "run_novel_agent_eval.py"),
        "patch_loader_sha256": sha256_file(
            SCRIPT_DIR / "run_natural_memory_native_scene_contrast_probe.py"
        ),
        "protected_splits_opened": ["publisher_validation_fresh_replication"],
        "prior_validation_artifacts_read": False,
    }


def write_or_validate_json(path: Path, value: Mapping[str, Any]) -> None:
    if path.is_file():
        if json.loads(path.read_text(encoding="utf-8")) != value:
            raise ValueError(f"Hybrid validation binding differs: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def output_path(output_dir: Path, task: str, condition: str) -> Path:
    validate_condition(task, condition)
    return output_dir / f"{task}.{condition}.jsonl"


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
                raise ValueError(f"Duplicate hybrid validation record: {path}:{index}")
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
    rows: Sequence[Mapping[str, Any]],
    *,
    task: str,
    condition: str,
    worker_index: int,
) -> None:
    expected = {int(row["source_index"]): row for row in rows}
    if not set(existing) <= set(expected):
        raise ValueError(f"Unexpected hybrid validation indices: {task}:{condition}")
    for index, record in existing.items():
        required = {
            "schema": SCHEMA,
            "protocol_payload_sha256": PROTOCOL_PAYLOAD_SHA256,
            "task": task,
            "condition": condition,
            "worker_index": worker_index,
            "world_size": WORLD_SIZE,
            "source_index": index,
            "row_sha256": expected[index]["row_sha256"],
        }
        if any(record.get(key) != value for key, value in required.items()):
            raise ValueError(f"Resumed hybrid record differs: {task}:{condition}:{index}")


def common_record_fields(
    row: Mapping[str, Any],
    *,
    task: str,
    condition: str,
    worker_index: int,
) -> dict[str, Any]:
    return {
        "schema": SCHEMA,
        "protocol_payload_sha256": PROTOCOL_PAYLOAD_SHA256,
        "task": task,
        "condition": condition,
        "worker_index": worker_index,
        "world_size": WORLD_SIZE,
        "source_index": row["source_index"],
        "row_sha256": row["row_sha256"],
    }


def attribution_record(
    model,
    tokenizer,
    row: Mapping[str, Any],
    *,
    device: str,
    worker_index: int,
) -> Mapping[str, Any]:
    messages = list(row["messages"])
    candidates = recovery.parse_candidates(str(messages[-1].get("content", "")))
    if not candidates:
        raise ValueError("Hybrid attribution row has no candidates")
    fields = common_record_fields(
        row,
        task="attribution",
        condition="base",
        worker_index=worker_index,
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
    worker_index: int,
    runtime_gate_state_sha256: str | None = None,
) -> Mapping[str, Any]:
    result = evaluator.generate_one(
        model=model,
        tokenizer=tokenizer,
        messages=list(row["messages"]),
        max_new_tokens=int(TASKS[task]["max_new_tokens"]),
        device=device,
        online_memory_protocol=(
            "legacy_write_only" if condition == "base" else "write_then_read"
        ),
    )
    parsed = result["parsed_json"]
    if task == "narrative":
        prediction: Any = recovery.recover_narrative(parsed)
    elif task == "scene":
        recovered = recovery.recover_scene(parsed)
        prediction = None if recovered is None else sorted(recovered)
    else:
        raise ValueError(f"Unsupported hybrid generation task: {task}")
    checkpoint = (
        {
            "checkpoint_step": SELECTED_STEP,
            "gate_state_sha256": SELECTED_GATE_STATE_SHA256,
            "runtime_gate_state_sha256": runtime_gate_state_sha256,
        }
        if condition == "checkpoint16_memory"
        else {}
    )
    return {
        **common_record_fields(
            row,
            task=task,
            condition=condition,
            worker_index=worker_index,
        ),
        **checkpoint,
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
    task: str,
    condition: str,
    model,
    tokenizer,
    rows: Sequence[Mapping[str, Any]],
    output_dir: Path,
    worker_index: int,
    device: str,
    runtime_gate_state_sha256: str | None = None,
) -> None:
    path = output_path(output_dir, task, condition)
    existing = read_completed(path)
    validate_resume(
        existing,
        rows,
        task=task,
        condition=condition,
        worker_index=worker_index,
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
                worker_index=worker_index,
            )
        else:
            record = generation_record(
                model,
                tokenizer,
                row,
                task=task,
                condition=condition,
                device=device,
                worker_index=worker_index,
                runtime_gate_state_sha256=runtime_gate_state_sha256,
            )
        append_record(path, record)
        print(
            f"HYBRID_VALIDATION_PROGRESS worker={worker_index}/{WORLD_SIZE} "
            f"task={task} condition={condition} row={index} "
            f"ordinal={ordinal}/{len(rows)}",
            flush=True,
        )


def condition_complete(
    *,
    task: str,
    condition: str,
    rows: Sequence[Mapping[str, Any]],
    output_dir: Path,
    worker_index: int,
) -> bool:
    existing = read_completed(output_path(output_dir, task, condition))
    validate_resume(
        existing,
        rows,
        task=task,
        condition=condition,
        worker_index=worker_index,
    )
    return len(existing) == len(rows)


def phase_complete(
    phase: Sequence[tuple[str, str]],
    *,
    rows_by_task: Mapping[str, Sequence[Mapping[str, Any]]],
    output_dir: Path,
    worker_index: int,
) -> bool:
    return all(
        condition_complete(
            task=task,
            condition=condition,
            rows=rows_by_task[task],
            output_dir=output_dir,
            worker_index=worker_index,
        )
        for task, condition in phase
    )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-model", type=Path, required=True)
    parser.add_argument("--memory-dir", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--hybrid-result", type=Path, required=True)
    parser.add_argument("--training-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--worker-index", type=int, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--attn-implementation", default="sdpa")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    validate_protocol()
    validate_worker_index(args.worker_index)
    base_model = args.base_model.expanduser().resolve(strict=True)
    memory_dir = args.memory_dir.expanduser().resolve(strict=True)
    dataset_root = args.dataset_root.expanduser().resolve(strict=True)
    hybrid_result = args.hybrid_result.expanduser().resolve(strict=True)
    training_root = args.training_root.expanduser().resolve(strict=True)
    output_dir = args.output_root.expanduser().resolve() / f"shard-{args.worker_index}"
    manifest = selected_manifest(training_root)
    binding = input_binding(
        base_model=base_model,
        memory_dir=memory_dir,
        dataset_root=dataset_root,
        hybrid_result=hybrid_result,
        training_root=training_root,
        manifest=manifest,
        worker_index=args.worker_index,
        dtype=args.dtype,
        attn_implementation=args.attn_implementation,
    )
    write_or_validate_json(output_dir / "input_binding.json", binding)
    rows_by_task = {
        task: selected_rows(
            load_rows(dataset_root, task),
            task=task,
            worker_index=args.worker_index,
        )
        for task in TASKS
    }
    load_kwargs = {
        "base_model": str(base_model),
        "device": args.device,
        "dtype": args.dtype,
        "attn_implementation": args.attn_implementation,
    }

    if not phase_complete(
        BASE_PHASE,
        rows_by_task=rows_by_task,
        output_dir=output_dir,
        worker_index=args.worker_index,
    ):
        model, tokenizer = load_model_and_tokenizer(**load_kwargs)
        model.eval()
        for task, condition in BASE_PHASE:
            run_condition(
                task=task,
                condition=condition,
                model=model,
                tokenizer=tokenizer,
                rows=rows_by_task[task],
                output_dir=output_dir,
                worker_index=args.worker_index,
                device=args.device,
            )
        del model, tokenizer
        gc.collect()
        torch.cuda.empty_cache()

    memory_phase = V9_PHASE + CHECKPOINT_PHASE
    if not phase_complete(
        memory_phase,
        rows_by_task=rows_by_task,
        output_dir=output_dir,
        worker_index=args.worker_index,
    ):
        model, tokenizer = load_model_and_tokenizer(memory_dir=str(memory_dir), **load_kwargs)
        model.eval()
        for task, condition in V9_PHASE:
            run_condition(
                task=task,
                condition=condition,
                model=model,
                tokenizer=tokenizer,
                rows=rows_by_task[task],
                output_dir=output_dir,
                worker_index=args.worker_index,
                device=args.device,
            )
        if not phase_complete(
            CHECKPOINT_PHASE,
            rows_by_task=rows_by_task,
            output_dir=output_dir,
            worker_index=args.worker_index,
        ):
            loaded = probe.load_gate_patch(
                model,
                patch_path=(
                    training_root
                    / f"checkpoint-{SELECTED_STEP}"
                    / "gate_patch.pt"
                ),
                manifest=manifest,
            )
            run_condition(
                task="scene",
                condition="checkpoint16_memory",
                model=model,
                tokenizer=tokenizer,
                rows=rows_by_task["scene"],
                output_dir=output_dir,
                worker_index=args.worker_index,
                device=args.device,
                runtime_gate_state_sha256=str(loaded["runtime_gate_state_sha256"]),
            )
        del model, tokenizer
        gc.collect()
        torch.cuda.empty_cache()

    print(
        f"HYBRID_VALIDATION_SHARD_COMPLETE worker={args.worker_index}/{WORLD_SIZE}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
