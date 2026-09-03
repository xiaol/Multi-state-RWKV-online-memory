#!/usr/bin/env python3
"""Materialize projected-slot teacher top-k logits for the locked train schedule."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Any

import torch
from torch.distributed.elastic.multiprocessing.errors import record

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[1]
for import_root in (SCRIPT_DIR, PROJECT_ROOT):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

from deltamem.core.delta import reset_delta_mem_states
from experiments.rethinking_rwkv_ms_gemma import common as model_common
from experiments.rethinking_rwkv_ms_gemma import natural_memory_distributed as distributed
from experiments.rethinking_rwkv_ms_gemma import recurrent_routed_posttrain_common as routed_common
from experiments.rethinking_rwkv_ms_gemma import run_natural_memory_native_recurrent_routed_posttrain as training
from experiments.rethinking_rwkv_ms_gemma import run_natural_memory_native_recurrent_routed_query_value_distill as runner


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=runner.DISTILL_CACHE_ROOT)
    parser.add_argument("--base-model", type=Path, default=routed_common.BASE_MODEL)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


@record
def main() -> int:
    args = parse_args()
    runner.configure()
    context = distributed.initialize_distributed_training(args.device, timeout_seconds=7200)
    if context is None or context.world_size != 4:
        raise ValueError("Teacher cache materialization requires four ranks")
    output = args.output_dir.expanduser().resolve()
    try:
        manifest = routed_common.validate_signed_json(
            routed_common.SPLIT_ROOT / "manifest.json",
            routed_common.SPLIT_MANIFEST_RECEIPT,
        )
        rows = routed_common.load_open_rows("train", manifest=manifest)
        schedule, payload, schedule_hash = runner.stage14_schedule(rows)
        keys = sorted(
            {
                (item.target.task, item.target.source_ordinal, item.target.row_sha256, item.prompt_variant)
                for item in schedule
            },
            key=lambda value: (routed_common.TASKS.index(value[0]), value[1], value[3]),
        )
        creation_error = None
        if context.is_primary:
            try:
                output.mkdir(parents=True, exist_ok=False)
            except BaseException as error:
                creation_error = error
        distributed.phase_consensus(context, phase="teacher-cache-output", error=creation_error)
        model, tokenizer = model_common.load_model_and_tokenizer(
            base_model=str(args.base_model.expanduser().resolve(strict=True)),
            device=str(context.device),
            dtype="bfloat16",
            attn_implementation="sdpa",
            delta_mem_root=PROJECT_ROOT,
            memory_dir=str(runner.TEACHER_ADAPTER.expanduser().resolve(strict=True)),
        )
        model.eval()
        for index, (task, source_ordinal, row_sha256, variant) in enumerate(keys):
            if index % context.world_size != context.process_rank:
                continue
            cache_path = output / f"{task}-{source_ordinal}-{variant}.pt"
            if cache_path.exists():
                raise ValueError(f"Teacher cache output is not fresh: {cache_path}")
            target = next(
                row for row in rows[task]
                if row.source_ordinal == source_ordinal and row.row_sha256 == row_sha256
            )
            example = routed_common.encode_row(tokenizer, target, prompt_variant=variant)
            batch = routed_common.evolution.collate_native_examples(
                [example],
                pad_token_id=int(tokenizer.pad_token_id),
                device=context.device,
            )
            try:
                with torch.inference_mode():
                    logits, audit = routed_common.direct_condition_logits(
                        model,
                        batch,
                        condition="correct_recurrent_state",
                        donor=None,
                        dtype=torch.bfloat16,
                    )
                    values, labels = training.selected_answer_logits(logits, batch.labels)
                if not audit["projected_carrier_references_fixed"] or not audit["projected_carrier_bytes_fixed"]:
                    raise RuntimeError("Teacher projected carrier audit failed")
                k = min(runner.DISTILL_TOP_K, int(values.size(-1)))
                top_values, top_indices = torch.topk(values.float(), k=k, dim=-1)
                torch.save(
                    {
                        "schema": "rwkv_ms_recurrent_routed_distill_teacher_cache.v1",
                        "task": task,
                        "source_ordinal": source_ordinal,
                        "row_sha256": row_sha256,
                        "prompt_variant": variant,
                        "schedule_sha256": schedule_hash,
                        "teacher_values": top_values.to(dtype=torch.float16).cpu(),
                        "teacher_indices": top_indices.to(dtype=torch.int32).cpu(),
                        "teacher_labels": labels.to(dtype=torch.int32).cpu(),
                    },
                    cache_path,
                )
            finally:
                del batch, example
                reset_delta_mem_states(model)
                routed_common.evolution.release_native_row_allocator_cache(context.device)
        torch.distributed.barrier(group=context.control_group)
        if context.is_primary:
            files = {}
            for path in sorted(output.glob("*.pt")):
                files[path.name] = {"bytes": path.stat().st_size, "sha256": sha256_file(path)}
            if len(files) != len(keys):
                raise RuntimeError("Teacher cache file count differs")
            manifest_value: dict[str, Any] = {
                "schema": "rwkv_ms_recurrent_routed_distill_teacher_cache_manifest.v1",
                "schedule_sha256": schedule_hash,
                "schedule_rows": len(schedule),
                "cache_rows": len(keys),
                "teacher_adapter": str(runner.TEACHER_ADAPTER.resolve()),
                "teacher_adapter_config_sha256": routed_common.sha256_file(runner.TEACHER_ADAPTER / "delta_mem_config.json"),
                "files": files,
            }
            manifest_value["receipt"] = {
                "algorithm": "sha256",
                "payload_scope": "canonical_manifest_without_receipt",
                "payload_sha256": routed_common.canonical_sha256(manifest_value),
            }
            (output / "manifest.json").write_text(
                json.dumps(manifest_value, ensure_ascii=True, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
        distributed.gather_objects(context, True)
    finally:
        try:
            del model, tokenizer
        except UnboundLocalError:
            pass
        distributed.destroy_distributed_training(context)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
