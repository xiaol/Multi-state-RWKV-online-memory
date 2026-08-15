#!/usr/bin/env python3
"""Evaluate one shard of the trained chunk-addressed RWKV bottleneck."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import time
from typing import Any, Mapping, Sequence

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

import torch
from transformers import AutoTokenizer


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = Path(__file__).resolve().parents[2]
for import_root in (SCRIPT_DIR, PROJECT_ROOT):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

from common import load_model_and_tokenizer  # noqa: E402
from deltamem.core.delta import iter_delta_mem_modules  # noqa: E402
from experiments.rethinking_rwkv_ms_gemma import (  # noqa: E402
    analyze_novel_agent_eval as recovery,
)
from experiments.rethinking_rwkv_ms_gemma import (  # noqa: E402
    run_natural_memory_native_projected_rwkv_hybrid_benchmark_eval as base_eval,
)
from experiments.rethinking_rwkv_ms_gemma import (  # noqa: E402
    run_natural_memory_native_rwkv_addressed_value_eval as addressed_eval,
)
from experiments.rethinking_rwkv_ms_gemma import (  # noqa: E402
    run_natural_memory_native_rwkv_chunk_addressed_value_causal_train as training,
)
from experiments.rethinking_rwkv_ms_gemma import (  # noqa: E402
    run_natural_memory_native_rwkv_chunk_addressed_value_screen as screen,
)


SCHEMA = "rwkv_ms_natural_memory_native_chunk_addressed_value_eval.v1"
INPUT_SCHEMA = "rwkv_ms_natural_memory_native_chunk_addressed_value_eval_input.v1"
PROTOCOL_PAYLOAD_SHA256 = training.PROTOCOL_PAYLOAD_SHA256
HF_MIRROR_ENDPOINT = training.HF_MIRROR_ENDPOINT
WORLD_SIZE = training.WORLD_SIZE
SEED = training.SEED
EVALUATION_ROWS = base_eval.EVALUATION_ROWS
AUTHORIZED_ROWS_PAYLOAD_SHA256 = base_eval.AUTHORIZED_ROWS_PAYLOAD_SHA256
DATASET_RELATIVE_PATH = base_eval.DATASET_RELATIVE_PATH
DATASET_SHA256 = base_eval.DATASET_SHA256
CONDITIONS = addressed_eval.CONDITIONS


def canonical_sha256(value: Any) -> str:
    return training.canonical_sha256(value)


def sha256_file(path: Path) -> str:
    return training.sha256_file(path)


def validate_train_result(
    result_path: Path,
    *,
    adapter_dir: Path,
) -> Mapping[str, Any]:
    result = json.loads(result_path.read_text(encoding="utf-8"))
    receipt = result.get("receipt")
    if not isinstance(receipt, Mapping):
        raise ValueError("Chunk-addressed training result receipt is missing")
    unsigned = dict(result)
    unsigned.pop("receipt")
    digest = canonical_sha256(unsigned)
    required = {
        "schema": training.SCHEMA,
        "status": "training_complete_open_evaluation_authorized",
        "passed": True,
        "protocol_payload_sha256": PROTOCOL_PAYLOAD_SHA256,
        "seed": SEED,
        "updates": training.TRAIN_UPDATES,
        "open_native_evaluation_authorized": True,
        "protected_splits_opened": [],
    }
    if (
        receipt.get("payload_sha256") != digest
        or any(result.get(key) != value for key, value in required.items())
    ):
        raise ValueError("Chunk-addressed training did not authorize evaluation")
    expected_files = result.get("adapter_files")
    if not isinstance(expected_files, Mapping):
        raise ValueError("Chunk-addressed adapter manifest is missing")
    for filename, metadata in expected_files.items():
        path = adapter_dir / str(filename)
        if not path.is_file() or sha256_file(path) != metadata.get("sha256"):
            raise ValueError(f"Chunk-addressed adapter file differs: {path}")
    if result.get("code_bindings", {}).get("runner_sha256") != sha256_file(
        Path(training.__file__)
    ):
        raise ValueError("Chunk-addressed training runner binding differs")
    return result


def generate_row_conditions(
    model,
    tokenizer,
    row: Mapping[str, Any],
    donor: Mapping[str, Any],
    *,
    module_names: Sequence[str],
    device: str,
    shard_index: int,
) -> list[Mapping[str, Any]]:
    correct_state = base_eval.prime_state(model, tokenizer, row, device=device)
    correct_alignment = screen.chunk_alignment_evidence(correct_state, module_names)
    if (
        correct_alignment["all_layer_occupied_slots_match"] is not True
        or correct_alignment["projected_values_exactly_zero_on_every_layer"]
        is not True
    ):
        raise RuntimeError("Correct chunk-addressed state alignment failed")
    correct_recurrent, correct_projected = base_eval.split_state(
        correct_state,
        module_names,
    )
    donor_state = base_eval.prime_state(model, tokenizer, donor, device=device)
    donor_alignment = screen.chunk_alignment_evidence(donor_state, module_names)
    if (
        donor_alignment["all_layer_occupied_slots_match"] is not True
        or donor_alignment["projected_values_exactly_zero_on_every_layer"] is not True
    ):
        raise RuntimeError("Donor chunk-addressed state alignment failed")
    donor_recurrent, donor_projected = base_eval.split_state(
        donor_state,
        module_names,
    )
    projected_digest = base_eval.tensor_digest(correct_projected)
    if base_eval.tensor_digest(donor_projected) == projected_digest:
        raise RuntimeError("Chunk-addressed donor carrier is unexpectedly equal")
    conditions = {
        "correct_recurrent_state": base_eval.merge_state(
            correct_recurrent,
            correct_projected,
        ),
        "zero_recurrent_state": base_eval.merge_state(
            base_eval.zero_recurrent_state(correct_recurrent),
            correct_projected,
        ),
        "matched_donor_recurrent_state": base_eval.merge_state(
            donor_recurrent,
            correct_projected,
        ),
        "layer_permuted_recurrent_state": base_eval.merge_state(
            base_eval.permute_recurrent_state(correct_recurrent, module_names),
            correct_projected,
        ),
        "empty_memory": addressed_eval.zero_state(correct_state),
    }
    generated_conditions = addressed_eval.batched_generate_from_states(
        model,
        tokenizer,
        row,
        conditions,
        device=device,
    )
    gold = recovery.strict_gold_boundaries(row["gold"])
    records: list[Mapping[str, Any]] = []
    for condition in CONDITIONS:
        state = conditions[condition]
        generated = generated_conditions[condition]
        recurrent, projected = base_eval.split_state(state, module_names)
        carrier_fixed = (
            condition == "empty_memory"
            or base_eval.tensor_digest(projected) == projected_digest
        )
        if not carrier_fixed:
            raise RuntimeError("Chunk-addressed carrier changed")
        records.append(
            {
                "schema": SCHEMA,
                "protocol_payload_sha256": PROTOCOL_PAYLOAD_SHA256,
                "condition": condition,
                "seed": SEED,
                "shard_index": shard_index,
                "world_size": WORLD_SIZE,
                "source_index": row["source_index"],
                "row_sha256": row["row_sha256"],
                "gold": sorted(gold),
                **generated,
                "score": base_eval.record_score(generated["prediction"], gold),
                "state_sha256": base_eval.tensor_digest(state),
                "condition_recurrent_sha256": base_eval.tensor_digest(recurrent),
                "projected_carrier_sha256": base_eval.tensor_digest(projected),
                "correct_projected_carrier_sha256": projected_digest,
                "projected_carrier_byte_identical": carrier_fixed,
                "chunk_slots_aligned": True,
                "projected_values_exactly_zero": True,
                "donor_source_index": donor["source_index"],
                "donor_row_sha256": donor["row_sha256"],
            }
        )
    by_condition = {record["condition"]: record for record in records}
    if (
        by_condition["zero_recurrent_state"]["prediction"]
        != by_condition["empty_memory"]["prediction"]
    ):
        raise RuntimeError("Zero recurrent and empty-memory predictions differ")
    return records


def input_binding(
    *,
    shard_index: int,
    partition_index: int,
    partitions_per_shard: int,
    partition_rows_payload_sha256: str,
    partition_rows: int,
    base_model: Path,
    dataset_root: Path,
    train_result: Path,
    training_result: Mapping[str, Any],
    adapter_dir: Path,
    donor_payload: Sequence[Mapping[str, Any]],
    module_names: Sequence[str],
) -> Mapping[str, Any]:
    return {
        "schema": INPUT_SCHEMA,
        "protocol_payload_sha256": PROTOCOL_PAYLOAD_SHA256,
        "seed": SEED,
        "shard_index": shard_index,
        "partition_index": partition_index,
        "partitions_per_shard": partitions_per_shard,
        "world_size": WORLD_SIZE,
        "base_model": str(base_model),
        "base_config_sha256": training.preflight.EXPECTED_BASE_CONFIG_SHA256,
        "dataset_root": str(dataset_root),
        "dataset_file": str(dataset_root / DATASET_RELATIVE_PATH),
        "dataset_sha256": DATASET_SHA256,
        "authorized_rows": EVALUATION_ROWS,
        "authorized_rows_payload_sha256": AUTHORIZED_ROWS_PAYLOAD_SHA256,
        "partition_rows": partition_rows,
        "partition_rows_payload_sha256": partition_rows_payload_sha256,
        "train_result": str(train_result),
        "train_result_file_sha256": sha256_file(train_result),
        "train_result_receipt": training_result["receipt"]["payload_sha256"],
        "adapter_dir": str(adapter_dir),
        "adapter_files": training_result["adapter_files"],
        "donor_mapping_payload_sha256": canonical_sha256(list(donor_payload)),
        "donor_mapping_rows": len(donor_payload),
        "conditions": list(CONDITIONS),
        "module_names_sha256": canonical_sha256(list(module_names)),
        "hf_endpoint": os.environ.get("HF_ENDPOINT"),
        "runner_sha256": sha256_file(Path(__file__)),
        "addressed_eval_helper_sha256": sha256_file(Path(addressed_eval.__file__)),
        "generation_runner_sha256": sha256_file(Path(base_eval.causal.__file__)),
    }


def run(args: argparse.Namespace) -> int:
    if os.environ.get("HF_ENDPOINT") != HF_MIRROR_ENDPOINT:
        raise ValueError(f"HF_ENDPOINT must be exactly {HF_MIRROR_ENDPOINT}")
    if not 0 <= args.shard_index < WORLD_SIZE:
        raise ValueError("Chunk-addressed shard index is outside the four-way split")
    if (
        args.partitions_per_shard < 1
        or not 0 <= args.partition_index < args.partitions_per_shard
    ):
        raise ValueError("Chunk-addressed partition is outside the shard split")
    base_model = args.base_model.expanduser().resolve(strict=True)
    dataset_root = args.dataset_root.expanduser().resolve(strict=True)
    train_result = args.train_result.expanduser().resolve(strict=True)
    adapter_dir = args.adapter_dir.expanduser().resolve(strict=True)
    training_result = validate_train_result(train_result, adapter_dir=adapter_dir)
    dataset_file = dataset_root / DATASET_RELATIVE_PATH
    if sha256_file(dataset_file) != DATASET_SHA256:
        raise ValueError("Chunk-addressed evaluation dataset hash differs")
    metadata = base_eval.raw_line_metadata(dataset_file)
    rows = base_eval.parse_authorized_rows(metadata)
    by_index = {int(row["source_index"]): row for row in rows}
    tokenizer = AutoTokenizer.from_pretrained(base_model, local_files_only=True)
    counts = base_eval.prompt_token_counts(tokenizer, rows)
    donor_mapping = base_eval.build_donor_mapping(rows, counts)
    donor_payload = base_eval.donor_mapping_payload(rows, counts, donor_mapping)
    all_shard_rows = [
        row
        for row in rows
        if int(row["source_index"]) % WORLD_SIZE == args.shard_index
    ]
    shard_rows = [
        row
        for ordinal, row in enumerate(all_shard_rows)
        if ordinal % args.partitions_per_shard == args.partition_index
    ]
    partition_payload = [
        {
            "source_index": int(row["source_index"]),
            "row_sha256": str(row["row_sha256"]),
        }
        for row in shard_rows
    ]
    output_dir = (
        args.output_root.expanduser().resolve()
        / f"shard-{args.shard_index}"
        / f"partition-{args.partition_index}-of-{args.partitions_per_shard}"
    )
    model, loaded_tokenizer = load_model_and_tokenizer(
        base_model=str(base_model),
        memory_dir=str(adapter_dir),
        device=args.device,
        dtype="bfloat16",
        attn_implementation="sdpa",
    )
    module_names = tuple(name for name, _ in iter_delta_mem_modules(model))
    if len(module_names) != training.preflight.EXPECTED_LAYERS:
        raise ValueError(f"Expected 42 chunk-addressed layers, found {len(module_names)}")
    if not all(
        module.rwkv_ms_hybrid_mode == "chunk_addressed_value"
        for _, module in iter_delta_mem_modules(model)
    ):
        raise ValueError("Loaded adapter is not chunk-addressed")
    expected_binding = input_binding(
        shard_index=args.shard_index,
        partition_index=args.partition_index,
        partitions_per_shard=args.partitions_per_shard,
        partition_rows_payload_sha256=canonical_sha256(partition_payload),
        partition_rows=len(shard_rows),
        base_model=base_model,
        dataset_root=dataset_root,
        train_result=train_result,
        training_result=training_result,
        adapter_dir=adapter_dir,
        donor_payload=donor_payload,
        module_names=module_names,
    )
    base_eval.write_or_validate_json(output_dir / "input_binding.json", expected_binding)
    existing = {
        condition: base_eval.read_completed(output_dir / f"{condition}.jsonl")
        for condition in CONDITIONS
    }
    expected_rows = {int(row["source_index"]): row for row in shard_rows}
    for condition, records in existing.items():
        if not set(records) <= set(expected_rows):
            raise ValueError(f"Unexpected chunk-addressed resumed rows: {condition}")
        for source_index, result in records.items():
            required = {
                "schema": SCHEMA,
                "protocol_payload_sha256": PROTOCOL_PAYLOAD_SHA256,
                "condition": condition,
                "seed": SEED,
                "shard_index": args.shard_index,
                "world_size": WORLD_SIZE,
                "source_index": source_index,
                "row_sha256": expected_rows[source_index]["row_sha256"],
            }
            if any(result.get(key) != value for key, value in required.items()):
                raise ValueError(f"Chunk-addressed resumed record differs: {source_index}")
    if all(len(existing[condition]) == len(shard_rows) for condition in CONDITIONS):
        print(f"CHUNK_ADDRESSED_EVAL_SHARD_COMPLETE shard={args.shard_index}", flush=True)
        return 0
    started = time.time()
    for ordinal, row in enumerate(shard_rows, start=1):
        source_index = int(row["source_index"])
        donor = by_index[donor_mapping[source_index]]
        records = generate_row_conditions(
            model,
            loaded_tokenizer,
            row,
            donor,
            module_names=module_names,
            device=args.device,
            shard_index=args.shard_index,
        )
        for result in records:
            if source_index in existing[result["condition"]]:
                continue
            base_eval.append_jsonl(
                output_dir / f"{result['condition']}.jsonl",
                result,
            )
        print(
            f"CHUNK_ADDRESSED_EVAL_PROGRESS shard={args.shard_index} "
            f"partition={args.partition_index}/{args.partitions_per_shard} "
            f"row={source_index} ordinal={ordinal}/{len(shard_rows)} "
            f"elapsed={time.time() - started:.1f}",
            flush=True,
        )
    print(
        f"CHUNK_ADDRESSED_EVAL_SHARD_COMPLETE shard={args.shard_index} "
        f"partition={args.partition_index}/{args.partitions_per_shard}",
        flush=True,
    )
    return 0


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shard-index", type=int, required=True)
    parser.add_argument("--partition-index", type=int, default=0)
    parser.add_argument("--partitions-per-shard", type=int, default=1)
    parser.add_argument("--base-model", type=Path, default=training.BASE_MODEL)
    parser.add_argument("--dataset-root", type=Path, default=training.DATASET_ROOT)
    parser.add_argument("--train-result", type=Path, required=True)
    parser.add_argument("--adapter-dir", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    return parser.parse_args(argv)


if __name__ == "__main__":
    raise SystemExit(run(parse_args()))
