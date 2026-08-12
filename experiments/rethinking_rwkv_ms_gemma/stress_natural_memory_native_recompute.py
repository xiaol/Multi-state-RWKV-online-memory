#!/usr/bin/env python3
"""Stress exact native failure rows under four-rank episode recomputation."""

from __future__ import annotations

import argparse
import gc
import json
import os
from pathlib import Path
import resource
from typing import Any, Mapping, Sequence

import torch
from torch.distributed.elastic.multiprocessing.errors import record


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = Path(__file__).resolve().parents[2]
for import_root in (SCRIPT_DIR, PROJECT_ROOT):
    if str(import_root) not in os.sys.path:
        os.sys.path.insert(0, str(import_root))

from deltamem.core.delta import (
    load_delta_mem_adapter,
    reset_delta_mem_states,
    snapshot_delta_mem_weights,
)
from experiments.rethinking_rwkv_ms_gemma import natural_memory_distributed as distributed
from experiments.rethinking_rwkv_ms_gemma import prepare_natural_memory_gate as source
from experiments.rethinking_rwkv_ms_gemma import run_natural_memory_gate as gate
from experiments.rethinking_rwkv_ms_gemma import run_natural_memory_native_evolution as evolution
from experiments.rethinking_rwkv_ms_gemma import (
    run_synthetic_compositional_associative_retrieval_v3 as runtime,
)


SCHEMA = "rwkv_ms_natural_memory_native_recompute_stress.v1"
STRESS_ROW_IDS = (
    "native:narrative:1:fe47c59e6160d3a6a12d",
    "native:narrative:165:430a5cb02fe68e302358",
    "native:scene:1262:287639295f2437b85bb9",
)


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _process_peak_rss_bytes() -> int:
    return int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) * 1024


def _load_runtime(
    context: distributed.DistributedTrainingContext,
    *,
    base_model: Path,
    adapter_path: Path,
    source_manifest: Path,
    native_dataset_root: Path,
) -> tuple[torch.nn.Module, Any, list[evolution.NativeFullRowExample], Mapping[str, Any]]:
    evolution.gate.configure_hf_mirror()
    protocol = evolution.load_evolution_protocol(
        "shared_qo_content_gated_attention_output"
    )
    evolution.validate_native_dataset_root(native_dataset_root)
    adapter_files = gate.snapshot_directory_files(adapter_path)
    if gate._sha256_json(adapter_files) != evolution.R12_ADAPTER_FILES_SHA256:
        raise ValueError("Stress warm-start R12 adapter hash differs")
    bundle = gate.load_profile_bundle(source_manifest, profile="development")
    if Path(bundle.model_binding["local_model_path"]).resolve() != base_model:
        raise ValueError("Stress R12 source manifest base model differs")
    source_delta_config = evolution.build_evolution_delta_config("attention_output")
    delta_config = evolution.build_evolution_delta_config(
        "shared_qo_content_gated_attention_output"
    )
    runtime.set_seed(evolution.SEED)
    model, tokenizer, _, trainable_names, _ = gate._load_model_and_tokenizer(
        {"model": {"path": str(base_model)}},
        device=context.device,
        dtype=torch.bfloat16,
        attn_implementation="sdpa",
        delta_config=delta_config,
    )
    loaded_config = load_delta_mem_adapter(
        model,
        adapter_path,
        initialize_missing_content_gate=True,
    )
    if loaded_config.to_dict() != source_delta_config.to_dict():
        raise ValueError("Stress warm-start configuration differs")
    audit = gate.audit_trainable_parameters(
        model,
        expected_trainable_names=trainable_names,
    )
    if audit["passed"] is not True:
        raise ValueError("Stress found trainable frozen-base parameters")
    initial_hash = runtime._state_dict_sha256(snapshot_delta_mem_weights(model))
    distributed.require_consensus(
        context,
        initial_hash,
        description="stress warm-start adapter state",
    )
    native_examples = evolution.load_native_examples(native_dataset_root, tokenizer)
    binding = {
        "schema": SCHEMA,
        "protocol_payload_sha256": protocol["receipt"]["payload_sha256"],
        "base_model": str(base_model),
        "base_model_config_sha256": source.sha256_file(base_model / "config.json"),
        "warm_start_adapter": str(adapter_path),
        "warm_start_adapter_files_sha256": evolution.R12_ADAPTER_FILES_SHA256,
        "source_manifest_sha256": source.sha256_file(source_manifest),
        "native_dataset_manifest_sha256": source.sha256_file(
            native_dataset_root / "manifest.json"
        ),
        "row_ids": list(STRESS_ROW_IDS),
        "world_size": context.world_size,
        "execution": "each_exact_row_forward_and_backward_on_every_rank",
        "hf_endpoint": os.environ.get("HF_ENDPOINT"),
        "protected_splits_opened": [],
    }
    distributed.require_consensus(
        context,
        evolution.canonical_sha256(binding),
        description="native recompute stress binding",
    )
    return model, tokenizer, native_examples, binding


def run_stress(
    context: distributed.DistributedTrainingContext,
    *,
    output_dir: Path,
    base_model: Path = evolution.BASE_MODEL,
    adapter_path: Path = evolution.R12_ADAPTER,
    source_manifest: Path = evolution.R12_SOURCE_MANIFEST,
    native_dataset_root: Path = evolution.NATIVE_DATASET_ROOT,
) -> Mapping[str, Any]:
    output_dir = output_dir.expanduser().resolve()
    if output_dir.exists():
        raise ValueError(f"Stress output must be fresh: {output_dir}")
    base_model = base_model.expanduser().resolve(strict=True)
    adapter_path = adapter_path.expanduser().resolve(strict=True)
    source_manifest = source_manifest.expanduser().resolve(strict=True)
    native_dataset_root = native_dataset_root.expanduser().resolve(strict=True)
    model, tokenizer, native_examples, binding = _load_runtime(
        context,
        base_model=base_model,
        adapter_path=adapter_path,
        source_manifest=source_manifest,
        native_dataset_root=native_dataset_root,
    )
    by_id = {example.row_id: example for example in native_examples}
    if any(row_id not in by_id for row_id in STRESS_ROW_IDS):
        missing = [row_id for row_id in STRESS_ROW_IDS if row_id not in by_id]
        raise ValueError(f"Stress rows are missing: {missing}")
    named_trainable = gate._named_trainable_parameters(model)
    model.train()
    local_results = []
    for row_id in STRESS_ROW_IDS:
        model.zero_grad(set_to_none=True)
        reset_delta_mem_states(model)
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(context.device)
        batch = evolution.collate_native_examples(
            [by_id[row_id]],
            pad_token_id=int(tokenizer.pad_token_id),
            device=context.device,
        )
        write_audit, logits = evolution.checkpointed_native_write_read(
            model,
            batch,
            dtype=torch.bfloat16,
        )
        loss, answer_tokens, ce_chunks = (
            evolution.checkpointed_native_answer_loss_sum_and_count(
                logits,
                batch.labels,
            )
        )
        loss.backward()
        gradients = distributed.validate_local_gradients(named_trainable)
        gate_gradients = evolution.audit_content_gate_gradients(named_trainable)
        if gradients["passed"] is not True or gate_gradients["passed"] is not True:
            raise RuntimeError(f"Stress row produced invalid gradients: {row_id}")
        torch.cuda.synchronize(context.device)
        local_results.append(
            {
                "process_rank": context.process_rank,
                "row_id": row_id,
                "write_tokens": int(batch.write_input_ids.size(1)),
                "read_tokens": int(batch.read_input_ids.size(1)),
                "answer_tokens": answer_tokens,
                "ce_chunks": ce_chunks,
                "loss_sum": float(loss.detach().float().item()),
                "write_audit": dict(write_audit),
                "gradient_audit": dict(gradients),
                "content_gate_gradient_audit": dict(gate_gradients),
                "cuda_memory": dict(distributed.cuda_memory_snapshot(context)),
                "process_peak_rss_bytes": _process_peak_rss_bytes(),
                "passed": True,
            }
        )
        reset_delta_mem_states(model)
        batch = None
        logits = None
        loss = None
        gc.collect()
        torch.cuda.empty_cache()
    gathered = distributed.gather_objects(context, local_results)
    flattened = [record for rank_results in gathered for record in rank_results]
    expected_pairs = {
        (rank, row_id)
        for rank in range(context.world_size)
        for row_id in STRESS_ROW_IDS
    }
    actual_pairs = {
        (int(record["process_rank"]), str(record["row_id"]))
        for record in flattened
    }
    passed = (
        actual_pairs == expected_pairs
        and len(flattened) == len(expected_pairs)
        and all(record.get("passed") is True for record in flattened)
    )
    result = {
        "schema": SCHEMA,
        "status": "passed" if passed else "failed",
        "input_binding": binding,
        "rank_row_results": flattened,
        "maximum_cuda_peak_allocated_bytes": max(
            int(record["cuda_memory"]["peak_allocated_bytes"])
            for record in flattened
        ),
        "maximum_cuda_peak_reserved_bytes": max(
            int(record["cuda_memory"]["peak_reserved_bytes"])
            for record in flattened
        ),
        "maximum_process_peak_rss_bytes": max(
            int(record["process_peak_rss_bytes"]) for record in flattened
        ),
        "protected_splits_opened": [],
        "passed": passed,
        "code_bindings": {
            "stress_driver_sha256": source.sha256_file(Path(__file__)),
            "runner_sha256": source.sha256_file(Path(evolution.__file__)),
            "delta_impl_sha256": source.sha256_file(
                PROJECT_ROOT / "deltamem/core/delta_impl.py"
            ),
        },
    }
    result["receipt"] = {
        "algorithm": "sha256",
        "payload_scope": "canonical_result_without_receipt",
        "payload_sha256": source.sha256_text(source.canonical_json(result)),
    }
    save_error: BaseException | None = None
    if context.is_primary:
        try:
            output_dir.mkdir(parents=True, exist_ok=False)
            _write_json(output_dir / "result.json", result)
        except BaseException as error:
            save_error = error
    distributed.phase_consensus(
        context,
        phase="native-recompute-stress-save",
        error=save_error,
    )
    if not passed:
        raise RuntimeError("Native recompute stress gate failed")
    return result if context.is_primary else {
        "status": "worker_complete",
        "rank": context.process_rank,
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--base-model", type=Path, default=evolution.BASE_MODEL)
    parser.add_argument("--adapter-path", type=Path, default=evolution.R12_ADAPTER)
    parser.add_argument("--source-manifest", type=Path, default=evolution.R12_SOURCE_MANIFEST)
    parser.add_argument(
        "--native-dataset-root",
        type=Path,
        default=evolution.NATIVE_DATASET_ROOT,
    )
    parser.add_argument("--device", default="cuda")
    return parser.parse_args(argv)


@record
def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    context = distributed.initialize_distributed_training(args.device)
    if context is None:
        raise ValueError("Native recompute stress requires four-rank torchrun")
    try:
        result = run_stress(
            context,
            output_dir=args.output_dir,
            base_model=args.base_model,
            adapter_path=args.adapter_path,
            source_manifest=args.source_manifest,
            native_dataset_root=args.native_dataset_root,
        )
    finally:
        distributed.destroy_distributed_training(context)
    print(json.dumps(result, ensure_ascii=True, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
