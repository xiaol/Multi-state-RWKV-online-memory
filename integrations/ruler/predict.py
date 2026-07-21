#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import inspect
import json
from pathlib import Path
import sys
import time


ROOT = Path(__file__).resolve().parents[2]
TASK_TOKEN_BUDGETS = {
    "niah_single_1": 128,
    "niah_single_2": 128,
    "niah_single_3": 128,
    "niah_multikey_1": 128,
    "niah_multikey_2": 128,
    "niah_multikey_3": 128,
    "niah_multivalue": 128,
    "niah_multiquery": 128,
    "vt": 30,
    "cwe": 120,
    "fwe": 50,
    "qa_1": 32,
    "qa_2": 32,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate matched base or RWKV-MS predictions for one RULER task."
    )
    parser.add_argument("--input-file", type=Path, required=True)
    parser.add_argument("--output-file", type=Path, required=True)
    parser.add_argument("--task", choices=sorted(TASK_TOKEN_BUDGETS), required=True)
    parser.add_argument("--variant", choices=("base", "hybrid"), required=True)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--adapter-dir", type=Path)
    parser.add_argument(
        "--runtime-root",
        "--delta-mem-root",
        dest="runtime_root",
        type=Path,
        default=ROOT,
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--dtype",
        choices=("bfloat16", "float16", "float32"),
        default="bfloat16",
    )
    parser.add_argument("--attn-implementation", default="sdpa")
    parser.add_argument("--max-new-tokens", type=int, default=0)
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument(
        "--prefill-chunk-tokens",
        type=int,
        default=0,
        help="Prefill with cached chunks of this size; 0 disables chunking.",
    )
    parser.add_argument("--chunk-index", type=int, default=0)
    parser.add_argument("--chunk-count", type=int, default=1)
    parser.add_argument("--overwrite-output", action="store_true")
    return parser.parse_args()


def load_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if not line:
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"Expected an object at {path}:{line_number}")
            rows.append(row)
    if not rows:
        raise ValueError(f"No rows found in {path}")
    return rows


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def eos_token_ids(model, tokenizer) -> set[int]:
    values = []
    generation_config = getattr(model, "generation_config", None)
    if generation_config is not None:
        values.append(getattr(generation_config, "eos_token_id", None))
    values.append(getattr(tokenizer, "eos_token_id", None))
    resolved: set[int] = set()
    for value in values:
        if value is None:
            continue
        if isinstance(value, (list, tuple, set)):
            resolved.update(int(item) for item in value)
        else:
            resolved.add(int(value))
    return resolved


def model_context_window(model) -> int:
    config = getattr(model, "config", None)
    candidates = (getattr(config, "text_config", None), config)
    for candidate in candidates:
        value = getattr(candidate, "max_position_embeddings", None)
        if isinstance(value, int) and value > 0:
            return value
    raise ValueError("Could not determine the model context window")


def logits_to_keep_kwargs(model, value: int) -> dict[str, int]:
    try:
        parameters = inspect.signature(model.forward).parameters
    except (AttributeError, TypeError, ValueError):
        return {}
    for name in ("logits_to_keep", "num_logits_to_keep"):
        if name in parameters:
            return {name: value}
    return {}


def synchronize(device: str) -> None:
    import torch

    if device.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.synchronize(device)


def reset_cuda_peak_memory_stats(device: str) -> None:
    import torch

    if device.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats(device)


def cuda_peak_memory_stats(device: str) -> tuple[int, int]:
    import torch

    if not device.startswith("cuda") or not torch.cuda.is_available():
        return 0, 0
    return (
        int(torch.cuda.max_memory_allocated(device)),
        int(torch.cuda.max_memory_reserved(device)),
    )


def validate_resume_manifest(
    manifest_path: Path,
    output_path: Path,
    expected_identity: dict[str, object],
) -> dict | None:
    if not manifest_path.is_file():
        if output_path.is_file() and output_path.stat().st_size:
            raise ValueError(f"Existing output has no identity manifest: {output_path}")
        return None
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(manifest, dict):
        raise ValueError(f"Expected an object in {manifest_path}")
    for key, expected in expected_identity.items():
        if manifest.get(key) != expected:
            raise ValueError(f"Resume metadata mismatch for {output_path}: {key}")
    status = manifest.get("status")
    if status not in ("running", "complete"):
        raise ValueError(f"Invalid prediction status in {manifest_path}: {status!r}")
    if status == "complete":
        if not output_path.is_file():
            raise FileNotFoundError(f"Completed prediction output is missing: {output_path}")
        if manifest.get("output_sha256") != sha256_file(output_path):
            raise ValueError(f"Completed prediction hash mismatch: {output_path}")
    return manifest


def validate_hybrid_prefill_chunking(prefill_chunk_tokens: int, adapter_config) -> None:
    granularity = getattr(adapter_config, "memory_write_granularity", None)
    if prefill_chunk_tokens > 0 and granularity != "token":
        raise ValueError(
            "Chunked hybrid prefill requires memory_write_granularity='token'; "
            f"found {granularity!r}"
        )


def prefill_with_chunks(
    model,
    input_ids,
    attention_mask,
    *,
    prefill_chunk_tokens: int,
    prefill_logits_kwargs: dict[str, int],
    write_message_ids=None,
    write_sentence_ids=None,
    set_write_message_ids=None,
    set_write_sentence_ids=None,
):
    """Run a monolithic or cached chunked prefill and return its final output."""
    prompt_tokens = int(input_ids.size(1))
    if prefill_chunk_tokens <= 0 or prefill_chunk_tokens >= prompt_tokens:
        if set_write_message_ids is not None:
            set_write_message_ids(model, write_message_ids)
        if set_write_sentence_ids is not None:
            set_write_sentence_ids(model, write_sentence_ids)
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=True,
            return_dict=True,
            **prefill_logits_kwargs,
        )
        return outputs, 1

    past_key_values = None
    final_outputs = None
    chunk_count = 0
    for start in range(0, prompt_tokens, prefill_chunk_tokens):
        end = min(start + prefill_chunk_tokens, prompt_tokens)
        if set_write_message_ids is not None:
            set_write_message_ids(
                model,
                None if write_message_ids is None else write_message_ids[:, start:end],
            )
        if set_write_sentence_ids is not None:
            set_write_sentence_ids(
                model,
                None if write_sentence_ids is None else write_sentence_ids[:, start:end],
            )
        outputs = model(
            input_ids=input_ids[:, start:end],
            attention_mask=attention_mask[:, :end],
            past_key_values=past_key_values,
            use_cache=True,
            return_dict=True,
            **prefill_logits_kwargs,
        )
        past_key_values = outputs.past_key_values
        chunk_count += 1
        if end < prompt_tokens:
            if past_key_values is None:
                raise RuntimeError("Chunked prefill requires the model to return a cache")
            del outputs
        else:
            final_outputs = outputs

    if final_outputs is None:
        raise RuntimeError("Chunked prefill produced no output")
    return final_outputs, chunk_count


def main() -> None:
    args = parse_args()
    if args.chunk_count < 1:
        raise ValueError("chunk-count must be >= 1")
    if args.chunk_index < 0 or args.chunk_index >= args.chunk_count:
        raise ValueError("chunk-index must satisfy 0 <= index < chunk-count")
    if args.max_new_tokens < 0 or args.max_samples < 0 or args.prefill_chunk_tokens < 0:
        raise ValueError(
            "max-new-tokens, max-samples, and prefill-chunk-tokens must be >= 0"
        )
    if args.variant == "hybrid":
        if args.adapter_dir is None:
            raise ValueError("hybrid prediction requires --adapter-dir")
        if not (args.adapter_dir / "delta_mem_adapter.pt").is_file():
            raise FileNotFoundError(f"Missing adapter weights in {args.adapter_dir}")
        if not (args.adapter_dir / "delta_mem_config.json").is_file():
            raise FileNotFoundError(f"Missing adapter config in {args.adapter_dir}")
        if not (args.runtime_root / "deltamem").is_dir():
            raise FileNotFoundError(f"Bundled deltamem package not found under {args.runtime_root}")
        sys.path.insert(0, str(args.runtime_root.resolve()))

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    reset_delta_mem_states = None
    set_delta_mem_write_enabled = None
    set_delta_mem_write_message_ids = None
    set_delta_mem_write_sentence_ids = None
    collect_delta_mem_state_stats = None
    if args.variant == "hybrid":
        from deltamem.core.delta import HFDeltaMemConfig, attach_delta_mem
        from deltamem.core.delta_impl import (
            collect_delta_mem_state_stats,
            load_delta_mem_adapter,
            reset_delta_mem_states,
            set_delta_mem_write_enabled,
            set_delta_mem_write_message_ids,
            set_delta_mem_write_sentence_ids,
        )

    rows = load_jsonl(args.input_file)
    selected = [
        (ordinal, row)
        for ordinal, row in enumerate(rows)
        if ordinal % args.chunk_count == args.chunk_index
    ]
    if args.max_samples:
        selected = selected[: args.max_samples]
    if not selected:
        raise ValueError("No rows selected for this chunk")

    args.output_file.parent.mkdir(parents=True, exist_ok=True)
    manifest_path = args.output_file.with_suffix(args.output_file.suffix + ".manifest.json")
    max_new_tokens = args.max_new_tokens or TASK_TOKEN_BUDGETS[args.task]
    resume_identity = {
        "variant": args.variant,
        "task": args.task,
        "input_file": str(args.input_file.resolve()),
        "input_sha256": sha256_file(args.input_file),
        "output_file": str(args.output_file.resolve()),
        "model_path": str(Path(args.model_path).expanduser().resolve()),
        "adapter_dir": None if args.adapter_dir is None else str(args.adapter_dir.resolve()),
        "adapter_sha256": None
        if args.adapter_dir is None
        else sha256_file(args.adapter_dir / "delta_mem_adapter.pt"),
        "adapter_config_sha256": None
        if args.adapter_dir is None
        else sha256_file(args.adapter_dir / "delta_mem_config.json"),
        "runtime_root": str(args.runtime_root.expanduser().resolve()),
        "device": args.device,
        "dtype": args.dtype,
        "attn_implementation": args.attn_implementation,
        "prefill_chunk_tokens": args.prefill_chunk_tokens,
        "max_new_tokens": max_new_tokens,
        "chunk_index": args.chunk_index,
        "chunk_count": args.chunk_count,
        "selected_rows": len(selected),
    }
    if args.overwrite_output:
        args.output_file.write_text("", encoding="utf-8")
        manifest_path.unlink(missing_ok=True)
    existing_manifest = validate_resume_manifest(
        manifest_path,
        args.output_file,
        resume_identity,
    )
    existing = (
        load_jsonl(args.output_file)
        if args.output_file.is_file() and args.output_file.stat().st_size
        else []
    )
    selected_ordinals = {ordinal for ordinal, _ in selected}
    completed_ordinals: set[int] = set()
    for existing_row in existing:
        if existing_row.get("variant") != args.variant or existing_row.get("task") != args.task:
            raise ValueError(f"Existing output belongs to another run: {args.output_file}")
        ordinal = int(existing_row["sample_ordinal"])
        if ordinal not in selected_ordinals:
            raise ValueError(f"Existing output has unexpected sample ordinal {ordinal}")
        if ordinal in completed_ordinals:
            raise ValueError(f"Existing output has duplicate sample ordinal {ordinal}")
        for key in ("input", "outputs"):
            if existing_row.get(key) != rows[ordinal].get(key):
                raise ValueError(f"Existing output row {ordinal} does not match source {key}")
        completed_ordinals.add(ordinal)
    if existing_manifest is not None and existing_manifest["status"] == "complete":
        if completed_ordinals != selected_ordinals:
            raise ValueError(f"Completed prediction output has incomplete coverage: {args.output_file}")
        if existing_manifest.get("completed_rows") != len(selected_ordinals):
            raise ValueError(f"Completed-row metadata mismatch: {args.output_file}")
    pending = [(ordinal, row) for ordinal, row in selected if ordinal not in completed_ordinals]

    dtype = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }[args.dtype]
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        dtype=dtype,
        device_map={"": args.device},
        attn_implementation=args.attn_implementation,
        local_files_only=True,
        low_cpu_mem_usage=True,
    ).eval()
    model.config.use_cache = True
    if args.variant == "hybrid":
        adapter_config = HFDeltaMemConfig.from_pretrained(args.adapter_dir)
        validate_hybrid_prefill_chunking(args.prefill_chunk_tokens, adapter_config)
        replaced_modules = attach_delta_mem(model, adapter_config)
        load_delta_mem_adapter(model, args.adapter_dir)
    else:
        replaced_modules = []

    context_window = model_context_window(model)
    stop_ids = eos_token_ids(model, tokenizer)
    model_device = next(model.parameters()).device
    prefill_logits_kwargs = logits_to_keep_kwargs(model, 1)

    manifest = {
        **resume_identity,
        "status": "running" if pending else "complete",
        "prefill_logits_to_keep": 1 if prefill_logits_kwargs else 0,
        "completed_rows": len(completed_ordinals),
        "replaced_modules": replaced_modules,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    with args.output_file.open("a", encoding="utf-8", buffering=1) as output_handle:
        for ordinal, row in pending:
            prompt = str(row.get("input", ""))
            if not prompt:
                raise ValueError(f"RULER row {ordinal} has an empty input")
            tokenized = tokenizer(
                prompt,
                return_tensors="pt",
                add_special_tokens=False,
            )
            input_ids = tokenized.input_ids.to(model_device)
            attention_mask = tokenized.attention_mask.to(model_device)
            prompt_tokens = int(input_ids.size(1))
            if prompt_tokens + max_new_tokens > context_window:
                raise ValueError(
                    f"Row {ordinal} requires {prompt_tokens + max_new_tokens} tokens, "
                    f"above the model limit {context_window}"
                )

            if args.variant == "hybrid":
                reset_delta_mem_states(model)
                set_delta_mem_write_enabled(model, True)
                write_message_ids = torch.zeros_like(input_ids)
                write_sentence_ids = torch.full_like(input_ids, -1)
            else:
                write_message_ids = None
                write_sentence_ids = None

            synchronize(args.device)
            reset_cuda_peak_memory_stats(args.device)
            started_at = time.perf_counter()
            try:
                with torch.inference_mode():
                    outputs, prefill_chunk_count = prefill_with_chunks(
                        model,
                        input_ids,
                        attention_mask,
                        prefill_chunk_tokens=args.prefill_chunk_tokens,
                        prefill_logits_kwargs=prefill_logits_kwargs,
                        write_message_ids=write_message_ids,
                        write_sentence_ids=write_sentence_ids,
                        set_write_message_ids=set_delta_mem_write_message_ids,
                        set_write_sentence_ids=set_delta_mem_write_sentence_ids,
                    )
                if args.variant == "hybrid":
                    set_delta_mem_write_message_ids(model, None)
                    set_delta_mem_write_sentence_ids(model, None)
                    set_delta_mem_write_enabled(model, False)

                past_key_values = outputs.past_key_values
                next_token_logits = outputs.logits[:, -1, :]
                generated_ids: list[int] = []
                with torch.inference_mode():
                    for _ in range(max_new_tokens):
                        next_token = next_token_logits.argmax(dim=-1, keepdim=True)
                        token_id = int(next_token.item())
                        if token_id in stop_ids:
                            break
                        generated_ids.append(token_id)
                        outputs = model(
                            input_ids=next_token,
                            past_key_values=past_key_values,
                            use_cache=True,
                            return_dict=True,
                        )
                        past_key_values = outputs.past_key_values
                        next_token_logits = outputs.logits[:, -1, :]
            finally:
                if args.variant == "hybrid":
                    set_delta_mem_write_message_ids(model, None)
                    set_delta_mem_write_sentence_ids(model, None)
                    set_delta_mem_write_enabled(model, True)

            synchronize(args.device)
            elapsed_seconds = time.perf_counter() - started_at
            cuda_peak_allocated_bytes, cuda_peak_reserved_bytes = cuda_peak_memory_stats(
                args.device
            )
            prediction = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
            state_stats = (
                collect_delta_mem_state_stats(model) if args.variant == "hybrid" else None
            )
            result = dict(row)
            result.update(
                {
                    "sample_ordinal": ordinal,
                    "task": args.task,
                    "variant": args.variant,
                    "pred": prediction,
                    "others": dict(row.get("others") or {}),
                    "prompt_tokens": prompt_tokens,
                    "generated_tokens": len(generated_ids),
                    "prefill_chunk_count": prefill_chunk_count,
                    "cuda_peak_allocated_bytes": cuda_peak_allocated_bytes,
                    "cuda_peak_reserved_bytes": cuda_peak_reserved_bytes,
                    "elapsed_seconds": elapsed_seconds,
                    "state_stats": state_stats,
                }
            )
            output_handle.write(json.dumps(result, ensure_ascii=False) + "\n")
            print(
                "RULER_PREDICTION="
                + json.dumps(
                    {
                        "sample_ordinal": ordinal,
                        "task": args.task,
                        "variant": args.variant,
                        "prompt_tokens": prompt_tokens,
                        "generated_tokens": len(generated_ids),
                        "prefill_chunk_count": prefill_chunk_count,
                        "cuda_peak_allocated_bytes": cuda_peak_allocated_bytes,
                        "cuda_peak_reserved_bytes": cuda_peak_reserved_bytes,
                        "elapsed_seconds": elapsed_seconds,
                        "prediction": prediction,
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
            del outputs, past_key_values, next_token_logits

    manifest["status"] = "complete"
    manifest["completed_rows"] = len(selected)
    manifest["output_sha256"] = sha256_file(args.output_file)
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
