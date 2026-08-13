#!/usr/bin/env python3
"""Run one append-only shard of the native scene causal study."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import re
import sys
import time
from typing import Any, Mapping, Sequence

import torch
from transformers import AutoTokenizer


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = Path(__file__).resolve().parents[2]
for import_root in (SCRIPT_DIR, PROJECT_ROOT):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

from common import (  # noqa: E402
    load_model_and_tokenizer,
    reset_delta_state,
    set_delta_write_enabled,
)
from deltamem.chat_templates import apply_chat_template  # noqa: E402
from deltamem.core.delta import (  # noqa: E402
    get_delta_mem_online_state,
    iter_delta_mem_modules,
    load_delta_mem_online_state,
)
from experiments.rethinking_rwkv_ms_gemma import (  # noqa: E402
    analyze_novel_agent_eval as recovery,
)
from experiments.rethinking_rwkv_ms_gemma import run_novel_agent_eval as evaluator  # noqa: E402


SCHEMA = "rwkv_ms_natural_memory_native_scene_causal_shard.v1"
INPUT_SCHEMA = "rwkv_ms_natural_memory_native_scene_causal_input.v1"
PROTOCOL = SCRIPT_DIR / "natural_memory_native_scene_causal_router_protocol_v1.json"
PROTOCOL_PAYLOAD_SHA256 = (
    "2829666c8d8e8bb5eb951b00e9d38d8e4cbd6db62f78beea9d95f440e140a0e2"
)
MEMORY_ADAPTER_SHA256 = (
    "b063940a9be0712f830a992e9114055e4488297b0842245e6e26563b303545a9"
)
BASE_CONFIG_SHA256 = (
    "33b10c02df3c2e8536cf323d29d53262aaa2f4d11dbe19bc729373fbe90295d4"
)
DATASET_SHA256 = (
    "b383625cee07e6a7565142e38bb0b0a4d4a2468b2c91171570115b7b311e1e68"
)
DATASET_RELATIVE_PATH = "v4-scene-boundary-detection/train_derived_development.jsonl"
EXPECTED_ROWS = 361
SELECTION_ROWS = 4
WORLD_SIZE = 4
CONDITIONS = ("zero_state", "donor_state", "layer_permuted_correct_state")
MAX_NEW_TOKENS = 128
REFERENCE_ARTIFACT_SHA256 = {
    "base": (
        "5e51f3d0e7d534a09c5c96a8f2a995c0c8383f07d5383e63151c235df7c0a4f6",
        "5bcbebc859a3ae54d9137308b579930d1b1c8d3e5a587e7a4a03c07f2dbc7010",
        "902a30495a40b07089bb4918d8c5d4aa66b64d36c958c452b0e4dbc1d94d52d9",
        "4e2f0c0cb4fff1c1beb3fec7131c3a982b9da1be4a76cc43651f6f38d9a84d91",
    ),
    "memory": (
        "f80b5c1068636759786e80c40bd7ea37df0eb66cb73e3e0a5b09edabc86014da",
        "9c18846305e40db7af4a046647693401ac1f569dc64f44e4c0f5434e726431cd",
        "43d5f18e82f7fd60ad425ad285e3172a36e82b6c8a942f46762cf6baed2efac2",
        "4be071a9196ca7d78edf64dd5aac0ee8f1982acb7c644153e223bb3d9f417dfa",
    ),
}
STATE_SUFFIXES = (
    "",
    ".__rwkv_ms_positions",
    ".__rwkv_ms_previous_source",
    ".__projected_kv_keys",
    ".__projected_kv_values",
    ".__projected_kv_occupied",
    ".__projected_kv_surprise",
)


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
        raise ValueError("Scene causal protocol receipt is missing")
    unsigned = dict(protocol)
    unsigned.pop("receipt")
    digest = canonical_sha256(unsigned)
    if digest != PROTOCOL_PAYLOAD_SHA256 or receipt.get("payload_sha256") != digest:
        raise ValueError("Scene causal protocol hash differs")
    return protocol


def load_rows(dataset_root: Path) -> list[dict[str, Any]]:
    path = dataset_root / DATASET_RELATIVE_PATH
    if sha256_file(path) != DATASET_SHA256:
        raise ValueError("Scene causal development dataset hash differs")
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            if not raw_line.strip():
                continue
            value = json.loads(raw_line)
            messages = value.get("messages")
            if not isinstance(messages, list) or len(messages) < 3:
                raise ValueError(f"Invalid scene causal row: {path}")
            gold = recovery.extract_json(str(messages[-1].get("content", "")))
            if not isinstance(gold, Mapping):
                raise ValueError(f"Invalid scene causal gold: {path}")
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
    if len(rows) != EXPECTED_ROWS:
        raise ValueError(f"Expected {EXPECTED_ROWS} scene rows, found {len(rows)}")
    return rows


def write_prompt_token_counts(tokenizer, rows: Sequence[Mapping[str, Any]]) -> list[int]:
    counts: list[int] = []
    for row in rows:
        rendered = apply_chat_template(
            tokenizer,
            list(row["messages"]),
            tokenize=False,
            add_generation_prompt=False,
        )
        counts.append(
            len(tokenizer(rendered, add_special_tokens=False).input_ids)
        )
    return counts


def build_donor_mapping(
    rows: Sequence[Mapping[str, Any]],
    token_counts: Sequence[int],
) -> dict[int, int]:
    if len(rows) != len(token_counts):
        raise ValueError("Scene causal donor token counts differ from rows")
    selected = [
        row for row in rows if int(row["source_index"]) >= SELECTION_ROWS
    ]
    gold = {
        int(row["source_index"]): recovery.strict_gold_boundaries(row["gold"])
        for row in selected
    }
    mapping: dict[int, int] = {}
    for row in selected:
        target = int(row["source_index"])
        candidates = [
            int(candidate["source_index"])
            for candidate in selected
            if int(candidate["source_index"]) != target
            and gold[int(candidate["source_index"])] != gold[target]
        ]
        if not candidates:
            raise ValueError(f"Scene causal row has no valid donor: {target}")
        mapping[target] = min(
            candidates,
            key=lambda donor: (abs(token_counts[target] - token_counts[donor]), donor),
        )
    return mapping


def donor_mapping_payload(
    rows: Sequence[Mapping[str, Any]],
    token_counts: Sequence[int],
    mapping: Mapping[int, int],
) -> list[dict[str, Any]]:
    by_index = {int(row["source_index"]): row for row in rows}
    return [
        {
            "source_index": source,
            "source_row_sha256": by_index[source]["row_sha256"],
            "source_write_tokens": token_counts[source],
            "donor_source_index": donor,
            "donor_row_sha256": by_index[donor]["row_sha256"],
            "donor_write_tokens": token_counts[donor],
            "absolute_write_token_delta": abs(token_counts[source] - token_counts[donor]),
        }
        for source, donor in sorted(mapping.items())
    ]


def selected_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    shard_index: int,
) -> list[Mapping[str, Any]]:
    return [
        row
        for row in rows
        if int(row["source_index"]) >= SELECTION_ROWS
        and int(row["source_index"]) % WORLD_SIZE == shard_index
    ]


def reference_artifacts(reference_root: Path) -> list[dict[str, Any]]:
    artifacts: list[dict[str, Any]] = []
    for condition, expected_hashes in REFERENCE_ARTIFACT_SHA256.items():
        for shard_index, expected_hash in enumerate(expected_hashes):
            path = reference_root / f"shard-{shard_index}" / f"scene.{condition}.jsonl"
            digest = sha256_file(path)
            if digest != expected_hash:
                raise ValueError(f"Scene causal reference artifact differs: {path}")
            artifacts.append(
                {
                    "condition": condition,
                    "shard_index": shard_index,
                    "path": str(path),
                    "sha256": digest,
                }
            )
    return artifacts


def input_binding(
    *,
    base_model: Path,
    memory_dir: Path,
    dataset_root: Path,
    reference_root: Path,
    shard_index: int,
    donor_payload: Sequence[Mapping[str, Any]],
    dtype: str,
    attn_implementation: str,
) -> Mapping[str, Any]:
    if sha256_file(base_model / "config.json") != BASE_CONFIG_SHA256:
        raise ValueError("Scene causal base config differs")
    if sha256_file(memory_dir / "delta_mem_adapter.pt") != MEMORY_ADAPTER_SHA256:
        raise ValueError("Scene causal memory adapter differs")
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
        "dataset_file": str(dataset_root / DATASET_RELATIVE_PATH),
        "dataset_sha256": DATASET_SHA256,
        "reference_root": str(reference_root),
        "reference_artifacts": reference_artifacts(reference_root),
        "donor_mapping_payload_sha256": canonical_sha256(list(donor_payload)),
        "donor_mapping_rows": len(donor_payload),
        "shard_index": shard_index,
        "world_size": WORLD_SIZE,
        "conditions": list(CONDITIONS),
        "dtype": dtype,
        "attn_implementation": attn_implementation,
        "runner_sha256": sha256_file(Path(__file__)),
        "generation_runner_sha256": sha256_file(
            SCRIPT_DIR / "run_novel_agent_eval.py"
        ),
    }


def write_or_validate_json(path: Path, value: Mapping[str, Any]) -> None:
    if path.is_file():
        if json.loads(path.read_text(encoding="utf-8")) != value:
            raise ValueError(f"Scene causal binding differs: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
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
                raise ValueError(f"Duplicate scene causal record: {path}:{index}")
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
    condition: str,
    shard_index: int,
) -> None:
    expected_by_index = {int(row["source_index"]): row for row in expected}
    if not set(existing) <= set(expected_by_index):
        raise ValueError(f"Unexpected resumed scene causal indices: {condition}")
    for index, record in existing.items():
        row = expected_by_index[index]
        required = {
            "schema": SCHEMA,
            "protocol_payload_sha256": PROTOCOL_PAYLOAD_SHA256,
            "condition": condition,
            "shard_index": shard_index,
            "world_size": WORLD_SIZE,
            "source_index": index,
            "row_sha256": row["row_sha256"],
        }
        if any(record.get(key) != value for key, value in required.items()):
            raise ValueError(f"Resumed scene causal record differs: {condition}:{index}")


def tensor_digest(state: Mapping[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for name in sorted(state):
        tensor = state[name].detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(json.dumps(list(tensor.shape), separators=(",", ":")).encode("ascii"))
        digest.update(tensor.view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def layer_ordinal(name: str) -> int:
    match = re.search(r"\.layers\.(\d+)\.", name)
    if match is None:
        raise ValueError(f"Cannot identify wrapped layer ordinal: {name}")
    return int(match.group(1))


def permute_layer_state(
    state: Mapping[str, torch.Tensor],
    module_names: Sequence[str],
) -> dict[str, torch.Tensor]:
    ordered = tuple(sorted(module_names, key=layer_ordinal))
    if len(ordered) != 42 or len(set(ordered)) != len(ordered):
        raise ValueError("Scene causal layer permutation requires 42 unique modules")
    expected_keys = {
        f"{module_name}{suffix}"
        for module_name in ordered
        for suffix in STATE_SUFFIXES
    }
    if set(state) != expected_keys:
        missing = sorted(expected_keys - set(state))
        extra = sorted(set(state) - expected_keys)
        raise ValueError(
            f"Scene causal online-state bundle differs: missing={missing[:3]} extra={extra[:3]}"
        )
    permuted: dict[str, torch.Tensor] = {}
    for target_index, target_name in enumerate(ordered):
        source_name = ordered[(target_index + 1) % len(ordered)]
        for suffix in STATE_SUFFIXES:
            target_key = f"{target_name}{suffix}"
            source_key = f"{source_name}{suffix}"
            target_tensor = state[target_key]
            source_tensor = state[source_key]
            if target_tensor.shape != source_tensor.shape or target_tensor.dtype != source_tensor.dtype:
                raise ValueError(
                    f"Scene causal layer-state tensor contract differs: {target_key} <- {source_key}"
                )
            permuted[target_key] = source_tensor.detach().cpu().clone()
    if tensor_digest(permuted) == tensor_digest(state):
        raise ValueError("Scene causal layer permutation did not change state digest")
    return permuted


def encode_prompt(tokenizer, messages: Sequence[Mapping[str, str]], *, generation: bool):
    rendered = apply_chat_template(
        tokenizer,
        list(messages),
        tokenize=False,
        add_generation_prompt=generation,
    )
    return tokenizer(rendered, return_tensors="pt", add_special_tokens=False)


def prime_messages(model, tokenizer, messages, *, device: str) -> Mapping[str, torch.Tensor]:
    reset_delta_state(model)
    encoded = encode_prompt(tokenizer, messages, generation=False)
    set_delta_write_enabled(model, True)
    with torch.inference_mode():
        model(
            input_ids=encoded.input_ids.to(device),
            attention_mask=encoded.attention_mask.to(device),
            use_cache=False,
            return_dict=True,
            logits_to_keep=1,
        )
    state = get_delta_mem_online_state(model)
    if not state:
        raise RuntimeError("Scene causal write produced no online state")
    return state


def generation_config(model, tokenizer):
    config = copy.deepcopy(model.generation_config)
    config.do_sample = False
    config.max_new_tokens = MAX_NEW_TOKENS
    config.use_cache = True
    config.temperature = None
    config.top_p = None
    config.top_k = None
    if tokenizer.pad_token_id is not None:
        config.pad_token_id = tokenizer.pad_token_id
    return config


def generate_read(
    model,
    tokenizer,
    target_messages,
    *,
    device: str,
) -> Mapping[str, Any]:
    encoded = encode_prompt(tokenizer, target_messages, generation=True)
    input_ids = encoded.input_ids.to(device)
    attention_mask = encoded.attention_mask.to(device)
    set_delta_write_enabled(model, False)
    if device.startswith("cuda"):
        torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()
    with torch.inference_mode():
        output_ids = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            generation_config=generation_config(model, tokenizer),
        )
    elapsed = time.perf_counter() - started
    generated_ids = output_ids[:, input_ids.size(1) :]
    raw = tokenizer.decode(generated_ids[0], skip_special_tokens=True).strip()
    parsed = recovery.extract_json(raw)
    recovered = recovery.recover_scene(parsed)
    return {
        "prediction": None if recovered is None else sorted(recovered),
        "raw_generation": raw,
        "input_tokens": int(input_ids.size(1)),
        "output_tokens": int(generated_ids.size(1)),
        "hit_max_new_tokens": int(generated_ids.size(1)) >= MAX_NEW_TOKENS,
        "elapsed_seconds": elapsed,
        "peak_cuda_memory_bytes": (
            int(torch.cuda.max_memory_allocated(device))
            if device.startswith("cuda")
            else None
        ),
    }


def generate_condition(
    model,
    tokenizer,
    row: Mapping[str, Any],
    *,
    condition: str,
    donor_row: Mapping[str, Any],
    donor_token_delta: int,
    module_names: Sequence[str],
    device: str,
    shard_index: int,
) -> Mapping[str, Any]:
    state_metadata: dict[str, Any]
    try:
        if condition == "zero_state":
            reset_delta_state(model)
            state_metadata = {"state_kind": "empty", "state_sha256": None}
        elif condition == "donor_state":
            state = prime_messages(
                model,
                tokenizer,
                donor_row["messages"],
                device=device,
            )
            state_metadata = {
                "state_kind": "different_gold_length_matched_donor",
                "state_sha256": tensor_digest(state),
                "donor_source_index": donor_row["source_index"],
                "donor_row_sha256": donor_row["row_sha256"],
                "absolute_write_token_delta": donor_token_delta,
            }
        elif condition == "layer_permuted_correct_state":
            correct_state = prime_messages(
                model,
                tokenizer,
                row["messages"],
                device=device,
            )
            permuted_state = permute_layer_state(correct_state, module_names)
            reset_delta_state(model)
            load_delta_mem_online_state(model, permuted_state)
            state_metadata = {
                "state_kind": "correct_state_complete_bundle_next_layer_cycle",
                "correct_state_sha256": tensor_digest(correct_state),
                "permuted_state_sha256": tensor_digest(permuted_state),
                "wrapped_layers": len(module_names),
            }
        else:
            raise ValueError(f"Unknown scene causal condition: {condition}")
        generated = generate_read(
            model,
            tokenizer,
            row["messages"],
            device=device,
        )
        return {
            "schema": SCHEMA,
            "protocol_payload_sha256": PROTOCOL_PAYLOAD_SHA256,
            "condition": condition,
            "shard_index": shard_index,
            "world_size": WORLD_SIZE,
            "source_index": row["source_index"],
            "row_sha256": row["row_sha256"],
            **state_metadata,
            **generated,
        }
    finally:
        reset_delta_state(model)
        set_delta_write_enabled(model, True)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-model", type=Path, required=True)
    parser.add_argument("--memory-dir", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--reference-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--shard-index", type=int, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--attn-implementation", default="sdpa")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    validate_protocol()
    if not 0 <= args.shard_index < WORLD_SIZE:
        raise ValueError("Scene causal study requires a valid four-way shard")
    base_model = args.base_model.expanduser().resolve(strict=True)
    memory_dir = args.memory_dir.expanduser().resolve(strict=True)
    dataset_root = args.dataset_root.expanduser().resolve(strict=True)
    reference_root = args.reference_root.expanduser().resolve(strict=True)
    output_root = args.output_root.expanduser().resolve()
    output_dir = output_root / f"shard-{args.shard_index}"

    rows = load_rows(dataset_root)
    tokenizer_for_mapping = AutoTokenizer.from_pretrained(
        base_model,
        local_files_only=True,
    )
    token_counts = write_prompt_token_counts(tokenizer_for_mapping, rows)
    donor_mapping = build_donor_mapping(rows, token_counts)
    donor_payload = donor_mapping_payload(rows, token_counts, donor_mapping)
    binding = input_binding(
        base_model=base_model,
        memory_dir=memory_dir,
        dataset_root=dataset_root,
        reference_root=reference_root,
        shard_index=args.shard_index,
        donor_payload=donor_payload,
        dtype=args.dtype,
        attn_implementation=args.attn_implementation,
    )
    write_or_validate_json(output_dir / "input_binding.json", binding)
    shard_rows = selected_rows(rows, shard_index=args.shard_index)
    by_index = {int(row["source_index"]): row for row in rows}
    existing_by_condition: dict[str, dict[int, Mapping[str, Any]]] = {}
    for condition in CONDITIONS:
        existing = read_completed(output_dir / f"{condition}.jsonl")
        validate_resume(
            existing,
            shard_rows,
            condition=condition,
            shard_index=args.shard_index,
        )
        existing_by_condition[condition] = existing
    if all(
        len(existing_by_condition[condition]) == len(shard_rows)
        for condition in CONDITIONS
    ):
        print(f"SCENE_CAUSAL_SHARD_COMPLETE shard={args.shard_index}", flush=True)
        return 0

    model, tokenizer = load_model_and_tokenizer(
        base_model=str(base_model),
        memory_dir=str(memory_dir),
        device=args.device,
        dtype=args.dtype,
        attn_implementation=args.attn_implementation,
    )
    module_names = tuple(name for name, _ in iter_delta_mem_modules(model))
    if len(module_names) != 42:
        raise ValueError(f"Expected 42 wrapped layers, found {len(module_names)}")
    for ordinal, row in enumerate(shard_rows, start=1):
        source = int(row["source_index"])
        donor_source = donor_mapping[source]
        donor_row = by_index[donor_source]
        donor_delta = abs(token_counts[source] - token_counts[donor_source])
        for condition in CONDITIONS:
            if source in existing_by_condition[condition]:
                continue
            record = generate_condition(
                model,
                tokenizer,
                row,
                condition=condition,
                donor_row=donor_row,
                donor_token_delta=donor_delta,
                module_names=module_names,
                device=args.device,
                shard_index=args.shard_index,
            )
            append_record(output_dir / f"{condition}.jsonl", record)
            print(
                f"SCENE_CAUSAL_PROGRESS shard={args.shard_index} "
                f"condition={condition} row={source} ordinal={ordinal}/{len(shard_rows)}",
                flush=True,
            )
    print(f"SCENE_CAUSAL_SHARD_COMPLETE shard={args.shard_index}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
