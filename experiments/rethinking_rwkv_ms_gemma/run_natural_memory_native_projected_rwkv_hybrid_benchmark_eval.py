#!/usr/bin/env python3
"""Evaluate one authorized shard of a trained projected/RWKV hybrid arm."""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
import sys
import time
from typing import Any, Iterable, Mapping, Sequence

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

import torch
from transformers import AutoTokenizer


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = Path(__file__).resolve().parents[2]
for import_root in (SCRIPT_DIR, PROJECT_ROOT):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

from common import load_model_and_tokenizer, reset_delta_state, set_delta_write_enabled  # noqa: E402
from deltamem.core.delta import (  # noqa: E402
    get_delta_mem_online_state,
    iter_delta_mem_modules,
    load_delta_mem_online_state,
)
from experiments.rethinking_rwkv_ms_gemma import (  # noqa: E402
    analyze_novel_agent_eval as recovery,
)
from experiments.rethinking_rwkv_ms_gemma import (  # noqa: E402
    run_natural_memory_native_projected_rwkv_hybrid_benchmark_train as training,
)
from experiments.rethinking_rwkv_ms_gemma import (  # noqa: E402
    run_natural_memory_native_scene_causal as causal,
)


SCHEMA = "rwkv_ms_natural_memory_native_projected_rwkv_hybrid_benchmark_eval.v1"
INPUT_SCHEMA = (
    "rwkv_ms_natural_memory_native_projected_rwkv_hybrid_benchmark_eval_input.v1"
)
PROTOCOL_PAYLOAD_SHA256 = training.PROTOCOL_PAYLOAD_SHA256
HF_MIRROR_ENDPOINT = training.HF_MIRROR_ENDPOINT
WORLD_SIZE = training.WORLD_SIZE
ARCHITECTURES = training.ARCHITECTURES
SEEDS = training.SEEDS
EVALUATION_ROWS = 220
EXCLUDED_INITIAL_ROWS = 4
PARTITION_NAMESPACE = "rwkv-ms-scale-v1:"
PROBE_SALT = "rwkv-ms-contrast-probe-v1:"
PROBE_ROWS = 64
AUTHORIZED_ROWS_PAYLOAD_SHA256 = (
    "0493e75da858d4ddebba580cc7b5aaaa32249527e5e44502e6ff06591cd82d09"
)
PROBE_PAYLOAD_SHA256 = (
    "5c8a10f1e373ec6661481caf79bae340fdf5a92ab6afa9cf04e22f6bda254994"
)
DATASET_RELATIVE_PATH = "v4-scene-boundary-detection/train_derived_development.jsonl"
DATASET_SHA256 = "b383625cee07e6a7565142e38bb0b0a4d4a2468b2c91171570115b7b311e1e68"
CONDITIONS = {
    "projected_control": ("correct_state",),
    "hybrid_candidate": (
        "correct_recurrent_state",
        "zero_recurrent_state",
        "matched_donor_recurrent_state",
        "layer_permuted_recurrent_state",
        "projected_only_bypass",
    ),
}
RECURRENT_SUFFIXES = (
    "",
    ".__rwkv_ms_positions",
    ".__rwkv_ms_previous_source",
)
PROJECTED_SUFFIXES = (
    ".__projected_kv_keys",
    ".__projected_kv_values",
    ".__projected_kv_occupied",
    ".__projected_kv_surprise",
)


def canonical_sha256(value: Any) -> str:
    return training.canonical_sha256(value)


def sha256_file(path: Path) -> str:
    return training.sha256_file(path)


def tensor_digest(state: Mapping[str, torch.Tensor]) -> str:
    return causal.tensor_digest(state)


def write_or_validate_json(path: Path, value: Mapping[str, Any]) -> None:
    if path.is_file():
        if json.loads(path.read_text(encoding="utf-8")) != value:
            raise ValueError(f"Evaluation binding differs: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=True, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def append_jsonl(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(value, ensure_ascii=True, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def raw_line_metadata(path: Path) -> tuple[dict[str, Any], ...]:
    """Hash every line before parsing any row content."""
    raw_lines = path.read_bytes().splitlines()
    if len(raw_lines) != 361:
        raise ValueError(f"Expected 361 raw development lines, found {len(raw_lines)}")
    return tuple(
        {
            "source_index": index,
            "row_sha256": hashlib.sha256(raw_line).hexdigest(),
            "raw_line": raw_line,
        }
        for index, raw_line in enumerate(raw_lines)
    )


def strength_partition(row_sha256: str) -> str:
    digest = hashlib.sha256(
        f"{PARTITION_NAMESPACE}{row_sha256}".encode("ascii")
    ).hexdigest()
    return "holdout" if int(digest[:8], 16) % 5 == 0 else "fit"


def authorized_metadata(
    metadata: Sequence[Mapping[str, Any]],
) -> tuple[dict[str, Any], ...]:
    fit = [
        row
        for row in metadata
        if int(row["source_index"]) >= EXCLUDED_INITIAL_ROWS
        and strength_partition(str(row["row_sha256"])) == "fit"
    ]
    if len(fit) != 284:
        raise ValueError("Authorized fit partition count differs")
    probe = sorted(
        fit,
        key=lambda row: (
            hashlib.sha256(
                f"{PROBE_SALT}{row['row_sha256']}".encode("ascii")
            ).hexdigest(),
            int(row["source_index"]),
        ),
    )[:PROBE_ROWS]
    probe_payload = [
        {
            "source_index": int(row["source_index"]),
            "row_sha256": str(row["row_sha256"]),
        }
        for row in probe
    ]
    if canonical_sha256(probe_payload) != PROBE_PAYLOAD_SHA256:
        raise ValueError("Locked contrast probe selection differs")
    probe_keys = {
        (int(row["source_index"]), str(row["row_sha256"])) for row in probe
    }
    selected = tuple(
        row
        for row in fit
        if (int(row["source_index"]), str(row["row_sha256"])) not in probe_keys
    )
    payload = [
        {
            "source_index": int(row["source_index"]),
            "row_sha256": str(row["row_sha256"]),
        }
        for row in selected
    ]
    if len(selected) != EVALUATION_ROWS:
        raise ValueError("Authorized native evaluation row count differs")
    if canonical_sha256(payload) != AUTHORIZED_ROWS_PAYLOAD_SHA256:
        raise ValueError("Authorized native evaluation row payload differs")
    return tuple(dict(row) for row in selected)


def parse_authorized_rows(
    metadata: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    selected = authorized_metadata(metadata)
    rows: list[dict[str, Any]] = []
    for record in selected:
        value = json.loads(bytes(record["raw_line"]).decode("utf-8"))
        messages = value.get("messages")
        if not isinstance(messages, list) or len(messages) < 3:
            raise ValueError(f"Invalid authorized scene row: {record['source_index']}")
        gold = recovery.extract_json(str(messages[-1].get("content", "")))
        if not isinstance(gold, Mapping):
            raise ValueError(f"Invalid authorized scene gold: {record['source_index']}")
        rows.append(
            {
                "source_index": int(record["source_index"]),
                "row_sha256": str(record["row_sha256"]),
                "messages": messages[:-1],
                "gold": dict(gold),
            }
        )
    return rows


def encode_prompt(tokenizer, messages: Sequence[Mapping[str, str]], *, generation: bool):
    return causal.encode_prompt(tokenizer, messages, generation=generation)


def prompt_token_counts(tokenizer, rows: Sequence[Mapping[str, Any]]) -> dict[int, int]:
    return {
        int(row["source_index"]): int(
            encode_prompt(tokenizer, row["messages"], generation=False)
            .input_ids.size(1)
        )
        for row in rows
    }


def build_donor_mapping(
    rows: Sequence[Mapping[str, Any]],
    token_counts: Mapping[int, int],
) -> dict[int, int]:
    gold = {
        int(row["source_index"]): recovery.strict_gold_boundaries(row["gold"])
        for row in rows
    }
    mapping: dict[int, int] = {}
    for row in rows:
        source = int(row["source_index"])
        candidates = [
            int(candidate["source_index"])
            for candidate in rows
            if int(candidate["source_index"]) != source
            and gold[int(candidate["source_index"])] != gold[source]
        ]
        if not candidates:
            raise ValueError(f"Authorized row has no different-gold donor: {source}")
        mapping[source] = min(
            candidates,
            key=lambda donor: (
                abs(token_counts[source] - token_counts[donor]),
                str(next(item for item in rows if int(item["source_index"]) == donor)["row_sha256"]),
                donor,
            ),
        )
    return mapping


def donor_mapping_payload(
    rows: Sequence[Mapping[str, Any]],
    token_counts: Mapping[int, int],
    mapping: Mapping[int, int],
) -> list[dict[str, Any]]:
    by_index = {int(row["source_index"]): row for row in rows}
    return [
        {
            "source_index": source,
            "source_row_sha256": by_index[source]["row_sha256"],
            "source_prompt_tokens": token_counts[source],
            "donor_source_index": donor,
            "donor_row_sha256": by_index[donor]["row_sha256"],
            "donor_prompt_tokens": token_counts[donor],
            "absolute_prompt_token_delta": abs(
                token_counts[source] - token_counts[donor]
            ),
        }
        for source, donor in sorted(mapping.items())
    ]


def validate_train_result(
    path: Path,
    *,
    architecture: str,
    seed: int,
    adapter_dir: Path,
) -> Mapping[str, Any]:
    result = json.loads(path.read_text(encoding="utf-8"))
    receipt = result.get("receipt")
    if not isinstance(receipt, Mapping):
        raise ValueError("Training result receipt is missing")
    unsigned = dict(result)
    unsigned.pop("receipt")
    if canonical_sha256(unsigned) != receipt.get("payload_sha256"):
        raise ValueError("Training result receipt differs")
    required = {
        "schema": training.SCHEMA,
        "status": "training_complete_evaluation_pending",
        "passed": True,
        "protocol_payload_sha256": PROTOCOL_PAYLOAD_SHA256,
        "architecture": architecture,
        "seed": seed,
        "updates": training.TRAIN_UPDATES,
        "protected_splits_opened": [],
    }
    if any(result.get(key) != value for key, value in required.items()):
        raise ValueError("Training result does not authorize evaluation")
    expected_files = result.get("adapter_files")
    if not isinstance(expected_files, Mapping):
        raise ValueError("Training result adapter manifest is missing")
    for filename, metadata in expected_files.items():
        file_path = adapter_dir / str(filename)
        if not file_path.is_file() or sha256_file(file_path) != metadata.get("sha256"):
            raise ValueError(f"Trained adapter file differs: {file_path}")
    training_binding = result.get("code_bindings", {})
    if training_binding.get("runner_sha256") != sha256_file(
        SCRIPT_DIR / "run_natural_memory_native_projected_rwkv_hybrid_benchmark_train.py"
    ):
        raise ValueError("Training runner binding differs")
    return result


def split_state(
    state: Mapping[str, torch.Tensor],
    module_names: Sequence[str],
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    expected = {
        f"{name}{suffix}"
        for name in module_names
        for suffix in (*RECURRENT_SUFFIXES, *PROJECTED_SUFFIXES)
    }
    if set(state) != expected:
        missing = sorted(expected - set(state))
        extra = sorted(set(state) - expected)
        raise ValueError(
            f"Hybrid online state bundle differs: missing={missing[:3]} extra={extra[:3]}"
        )
    recurrent = {
        key: state[key].detach().cpu().clone()
        for name in module_names
        for key in (f"{name}{suffix}" for suffix in RECURRENT_SUFFIXES)
    }
    projected = {
        key: state[key].detach().cpu().clone()
        for name in module_names
        for key in (f"{name}{suffix}" for suffix in PROJECTED_SUFFIXES)
    }
    return recurrent, projected


def merge_state(
    recurrent: Mapping[str, torch.Tensor],
    projected: Mapping[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    return {
        **{
            key: value.detach().cpu().clone()
            for key, value in recurrent.items()
        },
        **{
            key: value.detach().cpu().clone()
            for key, value in projected.items()
        },
    }


def zero_recurrent_state(
    recurrent: Mapping[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    return {key: torch.zeros_like(value) for key, value in recurrent.items()}


def permute_recurrent_state(
    recurrent: Mapping[str, torch.Tensor],
    module_names: Sequence[str],
) -> dict[str, torch.Tensor]:
    ordered = tuple(
        sorted(
            module_names,
            key=lambda name: int(name.split(".layers.", 1)[1].split(".", 1)[0]),
        )
    )
    if len(ordered) != 42 or len(set(ordered)) != 42:
        raise ValueError("Layer permutation requires 42 unique modules")
    permuted: dict[str, torch.Tensor] = {}
    for index, target in enumerate(ordered):
        source = ordered[(index + 1) % len(ordered)]
        for suffix in RECURRENT_SUFFIXES:
            target_key = f"{target}{suffix}"
            source_key = f"{source}{suffix}"
            if recurrent[target_key].shape != recurrent[source_key].shape:
                raise ValueError("Recurrent layer state shapes differ")
            permuted[target_key] = recurrent[source_key].detach().cpu().clone()
    if tensor_digest(permuted) == tensor_digest(recurrent):
        raise ValueError("Layer permutation did not change recurrent state")
    return permuted


@contextmanager
def projected_only_mode(model):
    modules = tuple(iter_delta_mem_modules(model))
    saved = [(module, module.memory_readout_mode) for _, module in modules]
    for module, _ in saved:
        module.memory_readout_mode = "projected_kv_slots"
    try:
        yield
    finally:
        for module, mode in saved:
            module.memory_readout_mode = mode


def generate_from_state(
    model,
    tokenizer,
    row: Mapping[str, Any],
    state: Mapping[str, torch.Tensor],
    *,
    device: str,
    bypass_projected: bool = False,
) -> Mapping[str, Any]:
    reset_delta_state(model)
    load_delta_mem_online_state(model, state)
    set_delta_write_enabled(model, False)
    try:
        if bypass_projected:
            with projected_only_mode(model):
                return causal.generate_read(
                    model, tokenizer, row["messages"], device=device
                )
        return causal.generate_read(model, tokenizer, row["messages"], device=device)
    finally:
        reset_delta_state(model)
        set_delta_write_enabled(model, True)


def prime_state(model, tokenizer, row: Mapping[str, Any], *, device: str):
    return causal.prime_messages(model, tokenizer, row["messages"], device=device)


def record_score(
    prediction: Sequence[int] | None,
    gold: set[int],
) -> Mapping[str, Any]:
    predicted = set() if prediction is None else {int(value) for value in prediction}
    tp = len(predicted & gold)
    fp = len(predicted - gold)
    fn = len(gold - predicted)
    return {
        "covered": prediction is not None,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "micro_f1": 0.0 if 2 * tp + fp + fn == 0 else 2 * tp / (2 * tp + fp + fn),
    }


def generate_row_conditions(
    model,
    tokenizer,
    row: Mapping[str, Any],
    donor: Mapping[str, Any],
    *,
    architecture: str,
    module_names: Sequence[str],
    device: str,
    shard_index: int,
) -> list[Mapping[str, Any]]:
    correct_state = prime_state(model, tokenizer, row, device=device)
    correct_recurrent, correct_projected = split_state(correct_state, module_names)
    correct_state_digest = tensor_digest(correct_state)
    projected_digest = tensor_digest(correct_projected)
    gold = recovery.strict_gold_boundaries(row["gold"])
    if architecture == "projected_control":
        generated = generate_from_state(
            model, tokenizer, row, correct_state, device=device
        )
        return [
            {
                "schema": SCHEMA,
                "protocol_payload_sha256": PROTOCOL_PAYLOAD_SHA256,
                "architecture": architecture,
                "condition": "correct_state",
                "shard_index": shard_index,
                "world_size": WORLD_SIZE,
                "source_index": row["source_index"],
                "row_sha256": row["row_sha256"],
                "gold": sorted(gold),
                **generated,
                "score": record_score(generated["prediction"], gold),
                "state_sha256": correct_state_digest,
                "projected_carrier_sha256": projected_digest,
            }
        ]

    donor_state = prime_state(model, tokenizer, donor, device=device)
    donor_recurrent, donor_projected = split_state(donor_state, module_names)
    if tensor_digest(donor_projected) == projected_digest:
        raise RuntimeError("Different donor unexpectedly produced identical projected carrier")
    condition_states = {
        "correct_recurrent_state": merge_state(correct_recurrent, correct_projected),
        "zero_recurrent_state": merge_state(
            zero_recurrent_state(correct_recurrent), correct_projected
        ),
        "matched_donor_recurrent_state": merge_state(donor_recurrent, correct_projected),
        "layer_permuted_recurrent_state": merge_state(
            permute_recurrent_state(correct_recurrent, module_names), correct_projected
        ),
    }
    records: list[Mapping[str, Any]] = []
    for condition in CONDITIONS["hybrid_candidate"]:
        state = (
            correct_state
            if condition == "correct_recurrent_state"
            else condition_states[condition]
            if condition != "projected_only_bypass"
            else condition_states["correct_recurrent_state"]
        )
        generated = generate_from_state(
            model,
            tokenizer,
            row,
            state,
            device=device,
            bypass_projected=condition == "projected_only_bypass",
        )
        recurrent, projected = split_state(state, module_names)
        if tensor_digest(projected) != projected_digest:
            raise RuntimeError("Hybrid projected carrier changed across an intervention")
        records.append(
            {
                "schema": SCHEMA,
                "protocol_payload_sha256": PROTOCOL_PAYLOAD_SHA256,
                "architecture": architecture,
                "condition": condition,
                "shard_index": shard_index,
                "world_size": WORLD_SIZE,
                "source_index": row["source_index"],
                "row_sha256": row["row_sha256"],
                "gold": sorted(gold),
                **generated,
                "score": record_score(generated["prediction"], gold),
                "state_sha256": tensor_digest(state),
                "correct_state_sha256": correct_state_digest,
                "correct_recurrent_sha256": tensor_digest(correct_recurrent),
                "donor_recurrent_sha256": tensor_digest(donor_recurrent),
                "condition_recurrent_sha256": tensor_digest(recurrent),
                "projected_carrier_sha256": tensor_digest(projected),
                "projected_carrier_byte_identical": True,
                "donor_source_index": donor["source_index"],
                "donor_row_sha256": donor["row_sha256"],
            }
        )
    by_condition = {record["condition"]: record for record in records}
    if (
        by_condition["zero_recurrent_state"]["prediction"]
        != by_condition["projected_only_bypass"]["prediction"]
    ):
        raise RuntimeError("Zero recurrent and projected bypass predictions differ")
    return records


def read_completed(path: Path) -> dict[int, Mapping[str, Any]]:
    if not path.is_file():
        return {}
    records: dict[int, Mapping[str, Any]] = {}
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        if not raw_line.strip():
            continue
        record = json.loads(raw_line)
        source = int(record["source_index"])
        if source in records:
            raise ValueError(f"Duplicate evaluation record: {path}:{source}")
        records[source] = record
    return records


def validate_resume(
    records: Mapping[int, Mapping[str, Any]],
    expected: Mapping[int, Mapping[str, Any]],
    *,
    architecture: str,
    condition: str,
    shard_index: int,
) -> None:
    if not set(records) <= set(expected):
        raise ValueError(f"Unexpected resumed rows for {condition}")
    for source, record in records.items():
        row = expected[source]
        required = {
            "schema": SCHEMA,
            "protocol_payload_sha256": PROTOCOL_PAYLOAD_SHA256,
            "architecture": architecture,
            "condition": condition,
            "shard_index": shard_index,
            "world_size": WORLD_SIZE,
            "source_index": source,
            "row_sha256": row["row_sha256"],
        }
        if any(record.get(key) != value for key, value in required.items()):
            raise ValueError(f"Resumed evaluation record differs: {condition}:{source}")


def input_binding(
    *,
    architecture: str,
    seed: int,
    shard_index: int,
    base_model: Path,
    dataset_root: Path,
    train_result: Path,
    training_result: Mapping[str, Any],
    adapter_dir: Path,
    donor_payload: Sequence[Mapping[str, Any]],
    module_names: Sequence[str] | None,
) -> Mapping[str, Any]:
    return {
        "schema": INPUT_SCHEMA,
        "protocol_payload_sha256": PROTOCOL_PAYLOAD_SHA256,
        "architecture": architecture,
        "seed": seed,
        "shard_index": shard_index,
        "world_size": WORLD_SIZE,
        "base_model": str(base_model),
        "base_config_sha256": training.recurrent_preflight.EXPECTED_BASE_CONFIG_SHA256,
        "dataset_root": str(dataset_root),
        "dataset_file": str(dataset_root / DATASET_RELATIVE_PATH),
        "dataset_sha256": DATASET_SHA256,
        "authorized_rows": EVALUATION_ROWS,
        "authorized_rows_payload_sha256": AUTHORIZED_ROWS_PAYLOAD_SHA256,
        "train_result": str(train_result),
        "train_result_file_sha256": sha256_file(train_result),
        "train_result_receipt": training_result["receipt"]["payload_sha256"],
        "adapter_dir": str(adapter_dir),
        "adapter_files": training_result["adapter_files"],
        "donor_mapping_payload_sha256": canonical_sha256(list(donor_payload)),
        "donor_mapping_rows": len(donor_payload),
        "conditions": list(CONDITIONS[architecture]),
        "module_names_sha256": (
            None if module_names is None else canonical_sha256(list(module_names))
        ),
        "hf_endpoint": os.environ.get("HF_ENDPOINT"),
        "runner_sha256": sha256_file(Path(__file__)),
        "generation_runner_sha256": sha256_file(
            SCRIPT_DIR / "run_natural_memory_native_scene_causal.py"
        ),
    }


def run(args: argparse.Namespace) -> int:
    if os.environ.get("HF_ENDPOINT") != HF_MIRROR_ENDPOINT:
        raise ValueError(f"HF_ENDPOINT must be exactly {HF_MIRROR_ENDPOINT}")
    if args.architecture not in ARCHITECTURES or args.seed not in SEEDS:
        raise ValueError("Evaluation architecture or seed is not locked")
    if not 0 <= args.shard_index < WORLD_SIZE:
        raise ValueError("Evaluation shard index is outside the four-way split")
    base_model = args.base_model.expanduser().resolve(strict=True)
    dataset_root = args.dataset_root.expanduser().resolve(strict=True)
    train_result = args.train_result.expanduser().resolve(strict=True)
    adapter_dir = args.adapter_dir.expanduser().resolve(strict=True)
    training_result = validate_train_result(
        train_result,
        architecture=args.architecture,
        seed=args.seed,
        adapter_dir=adapter_dir,
    )
    dataset_file = dataset_root / DATASET_RELATIVE_PATH
    if sha256_file(dataset_file) != DATASET_SHA256:
        raise ValueError("Authorized native evaluation dataset hash differs")
    metadata = raw_line_metadata(dataset_file)
    rows = parse_authorized_rows(metadata)
    by_index = {int(row["source_index"]): row for row in rows}
    tokenizer = AutoTokenizer.from_pretrained(base_model, local_files_only=True)
    counts = prompt_token_counts(tokenizer, rows)
    donor_mapping = build_donor_mapping(rows, counts)
    donor_payload = donor_mapping_payload(rows, counts, donor_mapping)
    shard_rows = [
        row for row in rows if int(row["source_index"]) % WORLD_SIZE == args.shard_index
    ]
    output_dir = (
        args.output_root.expanduser().resolve()
        / args.architecture
        / f"seed-{args.seed}"
        / f"shard-{args.shard_index}"
    )
    model, loaded_tokenizer = load_model_and_tokenizer(
        base_model=str(base_model),
        memory_dir=str(adapter_dir),
        device=args.device,
        dtype="bfloat16",
        attn_implementation="sdpa",
    )
    module_names = tuple(name for name, _ in iter_delta_mem_modules(model))
    if len(module_names) != 42:
        raise ValueError(f"Expected 42 wrapped layers, found {len(module_names)}")
    expected_binding = input_binding(
        architecture=args.architecture,
        seed=args.seed,
        shard_index=args.shard_index,
        base_model=base_model,
        dataset_root=dataset_root,
        train_result=train_result,
        training_result=training_result,
        adapter_dir=adapter_dir,
        donor_payload=donor_payload,
        module_names=module_names,
    )
    write_or_validate_json(output_dir / "input_binding.json", expected_binding)
    existing = {
        condition: read_completed(output_dir / f"{condition}.jsonl")
        for condition in CONDITIONS[args.architecture]
    }
    for condition, records in existing.items():
        validate_resume(
            records,
            {int(row["source_index"]): row for row in shard_rows},
            architecture=args.architecture,
            condition=condition,
            shard_index=args.shard_index,
        )
    if all(
        len(existing[condition]) == len(shard_rows)
        for condition in CONDITIONS[args.architecture]
    ):
        print(
            f"PROJECTED_RWKV_EVAL_SHARD_COMPLETE architecture={args.architecture} "
            f"seed={args.seed} shard={args.shard_index}",
            flush=True,
        )
        return 0
    started = time.time()
    for ordinal, row in enumerate(shard_rows, start=1):
        source = int(row["source_index"])
        donor = by_index[donor_mapping[source]]
        records = generate_row_conditions(
            model,
            loaded_tokenizer,
            row,
            donor,
            architecture=args.architecture,
            module_names=module_names,
            device=args.device,
            shard_index=args.shard_index,
        )
        for record in records:
            if source in existing[record["condition"]]:
                continue
            append_jsonl(output_dir / f"{record['condition']}.jsonl", record)
        print(
            f"PROJECTED_RWKV_EVAL_PROGRESS architecture={args.architecture} "
            f"seed={args.seed} shard={args.shard_index} row={source} "
            f"ordinal={ordinal}/{len(shard_rows)} elapsed={time.time()-started:.1f}",
            flush=True,
        )
    print(
        f"PROJECTED_RWKV_EVAL_SHARD_COMPLETE architecture={args.architecture} "
        f"seed={args.seed} shard={args.shard_index}",
        flush=True,
    )
    return 0


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--architecture", choices=ARCHITECTURES, required=True)
    parser.add_argument("--seed", type=int, choices=SEEDS, required=True)
    parser.add_argument("--shard-index", type=int, required=True)
    parser.add_argument("--base-model", type=Path, default=training.BASE_MODEL)
    parser.add_argument("--dataset-root", type=Path, default=training.DATASET_ROOT)
    parser.add_argument("--train-result", type=Path, required=True)
    parser.add_argument("--adapter-dir", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    return parser.parse_args(argv)


if __name__ == "__main__":
    raise SystemExit(run(parse_args()))
