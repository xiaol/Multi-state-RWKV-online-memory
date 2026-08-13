#!/usr/bin/env python3
"""Run one append-only shard of the native scene state-retrieval study."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
import sys
from typing import Any, Mapping, Sequence


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = Path(__file__).resolve().parents[2]
for import_root in (SCRIPT_DIR, PROJECT_ROOT):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

from common import load_model_and_tokenizer, reset_delta_state, set_delta_write_enabled  # noqa: E402
from experiments.rethinking_rwkv_ms_gemma import (  # noqa: E402
    prepare_natural_memory_native_scene_state_retrieval as mapping_builder,
)
from experiments.rethinking_rwkv_ms_gemma import (  # noqa: E402
    run_natural_memory_native_scene_causal as causal_runner,
)


SCHEMA = "rwkv_ms_natural_memory_native_scene_state_retrieval_shard.v1"
INPUT_SCHEMA = "rwkv_ms_natural_memory_native_scene_state_retrieval_input.v1"
SELECTION_SCHEMA = "rwkv_ms_natural_memory_native_scene_state_retrieval_selection.v1"
MAPPING = SCRIPT_DIR / "natural_memory_native_scene_state_retrieval_mapping_v1.json"
MAPPING_FILE_SHA256 = "87911f559e83bb8dd5cc29adfd5d799eb7b77ee4595aab8993f0437bec78b5d4"
MAPPING_PAYLOAD_SHA256 = "cfcb94dffa30199ef334cdc8f6241aaa547f9d7d9287bacd7ffb45dfa75c9902"
MEMORY_ADAPTER_SHA256 = causal_runner.MEMORY_ADAPTER_SHA256
BASE_CONFIG_SHA256 = causal_runner.BASE_CONFIG_SHA256
WORLD_SIZE = 4
PHASES = ("fit", "holdout")


def canonical_sha256(value: Any) -> str:
    return mapping_builder.canonical_sha256(value)


def sha256_file(path: Path) -> str:
    return mapping_builder.sha256_file(path)


def load_mapping() -> Mapping[str, Any]:
    if sha256_file(MAPPING) != MAPPING_FILE_SHA256:
        raise ValueError("State-retrieval mapping file hash differs")
    value = json.loads(MAPPING.read_text(encoding="utf-8"))
    receipt = value.get("receipt")
    if not isinstance(receipt, Mapping):
        raise ValueError("State-retrieval mapping receipt is missing")
    unsigned = dict(value)
    unsigned.pop("receipt")
    digest = canonical_sha256(unsigned)
    if digest != MAPPING_PAYLOAD_SHA256 or receipt.get("payload_sha256") != digest:
        raise ValueError("State-retrieval mapping payload differs")
    if tuple(value.get("candidate_methods", ())) != mapping_builder.CANDIDATE_METHODS:
        raise ValueError("State-retrieval mapping methods differ")
    return value


def validate_selection(path: Path, *, runner_sha256: str) -> Mapping[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    receipt = value.get("receipt")
    if not isinstance(receipt, Mapping):
        raise ValueError("State-retrieval selection receipt is missing")
    unsigned = dict(value)
    unsigned.pop("receipt")
    if canonical_sha256(unsigned) != receipt.get("payload_sha256"):
        raise ValueError("State-retrieval selection receipt differs")
    required = {
        "schema": SELECTION_SCHEMA,
        "protocol_payload_sha256": mapping_builder.PROTOCOL_PAYLOAD_SHA256,
        "amendment_payload_sha256": mapping_builder.AMENDMENT_PAYLOAD_SHA256,
        "mapping_payload_sha256": MAPPING_PAYLOAD_SHA256,
        "runner_sha256": runner_sha256,
        "phase_one_passed": True,
        "holdout_authorized": True,
    }
    if any(value.get(key) != expected for key, expected in required.items()):
        raise ValueError("State-retrieval selection binding differs")
    selected = value.get("selected_method")
    if selected not in mapping_builder.CANDIDATE_METHODS:
        raise ValueError("State-retrieval selected method differs")
    analyzer_sha256 = value.get("analyzer_sha256")
    if not isinstance(analyzer_sha256, str) or re.fullmatch(r"[0-9a-f]{64}", analyzer_sha256) is None:
        raise ValueError("State-retrieval analyzer hash is invalid")
    analyzer_path = SCRIPT_DIR / "analyze_natural_memory_native_scene_state_retrieval.py"
    if sha256_file(analyzer_path) != analyzer_sha256:
        raise ValueError("State-retrieval analyzer file hash differs")
    return value


def selected_target_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    phase: str,
    shard_index: int,
) -> list[Mapping[str, Any]]:
    return [
        row
        for row in rows
        if int(row["source_index"]) >= mapping_builder.EXCLUDED_TARGET_ROWS
        and mapping_builder.partition_for_hash(str(row["row_sha256"])) == phase
        and int(row["source_index"]) % WORLD_SIZE == shard_index
    ]


def read_completed(path: Path) -> dict[int, Mapping[str, Any]]:
    if not path.is_file():
        return {}
    records: dict[int, Mapping[str, Any]] = {}
    with path.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            if raw_line.strip():
                record = json.loads(raw_line)
                source_index = int(record["target_source_index"])
                if source_index in records:
                    raise ValueError(f"Duplicate state-retrieval output: {path}:{source_index}")
                records[source_index] = record
    return records


def append_record(path: Path, record: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def write_or_validate_json(path: Path, value: Mapping[str, Any]) -> None:
    if path.is_file():
        if json.loads(path.read_text(encoding="utf-8")) != value:
            raise ValueError(f"State-retrieval binding differs: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def input_binding(
    *,
    base_model: Path,
    memory_dir: Path,
    dataset_root: Path,
    phase: str,
    shard_index: int,
    methods: Sequence[str],
    selection: Mapping[str, Any] | None,
    selection_path: Path | None,
    dtype: str,
    attn_implementation: str,
) -> Mapping[str, Any]:
    if sha256_file(base_model / "config.json") != BASE_CONFIG_SHA256:
        raise ValueError("State-retrieval base config differs")
    if sha256_file(memory_dir / "delta_mem_adapter.pt") != MEMORY_ADAPTER_SHA256:
        raise ValueError("State-retrieval memory adapter differs")
    return {
        "schema": INPUT_SCHEMA,
        "protocol_payload_sha256": mapping_builder.PROTOCOL_PAYLOAD_SHA256,
        "amendment_payload_sha256": mapping_builder.AMENDMENT_PAYLOAD_SHA256,
        "mapping_file": str(MAPPING.resolve()),
        "mapping_file_sha256": MAPPING_FILE_SHA256,
        "mapping_payload_sha256": MAPPING_PAYLOAD_SHA256,
        "base_model": str(base_model),
        "base_config_sha256": BASE_CONFIG_SHA256,
        "memory_dir": str(memory_dir),
        "memory_adapter_sha256": MEMORY_ADAPTER_SHA256,
        "memory_config_sha256": sha256_file(memory_dir / "delta_mem_config.json"),
        "dataset_root": str(dataset_root),
        "target_file_sha256": mapping_builder.TARGET_SHA256,
        "bank_file_sha256": mapping_builder.BANK_SHA256,
        "phase": phase,
        "shard_index": shard_index,
        "world_size": WORLD_SIZE,
        "methods": list(methods),
        "selection_path": None if selection_path is None else str(selection_path),
        "selection_file_sha256": (
            None if selection_path is None else sha256_file(selection_path)
        ),
        "selection_payload_sha256": (
            None if selection is None else selection["receipt"]["payload_sha256"]
        ),
        "dtype": dtype,
        "attn_implementation": attn_implementation,
        "runner_sha256": sha256_file(Path(__file__)),
        "generation_runner_sha256": sha256_file(
            SCRIPT_DIR / "run_natural_memory_native_scene_causal.py"
        ),
    }


def validate_resume(
    existing: Mapping[int, Mapping[str, Any]],
    rows: Sequence[Mapping[str, Any]],
    *,
    phase: str,
    method: str,
    shard_index: int,
) -> None:
    expected = {int(row["source_index"]): row for row in rows}
    if not set(existing) <= set(expected):
        raise ValueError(f"Unexpected state-retrieval resumed indices: {phase}:{method}")
    for source_index, record in existing.items():
        required = {
            "schema": SCHEMA,
            "protocol_payload_sha256": mapping_builder.PROTOCOL_PAYLOAD_SHA256,
            "amendment_payload_sha256": mapping_builder.AMENDMENT_PAYLOAD_SHA256,
            "mapping_payload_sha256": MAPPING_PAYLOAD_SHA256,
            "phase": phase,
            "method": method,
            "shard_index": shard_index,
            "world_size": WORLD_SIZE,
            "target_source_index": source_index,
            "target_row_sha256": expected[source_index]["row_sha256"],
        }
        if any(record.get(key) != value for key, value in required.items()):
            raise ValueError(f"State-retrieval resumed record differs: {phase}:{method}:{source_index}")


def generate_record(
    model,
    tokenizer,
    target_row: Mapping[str, Any],
    bank_row: Mapping[str, Any],
    mapping_method: Mapping[str, Any],
    *,
    phase: str,
    method: str,
    shard_index: int,
    device: str,
) -> Mapping[str, Any]:
    try:
        state = causal_runner.prime_messages(
            model,
            tokenizer,
            bank_row["messages"],
            device=device,
        )
        state_sha256 = causal_runner.tensor_digest(state)
        generated = causal_runner.generate_read(
            model,
            tokenizer,
            target_row["messages"],
            device=device,
        )
        return {
            "schema": SCHEMA,
            "protocol_payload_sha256": mapping_builder.PROTOCOL_PAYLOAD_SHA256,
            "amendment_payload_sha256": mapping_builder.AMENDMENT_PAYLOAD_SHA256,
            "mapping_payload_sha256": MAPPING_PAYLOAD_SHA256,
            "phase": phase,
            "method": method,
            "shard_index": shard_index,
            "world_size": WORLD_SIZE,
            "target_source_index": target_row["source_index"],
            "target_row_sha256": target_row["row_sha256"],
            "bank_source_index": bank_row["source_index"],
            "bank_row_sha256": bank_row["row_sha256"],
            "bank_state_sha256": state_sha256,
            "target_write_tokens": mapping_method["target_write_tokens"],
            "bank_write_tokens": mapping_method["bank_write_tokens"],
            "absolute_write_token_delta": mapping_method["absolute_write_token_delta"],
            "char_tfidf_cosine": mapping_method["char_tfidf_cosine"],
            "selection_score": mapping_method["selection_score"],
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
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--phase", choices=PHASES, required=True)
    parser.add_argument("--selection", type=Path)
    parser.add_argument("--shard-index", type=int, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--attn-implementation", default="sdpa")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    mapping_builder.validate_protocol()
    mapping = load_mapping()
    if not 0 <= args.shard_index < WORLD_SIZE:
        raise ValueError("State-retrieval study requires a valid four-way shard")
    runner_sha256 = sha256_file(Path(__file__))
    selection: Mapping[str, Any] | None = None
    selection_path: Path | None = None
    if args.phase == "fit":
        if args.selection is not None:
            raise ValueError("State-retrieval fit phase forbids a selection file")
        methods = mapping_builder.CANDIDATE_METHODS
    else:
        if args.selection is None:
            raise ValueError("State-retrieval holdout phase requires a signed selection")
        selection_path = args.selection.expanduser().resolve(strict=True)
        selection = validate_selection(selection_path, runner_sha256=runner_sha256)
        methods = (str(selection["selected_method"]),)

    base_model = args.base_model.expanduser().resolve(strict=True)
    memory_dir = args.memory_dir.expanduser().resolve(strict=True)
    dataset_root = args.dataset_root.expanduser().resolve(strict=True)
    output_root = args.output_root.expanduser().resolve()
    output_dir = output_root / args.phase / f"shard-{args.shard_index}"
    target_rows = mapping_builder.load_prompt_rows(
        dataset_root / mapping_builder.TARGET_RELATIVE_PATH,
        expected_sha256=mapping_builder.TARGET_SHA256,
        expected_rows=mapping_builder.EXPECTED_TARGET_ROWS,
    )
    bank_rows = mapping_builder.load_prompt_rows(
        dataset_root / mapping_builder.BANK_RELATIVE_PATH,
        expected_sha256=mapping_builder.BANK_SHA256,
        expected_rows=mapping_builder.EXPECTED_BANK_ROWS,
    )
    shard_rows = selected_target_rows(
        target_rows,
        phase=args.phase,
        shard_index=args.shard_index,
    )
    mapping_by_index = {
        int(record["target_source_index"]): record for record in mapping["records"]
    }
    bank_by_index = {int(row["source_index"]): row for row in bank_rows}
    binding = input_binding(
        base_model=base_model,
        memory_dir=memory_dir,
        dataset_root=dataset_root,
        phase=args.phase,
        shard_index=args.shard_index,
        methods=methods,
        selection=selection,
        selection_path=selection_path,
        dtype=args.dtype,
        attn_implementation=args.attn_implementation,
    )
    write_or_validate_json(output_dir / "input_binding.json", binding)
    existing_by_method: dict[str, dict[int, Mapping[str, Any]]] = {}
    for method in methods:
        existing = read_completed(output_dir / f"{method}.jsonl")
        validate_resume(
            existing,
            shard_rows,
            phase=args.phase,
            method=method,
            shard_index=args.shard_index,
        )
        existing_by_method[method] = existing
    if all(len(existing_by_method[method]) == len(shard_rows) for method in methods):
        print(
            f"STATE_RETRIEVAL_SHARD_COMPLETE phase={args.phase} shard={args.shard_index}",
            flush=True,
        )
        return 0

    model, tokenizer = load_model_and_tokenizer(
        base_model=str(base_model),
        memory_dir=str(memory_dir),
        device=args.device,
        dtype=args.dtype,
        attn_implementation=args.attn_implementation,
    )
    for ordinal, target_row in enumerate(shard_rows, start=1):
        source_index = int(target_row["source_index"])
        mapping_record = mapping_by_index[source_index]
        if mapping_record["partition"] != args.phase:
            raise ValueError(f"State-retrieval mapping phase differs: {source_index}")
        for method in methods:
            if source_index in existing_by_method[method]:
                continue
            method_mapping = {
                "target_write_tokens": mapping_record["target_write_tokens"],
                **mapping_record["methods"][method],
            }
            bank_row = bank_by_index[int(method_mapping["bank_source_index"])]
            if bank_row["row_sha256"] != method_mapping["bank_row_sha256"]:
                raise ValueError(f"State-retrieval bank row hash differs: {source_index}:{method}")
            record = generate_record(
                model,
                tokenizer,
                target_row,
                bank_row,
                method_mapping,
                phase=args.phase,
                method=method,
                shard_index=args.shard_index,
                device=args.device,
            )
            append_record(output_dir / f"{method}.jsonl", record)
            print(
                f"STATE_RETRIEVAL_PROGRESS phase={args.phase} shard={args.shard_index} "
                f"method={method} row={source_index} ordinal={ordinal}/{len(shard_rows)}",
                flush=True,
            )
    print(
        f"STATE_RETRIEVAL_SHARD_COMPLETE phase={args.phase} shard={args.shard_index}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
