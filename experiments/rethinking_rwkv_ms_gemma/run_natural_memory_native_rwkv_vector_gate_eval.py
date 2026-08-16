#!/usr/bin/env python3
"""Evaluate vector-gate RWKV control against a fixed projected carrier."""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import json
import os
from pathlib import Path
import sys
from typing import Any, Iterator, Mapping, Sequence

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

import torch


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = Path(__file__).resolve().parents[2]
for import_root in (SCRIPT_DIR, PROJECT_ROOT):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

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
    run_natural_memory_native_rwkv_vector_gate_causal_train as training,
)


SCHEMA = "rwkv_ms_natural_memory_native_vector_gate_eval.v1"
INPUT_SCHEMA = "rwkv_ms_natural_memory_native_vector_gate_eval_input.v1"
PROTOCOL = (
    SCRIPT_DIR
    / "natural_memory_native_rwkv_vector_gate_generation_protocol_v1.json"
)
PROTOCOL_PAYLOAD_SHA256 = (
    "8b404957979865498b77035fa3c0f9c616baa5625c39a2eab2ca798dd57d5732"
)
TRAINING_PROTOCOL_PAYLOAD_SHA256 = training.PROTOCOL_PAYLOAD_SHA256
TRAIN_RESULT_FILE_SHA256 = (
    "deaa6f2004ec61f7cd8d0040a3799db0dfa0624fce03cd6a5975a95aa89ecc5e"
)
TRAIN_RESULT_RECEIPT = (
    "2b042650a1ef0f59fe9035eb55bba23ec1dd648513a3b1f44b53414f733766bc"
)
ADAPTER_CONFIG_SHA256 = (
    "28830f91f455c77f6b565cacb3069262d991d2030e3564961e0c957f13c8245f"
)
HF_MIRROR_ENDPOINT = training.shared.HF_MIRROR_ENDPOINT
WORLD_SIZE = 4
SEED = training.SEED
EVALUATION_ROWS = base_eval.EVALUATION_ROWS
AUTHORIZED_ROWS_PAYLOAD_SHA256 = base_eval.AUTHORIZED_ROWS_PAYLOAD_SHA256
DATASET_RELATIVE_PATH = base_eval.DATASET_RELATIVE_PATH
DATASET_SHA256 = base_eval.DATASET_SHA256
STATE_CONDITIONS = (
    "correct_recurrent_state",
    "zero_recurrent_state",
    "matched_donor_recurrent_state",
    "layer_permuted_recurrent_state",
)
CONDITIONS = (*STATE_CONDITIONS, "projected_only_bypass")
BATCH_CONDITIONS = (
    "zero_recurrent_state",
    "correct_recurrent_state",
    "matched_donor_recurrent_state",
    "layer_permuted_recurrent_state",
)


def canonical_sha256(value: Any) -> str:
    return training.shared.canonical_sha256(value)


def sha256_file(path: Path) -> str:
    return training.shared.sha256_file(path)


def validate_protocol() -> Mapping[str, Any]:
    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    receipt = protocol.get("receipt")
    if not isinstance(receipt, Mapping):
        raise ValueError("Vector-gate generation protocol receipt is missing")
    unsigned = dict(protocol)
    unsigned.pop("receipt")
    digest = canonical_sha256(unsigned)
    frozen = protocol.get("frozen_inputs", {})
    generation = protocol.get("generation", {})
    if (
        digest != PROTOCOL_PAYLOAD_SHA256
        or receipt.get("payload_sha256") != digest
        or frozen.get("dataset_sha256") != DATASET_SHA256
        or frozen.get("authorized_rows") != EVALUATION_ROWS
        or frozen.get("authorized_rows_payload_sha256")
        != AUTHORIZED_ROWS_PAYLOAD_SHA256
        or frozen.get("seed") != SEED
        or generation.get("conditions") != list(CONDITIONS)
        or protocol.get("protected_splits_opened_by_this_protocol") != []
    ):
        raise ValueError("Vector-gate generation protocol differs")
    basis = protocol.get("authorization_basis", {})
    if (
        basis.get("training_result_file_sha256") != TRAIN_RESULT_FILE_SHA256
        or basis.get("training_result_receipt") != TRAIN_RESULT_RECEIPT
        or basis.get("adapter_config_file_sha256") != ADAPTER_CONFIG_SHA256
        or basis.get("open_native_generation_authorized") is not True
    ):
        raise ValueError("Vector-gate generation authorization differs")
    return protocol


def validate_train_result(
    result_path: Path,
    *,
    adapter_dir: Path,
) -> Mapping[str, Any]:
    validate_protocol()
    if sha256_file(result_path) != TRAIN_RESULT_FILE_SHA256:
        raise ValueError("Vector-gate training result file differs")
    result = json.loads(result_path.read_text(encoding="utf-8"))
    receipt = result.get("receipt")
    if not isinstance(receipt, Mapping):
        raise ValueError("Vector-gate training result receipt is missing")
    unsigned = dict(result)
    unsigned.pop("receipt")
    required = {
        "schema": training.SCHEMA,
        "status": "vector_gate_heldout_passed_generation_authorized",
        "passed": True,
        "protocol_payload_sha256": TRAINING_PROTOCOL_PAYLOAD_SHA256,
        "seed": SEED,
        "updates": training.shared.UPDATES,
        "open_native_generation_authorized": True,
        "protected_splits_opened": [],
    }
    if (
        canonical_sha256(unsigned) != TRAIN_RESULT_RECEIPT
        or receipt.get("payload_sha256") != TRAIN_RESULT_RECEIPT
        or any(result.get(key) != value for key, value in required.items())
        or result.get("input_binding", {}).get("selected_candidate")
        != training.SELECTED_CANDIDATE
    ):
        raise ValueError("Vector-gate training did not authorize generation")
    expected_files = result.get("adapter_files")
    if not isinstance(expected_files, Mapping):
        raise ValueError("Scalar-agreement adapter manifest is missing")
    for filename, metadata in expected_files.items():
        path = adapter_dir / str(filename)
        if not path.is_file() or sha256_file(path) != metadata.get("sha256"):
            raise ValueError(f"Scalar-agreement adapter file differs: {path}")
    if expected_files.get("delta_mem_config.json", {}).get("sha256") != (
        ADAPTER_CONFIG_SHA256
    ):
        raise ValueError("Scalar-agreement serialized config binding differs")
    if result.get("code_bindings", {}).get("runner_sha256") != sha256_file(
        Path(training.__file__)
    ):
        raise ValueError("Vector-gate training runner binding differs")
    return result


@contextmanager
def _state_condition_batch() -> Iterator[None]:
    previous = addressed_eval.CONDITIONS
    addressed_eval.CONDITIONS = BATCH_CONDITIONS
    try:
        yield
    finally:
        addressed_eval.CONDITIONS = previous


def _restore_and_assert_vector_gate_modules(model) -> None:
    modules = tuple(iter_delta_mem_modules(model))
    if len(modules) != training.shared.preflight.EXPECTED_LAYERS:
        raise ValueError(f"Expected 42 vector-gate layers, found {len(modules)}")
    modes = {module.rwkv_ms_hybrid_mode for _, module in modules}
    if modes == {"recurrent_value"}:
        for _, module in modules:
            module.rwkv_ms_hybrid_mode = "vector_gate"
    elif modes != {"vector_gate"}:
        raise ValueError(f"Unexpected loaded hybrid modes: {sorted(modes)!r}")
    if not all(
        module.memory_readout_mode == "projected_kv_rwkv_hybrid"
        and module.rwkv_ms_hybrid_mode == "vector_gate"
        and module.rwkv_ms_hybrid_gain == 0.125
        and module.rwkv_ms_read_temperature == 16.0
        and module.rwkv_ms_read_top_k == 2
        and module.rwkv_ms_detach_read_scores is True
        for _, module in modules
    ):
        raise ValueError("Loaded adapter is not the locked vector-gate hybrid")


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
    _restore_and_assert_vector_gate_modules(model)
    correct_state = base_eval.prime_state(model, tokenizer, row, device=device)
    correct_recurrent, correct_projected = base_eval.split_state(
        correct_state, module_names
    )
    donor_state = base_eval.prime_state(model, tokenizer, donor, device=device)
    donor_recurrent, donor_projected = base_eval.split_state(donor_state, module_names)
    projected_digest = base_eval.tensor_digest(correct_projected)
    if base_eval.tensor_digest(donor_projected) == projected_digest:
        raise RuntimeError("Scalar-agreement donor projected carrier is unexpectedly equal")
    states = {
        "zero_recurrent_state": base_eval.merge_state(
            base_eval.zero_recurrent_state(correct_recurrent), correct_projected
        ),
        "correct_recurrent_state": base_eval.merge_state(
            correct_recurrent, correct_projected
        ),
        "matched_donor_recurrent_state": base_eval.merge_state(
            donor_recurrent, correct_projected
        ),
        "layer_permuted_recurrent_state": base_eval.merge_state(
            base_eval.permute_recurrent_state(correct_recurrent, module_names),
            correct_projected,
        ),
    }
    with _state_condition_batch():
        generated = dict(
            addressed_eval.batched_generate_from_states(
                model, tokenizer, row, states, device=device
            )
        )
        bypass_states = {
            condition: states["zero_recurrent_state"]
            for condition in BATCH_CONDITIONS
        }
        with base_eval.projected_only_mode(model):
            bypass_batch = addressed_eval.batched_generate_from_states(
                model, tokenizer, row, bypass_states, device=device
            )
    generated["projected_only_bypass"] = bypass_batch["zero_recurrent_state"]
    gold = recovery.strict_gold_boundaries(row["gold"])
    records: list[Mapping[str, Any]] = []
    for condition in CONDITIONS:
        state = states.get(condition, states["zero_recurrent_state"])
        recurrent, projected = base_eval.split_state(state, module_names)
        carrier_fixed = base_eval.tensor_digest(projected) == projected_digest
        if not carrier_fixed:
            raise RuntimeError("Scalar-agreement projected carrier changed")
        generation = generated[condition]
        records.append(
            {
                "schema": SCHEMA,
                "protocol_payload_sha256": PROTOCOL_PAYLOAD_SHA256,
                "training_protocol_payload_sha256": (
                    TRAINING_PROTOCOL_PAYLOAD_SHA256
                ),
                "condition": condition,
                "seed": SEED,
                "shard_index": shard_index,
                "world_size": WORLD_SIZE,
                "source_index": row["source_index"],
                "row_sha256": row["row_sha256"],
                "gold": sorted(gold),
                **generation,
                "score": base_eval.record_score(generation["prediction"], gold),
                "state_sha256": base_eval.tensor_digest(state),
                "condition_recurrent_sha256": base_eval.tensor_digest(recurrent),
                "projected_carrier_sha256": base_eval.tensor_digest(projected),
                "correct_projected_carrier_sha256": projected_digest,
                "projected_carrier_byte_identical": carrier_fixed,
                "projected_carrier_active": True,
                "serialized_adapter_hybrid_mode": "recurrent_value",
                "runtime_hybrid_mode": "vector_gate",
                "runtime_mode_restoration_changed_no_parameters": True,
                "generation_batch_shape_control": "four_by_four_same_position",
                "rwkv_recurrent_vector_controller_active": (
                    condition != "projected_only_bypass"
                ),
                "explicit_projected_only_bypass": (
                    condition == "projected_only_bypass"
                ),
                "donor_source_index": donor["source_index"],
                "donor_row_sha256": donor["row_sha256"],
            }
        )
    by_condition = {record["condition"]: record for record in records}
    if (
        by_condition["zero_recurrent_state"]["prediction"]
        != by_condition["projected_only_bypass"]["prediction"]
    ):
        raise RuntimeError("Zero recurrent and projected-only predictions differ")
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
        "protocol_file_sha256": sha256_file(PROTOCOL),
        "training_protocol_payload_sha256": TRAINING_PROTOCOL_PAYLOAD_SHA256,
        "seed": SEED,
        "shard_index": shard_index,
        "partition_index": partition_index,
        "partitions_per_shard": partitions_per_shard,
        "world_size": WORLD_SIZE,
        "base_model": str(base_model),
        "base_config_sha256": training.shared.preflight.EXPECTED_BASE_CONFIG_SHA256,
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
        "hybrid_mode": "vector_gate",
        "serialized_adapter_hybrid_mode": "recurrent_value",
        "runtime_mode_restoration_changed_no_parameters": True,
        "generation_batch_shape_control": "four_by_four_same_position",
        "projected_carrier_active": True,
        "rwkv_recurrent_vector_controller_active": True,
        "zero_recurrent_identity_requires_explicit_bypass_match": True,
        "hf_endpoint": os.environ.get("HF_ENDPOINT"),
        "runner_sha256": sha256_file(Path(__file__)),
        "addressed_eval_helper_sha256": sha256_file(Path(addressed_eval.__file__)),
        "generation_runner_sha256": sha256_file(Path(base_eval.causal.__file__)),
    }


@contextmanager
def evaluation_bindings() -> Iterator[None]:
    bindings = {
        "SCHEMA": SCHEMA,
        "INPUT_SCHEMA": INPUT_SCHEMA,
        "PROTOCOL_PAYLOAD_SHA256": PROTOCOL_PAYLOAD_SHA256,
        "SEED": SEED,
        "CONDITIONS": CONDITIONS,
        "validate_train_result": validate_train_result,
        "generate_row_conditions": generate_row_conditions,
        "input_binding": input_binding,
    }
    previous = {name: getattr(addressed_eval, name) for name in bindings}
    try:
        for name, value in bindings.items():
            setattr(addressed_eval, name, value)
        yield
    finally:
        for name, value in previous.items():
            setattr(addressed_eval, name, value)


def run(args: argparse.Namespace) -> int:
    validate_protocol()
    with evaluation_bindings():
        return addressed_eval.run(args)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shard-index", type=int, required=True)
    parser.add_argument("--partition-index", type=int, default=0)
    parser.add_argument("--partitions-per-shard", type=int, default=1)
    parser.add_argument("--base-model", type=Path, default=training.shared.BASE_MODEL)
    parser.add_argument("--dataset-root", type=Path, default=training.shared.DATASET_ROOT)
    parser.add_argument("--train-result", type=Path, required=True)
    parser.add_argument("--adapter-dir", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    return parser.parse_args(argv)


if __name__ == "__main__":
    raise SystemExit(run(parse_args()))
