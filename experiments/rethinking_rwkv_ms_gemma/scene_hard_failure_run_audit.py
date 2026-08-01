#!/usr/bin/env python3
"""Audit fresh hard-scene checkpoints against their seeded adapter."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile
from typing import Any, Mapping, Sequence

from experiments.rethinking_rwkv_ms_gemma import (
    scene_hard_failure_train_contract as contract,
)
from experiments.rethinking_rwkv_ms_gemma.scene_memory_v6_run_audit import (
    load_finite_adapter,
)


AUDIT_SCHEMA = "rwkv_ms_scene_hard_failure_checkpoint_audit.v2"
AUDIT_FILENAME = "scene_hard_failure_checkpoint_audit.json"

ADAPTER_TENSOR_SUFFIXES = (
    "delta_scale_raw",
    "memory_q_proj",
    "memory_k_proj",
    "memory_v_proj",
    "delta_q_proj",
    "delta_k_proj",
    "delta_v_proj",
    "delta_o_proj",
    "beta_proj",
    "beta_bias",
    "hrm_rwkv7_core.x_r",
    "hrm_rwkv7_core.x_w",
    "hrm_rwkv7_core.x_k",
    "hrm_rwkv7_core.x_v",
    "hrm_rwkv7_core.x_a",
    "hrm_rwkv7_core.x_g",
    "hrm_rwkv7_core.w1",
    "hrm_rwkv7_core.w2",
    "hrm_rwkv7_core.w0",
    "hrm_rwkv7_core.a1",
    "hrm_rwkv7_core.a2",
    "hrm_rwkv7_core.a0",
    "hrm_rwkv7_core.g1",
    "hrm_rwkv7_core.g2",
    "hrm_rwkv7_core.k_k",
    "hrm_rwkv7_core.k_a",
    "hrm_rwkv7_core.receptance.weight",
    "hrm_rwkv7_core.key.weight",
    "hrm_rwkv7_core.value.weight",
    "hrm_rwkv7_core.output.weight",
    "hrm_rwkv7_core.ln_x.weight",
    "hrm_rwkv7_core.ln_x.bias",
)
FROZEN_ADAPTER_TENSOR_SUFFIXES = (
    "memory_q_proj",
    "memory_k_proj",
    "delta_k_proj",
    "delta_v_proj",
    "hrm_rwkv7_core.ln_x.bias",
)
TRAINABLE_ADAPTER_TENSOR_SUFFIXES = tuple(
    suffix
    for suffix in ADAPTER_TENSOR_SUFFIXES
    if suffix not in FROZEN_ADAPTER_TENSOR_SUFFIXES
)

# These second factors and their inputs have zero first-step gradients because
# w1/a1/g1 are initialized to zero. They become reachable after AdamW updates
# the first factors. A fresh one-step smoke must prove the other 22 families
# changed without falsely claiming full 27-family coverage.
FIRST_UPDATE_DELAYED_TENSOR_SUFFIXES = (
    "hrm_rwkv7_core.x_w",
    "hrm_rwkv7_core.x_a",
    "hrm_rwkv7_core.x_g",
    "hrm_rwkv7_core.w2",
    "hrm_rwkv7_core.a2",
)
FIRST_UPDATE_REQUIRED_TENSOR_SUFFIXES = tuple(
    suffix
    for suffix in TRAINABLE_ADAPTER_TENSOR_SUFFIXES
    if suffix not in FIRST_UPDATE_DELAYED_TENSOR_SUFFIXES
)

EXPECTED_ADAPTER_TENSOR_COUNT = len(contract.TARGET_LAYERS) * len(
    ADAPTER_TENSOR_SUFFIXES
)
EXPECTED_TRAINABLE_TENSOR_COUNT = len(contract.TARGET_LAYERS) * len(
    TRAINABLE_ADAPTER_TENSOR_SUFFIXES
)


class AuditError(ValueError):
    """Raised when a hard-scene checkpoint differs from the locked run."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AuditError(message)


def load_json_object(path: Path, *, description: str) -> dict[str, Any]:
    require(path.is_file() and not path.is_symlink(), f"{description}_missing")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise AuditError(f"{description}_invalid") from exc
    require(isinstance(payload, dict), f"{description}_must_be_object")
    return payload


def write_json_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=True, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _expected_tensor_names(suffixes: Sequence[str]) -> list[str]:
    return [
        f"model.language_model.layers.{layer}.self_attn.{suffix}"
        for layer in contract.TARGET_LAYERS
        for suffix in suffixes
    ]


def _adapter_topology_sha256(state: Mapping[str, Any]) -> str:
    topology = [
        {
            "name": name,
            "shape": list(tensor.shape),
            "dtype": str(tensor.dtype),
        }
        for name, tensor in state.items()
    ]
    return hashlib.sha256(
        json.dumps(topology, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _validate_rank4_adapter_shapes(state: Mapping[str, Any]) -> None:
    import torch

    expected_names = _expected_tensor_names(ADAPTER_TENSOR_SUFFIXES)
    require(list(state) == expected_names, "adapter_exact_qo_rank4_topology_differs")
    trainable = set(TRAINABLE_ADAPTER_TENSOR_SUFFIXES)
    frozen = set(FROZEN_ADAPTER_TENSOR_SUFFIXES)
    rank = contract.RANK
    low_rank_width = 32
    vector_suffixes = {
        "delta_scale_raw",
        "beta_bias",
        "hrm_rwkv7_core.x_r",
        "hrm_rwkv7_core.x_w",
        "hrm_rwkv7_core.x_k",
        "hrm_rwkv7_core.x_v",
        "hrm_rwkv7_core.x_a",
        "hrm_rwkv7_core.x_g",
        "hrm_rwkv7_core.w0",
        "hrm_rwkv7_core.a0",
        "hrm_rwkv7_core.k_k",
        "hrm_rwkv7_core.k_a",
        "hrm_rwkv7_core.ln_x.weight",
        "hrm_rwkv7_core.ln_x.bias",
    }
    rank_by_hidden_suffixes = {
        "memory_q_proj",
        "memory_k_proj",
        "memory_v_proj",
        "beta_proj",
    }
    output_by_rank_suffixes = {
        "delta_q_proj",
        "delta_k_proj",
        "delta_v_proj",
        "delta_o_proj",
    }
    rank_by_low_rank_suffixes = {
        "hrm_rwkv7_core.w1",
        "hrm_rwkv7_core.a1",
        "hrm_rwkv7_core.g1",
    }
    low_rank_by_rank_suffixes = {
        "hrm_rwkv7_core.w2",
        "hrm_rwkv7_core.a2",
        "hrm_rwkv7_core.g2",
    }
    rank_square_suffixes = {
        "hrm_rwkv7_core.receptance.weight",
        "hrm_rwkv7_core.key.weight",
        "hrm_rwkv7_core.value.weight",
        "hrm_rwkv7_core.output.weight",
    }

    for layer in contract.TARGET_LAYERS:
        prefix = f"model.language_model.layers.{layer}.self_attn."
        memory_v = state[prefix + "memory_v_proj"]
        require(memory_v.ndim == 2, f"layer_{layer}_memory_v_proj_shape_differs")
        hidden_size = memory_v.shape[1]
        require(hidden_size > 0, f"layer_{layer}_hidden_size_invalid")
        for suffix in ADAPTER_TENSOR_SUFFIXES:
            tensor = state[prefix + suffix]
            if suffix in vector_suffixes:
                expected_shape = (rank,)
            elif suffix in rank_by_hidden_suffixes:
                expected_shape = (rank, hidden_size)
            elif suffix in output_by_rank_suffixes:
                require(
                    tensor.ndim == 2 and tensor.shape[0] > 0,
                    f"layer_{layer}_{suffix}_shape_differs",
                )
                expected_shape = (tensor.shape[0], rank)
            elif suffix in rank_by_low_rank_suffixes:
                expected_shape = (rank, low_rank_width)
            elif suffix in low_rank_by_rank_suffixes:
                expected_shape = (low_rank_width, rank)
            elif suffix in rank_square_suffixes:
                expected_shape = (rank, rank)
            else:  # pragma: no cover - constants above partition every suffix
                raise AssertionError(f"unclassified adapter suffix: {suffix}")
            require(
                tuple(tensor.shape) == expected_shape,
                f"layer_{layer}_{suffix}_shape_differs",
            )
            expected_dtype = torch.float32 if suffix in trainable else torch.bfloat16
            require(
                tensor.dtype == expected_dtype,
                f"layer_{layer}_{suffix}_dtype_differs",
            )
        require(
            state[prefix + "delta_o_proj"].shape[0] == hidden_size,
            f"layer_{layer}_o_projection_hidden_size_differs",
        )
        require(
            frozen | trainable == set(ADAPTER_TENSOR_SUFFIXES)
            and not frozen & trainable,
            "adapter_trainability_partition_invalid",
        )


def _validate_initial_adapter_topology(
    topology: object,
    initial_adapter: Mapping[str, Any],
) -> list[str]:
    require(isinstance(topology, Mapping), "trainable_topology_missing")
    expected_replaced = [
        f"model.language_model.layers.{layer}.self_attn"
        for layer in contract.TARGET_LAYERS
    ]
    expected_trainable = _expected_tensor_names(TRAINABLE_ADAPTER_TENSOR_SUFFIXES)
    require(
        topology.get("replaced_modules") == expected_replaced,
        "replaced_module_topology_differs",
    )
    require(
        topology.get("trainable_names") == expected_trainable,
        "trainable_qo_rank4_topology_differs",
    )
    require(
        topology.get("adapter_tensor_count") == EXPECTED_ADAPTER_TENSOR_COUNT,
        "adapter_tensor_count_differs",
    )
    require(
        topology.get("adapter_parameter_count")
        == sum(int(tensor.numel()) for tensor in initial_adapter.values()),
        "adapter_parameter_count_differs",
    )
    require(
        topology.get("adapter_topology_sha256")
        == _adapter_topology_sha256(initial_adapter),
        "adapter_topology_sha256_differs",
    )
    return expected_trainable


def _validate_optimizer_state(path: Path, *, checkpoint_step: int) -> dict[str, Any]:
    import torch

    require(path.is_file() and not path.is_symlink(), "checkpoint_optimizer_missing")
    try:
        payload = torch.load(path, map_location="cpu", weights_only=True)
    except Exception as exc:
        raise AuditError("checkpoint_optimizer_invalid") from exc
    require(isinstance(payload, Mapping), "checkpoint_optimizer_must_be_mapping")
    groups = payload.get("param_groups")
    states = payload.get("state")
    require(isinstance(groups, list) and groups, "checkpoint_optimizer_groups_missing")
    require(isinstance(states, Mapping), "checkpoint_optimizer_state_missing")
    parameter_ids: list[Any] = []
    for group in groups:
        require(isinstance(group, Mapping), "checkpoint_optimizer_group_invalid")
        group_parameters = group.get("params")
        require(isinstance(group_parameters, list), "checkpoint_optimizer_params_invalid")
        parameter_ids.extend(group_parameters)
    require(
        len(parameter_ids) == len(set(parameter_ids)) == EXPECTED_TRAINABLE_TENSOR_COUNT,
        "checkpoint_optimizer_parameter_count_differs",
    )
    require(set(states) == set(parameter_ids), "checkpoint_optimizer_state_keys_differ")
    for parameter_id in parameter_ids:
        parameter_state = states[parameter_id]
        require(isinstance(parameter_state, Mapping), "checkpoint_optimizer_entry_invalid")
        require(
            {"step", "exp_avg", "exp_avg_sq"} <= set(parameter_state),
            "checkpoint_optimizer_entry_fields_missing",
        )
        step_value = parameter_state["step"]
        if isinstance(step_value, torch.Tensor):
            require(step_value.numel() == 1, "checkpoint_optimizer_step_not_scalar")
            step_value = step_value.item()
        require(
            isinstance(step_value, (int, float))
            and not isinstance(step_value, bool)
            and math.isfinite(float(step_value))
            and float(step_value) == float(checkpoint_step),
            "checkpoint_optimizer_step_differs",
        )
        for field in ("exp_avg", "exp_avg_sq"):
            tensor = parameter_state[field]
            require(
                isinstance(tensor, torch.Tensor)
                and bool(torch.isfinite(tensor).all()),
                f"checkpoint_optimizer_{field}_nonfinite_or_missing",
            )
    return {
        "optimizer_parameter_state_count": len(states),
        "declared_trainable_adapter_tensor_count": EXPECTED_TRAINABLE_TENSOR_COUNT,
        "all_optimizer_parameter_states_at_checkpoint_step": True,
    }


def adapter_change_record(
    initial: Mapping[str, Any],
    candidate: Mapping[str, Any],
    *,
    trainable_names: Sequence[str],
    checkpoint_step: int,
    smoke: bool,
) -> dict[str, Any]:
    import torch

    _validate_rank4_adapter_shapes(initial)
    require(list(candidate) == list(initial), "checkpoint_adapter_topology_order_differs")
    require(
        list(trainable_names) == _expected_tensor_names(TRAINABLE_ADAPTER_TENSOR_SUFFIXES),
        "declared_trainable_qo_rank4_topology_differs",
    )
    trainable = set(trainable_names)
    changed_trainable: list[str] = []
    changed_nontrainable: list[str] = []
    changed_layers = {suffix: set() for suffix in TRAINABLE_ADAPTER_TENSOR_SUFFIXES}
    maximum_absolute_delta = 0.0
    for name, initial_tensor in initial.items():
        candidate_tensor = candidate[name]
        require(
            initial_tensor.shape == candidate_tensor.shape
            and initial_tensor.dtype == candidate_tensor.dtype,
            f"checkpoint_tensor_metadata_differs:{name}",
        )
        require(
            bool(torch.isfinite(candidate_tensor).all()),
            f"checkpoint_tensor_nonfinite:{name}",
        )
        if torch.equal(initial_tensor, candidate_tensor):
            continue
        suffix = name.rsplit(".self_attn.", 1)[1]
        layer = int(name.split(".layers.", 1)[1].split(".", 1)[0])
        if name in trainable:
            changed_trainable.append(name)
            changed_layers[suffix].add(layer)
        else:
            changed_nontrainable.append(name)
        maximum_absolute_delta = max(
            maximum_absolute_delta,
            float((candidate_tensor.float() - initial_tensor.float()).abs().max().item()),
        )
    require(not changed_nontrainable, "checkpoint_changed_frozen_adapter_tensor")
    require(
        math.isfinite(maximum_absolute_delta) and maximum_absolute_delta > 0.0,
        "checkpoint_adapter_delta_not_positive_finite",
    )

    expected_layers = set(contract.TARGET_LAYERS)
    for suffix in FIRST_UPDATE_REQUIRED_TENSOR_SUFFIXES:
        require(
            changed_layers[suffix] == expected_layers,
            f"checkpoint_missing_required_family_layer_change:{suffix}",
        )
    if checkpoint_step == 1:
        for suffix in FIRST_UPDATE_DELAYED_TENSOR_SUFFIXES:
            require(
                not changed_layers[suffix],
                f"first_update_delayed_family_changed:{suffix}",
            )

    for layer in contract.TARGET_LAYERS:
        prefix = f"model.language_model.layers.{layer}.self_attn."
        initial_scale = initial[prefix + "delta_scale_raw"]
        candidate_scale = candidate[prefix + "delta_scale_raw"]
        require(
            not torch.equal(initial_scale[0], candidate_scale[0])
            and not torch.equal(initial_scale[3], candidate_scale[3]),
            f"layer_{layer}_active_qo_scale_entries_unchanged",
        )
        require(
            torch.equal(initial_scale[1:3], candidate_scale[1:3]),
            f"layer_{layer}_inactive_kv_scale_entries_changed",
        )

    coverage = {
        suffix: len(changed_layers[suffix])
        for suffix in TRAINABLE_ADAPTER_TENSOR_SUFFIXES
    }
    missing = {
        suffix: sorted(expected_layers - changed_layers[suffix])
        for suffix in TRAINABLE_ADAPTER_TENSOR_SUFFIXES
        if changed_layers[suffix] != expected_layers
    }
    full_coverage = not missing
    full_coverage_required = (
        not smoke and checkpoint_step == contract.TOTAL_OPTIMIZER_STEPS
    )
    if full_coverage_required:
        require(full_coverage, "final_checkpoint_trainable_family_layer_coverage_incomplete")

    return {
        "changed_tensor_count": len(changed_trainable),
        "changed_trainable_tensor_count": len(changed_trainable),
        "changed_nontrainable_tensor_count": 0,
        "maximum_absolute_delta": maximum_absolute_delta,
        "trainable_tensor_family_count": len(TRAINABLE_ADAPTER_TENSOR_SUFFIXES),
        "target_layer_count": len(contract.TARGET_LAYERS),
        "expected_trainable_tensor_count": EXPECTED_TRAINABLE_TENSOR_COUNT,
        "trainable_family_layer_coverage": coverage,
        "missing_trainable_family_layers": missing,
        "full_trainable_family_coverage": full_coverage,
        "full_trainable_family_coverage_required": full_coverage_required,
        "first_update_required_family_layer_coverage": {
            suffix: coverage[suffix]
            for suffix in FIRST_UPDATE_REQUIRED_TENSOR_SUFFIXES
        },
        "first_update_structurally_delayed_families": list(
            FIRST_UPDATE_DELAYED_TENSOR_SUFFIXES
        ),
        "frozen_adapter_tensor_count": len(contract.TARGET_LAYERS)
        * len(FROZEN_ADAPTER_TENSOR_SUFFIXES),
        "frozen_adapter_tensors_unchanged": True,
        "inactive_kv_delta_scale_entries_unchanged": len(contract.TARGET_LAYERS) * 2,
        "first_changed_trainable_tensors": changed_trainable[:8],
    }


def _validate_protocol(protocol: Mapping[str, Any], *, smoke: bool) -> None:
    expected_steps = [1] if smoke else list(contract.CHECKPOINT_STEPS)
    expected_endpoints = [1] if smoke else list(contract.GENERATION_ENDPOINT_STEPS)
    expected = {
        "schema_version": contract.OBJECTIVE_SCHEMA_VERSION,
        "memory_objective_version": contract.OBJECTIVE_VERSION,
        "gradient_accumulation_steps": contract.GRADIENT_ACCUMULATION_STEPS,
        "max_steps": 1 if smoke else contract.TOTAL_OPTIMIZER_STEPS,
        "save_steps": contract.SAVE_STEPS,
        "save_total_limit": 1 if smoke else contract.SAVE_TOTAL_LIMIT,
        "scene_generation_hard_failure_run_mode": (
            contract.ONE_PAIR_SMOKE_RUN_MODE if smoke else contract.PRODUCTION_RUN_MODE
        ),
        "scene_generation_hard_failure_production_eligible": not smoke,
        "scene_generation_row_objective_audit_filename": (
            contract.ROW_OBJECTIVE_AUDIT_FILENAME
        ),
        "scene_generation_row_objective_audit_schema": contract.ROW_OBJECTIVE_AUDIT_SCHEMA,
    }
    mismatches = [
        name for name, value in expected.items() if protocol.get(name) != value
    ]
    require(not mismatches, "training_protocol_differs: " + ", ".join(mismatches))
    require(
        "scene_generation_v15_run_mode" not in protocol
        and "scene_generation_v15_production_eligible" not in protocol,
        "training_protocol_contains_v15_run_fields",
    )
    schedule = protocol.get("train_schedule")
    require(isinstance(schedule, Mapping), "training_protocol_schedule_missing")
    expected_schedule = {
        "checkpoint_steps": expected_steps,
        "optimizer_checkpoint_steps": expected_steps,
        "generation_endpoint_steps": expected_endpoints,
        "microbatch_cycle_size": 1,
        "continuation_policy": "forbidden_fresh_only",
    }
    schedule_mismatches = [
        name for name, value in expected_schedule.items() if schedule.get(name) != value
    ]
    require(
        not schedule_mismatches,
        "training_protocol_schedule_differs: " + ", ".join(schedule_mismatches),
    )


def _validate_trainer_state(state: Mapping[str, Any], *, step: int) -> None:
    require(state.get("global_step") == step, "trainer_global_step_differs")
    history = state.get("log_history")
    require(isinstance(history, list) and history, "trainer_log_history_missing")
    step_records = [
        record
        for record in history
        if isinstance(record, Mapping) and record.get("step") == step
    ]
    require(step_records, "trainer_step_log_missing")
    latest = step_records[-1]
    for field in ("loss", "grad_norm"):
        value = latest.get(field)
        require(
            isinstance(value, (int, float))
            and not isinstance(value, bool)
            and math.isfinite(float(value)),
            f"trainer_{field}_nonfinite_or_missing",
        )
    require(float(latest["grad_norm"]) > 0.0, "trainer_grad_norm_not_positive")


def _validate_row_audit(
    audit: Mapping[str, Any],
    *,
    step: int,
    smoke: bool,
) -> None:
    require(audit.get("schema") == contract.ROW_OBJECTIVE_AUDIT_SCHEMA, "row_audit_schema_differs")
    require(audit.get("memory_objective_version") == contract.OBJECTIVE_VERSION, "row_audit_objective_differs")
    require(audit.get("checkpoint_optimizer_step") == step, "row_audit_step_differs")
    require(audit.get("completed_pair_presentations") == step, "row_audit_presentation_count_differs")
    require(
        audit.get("run_mode")
        == (contract.ONE_PAIR_SMOKE_RUN_MODE if smoke else contract.PRODUCTION_RUN_MODE),
        "row_audit_run_mode_differs",
    )
    require(audit.get("production_eligible") is (not smoke), "row_audit_eligibility_differs")
    pairs = audit.get("pair_presentations")
    rows = audit.get("rows")
    schedule = audit.get("pair_schedule")
    require(isinstance(pairs, list) and len(pairs) == step, "row_audit_pairs_incomplete")
    require(isinstance(rows, list) and len(rows) == step * 2, "row_audit_rows_incomplete")
    require(isinstance(schedule, list) and len(schedule) == step, "row_audit_schedule_incomplete")
    require(
        audit.get("generation_endpoint")
        is (step in contract.GENERATION_ENDPOINT_STEPS),
        "row_audit_generation_endpoint_differs",
    )


def audit_checkpoint(
    *,
    run_root: Path,
    checkpoint_step: int,
    smoke: bool,
    receipt: Path | None = None,
) -> dict[str, Any]:
    run_root = run_root.expanduser().resolve()
    contract.reject_protected_path(run_root, description="hard_failure_run_root")
    contract.validate_data_contract()
    source_lock = contract.validate_source_lock()
    allowed_steps = (1,) if smoke else contract.CHECKPOINT_STEPS
    require(checkpoint_step in allowed_steps, "checkpoint_step_outside_contract")
    checkpoint = run_root / "trainer" / f"checkpoint-{checkpoint_step}"
    initial = run_root / "initial_adapter"
    require(checkpoint.is_dir() and initial.is_dir(), "checkpoint_or_initial_adapter_missing")

    initial_manifest = load_json_object(
        initial / "initial_adapter_manifest.json",
        description="initial_adapter_manifest",
    )
    initial_adapter = load_finite_adapter(initial / "delta_mem_adapter.pt")
    checkpoint_adapter = load_finite_adapter(checkpoint / "delta_mem_adapter.pt")
    trainable_names = _validate_initial_adapter_topology(
        initial_manifest.get("topology"),
        initial_adapter,
    )
    change = adapter_change_record(
        initial_adapter,
        checkpoint_adapter,
        trainable_names=trainable_names,
        checkpoint_step=checkpoint_step,
        smoke=smoke,
    )
    optimizer_evidence = _validate_optimizer_state(
        checkpoint / "optimizer.pt",
        checkpoint_step=checkpoint_step,
    )

    protocol = load_json_object(
        checkpoint / "training_protocol.json",
        description="checkpoint_training_protocol",
    )
    trainer_state = load_json_object(
        checkpoint / "trainer_state.json",
        description="checkpoint_trainer_state",
    )
    row_audit = load_json_object(
        checkpoint / contract.ROW_OBJECTIVE_AUDIT_FILENAME,
        description="checkpoint_row_audit",
    )
    _validate_protocol(protocol, smoke=smoke)
    _validate_trainer_state(trainer_state, step=checkpoint_step)
    _validate_row_audit(row_audit, step=checkpoint_step, smoke=smoke)

    payload: dict[str, Any] = {
        "schema": AUDIT_SCHEMA,
        "audited_at": datetime.now(timezone.utc).isoformat(),
        "run_root": str(run_root),
        "checkpoint": str(checkpoint),
        "checkpoint_optimizer_step": checkpoint_step,
        "run_mode": (
            contract.ONE_PAIR_SMOKE_RUN_MODE if smoke else contract.PRODUCTION_RUN_MODE
        ),
        "objective_version": contract.OBJECTIVE_VERSION,
        "source_lock_sha256": source_lock["lock_sha256"],
        "adapter_change": change,
        "trainable_tensor_family_count": change["trainable_tensor_family_count"],
        "target_layer_count": change["target_layer_count"],
        "trainable_family_layer_coverage": change[
            "trainable_family_layer_coverage"
        ],
        "full_trainable_family_coverage": change[
            "full_trainable_family_coverage"
        ],
        "optimizer_update": optimizer_evidence,
        "optimizer_contains_only_declared_trainable_adapter_state_count": True,
        "base_model_parameter_values_not_materialized_in_adapter_checkpoint": True,
        "nontrainable_adapter_tensors_unchanged": True,
        "row_audit_complete": True,
    }
    payload["receipt_sha256"] = contract.canonical_sha256(payload)
    destination = checkpoint / AUDIT_FILENAME if receipt is None else receipt.resolve()
    require(
        destination == checkpoint / AUDIT_FILENAME,
        "checkpoint_audit_receipt_path_differs",
    )
    write_json_atomic(destination, payload)
    return payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--checkpoint-step", type=int, required=True)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--receipt", type=Path)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    try:
        payload = audit_checkpoint(
            run_root=args.run_root,
            checkpoint_step=args.checkpoint_step,
            smoke=args.smoke,
            receipt=args.receipt,
        )
    except (AuditError, ValueError, OSError) as exc:
        print(f"ERROR: scene_hard_failure_checkpoint_audit_failed: {exc}")
        return 2
    print(
        "checkpoint_audit=valid "
        f"step={payload['checkpoint_optimizer_step']} "
        f"sha256={payload['receipt_sha256']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
