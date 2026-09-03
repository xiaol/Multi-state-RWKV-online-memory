#!/usr/bin/env python3
"""Continue direct native-PLE RWKV training after the v10 generation diagnosis."""

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
from torch.distributed.elastic.multiprocessing.errors import record


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[1]
for import_root in (SCRIPT_DIR, PROJECT_ROOT):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

from experiments.rethinking_rwkv_ms_gemma import (  # noqa: E402
    evaluate_natural_memory_native_rwkv_direct_ple_development as evaluator,
)
from experiments.rethinking_rwkv_ms_gemma import (  # noqa: E402
    recurrent_routed_posttrain_common as common,
)
from experiments.rethinking_rwkv_ms_gemma import (  # noqa: E402
    run_natural_memory_native_rwkv_direct_ple_causal_train as direct,
)
from experiments.rethinking_rwkv_ms_gemma import (  # noqa: E402
    natural_memory_distributed as distributed,
)


PROTOCOL = SCRIPT_DIR / "natural_memory_native_rwkv_direct_ple_continuation_protocol_v5.json"
PROTOCOL_PAYLOAD_SHA256 = "29657c8d7172a402e808276890d464de8051450023b4d511f11ae22f815cafdc"
SCHEMA = "rwkv_ms_natural_memory_native_rwkv_direct_ple_continuation.v5"
STEP_SCHEMA = "rwkv_ms_natural_memory_native_rwkv_direct_ple_continuation_step.v5"
INPUT_SCHEMA = "rwkv_ms_natural_memory_native_rwkv_direct_ple_continuation_input.v5"
PREDECESSOR_RESULT = (
    SCRIPT_DIR
    / "local_artifacts/natural_memory_native_rwkv_direct_ple_train_v10/result.json"
)
PREDECESSOR_ADAPTER = PREDECESSOR_RESULT.parent / "adapter"
PREDECESSOR_RESULT_SHA256 = "982f15eb55a0cbb68e4dc69d3f4d9afd3c2f8e02eaec7baace32aa199edbbb45"
PREDECESSOR_RESULT_RECEIPT = "1d1e29197421cc932949c958d48fd63202e034bbc08aa02cfad3a7817ad99bec"
PREDECESSOR_ADAPTER_FP32_SNAPSHOT_SHA256 = "86c4566326c65a2bbb881e0cd18d7ce84cad3c9de0b1d76ac0e347f0351c6161"
V1_PROTOCOL_RECEIPT = "a78f692aeef57f3e066a014a9d173ff45d45654f4eb3af921bda52d0621da2d9"
V1_FAILURE_LOG = (
    SCRIPT_DIR
    / "local_artifacts/natural_memory_native_rwkv_direct_ple_continuation_preflight_v1.log"
)
V1_FAILURE_LOG_SHA256 = "b9c27b26480728427e1487e1d6dd438f80fc2f34a2448ee2388e44eb3fe398c4"
V2_PROTOCOL_RECEIPT = "1652f9ba87e91bdbca8b73418b740060902ccc3a34874a6001bbbc8c8fef115b"
V2_PREFLIGHT_RESULT = (
    SCRIPT_DIR
    / "local_artifacts/natural_memory_native_rwkv_direct_ple_continuation_preflight_v2/result.json"
)
V2_PREFLIGHT_RESULT_SHA256 = "dd2a75fd5da34f60dab5f0f4061c0091df5814517907b6d537a230e4e5be9194"
V2_PREFLIGHT_RESULT_RECEIPT = "6a259e773e48259a05468b140c181b348f8f9a5bd2af85cd0c1f8288dc72e7e6"
V2_FAILURE_LOG = (
    SCRIPT_DIR
    / "local_artifacts/natural_memory_native_rwkv_direct_ple_continuation_preflight_v2.log"
)
V2_FAILURE_LOG_SHA256 = "14f122006891534fc81e61391bcf50105a0ecbd6f70b70ea60179cc925d7f53e"
V3_PROTOCOL_RECEIPT = "12d21777f60d8d996a232b5ff2316420cb599bfd2b8ba40887ce182b2ddc3188"
V4_PROTOCOL_RECEIPT = "88dc9e6fe49e70e8f99195be0d7e1ff3204899da0067c187e3623eaec04e0b1f"
V4_RUNNER_SHA256 = "e311f218195d2fd43f3f0e82c52c564fec71a5521b4e769cc4cd7f58f331196f"
V4_PREFLIGHT_RESULT = (
    SCRIPT_DIR
    / "local_artifacts/natural_memory_native_rwkv_direct_ple_continuation_preflight_v3/result.json"
)
V4_PREFLIGHT_RESULT_SHA256 = "211dc3f2cce05e89a40216abfa804f22abd7ab7c402f11c1a6e46c7bc455d924"
V4_PREFLIGHT_RESULT_RECEIPT = "945a0c0de09851c30c320b57bccb27d59a12248fd0711ebb98766cc89cbd2f9f"
V4_PREFLIGHT_LOG = (
    SCRIPT_DIR
    / "local_artifacts/natural_memory_native_rwkv_direct_ple_continuation_preflight_v3.log"
)
V4_PREFLIGHT_LOG_SHA256 = "5b0f0c99162b0573e4627e8755d2408e7f562ab4779d1e27d9415d1db24be518"
V4_INVALID_TRAIN = (
    SCRIPT_DIR
    / "local_artifacts/natural_memory_native_rwkv_direct_ple_continuation_train_v1"
)
V4_INVALID_TRAIN_INPUT_SHA256 = "6350cdbeeb538db0416660d70ff279dc246f41d4ab0bb151947059bee5ae0798"
V4_INVALID_TRAIN_PROGRESS_SHA256 = "d97619cfd49d22c6d4b0eeaaf34e007719c0e87ac03fc457b749b82795146422"
V4_INVALID_TRAIN_LOG = V4_INVALID_TRAIN.with_suffix(".log")
V4_INVALID_TRAIN_LOG_SHA256 = "cd58e155ab081c7eecb5c9e4ef726fab64fef4fa693dbef7fc55b6a6abc27a38"
V4_CAPTURED_PARENT_PROTOCOL_SHA256 = "576537822ca7079d15fc6d0ce618a94b8631286c47008f136ef1b6ed725d191d"
SEED = 20260903
UPDATES = 32
PREFLIGHT_UPDATES = 1
LEARNING_RATE = 2.5e-5
MAX_GRAD_NORM = 0.1
MARGIN = 0.05
PLE_GAIN = 1.0
PLE_INPUT_GAIN = 0.125
CONTROL_WEIGHTS = {
    "zero_recurrent_state": 0.25,
    "matched_donor_recurrent_state": 1.0,
    "slot_shuffled_recurrent_state": 0.125,
    "layer_permuted_recurrent_state": 0.125,
}
DIAGNOSTICS = (
    (
        SCRIPT_DIR
        / "local_artifacts/natural_memory_native_rwkv_direct_ple_generation_diagnostic_v1/result.json",
        "rwkv_ms_natural_memory_native_rwkv_direct_ple_generation_diagnostic.v1",
        "7d08ea3ae65b25b1b4895fc413e4982ffca11c3053d63d2cecbfb2ac8b931805",
        "010eb117e9b140e41a86e864f6223addba99b91f368d9efa2d6727670bc7a6d3",
    ),
    (
        SCRIPT_DIR
        / "local_artifacts/natural_memory_native_rwkv_direct_ple_gain_x2_diagnostic_v1/result.json",
        "rwkv_ms_natural_memory_native_rwkv_direct_ple_gain_diagnostic.v1",
        "9c1143e8f8435103ef4073966849ef9da403fa25f8f9da9834bcee35d7467a27",
        "0a94e507b6486f6558022382a58de3a91c6a321f9822c1c07d0807671449ed95",
    ),
    (
        SCRIPT_DIR
        / "local_artifacts/natural_memory_native_rwkv_direct_ple_gain_x4_diagnostic_v1/result.json",
        "rwkv_ms_natural_memory_native_rwkv_direct_ple_gain_diagnostic.v1",
        "cec2d9bd4f29585dfc8efe16a4e2ea21c2957f9b59a735120d4445909971b290",
        "52b1bb785f417b03f7cc2f762c1188ef12606f220aaea885260901001ae76712",
    ),
    (
        SCRIPT_DIR
        / "local_artifacts/natural_memory_native_rwkv_direct_ple_gain_x8_diagnostic_v1/result.json",
        "rwkv_ms_natural_memory_native_rwkv_direct_ple_gain_diagnostic.v1",
        "8b4d1a5d4a7daaf1323ce4cb6f46d6ff862c914a3ae694f76ebd4668a4ac7844",
        "7c539c45778fcf58f96026584384ea59697d7bc2d2e5eaa976cc75080605816d",
    ),
)
ORIGINAL_BUILD_TRAINING_SCHEDULE = common.build_training_schedule
ORIGINAL_GRADIENT_AUDIT = direct._gradient_audit
SCHEDULE_AUDIT: dict[str, Any] = {}


def validate_signed_artifact(
    path: Path,
    *,
    schema: str,
    file_sha256: str,
    receipt_sha256: str,
) -> Mapping[str, Any]:
    resolved = path.expanduser().resolve(strict=True)
    value = json.loads(resolved.read_text(encoding="utf-8"))
    receipt = value.get("receipt")
    unsigned = dict(value)
    unsigned.pop("receipt", None)
    if (
        evaluator.sha256_file(resolved) != file_sha256
        or value.get("schema") != schema
        or not isinstance(receipt, Mapping)
        or receipt.get("payload_sha256") != receipt_sha256
        or evaluator.canonical_sha256(unsigned) != receipt_sha256
        or value.get("final_rows_opened") is not False
        or value.get("publisher_validation_opened") is not False
        or value.get("publisher_test_opened") is not False
    ):
        raise ValueError(f"Continuation evidence differs: {resolved}")
    return value


def validate_v4_binding_failure() -> None:
    validate_signed_artifact(
        V4_PREFLIGHT_RESULT,
        schema="rwkv_ms_natural_memory_native_rwkv_direct_ple_continuation.v4",
        file_sha256=V4_PREFLIGHT_RESULT_SHA256,
        receipt_sha256=V4_PREFLIGHT_RESULT_RECEIPT,
    )
    frozen_files = {
        V4_PREFLIGHT_LOG: V4_PREFLIGHT_LOG_SHA256,
        V4_INVALID_TRAIN / "input_binding.json": V4_INVALID_TRAIN_INPUT_SHA256,
        V4_INVALID_TRAIN / "training_progress.jsonl": V4_INVALID_TRAIN_PROGRESS_SHA256,
        V4_INVALID_TRAIN_LOG: V4_INVALID_TRAIN_LOG_SHA256,
    }
    for path, expected_sha256 in frozen_files.items():
        if evaluator.sha256_file(path.resolve(strict=True)) != expected_sha256:
            raise ValueError(f"V4 binding-failure evidence differs: {path}")
    input_binding = json.loads(
        (V4_INVALID_TRAIN / "input_binding.json").read_text(encoding="utf-8")
    )
    progress = [
        json.loads(line)
        for line in (V4_INVALID_TRAIN / "training_progress.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
        if line.strip()
    ]
    if (
        input_binding.get("schema")
        != "rwkv_ms_natural_memory_native_rwkv_direct_ple_continuation_input.v4"
        or input_binding.get("protocol_payload_sha256") != V4_PROTOCOL_RECEIPT
        or input_binding.get("learning_rate") != LEARNING_RATE
        or len(progress) != 1
        or progress[0].get("schema")
        != "rwkv_ms_natural_memory_native_rwkv_direct_ple_continuation_step.v4"
        or progress[0].get("step") != 1
        or progress[0].get("learning_rate") != 1e-4
        or progress[0].get("protocol_payload_sha256")
        != V4_CAPTURED_PARENT_PROTOCOL_SHA256
    ):
        raise ValueError("V4 captured-default binding failure is not reproduced")


def validate_progress_bindings(
    output_dir: Path,
    *,
    updates: int,
) -> Mapping[str, Any]:
    progress_path = output_dir / "training_progress.jsonl"
    records = [
        json.loads(line)
        for line in progress_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    expected_controls = list(common.CONDITIONS[1:])
    for index, record in enumerate(records, start=1):
        baseline_anchor = record.get("baseline_anchor", {})
        if (
            record.get("schema") != STEP_SCHEMA
            or record.get("step") != index
            or record.get("learning_rate") != LEARNING_RATE
            or record.get("max_gradient_norm") != MAX_GRAD_NORM
            or record.get("contrast_margin") != MARGIN
            or record.get("control_weights") != CONTROL_WEIGHTS
            or record.get("always_active_controls") != expected_controls
            or record.get("backward_control_names")
            != list(direct.BACKWARD_CONTROL_NAMES)
            or record.get("learning_rate_multipliers") != {}
            or record.get("protocol_payload_sha256") != PROTOCOL_PAYLOAD_SHA256
            or baseline_anchor.get("weight") != 0.0
        ):
            raise ValueError(f"Continuation step {index} binding differs")
    if len(records) != updates:
        raise ValueError(
            f"Continuation progress rows differ: {len(records)} != {updates}"
        )
    return {
        "passed": True,
        "rows": len(records),
        "progress_sha256": evaluator.sha256_file(progress_path),
        "learning_rate": LEARNING_RATE,
        "max_gradient_norm": MAX_GRAD_NORM,
        "contrast_margin": MARGIN,
        "control_weights": CONTROL_WEIGHTS,
        "protocol_payload_sha256": PROTOCOL_PAYLOAD_SHA256,
    }


def validate_predecessor() -> Mapping[str, Any]:
    predecessor = evaluator.validate_training_result(
        PREDECESSOR_RESULT,
        PREDECESSOR_ADAPTER,
    )
    if (
        evaluator.sha256_file(PREDECESSOR_RESULT) != PREDECESSOR_RESULT_SHA256
        or predecessor["receipt"]["payload_sha256"]
        != PREDECESSOR_RESULT_RECEIPT
        or predecessor["training"]["final_adapter_sha256"]
        != PREDECESSOR_ADAPTER_FP32_SNAPSHOT_SHA256
    ):
        raise ValueError("Direct PLE continuation predecessor differs")
    return predecessor


def continuation_config() -> Any:
    source = json.loads(
        (PREDECESSOR_ADAPTER / "delta_mem_config.json").read_text(encoding="utf-8")
    )
    return direct.HFDeltaMemConfig.from_dict(
        {
            **source,
            "rwkv_ms_ple_gain": PLE_GAIN,
            "rwkv_ms_ple_input_gain": PLE_INPUT_GAIN,
        }
    )


def load_predecessor(model: torch.nn.Module) -> Mapping[str, Any]:
    predecessor = validate_predecessor()
    source = torch.load(
        PREDECESSOR_ADAPTER / "delta_mem_adapter.pt",
        map_location="cpu",
        weights_only=True,
    )
    expected = direct.get_delta_mem_state_dict(model)
    named_parameters = dict(model.named_parameters())
    if set(source) != set(expected):
        missing = sorted(set(expected) - set(source))
        unexpected = sorted(set(source) - set(expected))
        raise ValueError(
            f"Continuation predecessor tensor set differs: missing={missing[:8]} "
            f"unexpected={unexpected[:8]}"
        )
    copied = 0
    promoted_before_copy = 0
    for key, expected_value in expected.items():
        value = source[key]
        parameter = named_parameters.get(key)
        if parameter is None:
            raise ValueError(f"Continuation model parameter is missing: {key}")
        if tuple(value.shape) != tuple(expected_value.shape):
            raise ValueError(f"Continuation predecessor shape differs: {key}")
        if value.dtype == torch.float32 and parameter.dtype != torch.float32:
            parameter.data = parameter.data.float()
            promoted_before_copy += 1
        parameter.data.copy_(value.to(device=parameter.device, dtype=parameter.dtype))
        copied += 1
    digest = direct.runtime._state_dict_sha256(
        {key: value.float() for key, value in source.items()}
    )
    if digest != predecessor["training"]["final_adapter_sha256"]:
        raise ValueError("Continuation predecessor state digest differs")
    loaded_snapshot = direct.trainer.snapshot_delta_mem_weights(model)
    loaded_digest = direct.runtime._state_dict_sha256(loaded_snapshot)
    if loaded_digest != digest:
        mismatches = [
            key
            for key, value in source.items()
            if not torch.equal(loaded_snapshot[key], value.float())
        ]
        raise ValueError(
            "Continuation loaded model differs from predecessor state: "
            f"mismatches={len(mismatches)} sample={mismatches[:8]}"
        )
    return {
        "source": str(PREDECESSOR_ADAPTER.resolve(strict=True)),
        "source_result": str(PREDECESSOR_RESULT.resolve(strict=True)),
        "source_result_sha256": PREDECESSOR_RESULT_SHA256,
        "source_result_receipt": PREDECESSOR_RESULT_RECEIPT,
        "source_adapter_sha256": digest,
        "loaded_adapter_sha256": loaded_digest,
        "copied_parameter_tensors": copied,
        "promoted_before_copy_parameter_tensors": promoted_before_copy,
        "all_predecessor_parameters_loaded_exactly": True,
    }


def preserve_rwkv_factors(model: torch.nn.Module) -> Mapping[str, Any]:
    state = direct.get_delta_mem_state_dict(model)
    factors = {
        key: value
        for key, value in state.items()
        if any(key.endswith(f".{name}") for name in ("w1", "a1", "g1"))
    }
    expected = common.EXPECTED_LAYERS * 3
    if len(factors) != expected:
        raise ValueError(
            f"Continuation RWKV low-rank factor count differs: {len(factors)}"
        )
    return {
        "rule": "preserve_predecessor_without_rebootstrap",
        "parameter_tensors": len(factors),
        "state_sha256": direct.runtime._state_dict_sha256(factors),
    }


def continuation_schedule(
    rows_by_task: Mapping[str, Sequence[common.SourceRow]],
    *,
    updates: int,
) -> tuple[tuple[common.ScheduledRow, ...], list[dict[str, Any]]]:
    predecessor_schedule, predecessor_payload = ORIGINAL_BUILD_TRAINING_SCHEDULE(
        rows_by_task,
        updates=UPDATES,
    )
    excluded = {
        task: {
            ordinal
            for row in predecessor_schedule
            if row.target.task == task
            for ordinal in (
                row.target.source_ordinal,
                row.donor.source_ordinal,
            )
        }
        for task in common.TASKS
    }
    remaining = {
        task: tuple(
            row
            for row in rows_by_task[task]
            if row.source_ordinal not in excluded[task]
        )
        for task in common.TASKS
    }
    schedule, payload = ORIGINAL_BUILD_TRAINING_SCHEDULE(
        remaining,
        updates=updates,
    )
    overlap = [
        (row.target.task, row.target.source_ordinal, row.donor.source_ordinal)
        for row in schedule
        if row.target.source_ordinal in excluded[row.target.task]
        or row.donor.source_ordinal in excluded[row.target.task]
    ]
    if overlap:
        raise ValueError(f"Continuation schedule overlaps v10 rows: {overlap[:8]}")
    global SCHEDULE_AUDIT
    SCHEDULE_AUDIT = {
        "policy": "exclude_every_v10_target_and_donor",
        "predecessor_schedule_sha256": evaluator.canonical_sha256(
            predecessor_payload
        ),
        "excluded_rows": {
            task: len(excluded[task]) for task in common.TASKS
        },
        "remaining_rows": {
            task: len(remaining[task]) for task in common.TASKS
        },
        "continuation_schedule_sha256": evaluator.canonical_sha256(payload),
        "overlap_rows": overlap,
    }
    return schedule, payload


def continuation_gradient_audit(
    named_trainable: Sequence[tuple[str, torch.nn.Parameter]],
) -> Mapping[str, Any]:
    audit = dict(ORIGINAL_GRADIENT_AUDIT(named_trainable))
    families = audit["families"]
    core = families.get("rwkv_core", {})
    non_core_fully_active = all(
        family["active"] == family["tensors"]
        for name, family in families.items()
        if name != "rwkv_core"
    )
    every_family_active = all(
        family["active"] > 0 for family in families.values()
    )
    passed = bool(
        audit["global_finite_fp32_tensors"] == audit["trainable_tensors"]
        and every_family_active
        and non_core_fully_active
        and core.get("tensors") == 882
        and int(core.get("active", 0)) >= 672
    )
    audit["passed"] = passed
    audit["continuation_criterion"] = {
        "all_gradients_finite_fp32": (
            audit["global_finite_fp32_tensors"] == audit["trainable_tensors"]
        ),
        "every_parameter_family_has_nonzero_gradient": every_family_active,
        "all_non_core_tensors_have_nonzero_gradient": non_core_fully_active,
        "minimum_nonzero_rwkv_core_tensors": 672,
        "nonzero_rwkv_core_tensors": int(core.get("active", 0)),
        "total_rwkv_core_tensors": int(core.get("tensors", 0)),
    }
    return audit


def validate_protocol(updates: int) -> Mapping[str, Any]:
    if os.environ.get("HF_ENDPOINT") != common.HF_MIRROR_ENDPOINT:
        raise ValueError(f"HF_ENDPOINT must be exactly {common.HF_MIRROR_ENDPOINT}")
    if updates not in {PREFLIGHT_UPDATES, UPDATES}:
        raise ValueError("Direct PLE continuation updates must be 1 or 32")
    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    receipt = protocol.get("receipt")
    unsigned = dict(protocol)
    unsigned.pop("receipt", None)
    architecture = protocol.get("architecture", {})
    training = protocol.get("training", {})
    frozen = protocol.get("frozen_inputs", {})
    if (
        not isinstance(receipt, Mapping)
        or evaluator.canonical_sha256(unsigned) != PROTOCOL_PAYLOAD_SHA256
        or receipt.get("payload_sha256") != PROTOCOL_PAYLOAD_SHA256
        or protocol.get("schema") != SCHEMA
        or frozen.get("base_model_revision") != common.BASE_MODEL_REVISION
        or frozen.get("base_model_weights_sha256")
        != common.BASE_MODEL_WEIGHTS_SHA256
        or frozen.get("predecessor_result_sha256")
        != PREDECESSOR_RESULT_SHA256
        or frozen.get("predecessor_result_receipt")
        != PREDECESSOR_RESULT_RECEIPT
        or frozen.get("predecessor_adapter_fp32_snapshot_sha256")
        != PREDECESSOR_ADAPTER_FP32_SNAPSHOT_SHA256
        or frozen.get("split_manifest_receipt") != common.SPLIT_MANIFEST_RECEIPT
        or frozen.get("open_split_receipt") != common.OPEN_SPLIT_RECEIPT
        or frozen.get("v1_protocol_receipt") != V1_PROTOCOL_RECEIPT
        or frozen.get("v1_preflight_failure_log_sha256")
        != V1_FAILURE_LOG_SHA256
        or evaluator.sha256_file(V1_FAILURE_LOG.resolve(strict=True))
        != V1_FAILURE_LOG_SHA256
        or frozen.get("v2_protocol_receipt") != V2_PROTOCOL_RECEIPT
        or frozen.get("v2_preflight_result_sha256")
        != V2_PREFLIGHT_RESULT_SHA256
        or frozen.get("v2_preflight_result_receipt")
        != V2_PREFLIGHT_RESULT_RECEIPT
        or frozen.get("v2_preflight_failure_log_sha256")
        != V2_FAILURE_LOG_SHA256
        or evaluator.sha256_file(V2_FAILURE_LOG.resolve(strict=True))
        != V2_FAILURE_LOG_SHA256
        or frozen.get("v3_protocol_receipt") != V3_PROTOCOL_RECEIPT
        or frozen.get("v4_protocol_receipt") != V4_PROTOCOL_RECEIPT
        or frozen.get("v4_runner_sha256") != V4_RUNNER_SHA256
        or frozen.get("v4_preflight_result_sha256")
        != V4_PREFLIGHT_RESULT_SHA256
        or frozen.get("v4_preflight_result_receipt")
        != V4_PREFLIGHT_RESULT_RECEIPT
        or frozen.get("v4_preflight_log_sha256") != V4_PREFLIGHT_LOG_SHA256
        or frozen.get("v4_invalid_train_input_binding_sha256")
        != V4_INVALID_TRAIN_INPUT_SHA256
        or frozen.get("v4_invalid_train_progress_sha256")
        != V4_INVALID_TRAIN_PROGRESS_SHA256
        or frozen.get("v4_invalid_train_log_sha256")
        != V4_INVALID_TRAIN_LOG_SHA256
        or frozen.get("protected_splits_opened") != []
        or architecture.get("hybrid_mode") != direct.HYBRID_MODE
        or architecture.get("ple_gain") != PLE_GAIN
        or architecture.get("ple_input_gain") != PLE_INPUT_GAIN
        or architecture.get("task_router") is not False
        or architecture.get("template_matcher") is not False
        or architecture.get("dual_pass_selector") is not False
        or architecture.get("benchmark_specific_decoder") is not False
        or training.get("optimizer_updates") != UPDATES
        or training.get("preflight_optimizer_updates") != PREFLIGHT_UPDATES
        or training.get("learning_rate") != LEARNING_RATE
        or training.get("max_gradient_norm") != MAX_GRAD_NORM
        or training.get("contrast_margin") != MARGIN
        or training.get("control_weights") != CONTROL_WEIGHTS
        or training.get("always_active_controls") != list(common.CONDITIONS[1:])
        or training.get("backward_control_names")
        != list(direct.BACKWARD_CONTROL_NAMES)
        or training.get("precision_load_order")
        != (
            "Resolve actual model parameters by name, promote every fp32 predecessor "
            "destination before copying, copy all 1638 tensors, then require the "
            "loaded fp32 snapshot hash to equal the v10 final snapshot hash before "
            "training."
        )
        or training.get("protocol_argument_binding")
        != (
            "Pass learning_rate, max_grad_norm, margin, control_weights, and "
            "protocol_payload_sha256 explicitly to the shared trainer; reject "
            "the result unless every progress row exactly matches these "
            "preregistered values."
        )
        or training.get("first_update_gradient_audit")
        != {
            "all_gradients_finite_fp32": True,
            "every_parameter_family_has_nonzero_gradient": True,
            "all_non_core_tensors_have_nonzero_gradient": True,
            "minimum_nonzero_rwkv_core_tensors": 672,
            "total_rwkv_core_tensors": 882,
        }
        or training.get("final_rows_opened_during_training") is not False
        or protocol.get("required_gates", {}).get(
            "every_step_binding_matches_protocol"
        )
        is not True
        or protocol.get("protected_splits_opened_by_this_protocol") != []
    ):
        raise ValueError("Direct PLE continuation protocol differs")
    validate_predecessor()
    validate_v4_binding_failure()
    validate_signed_artifact(
        V2_PREFLIGHT_RESULT,
        schema="rwkv_ms_natural_memory_native_rwkv_direct_ple_continuation.v2",
        file_sha256=V2_PREFLIGHT_RESULT_SHA256,
        receipt_sha256=V2_PREFLIGHT_RESULT_RECEIPT,
    )
    for path, schema, file_sha256, receipt_sha256 in DIAGNOSTICS:
        validate_signed_artifact(
            path,
            schema=schema,
            file_sha256=file_sha256,
            receipt_sha256=receipt_sha256,
        )
    return protocol


@contextmanager
def bindings() -> Iterator[None]:
    direct_values = {
        "PROTOCOL": direct.PROTOCOL,
        "PROTOCOL_PAYLOAD_SHA256": direct.PROTOCOL_PAYLOAD_SHA256,
        "SCHEMA": direct.SCHEMA,
        "STEP_SCHEMA": direct.STEP_SCHEMA,
        "INPUT_SCHEMA": direct.INPUT_SCHEMA,
        "SEED": direct.SEED,
        "UPDATES": direct.UPDATES,
        "PRELIGHT_UPDATES": direct.PRELIGHT_UPDATES,
        "LEARNING_RATE": direct.LEARNING_RATE,
        "MAX_GRAD_NORM": direct.MAX_GRAD_NORM,
        "MARGIN": direct.MARGIN,
        "PLE_GAIN": direct.PLE_GAIN,
        "PLE_INPUT_GAIN": direct.PLE_INPUT_GAIN,
        "_config": direct._config,
        "_load_warmstart": direct._load_warmstart,
        "_bootstrap_rwkv_low_rank_factors": direct._bootstrap_rwkv_low_rank_factors,
        "_gradient_audit": direct._gradient_audit,
        "validate_training_protocol": direct.validate_training_protocol,
    }
    replacements = {
        "PROTOCOL": PROTOCOL,
        "PROTOCOL_PAYLOAD_SHA256": PROTOCOL_PAYLOAD_SHA256,
        "SCHEMA": SCHEMA,
        "STEP_SCHEMA": STEP_SCHEMA,
        "INPUT_SCHEMA": INPUT_SCHEMA,
        "SEED": SEED,
        "UPDATES": UPDATES,
        "PRELIGHT_UPDATES": PREFLIGHT_UPDATES,
        "LEARNING_RATE": LEARNING_RATE,
        "MAX_GRAD_NORM": MAX_GRAD_NORM,
        "MARGIN": MARGIN,
        "PLE_GAIN": PLE_GAIN,
        "PLE_INPUT_GAIN": PLE_INPUT_GAIN,
        "_config": continuation_config,
        "_load_warmstart": load_predecessor,
        "_bootstrap_rwkv_low_rank_factors": preserve_rwkv_factors,
        "_gradient_audit": continuation_gradient_audit,
        "validate_training_protocol": validate_protocol,
    }
    for name, value in replacements.items():
        setattr(direct, name, value)
    try:
        with direct.bindings():
            previous_schedule = common.build_training_schedule
            previous_train = direct.trainer.train

            def strengthened_train(*args: Any, **kwargs: Any) -> Mapping[str, Any]:
                kwargs["learning_rate"] = LEARNING_RATE
                kwargs["max_grad_norm"] = MAX_GRAD_NORM
                kwargs["margin"] = MARGIN
                kwargs["control_weights"] = CONTROL_WEIGHTS
                kwargs["protocol_payload_sha256"] = PROTOCOL_PAYLOAD_SHA256
                return previous_train(*args, **kwargs)

            common.build_training_schedule = continuation_schedule
            direct.trainer.train = strengthened_train
            try:
                yield
            finally:
                direct.trainer.train = previous_train
                common.build_training_schedule = previous_schedule
    finally:
        for name, value in direct_values.items():
            setattr(direct, name, value)


def postprocess_result(
    result: Mapping[str, Any],
    *,
    output_dir: Path,
) -> Mapping[str, Any]:
    value = dict(result)
    training = dict(value["training"])
    training["step_binding_audit"] = validate_progress_bindings(
        output_dir,
        updates=int(value["updates"]),
    )
    value["training"] = training
    input_binding = dict(value["input_binding"])
    input_binding["continuation"] = {
        "predecessor_result": str(PREDECESSOR_RESULT.resolve(strict=True)),
        "predecessor_result_sha256": PREDECESSOR_RESULT_SHA256,
        "predecessor_result_receipt": PREDECESSOR_RESULT_RECEIPT,
        "ple_gain": PLE_GAIN,
        "ple_input_gain": PLE_INPUT_GAIN,
        "control_weights": CONTROL_WEIGHTS,
        "schedule_audit": SCHEDULE_AUDIT,
        "runner_sha256": evaluator.sha256_file(Path(__file__)),
    }
    value["input_binding"] = input_binding
    code_bindings = dict(value["code_bindings"])
    code_bindings["continuation_runner_sha256"] = evaluator.sha256_file(
        Path(__file__)
    )
    code_bindings["continuation_protocol_sha256"] = evaluator.sha256_file(
        PROTOCOL
    )
    value["code_bindings"] = code_bindings
    value.pop("receipt", None)
    value["receipt"] = {
        "algorithm": "sha256",
        "payload_scope": "canonical_result_without_receipt",
        "payload_sha256": evaluator.canonical_sha256(value),
    }
    evaluator.write_json(output_dir / "input_binding.json", input_binding)
    evaluator.write_json(output_dir / "result.json", value)
    return value


def run(
    *,
    context: distributed.DistributedTrainingContext,
    output_dir: Path,
    updates: int,
    base_model: Path,
) -> Mapping[str, Any]:
    with bindings():
        result = direct.trainer.run(
            context=context,
            output_dir=output_dir,
            updates=updates,
            base_model=base_model,
        )
        postprocess_error = None
        if context.is_primary:
            try:
                result = postprocess_result(
                    result,
                    output_dir=output_dir.expanduser().resolve(strict=True),
                )
            except BaseException as error:
                postprocess_error = error
        distributed.phase_consensus(
            context,
            phase="direct-ple-continuation-result-binding",
            error=postprocess_error,
        )
        return result


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--updates", type=int, required=True, choices=(1, 32))
    parser.add_argument("--base-model", type=Path, default=common.BASE_MODEL)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args(argv)


@record
def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    context = distributed.initialize_distributed_training(
        args.device,
        timeout_seconds=7200,
    )
    if context is None or context.world_size != direct.WORLD_SIZE:
        raise ValueError("Direct PLE continuation requires exactly four ranks")
    try:
        result = run(
            context=context,
            output_dir=args.output_dir,
            updates=args.updates,
            base_model=args.base_model,
        )
    finally:
        distributed.destroy_distributed_training(context)
    print(
        json.dumps(
            {
                "rank": context.process_rank,
                "status": result["status"],
                "passed": result["passed"],
                "result_receipt": (
                    result.get("receipt", {}).get("payload_sha256")
                    if context.is_primary
                    else None
                ),
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
