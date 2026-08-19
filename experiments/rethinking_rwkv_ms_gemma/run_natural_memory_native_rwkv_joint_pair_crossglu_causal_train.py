#!/usr/bin/env python3
"""Train and evaluate the signed joint-pair CrossGLU causal endpoint."""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import gc
import json
import os
from pathlib import Path
import sys
from typing import Any, Iterator, Mapping, Sequence

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

import torch
import torch.distributed as dist
from torch.distributed.elastic.multiprocessing.errors import record
from torch.utils.checkpoint import checkpoint


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[1]
for import_root in (SCRIPT_DIR, PROJECT_ROOT):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

from deltamem.core.delta import reset_delta_mem_states  # noqa: E402
from experiments.rethinking_rwkv_ms_gemma import (  # noqa: E402
    run_natural_memory_native_rwkv_joint_pair_crossglu_mechanics as mechanics,
)
from experiments.rethinking_rwkv_ms_gemma import (  # noqa: E402
    run_natural_memory_native_rwkv_query_state_bilinear_crossfit as crossfit,
)


causal_train = mechanics.causal_train
contrast = mechanics.contrast
distributed = mechanics.candidate.shared.distributed
evolution = mechanics.evolution
hardware = mechanics.hardware
value_identity = mechanics.value_identity

SCHEMA = "rwkv_ms_natural_memory_native_joint_pair_crossglu_causal_train.v1"
STEP_SCHEMA = "rwkv_ms_natural_memory_native_joint_pair_crossglu_causal_train_step.v1"
INPUT_SCHEMA = "rwkv_ms_natural_memory_native_joint_pair_crossglu_causal_train_input.v1"
PROTOCOL = SCRIPT_DIR / "natural_memory_native_rwkv_joint_pair_crossglu_causal_train_protocol_v1.json"
PROTOCOL_FILE_SHA256 = "8b85490ea1c46c38049bcc92deb535e0bc6eadae0cd5a194a61a6c744ee8bbdb"
PROTOCOL_PAYLOAD_SHA256 = "80901c5eed1d95f66b2e89af25c2d0b0be929a29059f5e08095cd8110cf46edd"
MECHANICS_RESULT = SCRIPT_DIR / "local_artifacts/natural_memory_native_rwkv_joint_pair_crossglu_mechanics_v22/result.json"
MECHANICS_RESULT_FILE_SHA256 = "dba313f9a2a441ed1c81fac0a83cc42b4ca6f94975d8e8b5b483db654296bcc3"
MECHANICS_RESULT_RECEIPT = "ba9a986ef0bc937c652e82204c8edb3efc80855136e1261c30c9fe15e3954ddd"
CROSSFIT_RESULT = crossfit.SCRIPT_DIR / "local_artifacts/natural_memory_native_rwkv_query_state_bilinear_crossfit_v1/result.json"
CROSSFIT_RESULT_FILE_SHA256 = "5e41c4569273fd5841381fcb6c5738b26212dd326b4f7cf56b589528df346ba3"
CROSSFIT_RESULT_RECEIPT = "89392eaeffa50c0bed9109fd8db3d33a5625eb4ff7117f81d710cf9b5be93945"
CROSSFIT_SPLIT_PAYLOAD_SHA256 = "3b6306fdd2120faf9b2a20f370ad08a7fa7eaffa347e6f8deaa91e3cc1a7bf3c"
RECONSTRUCTION_STATE_SHA256 = "1bbb5e8ff1b61aa2309c81d4a0235a9b74e0c8eb855604c03c6bbea4b939790f"
TRAINING_SCHEDULE_PREFIX_SHA256 = "108b83ce2f2dd590c6ad45c7d46affeb4fd01afddee08dce35b2d5d18219876d"
HELDOUT_PAYLOAD_SHA256 = "84ae348079351f96f70172bfc8c02b3ae44f54bc74db260e8ba4d7c9164117a2"
HF_MIRROR_ENDPOINT = "https://hf-mirror.com"
BASE_MODEL = contrast.BASE_MODEL
DATASET_ROOT = contrast.NATIVE_DATASET_ROOT
WORLD_SIZE = 4
SEED = 116
UPDATES = 8
GLOBAL_BATCH_SIZE = 4
LOCAL_ROWS = 1
TRAINABLE_TENSORS = 126
TRAINABLE_ELEMENTS = 172032
PASS_STATUS = "joint_pair_crossglu_heldout_causal_passed_generation_blocked"
FAIL_STATUS = "joint_pair_crossglu_heldout_causal_failed_generation_blocked"
HELDOUT_SOURCES = (
    7, 12, 14, 19, 24, 26, 29, 30, 33, 52, 59, 62, 66, 73, 80, 96,
    100, 101, 119, 121, 128, 129, 139, 144, 152, 172, 177, 181, 201,
    211, 216, 218, 225, 233, 236, 247, 255, 259, 279, 296, 321, 322,
    356, 358,
)


def canonical_sha256(value: Any) -> str:
    return distributed.canonical_sha256(value)


def sha256_file(path: Path) -> str:
    return mechanics.sha256_file(path)


def write_json(path: Path, value: Mapping[str, Any]) -> None:
    if path.exists():
        raise ValueError(f"Joint-pair causal artifact must be fresh: {path}")
    path.write_text(json.dumps(value, ensure_ascii=True, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _validate_receipted_result(path: Path, *, file_sha256: str, receipt_sha256: str) -> Mapping[str, Any]:
    if sha256_file(path) != file_sha256:
        raise ValueError(f"Signed result file differs: {path}")
    result = json.loads(path.read_text(encoding="utf-8"))
    unsigned = dict(result)
    receipt = unsigned.pop("receipt", {})
    if canonical_sha256(unsigned) != receipt_sha256 or receipt.get("payload_sha256") != receipt_sha256:
        raise ValueError(f"Signed result receipt differs: {path}")
    return result


def validate_protocol() -> tuple[Mapping[str, Any], Mapping[str, Any], Mapping[str, Any]]:
    if sha256_file(PROTOCOL) != PROTOCOL_FILE_SHA256:
        raise ValueError("Joint-pair causal protocol file differs")
    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    unsigned = dict(protocol)
    receipt = unsigned.pop("receipt", {})
    if (
        canonical_sha256(unsigned) != PROTOCOL_PAYLOAD_SHA256
        or receipt.get("payload_sha256") != PROTOCOL_PAYLOAD_SHA256
        or protocol.get("generation_authorized") is not False
        or protocol.get("causal_endpoint_authorized") is not True
        or protocol.get("delta_mem_adapter_saved") is not False
        or protocol.get("protected_splits_opened_by_this_protocol") != []
    ):
        raise ValueError("Joint-pair causal protocol differs")
    training = protocol.get("training", {})
    architecture = protocol.get("architecture", {})
    endpoint = protocol.get("heldout_causal_endpoint", {})
    if (
        training.get("optimizer_updates") != UPDATES
        or training.get("global_batch_rows") != GLOBAL_BATCH_SIZE
        or training.get("local_rows_per_rank") != LOCAL_ROWS
        or training.get("optimizer_state_cpu_offload_enabled") is not True
        or training.get("all_126_bridge_tensors_require_finite_nonzero_global_gradient_each_update") is not True
        or architecture.get("trainable_parameter_tensors") != TRAINABLE_TENSORS
        or architecture.get("trainable_parameter_elements") != TRAINABLE_ELEMENTS
        or endpoint.get("rows") != len(HELDOUT_SOURCES)
        or endpoint.get("source_indices") != list(HELDOUT_SOURCES)
        or endpoint.get("source_donor_payload_sha256") != HELDOUT_PAYLOAD_SHA256
    ):
        raise ValueError("Joint-pair causal training contract differs")
    mechanics_result = _validate_receipted_result(
        MECHANICS_RESULT,
        file_sha256=MECHANICS_RESULT_FILE_SHA256,
        receipt_sha256=MECHANICS_RESULT_RECEIPT,
    )
    required_checks = tuple(mechanics_result.get("checks", {}))
    if (
        mechanics_result.get("status") != "joint_pair_crossglu_mechanics_passed_generation_blocked"
        or mechanics_result.get("passed") is not True
        or mechanics_result.get("causal_endpoint_authorized") is not True
        or not all(mechanics_result.get("checks", {}).get(name) is True for name in required_checks)
        or mechanics_result.get("protected_splits_opened") != []
    ):
        raise ValueError("Joint-pair mechanics did not authorize training")
    crossfit_result = _validate_receipted_result(
        CROSSFIT_RESULT,
        file_sha256=CROSSFIT_RESULT_FILE_SHA256,
        receipt_sha256=CROSSFIT_RESULT_RECEIPT,
    )
    split = crossfit_result.get("crossfit_split", {})
    split_unsigned = dict(split)
    split_receipt = split_unsigned.pop("payload_sha256", None)
    if (
        crossfit_result.get("status") != "bilinear_crossfit_passed_causal_training_design_authorized"
        or crossfit_result.get("passed") is not True
        or split_receipt != CROSSFIT_SPLIT_PAYLOAD_SHA256
        or canonical_sha256(split_unsigned) != CROSSFIT_SPLIT_PAYLOAD_SHA256
        or split.get("heldout_sources") != list(HELDOUT_SOURCES)
    ):
        raise ValueError("Signed bilinear cross-fit binding differs")
    return protocol, mechanics_result, crossfit_result


def _recompute_batch(batch: Any, tensors: Sequence[torch.Tensor]) -> Any:
    return evolution.NativeFullRowBatch(
        examples=batch.examples,
        write_input_ids=tensors[0],
        write_attention_mask=tensors[1],
        read_input_ids=tensors[2],
        read_attention_mask=tensors[3],
        labels=batch.labels,
    )


def _set_original_write_mode(modules: Sequence[tuple[str, Any]]) -> None:
    mechanics._set_original_write_mode(modules)


def _set_joint_read_mode(modules: Sequence[tuple[str, Any]]) -> None:
    mechanics._set_joint_read_mode(modules)
    for _, module in modules:
        module.rwkv_ms_outer_ffn_enabled = False


def _set_queries(modules: Sequence[tuple[str, Any]], values: Mapping[str, torch.Tensor], seq_len: int) -> None:
    mechanics._set_queries(modules, values, seq_len)


def _normalize_logits(logits: torch.Tensor) -> torch.Tensor:
    return logits.unsqueeze(0) if logits.ndim == 2 else logits


def _prepare_read(
    model: torch.nn.Module,
    modules: Sequence[tuple[str, Any]],
    *,
    projected: Mapping[str, Mapping[str, torch.Tensor]],
    recurrent: Mapping[str, Mapping[str, torch.Tensor]],
    query: Mapping[str, torch.Tensor],
    seq_len: int,
    rotate_recurrent_layers: bool,
) -> bool:
    fixed = causal_train.install_intervened_state(
        modules,
        projected=projected,
        recurrent=recurrent,
        rotate_recurrent_layers=rotate_recurrent_layers,
    )
    _set_joint_read_mode(modules)
    _set_queries(modules, query, seq_len)
    value_identity.set_fixed_target_values(model, dict(query))
    return bool(fixed)


def _zero_recurrent(state: Mapping[str, Mapping[str, torch.Tensor]]) -> dict[str, dict[str, torch.Tensor]]:
    return mechanics.zero_recurrent(state)


def _promote_trainable_bridge_parameters(model: torch.nn.Module) -> None:
    for name, parameter in model.named_parameters():
        if "rwkv_joint_pair_crossglu" in name and ".identity." not in name:
            parameter.data = parameter.data.float()


def _read_projected_only(model: torch.nn.Module, target: Any, modules: Sequence[tuple[str, Any]]) -> torch.Tensor:
    saved = [(module, module.memory_readout_mode, module.rwkv_ms_hybrid_mode, module.rwkv_ms_hybrid_gain) for _, module in modules]
    for module, _, _, _ in saved:
        module.memory_readout_mode = "projected_kv_rwkv_hybrid"
        module.rwkv_ms_hybrid_mode = "residual"
        module.rwkv_ms_outer_ffn_enabled = False
    try:
        return _normalize_logits(evolution._native_read(model, target, dtype=torch.bfloat16))
    finally:
        for module, readout_mode, hybrid_mode, gain in saved:
            module.memory_readout_mode = readout_mode
            module.rwkv_ms_hybrid_mode = hybrid_mode
            module.rwkv_ms_hybrid_gain = gain


def checkpointed_native_write_read(model: torch.nn.Module, batch: Any, *, dtype: torch.dtype) -> tuple[Mapping[str, Any], torch.Tensor]:
    modules = causal_train.ordered_modules(model)

    def write_read(*tensors: torch.Tensor) -> torch.Tensor:
        target = _recompute_batch(batch, tensors)
        _set_original_write_mode(modules)
        value_identity.clear(model)
        evolution._native_write(model, target, dtype=dtype)
        correct_state = causal_train.capture_online_state_references(modules)
        target_values = value_identity.capture_write_values(model)
        value_identity.set_fixed_target_values(model, target_values)
        fixed = _prepare_read(
            model,
            modules,
            projected=correct_state,
            recurrent=correct_state,
            query=target_values,
            seq_len=int(target.read_input_ids.shape[1]),
            rotate_recurrent_layers=False,
        )
        if not fixed:
            raise RuntimeError("Joint-pair projected carrier changed on correct read")
        return _normalize_logits(evolution._native_read(model, target, dtype=dtype))

    logits = checkpoint(
        write_read,
        batch.write_input_ids,
        batch.write_attention_mask,
        batch.read_input_ids,
        batch.read_attention_mask,
        use_reentrant=False,
    )
    occupied_rows = 0
    occupied_total = 0
    for _, module in modules:
        occupied = module.projected_kv_occupied
        if occupied is None or occupied.ndim != 2:
            raise RuntimeError("Joint-pair checkpoint did not preserve occupancy")
        occupied_rows += int(occupied.any(dim=-1).sum().item())
        occupied_total += int(occupied.size(0))
    return {"occupied_rows": occupied_rows, "occupied_total": occupied_total}, logits


def checkpointed_intervened_write_read(
    model: torch.nn.Module,
    target_batch: Any,
    *,
    donor_batch: Any | None,
    rotate_recurrent_layers: bool,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, Mapping[str, bool]]:
    modules = causal_train.ordered_modules(model)
    audit = {"projected_carrier_references_fixed": True}

    def write_read(*tensors: torch.Tensor) -> torch.Tensor:
        target = _recompute_batch(target_batch, tensors)
        _set_original_write_mode(modules)
        value_identity.clear(model)
        evolution._native_write(model, target, dtype=dtype)
        correct_state = causal_train.capture_online_state_references(modules)
        target_values = value_identity.capture_write_values(model)
        value_identity.set_fixed_target_values(model, target_values)
        recurrent_state = correct_state
        if donor_batch is not None:
            donor = evolution.NativeFullRowBatch(
                examples=donor_batch.examples,
                write_input_ids=tensors[4],
                write_attention_mask=tensors[5],
                read_input_ids=tensors[2],
                read_attention_mask=tensors[3],
                labels=target_batch.labels,
            )
            _set_original_write_mode(modules)
            evolution._native_write(model, donor, dtype=dtype)
            recurrent_state = causal_train.capture_online_state_references(modules)
        audit["projected_carrier_references_fixed"] = bool(
            audit["projected_carrier_references_fixed"]
            and _prepare_read(
                model,
                modules,
                projected=correct_state,
                recurrent=recurrent_state,
                query=target_values,
                seq_len=int(target.read_input_ids.shape[1]),
                rotate_recurrent_layers=rotate_recurrent_layers,
            )
        )
        return _normalize_logits(evolution._native_read(model, target, dtype=dtype))

    inputs = [
        target_batch.write_input_ids,
        target_batch.write_attention_mask,
        target_batch.read_input_ids,
        target_batch.read_attention_mask,
    ]
    if donor_batch is not None:
        inputs.extend((donor_batch.write_input_ids, donor_batch.write_attention_mask))
    return checkpoint(write_read, *inputs, use_reentrant=False), audit


def evaluate_condition_ce(model: torch.nn.Module, batch: Any, *, no_state: bool, dtype: torch.dtype) -> tuple[float, int]:
    if not no_state:
        return contrast.evaluate_condition_ce_original(model, batch, no_state=False, dtype=dtype)
    modules = causal_train.ordered_modules(model)
    try:
        with torch.no_grad():
            _set_original_write_mode(modules)
            value_identity.clear(model)
            evolution._native_write(model, batch, dtype=dtype)
            correct_state = causal_train.capture_online_state_references(modules)
            target_values = value_identity.capture_write_values(model)
            value_identity.set_fixed_target_values(model, target_values)
            fixed = _prepare_read(
                model,
                modules,
                projected=correct_state,
                recurrent=_zero_recurrent(correct_state),
                query=target_values,
                seq_len=int(batch.read_input_ids.shape[1]),
                rotate_recurrent_layers=False,
            )
            if not fixed:
                raise RuntimeError("Zero-recurrent projected carrier changed")
            logits = _normalize_logits(evolution._native_read(model, batch, dtype=dtype))
            return contrast.detached_answer_ce(logits, batch.labels)
    finally:
        value_identity.clear(model)
        reset_delta_mem_states(model)


def gradient_audit(named: Sequence[tuple[str, torch.nn.Parameter]]) -> Mapping[str, Any]:
    if len(named) != TRAINABLE_TENSORS:
        raise RuntimeError(f"Expected {TRAINABLE_TENSORS} bridge gradients, got {len(named)}")
    finite_values: list[int] = []
    active_values: list[int] = []
    for name, parameter in named:
        if "rwkv_joint_pair_crossglu" not in name or ".identity." in name:
            raise RuntimeError(f"Non-bridge parameter entered gradient audit: {name}")
        gradient = parameter.grad
        finite = bool(gradient is not None and gradient.dtype == torch.float32 and torch.isfinite(gradient).all().item())
        finite_values.append(int(finite))
        active_values.append(int(finite and bool(gradient.abs().max().gt(0).item())))
    finite_tensor = torch.tensor(finite_values, device=named[0][1].device, dtype=torch.int32)
    active_tensor = torch.tensor(active_values, device=named[0][1].device, dtype=torch.int32)
    dist.all_reduce(finite_tensor, op=dist.ReduceOp.SUM)
    dist.all_reduce(active_tensor, op=dist.ReduceOp.SUM)
    return {
        "trainable_tensors": len(named),
        "global_finite_fp32_tensors": int(finite_tensor.gt(0).sum().item()),
        "global_finite_nonzero_tensors": int(active_tensor.gt(0).sum().item()),
        "passed": bool(finite_tensor.gt(0).all().item() and active_tensor.gt(0).all().item()),
    }


def _promote_bridge_parameters_to_fp32(model: torch.nn.Module) -> None:
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if "rwkv_joint_pair_crossglu" not in name or ".identity." in name:
            raise RuntimeError(f"Unexpected trainable parameter during bridge promotion: {name}")
        if not parameter.is_floating_point():
            raise TypeError(f"Trainable bridge parameter must be floating point: {name}")
        parameter.data = parameter.data.float()


@contextmanager
def training_bindings() -> Iterator[None]:
    names = (
        "GLOBAL_BATCH_SIZE", "LOCAL_ROWS", "MIN_ACCEPTED_ROWS_PER_UPDATE",
        "MAX_TOTAL_REJECTED_ROWS", "OFFLOAD_OPTIMIZER_STATE_DURING_ROWS",
        "SERIALIZE_CONTROL_BRANCH_GRAPHS", "FIRST_UPDATE_GRADIENT_AUDITOR",
        "STEP_SCHEMA", "PROTOCOL_PAYLOAD_SHA256",
    )
    previous = {name: getattr(causal_train, name) for name in names}
    previous_correct = evolution.checkpointed_native_write_read
    previous_control = causal_train.checkpointed_intervened_write_read
    previous_condition = contrast.evaluate_condition_ce
    try:
        causal_train.GLOBAL_BATCH_SIZE = GLOBAL_BATCH_SIZE
        causal_train.LOCAL_ROWS = LOCAL_ROWS
        causal_train.MIN_ACCEPTED_ROWS_PER_UPDATE = GLOBAL_BATCH_SIZE
        causal_train.MAX_TOTAL_REJECTED_ROWS = 0
        causal_train.OFFLOAD_OPTIMIZER_STATE_DURING_ROWS = True
        causal_train.SERIALIZE_CONTROL_BRANCH_GRAPHS = True
        causal_train.FIRST_UPDATE_GRADIENT_AUDITOR = gradient_audit
        causal_train.STEP_SCHEMA = STEP_SCHEMA
        causal_train.PROTOCOL_PAYLOAD_SHA256 = PROTOCOL_PAYLOAD_SHA256
        evolution.checkpointed_native_write_read = checkpointed_native_write_read
        causal_train.checkpointed_intervened_write_read = checkpointed_intervened_write_read
        contrast.evaluate_condition_ce_original = previous_condition
        contrast.evaluate_condition_ce = evaluate_condition_ce
        yield
    finally:
        contrast.evaluate_condition_ce = previous_condition
        if hasattr(contrast, "evaluate_condition_ce_original"):
            delattr(contrast, "evaluate_condition_ce_original")
        causal_train.checkpointed_intervened_write_read = previous_control
        evolution.checkpointed_native_write_read = previous_correct
        for name, value in previous.items():
            setattr(causal_train, name, value)


def heldout_payload(rows_by_source: Mapping[int, Mapping[str, Any]], donor_mapping: Mapping[int, int]) -> list[dict[str, Any]]:
    return [
        {
            "source_index": source,
            "row_sha256": rows_by_source[source]["row_sha256"],
            "donor_source_index": int(donor_mapping[source]),
            "donor_row_sha256": rows_by_source[int(donor_mapping[source])]["row_sha256"],
        }
        for source in HELDOUT_SOURCES
    ]


def _endpoint_read(
    model: torch.nn.Module,
    target: Any,
    modules: Sequence[tuple[str, Any]],
    projected: Mapping[str, Mapping[str, torch.Tensor]],
    recurrent: Mapping[str, Mapping[str, torch.Tensor]],
    query: Mapping[str, torch.Tensor],
    *,
    seq_len: int,
    rotate_recurrent_layers: bool = False,
) -> tuple[torch.Tensor, bool, Mapping[str, torch.Tensor], Mapping[str, torch.Tensor], torch.Tensor, torch.Tensor]:
    reset_delta_mem_states(model)
    value_identity.set_fixed_target_values(model, dict(query))
    fixed = _prepare_read(
        model,
        modules,
        projected=projected,
        recurrent=recurrent,
        query=query,
        seq_len=seq_len,
        rotate_recurrent_layers=rotate_recurrent_layers,
    )
    with torch.no_grad():
        logits = _normalize_logits(evolution._native_read(model, target, dtype=torch.bfloat16))
        query_vectors, state_vectors = crossfit.aggregate_vectors(
            value_identity.capture(model), target.labels
        )
    gates: dict[str, torch.Tensor] = {}
    values: dict[str, torch.Tensor] = {}
    for name, module in modules:
        if module.rwkv_joint_pair_crossglu_last_gate is None or module.rwkv_joint_pair_crossglu_last_value is None:
            raise RuntimeError(f"Joint bridge diagnostics missing for {name}")
        gates[name] = module.rwkv_joint_pair_crossglu_last_gate.detach()
        values[name] = module.rwkv_joint_pair_crossglu_last_value.detach()
    return logits, bool(fixed), gates, values, query_vectors, state_vectors


def _identity_scores(
    modules: Sequence[tuple[str, Any]],
    query: torch.Tensor,
    state: torch.Tensor,
) -> torch.Tensor:
    scores = torch.stack(
        [module.rwkv_joint_pair_crossglu.identity.score(query[index], state[index]) for index, (_, module) in enumerate(modules)]
    )
    if not bool(torch.isfinite(scores).all().item()):
        raise RuntimeError("Joint-pair endpoint identity scores are non-finite")
    return scores


def _evaluate_endpoint_row(model: torch.nn.Module, target: Any, donor: Any, *, source: int, donor_source: int, pad_token_id: int, device: torch.device) -> Mapping[str, Any]:
    modules = causal_train.ordered_modules(model)
    target_batch = evolution.collate_native_examples([target], pad_token_id=pad_token_id, device=device)
    donor_batch = contrast.build_donor_batch(target_batch, donor, device=device)
    try:
        with torch.no_grad():
            _set_original_write_mode(modules)
            value_identity.clear(model)
            evolution._native_write(model, target_batch, dtype=torch.bfloat16)
            correct_state = causal_train.capture_online_state_references(modules)
            target_values = value_identity.capture_write_values(model)
            seq_len = int(target_batch.read_input_ids.shape[1])
            zero_state = _zero_recurrent(correct_state)
            logits_correct, fixed_correct, gates_correct, values_correct, query_correct, state_correct = _endpoint_read(
                model, target_batch, modules, correct_state, correct_state, target_values, seq_len=seq_len
            )
            logits_zero, fixed_zero, gates_zero, values_zero, query_zero, state_zero_read = _endpoint_read(
                model, target_batch, modules, correct_state, zero_state, target_values, seq_len=seq_len
            )
            _set_original_write_mode(modules)
            evolution._native_write(model, donor_batch, dtype=torch.bfloat16)
            donor_state = causal_train.capture_online_state_references(modules)
            logits_donor, fixed_donor, gates_donor, values_donor, query_donor, state_donor_read = _endpoint_read(
                model, target_batch, modules, correct_state, donor_state, target_values, seq_len=seq_len
            )
            logits_permuted, fixed_permuted, gates_permuted, values_permuted, query_permuted, state_permuted = _endpoint_read(
                model, target_batch, modules, correct_state, correct_state, target_values,
                seq_len=seq_len, rotate_recurrent_layers=True,
            )
            reset_delta_mem_states(model)
            fixed_projected_only = _prepare_read(
                model,
                modules,
                projected=correct_state,
                recurrent=zero_state,
                query=target_values,
                seq_len=seq_len,
                rotate_recurrent_layers=False,
            )
            projected_only = _read_projected_only(model, target_batch, modules)
            branches = {
                "correct": logits_correct,
                "zero": logits_zero,
                "donor": logits_donor,
                "layer_permuted": logits_permuted,
                "projected_only_bypass": projected_only,
            }
            ce_with_tokens = {name: contrast.detached_answer_ce(logits, target_batch.labels) for name, logits in branches.items()}
            if len({tokens for _, tokens in ce_with_tokens.values()}) != 1:
                raise RuntimeError("Heldout endpoint answer token counts differ")
            query_reference = {name: module.rwkv_joint_pair_crossglu_query.detach().clone() for name, module in modules}
            query_fixed = all(torch.equal(query_reference[name], target_values[name].expand_as(query_reference[name])) for name in query_reference)
            identity_correct = _identity_scores(modules, query_correct, state_correct)
            identity_donor = _identity_scores(modules, query_donor, state_donor_read)
            identity_permuted = _identity_scores(modules, query_permuted, state_permuted)
            identity_query_fixed = bool(torch.equal(query_correct, query_donor) and torch.equal(query_correct, query_permuted))
            bridge_values = (*values_correct.values(), *values_zero.values(), *values_donor.values(), *values_permuted.values())
            all_values_finite = all(bool(torch.isfinite(value).all().item()) for value in bridge_values)
            all_gates_finite = all(bool(torch.isfinite(value).all().item()) for value in (*gates_correct.values(), *gates_zero.values(), *gates_donor.values(), *gates_permuted.values()))
        ce = {name: value[0] for name, value in ce_with_tokens.items()}
        return {
            "source_index": source,
            "donor_source_index": donor_source,
            "ce": ce,
            "ce_margins": {
                "zero_minus_correct": ce["zero"] - ce["correct"],
                "donor_minus_correct": ce["donor"] - ce["correct"],
                "layer_permuted_minus_correct": ce["layer_permuted"] - ce["correct"],
            },
            "fixed_carrier": {
                "correct": fixed_correct,
                "zero": fixed_zero,
                "donor": fixed_donor,
                "layer_permuted": fixed_permuted,
                "projected_only_bypass": fixed_projected_only,
            },
            "all_logits_finite": all(bool(torch.isfinite(value).all().item()) for value in branches.values()),
            "all_bridge_values_finite": bool(all_values_finite and all_gates_finite),
            "query_fixed_equal": query_fixed,
            "identity": {
                "correct_mean": float(identity_correct.mean().item()),
                "donor_mean": float(identity_donor.mean().item()),
                "layer_permuted_mean": float(identity_permuted.mean().item()),
                "correct_minus_donor": float((identity_correct - identity_donor).mean().item()),
                "correct_minus_layer_permuted": float((identity_correct - identity_permuted).mean().item()),
                "donor_pairwise_positive_fraction": float((identity_correct - identity_donor).gt(0).float().mean().item()),
                "layer_permuted_pairwise_positive_fraction": float((identity_correct - identity_permuted).gt(0).float().mean().item()),
                "query_fixed": identity_query_fixed,
            },
            "zero_logits_byte_equal_projected_only_bypass": bool(torch.equal(logits_zero, projected_only)),
            "answer_tokens": next(iter({tokens for _, tokens in ce_with_tokens.values()})),
        }
    finally:
        value_identity.clear(model)
        reset_delta_mem_states(model)
        evolution.release_native_row_allocator_cache(device)


def evaluate_heldout_endpoint(model: torch.nn.Module, examples: Mapping[int, Any], donor_mapping: Mapping[int, int], *, context: Any, pad_token_id: int, protocol: Mapping[str, Any]) -> Mapping[str, Any]:
    model.eval()
    local_rows = []
    for index, source in enumerate(HELDOUT_SOURCES):
        if index % WORLD_SIZE != context.process_rank:
            continue
        donor_source = int(donor_mapping[source])
        local_rows.append(_evaluate_endpoint_row(model, examples[source], examples[donor_source], source=source, donor_source=donor_source, pad_token_id=pad_token_id, device=context.device))
    gathered = distributed.gather_objects(context, local_rows)
    rows = sorted([row for rank_rows in gathered for row in rank_rows], key=lambda row: HELDOUT_SOURCES.index(int(row["source_index"])))
    count = len(rows)
    ce = {name: sum(float(row["ce"][name]) for row in rows) / max(count, 1) for name in ("correct", "zero", "donor", "layer_permuted", "projected_only_bypass")}
    margin_names = ("zero_minus_correct", "donor_minus_correct", "layer_permuted_minus_correct")
    margins = {name: sum(float(row["ce_margins"][name]) for row in rows) / max(count, 1) for name in margin_names}
    donor_fraction = sum(row["ce_margins"]["donor_minus_correct"] > 0.0 for row in rows) / max(count, 1)
    identity = {
        "correct_mean": sum(row["identity"]["correct_mean"] for row in rows) / max(count, 1),
        "donor_mean": sum(row["identity"]["donor_mean"] for row in rows) / max(count, 1),
        "layer_permuted_mean": sum(row["identity"]["layer_permuted_mean"] for row in rows) / max(count, 1),
        "correct_minus_donor": sum(row["identity"]["correct_minus_donor"] for row in rows) / max(count, 1),
        "correct_minus_layer_permuted": sum(row["identity"]["correct_minus_layer_permuted"] for row in rows) / max(count, 1),
        "donor_pairwise_positive_fraction": sum(
            row["identity"]["donor_pairwise_positive_fraction"] for row in rows
        ) / max(count, 1),
        "layer_permuted_pairwise_positive_fraction": sum(
            row["identity"]["layer_permuted_pairwise_positive_fraction"]
            for row in rows
        ) / max(count, 1),
    }
    identity_contract = protocol["heldout_causal_endpoint"]["required_retained_identity"]
    checks = {
        "rows_complete": count == len(HELDOUT_SOURCES),
        "sources_exact": [row["source_index"] for row in rows] == list(HELDOUT_SOURCES),
        "all_condition_logits_finite": all(row["all_logits_finite"] for row in rows),
        "all_bridge_values_finite": all(row["all_bridge_values_finite"] for row in rows),
        "projected_carrier_fixed_every_intervention": all(all(row["fixed_carrier"].values()) for row in rows),
        "query_fixed_equal_every_intervention": all(row["query_fixed_equal"] for row in rows),
        "zero_equals_projected_only_bypass_every_row": all(row["zero_logits_byte_equal_projected_only_bypass"] for row in rows),
        "zero_minus_correct_mean_positive": margins["zero_minus_correct"] > protocol["heldout_causal_endpoint"]["required_ce_margins"]["zero_minus_correct_mean"],
        "donor_minus_correct_mean_positive": margins["donor_minus_correct"] >= protocol["heldout_causal_endpoint"]["required_ce_margins"]["donor_minus_correct_mean"],
        "layer_permuted_minus_correct_mean_positive": margins["layer_permuted_minus_correct"] > protocol["heldout_causal_endpoint"]["required_ce_margins"]["layer_permuted_minus_correct_mean"],
        "donor_positive_row_fraction": donor_fraction >= protocol["heldout_causal_endpoint"]["required_ce_margins"]["minimum_donor_positive_row_fraction"],
        "identity_query_fixed_every_intervention": all(
            row["identity"]["query_fixed"] for row in rows
        ),
        "donor_identity_pairwise_fraction": identity["donor_pairwise_positive_fraction"]
        >= identity_contract["donor_pairwise_positive_fraction_minimum"],
        "donor_identity_mean_margin": identity["correct_minus_donor"]
        >= identity_contract["donor_mean_correct_minus_control_score_minimum"],
        "layer_permuted_identity_pairwise_fraction": identity[
            "layer_permuted_pairwise_positive_fraction"
        ]
        >= identity_contract["layer_permuted_pairwise_positive_fraction_minimum"],
    }
    return {
        "rows": count,
        "mean_ce": ce,
        "mean_ce_margins": margins,
        "donor_ce_positive_row_fraction": donor_fraction,
        "learned_identity": identity,
        "checks": checks,
        "passed": all(checks.values()),
        "rank_rows": list(gathered),
    }


def save_bridge_checkpoint(named_trainable: Sequence[tuple[str, torch.nn.Parameter]], path: Path) -> Mapping[str, Any]:
    from safetensors.torch import save_file

    state = {name: parameter.detach().float().cpu().contiguous() for name, parameter in named_trainable}
    if len(state) != TRAINABLE_TENSORS or sum(value.numel() for value in state.values()) != TRAINABLE_ELEMENTS:
        raise RuntimeError("Joint-pair checkpoint tensor count differs")
    save_file(state, str(path), metadata={"schema": SCHEMA, "protocol_payload_sha256": PROTOCOL_PAYLOAD_SHA256})
    return {"path": path.name, "sha256": sha256_file(path), "tensors": len(state), "elements": sum(value.numel() for value in state.values())}


def run(*, context: Any, output_dir: Path, base_model: Path = BASE_MODEL, dataset_root: Path = DATASET_ROOT) -> Mapping[str, Any]:
    if context.world_size != WORLD_SIZE or not hardware.four_distinct_a100s(context.rank_devices):
        raise RuntimeError("Joint-pair causal training requires exactly four distinct A100s")
    if os.environ.get("HF_ENDPOINT") != HF_MIRROR_ENDPOINT:
        raise RuntimeError(f"HF_ENDPOINT must be exactly {HF_MIRROR_ENDPOINT}")
    protocol, mechanics_result, crossfit_result = validate_protocol()
    base_model = base_model.expanduser().resolve(strict=True)
    dataset_root = dataset_root.expanduser().resolve(strict=True)
    if sha256_file(base_model / "config.json") != protocol["frozen_inputs"]["base_config_sha256"]:
        raise ValueError("Pinned base-model config differs")
    resolved_output = output_dir.expanduser().resolve()
    fresh_error = ValueError(f"Joint-pair output must be fresh: {resolved_output}") if context.is_primary and resolved_output.exists() else None
    distributed.phase_consensus(context, phase="joint-pair-causal-fresh", error=fresh_error)
    create_error: BaseException | None = None
    if context.is_primary:
        try:
            resolved_output.mkdir(parents=True, exist_ok=False)
        except BaseException as error:
            create_error = error
    distributed.phase_consensus(context, phase="joint-pair-causal-create", error=create_error)
    head = mechanics.reconstruct_crossfit_head(context)
    if canonical_sha256({name: value.detach().tolist() for name, value in head.state_dict().items()}) != RECONSTRUCTION_STATE_SHA256:
        raise RuntimeError("Reconstructed cross-fit head differs from signed training input")
    distributed.require_consensus(context, RECONSTRUCTION_STATE_SHA256, description="reconstructed joint-pair head")
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    model, tokenizer, model_audit = mechanics.candidate.load_model(base_model, device=context.device)
    installation = mechanics.install_bridges(model, head)
    value_identity_audit = value_identity.install(model)
    installation = {**dict(installation), "value_identity": value_identity_audit}
    _promote_trainable_bridge_parameters(model)
    named_trainable, trainable_audit = mechanics.freeze_and_select_bridges(model)
    _promote_trainable_bridge_parameters(model)
    _promote_bridge_parameters_to_fp32(model)
    if len(named_trainable) != TRAINABLE_TENSORS or sum(parameter.numel() for _, parameter in named_trainable) != TRAINABLE_ELEMENTS or any(parameter.dtype != torch.float32 for _, parameter in named_trainable):
        raise RuntimeError("Joint-pair trainable isolation differs")
    training_rows = contrast.load_scene_rows(tokenizer, dataset_root)
    donor_mapping, donor_deltas, donor_payload = contrast.build_donor_mapping(training_rows)
    schedule, schedule_payload = contrast.build_schedule(training_rows, donor_mapping, donor_deltas)
    frozen = protocol["frozen_inputs"]
    if canonical_sha256(donor_payload) != frozen["training_donor_mapping_payload_sha256"] or canonical_sha256(schedule_payload[:UPDATES]) != TRAINING_SCHEDULE_PREFIX_SHA256:
        raise RuntimeError("Joint-pair training schedule differs")
    examples, rows_by_source, endpoint_mapping, split_payload = crossfit.authorized_examples(tokenizer, dataset_root)
    signed_split = dict(crossfit_result["crossfit_split"])
    signed_split.pop("payload_sha256")
    endpoint_payload = heldout_payload(rows_by_source, endpoint_mapping)
    if split_payload != signed_split or canonical_sha256(split_payload) != CROSSFIT_SPLIT_PAYLOAD_SHA256 or canonical_sha256(endpoint_payload) != HELDOUT_PAYLOAD_SHA256:
        raise RuntimeError("Joint-pair heldout endpoint binding differs")
    input_binding = {
        "schema": INPUT_SCHEMA,
        "protocol_payload_sha256": PROTOCOL_PAYLOAD_SHA256,
        "protocol_objective": protocol["objective"],
        "mechanics_result_file_sha256": MECHANICS_RESULT_FILE_SHA256,
        "mechanics_result_receipt": MECHANICS_RESULT_RECEIPT,
        "mechanics_status": mechanics_result["status"],
        "crossfit_result_file_sha256": CROSSFIT_RESULT_FILE_SHA256,
        "crossfit_result_receipt": CROSSFIT_RESULT_RECEIPT,
        "crossfit_split_payload_sha256": CROSSFIT_SPLIT_PAYLOAD_SHA256,
        "reconstruction_state_sha256": RECONSTRUCTION_STATE_SHA256,
        "seed": SEED,
        "updates": UPDATES,
        "world_size": context.world_size,
        "global_batch_size": GLOBAL_BATCH_SIZE,
        "local_rows": LOCAL_ROWS,
        "trainable_tensors": TRAINABLE_TENSORS,
        "trainable_elements": TRAINABLE_ELEMENTS,
        "base_model": str(base_model),
        "dataset_root": str(dataset_root),
        "training_schedule_prefix_sha256": canonical_sha256(schedule_payload[:UPDATES]),
        "heldout_payload_sha256": canonical_sha256(endpoint_payload),
        "heldout_sources": list(HELDOUT_SOURCES),
        "hf_endpoint": os.environ.get("HF_ENDPOINT"),
        "rank_devices": list(context.rank_devices),
        "model_audit": model_audit,
        "installation": installation,
        "trainable_audit": trainable_audit,
        "runner_sha256": sha256_file(Path(__file__)),
        "protected_splits_opened": [],
    }
    distributed.require_consensus(context, canonical_sha256(input_binding), description="joint-pair causal input binding")
    binding_error: BaseException | None = None
    if context.is_primary:
        try:
            write_json(resolved_output / "input_binding.json", input_binding)
        except BaseException as error:
            binding_error = error
    distributed.phase_consensus(context, phase="joint-pair-causal-binding-save", error=binding_error)
    with training_bindings():
        training = causal_train.train(model, training_rows, schedule, updates=UPDATES, context=context, pad_token_id=int(tokenizer.pad_token_id), output_dir=resolved_output, named_trainable=named_trainable)
    training_passed = bool(training["updates"] == UPDATES and training["trainable_subset_changed"] is True and training["recurrent_subset_changed"] is False and training["maximum_global_inactive_parameter_tensors"] == 0 and training["projected_carrier_fixed_every_row"] is True and training["optimizer_state_cpu_offload"]["enabled"] is True and training["control_branch_graph_serialization"]["enabled"] is True and training["first_update_gradient_audit"]["passed"] is True)
    endpoint = evaluate_heldout_endpoint(model, examples, endpoint_mapping, context=context, pad_token_id=int(tokenizer.pad_token_id), protocol=protocol)
    rank_runtime = distributed.gather_objects(context, {"rank": context.process_rank, "peak_cuda_memory_bytes": training["peak_cuda_memory_bytes"]})
    save_error: BaseException | None = None
    result: dict[str, Any] = {}
    if context.is_primary:
        try:
            checkpoint_audit = save_bridge_checkpoint(named_trainable, resolved_output / "joint_pair_crossglu.safetensors")
            passed = bool(training_passed and endpoint["passed"])
            result = {
                "schema": SCHEMA,
                "status": PASS_STATUS if passed else FAIL_STATUS,
                "passed": passed,
                "generation_authorized": False,
                "native_generation_benchmark_authorized": False,
                "causal_endpoint_authorized": True,
                "protocol_payload_sha256": PROTOCOL_PAYLOAD_SHA256,
                "seed": SEED,
                "updates": UPDATES,
                "input_binding": input_binding,
                "training": training,
                "training_passed": training_passed,
                "heldout_causal_endpoint": endpoint,
                "joint_pair_crossglu_checkpoint": checkpoint_audit,
                "delta_mem_adapter_saved": False,
                "rank_runtime": list(rank_runtime),
                "protected_splits_opened": [],
                "claim_policy": protocol["claim_policy"],
            }
            result["receipt"] = {"algorithm": "sha256", "payload_scope": "canonical_result_without_receipt", "payload_sha256": canonical_sha256(result)}
            write_json(resolved_output / "result.json", result)
        except BaseException as error:
            save_error = error
    distributed.phase_consensus(context, phase="joint-pair-causal-result-save", error=save_error)
    del model, training_rows, examples
    gc.collect()
    torch.cuda.empty_cache()
    return result if context.is_primary else {"status": "worker_complete", "passed": bool(training_passed and endpoint["passed"]), "generation_authorized": False}


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--base-model", type=Path, default=BASE_MODEL)
    parser.add_argument("--dataset-root", type=Path, default=DATASET_ROOT)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args(argv)


@record
def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    context = distributed.initialize_distributed_training(args.device)
    if context is None:
        raise RuntimeError("Run with torchrun --nproc_per_node=4")
    try:
        result = run(context=context, output_dir=args.output_dir, base_model=args.base_model, dataset_root=args.dataset_root)
    finally:
        distributed.destroy_distributed_training(context)
    print(json.dumps({"rank": context.process_rank, "status": result["status"], "passed": result["passed"], "generation_authorized": False}, ensure_ascii=True, sort_keys=True), flush=True)
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
