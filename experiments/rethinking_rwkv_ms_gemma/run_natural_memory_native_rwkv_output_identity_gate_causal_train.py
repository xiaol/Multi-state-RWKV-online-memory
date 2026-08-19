#!/usr/bin/env python3
"""Train and evaluate the signed output-coupled RWKV identity gate."""

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
    rwkv_projected_value_identity as value_identity,
)
from experiments.rethinking_rwkv_ms_gemma import (  # noqa: E402
    run_natural_memory_native_rwkv_output_identity_gate_mechanics as mechanics,
)
from experiments.rethinking_rwkv_ms_gemma import (  # noqa: E402
    run_natural_memory_native_rwkv_query_state_bilinear_crossfit as crossfit,
)


causal_train = mechanics.causal_train
contrast = mechanics.contrast
distributed = mechanics.distributed
evolution = mechanics.evolution
hardware = crossfit.hardware

SCHEMA = "rwkv_ms_natural_memory_native_output_identity_gate_causal_train.v1"
STEP_SCHEMA = "rwkv_ms_natural_memory_native_output_identity_gate_causal_train_step.v1"
INPUT_SCHEMA = "rwkv_ms_natural_memory_native_output_identity_gate_causal_train_input.v1"
PROTOCOL = SCRIPT_DIR / "natural_memory_native_rwkv_output_identity_gate_causal_train_protocol_v1.json"
PROTOCOL_FILE_SHA256 = "d87201740fd4fb3508cc7819a29f9d3311c3c69f35607ad8fc8add92714106fa"
PROTOCOL_PAYLOAD_SHA256 = "c95190aaf38d7175fdd3b6c003e89d771db5a47911660e888131a86bc10eebed"
MECHANICS_RESULT = SCRIPT_DIR / "local_artifacts/natural_memory_native_rwkv_output_identity_gate_mechanics_v2/result.json"
MECHANICS_RESULT_FILE_SHA256 = "089d4c6392dc83259ac4bcc44632b5b7f459eed6f843e23898cd07d7f9ccc907"
MECHANICS_RESULT_RECEIPT = "c528311da90fe9b56a7dc7660798f242249170242261c418c9dcdc10547bb4a5"
CROSSFIT_RESULT = crossfit.SCRIPT_DIR / "local_artifacts/natural_memory_native_rwkv_query_state_bilinear_crossfit_v1/result.json"
CROSSFIT_RESULT_FILE_SHA256 = "5e41c4569273fd5841381fcb6c5738b26212dd326b4f7cf56b589528df346ba3"
CROSSFIT_RESULT_RECEIPT = "89392eaeffa50c0bed9109fd8db3d33a5625eb4ff7117f81d710cf9b5be93945"
CROSSFIT_SPLIT_PAYLOAD_SHA256 = "3b6306fdd2120faf9b2a20f370ad08a7fa7eaffa347e6f8deaa91e3cc1a7bf3c"
RECONSTRUCTION_STATE_SHA256 = "1bbb5e8ff1b61aa2309c81d4a0235a9b74e0c8eb855604c03c6bbea4b939790f"
TRAINING_SCHEDULE_PREFIX_SHA256 = "108b83ce2f2dd590c6ad45c7d46affeb4fd01afddee08dce35b2d5d18219876d"
HELDOUT_PAYLOAD_SHA256 = "84ae348079351f96f70172bfc8c02b3ae44f54bc74db260e8ba4d7c9164117a2"
HELDOUT_SOURCES = (
    7, 12, 14, 19, 24, 26, 29, 30, 33, 52, 59, 62, 66, 73, 80, 96,
    100, 101, 119, 121, 128, 129, 139, 144, 152, 172, 177, 181, 201,
    211, 216, 218, 225, 233, 236, 247, 255, 259, 279, 296, 321, 322,
    356, 358,
)
HF_MIRROR_ENDPOINT = "https://hf-mirror.com"
BASE_MODEL = contrast.BASE_MODEL
DATASET_ROOT = contrast.NATIVE_DATASET_ROOT
WORLD_SIZE = 4
SEED = 116
UPDATES = 8
GLOBAL_BATCH_SIZE = 4
LOCAL_ROWS = 1
PASS_STATUS = "output_identity_gate_heldout_causal_passed_generation_blocked"
FAIL_STATUS = "output_identity_gate_heldout_causal_failed_generation_blocked"


def canonical_sha256(value: Any) -> str:
    return distributed.canonical_sha256(value)


def sha256_file(path: Path) -> str:
    return mechanics.sha256_file(path)


def write_json(path: Path, value: Mapping[str, Any]) -> None:
    if path.exists():
        raise ValueError(f"Output identity-gate artifact must be fresh: {path}")
    path.write_text(
        json.dumps(value, ensure_ascii=True, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _validate_receipted_result(
    path: Path,
    *,
    file_sha256: str,
    receipt_sha256: str,
) -> Mapping[str, Any]:
    if sha256_file(path) != file_sha256:
        raise ValueError(f"Signed result file differs: {path}")
    result = json.loads(path.read_text(encoding="utf-8"))
    unsigned = dict(result)
    receipt = unsigned.pop("receipt", {})
    if (
        canonical_sha256(unsigned) != receipt_sha256
        or receipt.get("payload_sha256") != receipt_sha256
    ):
        raise ValueError(f"Signed result receipt differs: {path}")
    return result


def validate_protocol() -> tuple[Mapping[str, Any], Mapping[str, Any], Mapping[str, Any]]:
    if sha256_file(PROTOCOL) != PROTOCOL_FILE_SHA256:
        raise ValueError("Output identity-gate causal protocol file differs")
    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    unsigned = dict(protocol)
    receipt = unsigned.pop("receipt", {})
    if (
        canonical_sha256(unsigned) != PROTOCOL_PAYLOAD_SHA256
        or receipt.get("payload_sha256") != PROTOCOL_PAYLOAD_SHA256
        or protocol.get("generation_authorized") is not False
        or protocol.get("delta_mem_adapter_saved") is not False
        or protocol.get("protected_splits_opened_by_this_protocol") != []
    ):
        raise ValueError("Output identity-gate causal protocol differs")
    training = protocol.get("training", {})
    endpoint = protocol.get("heldout_causal_endpoint", {})
    architecture = protocol.get("architecture", {})
    if (
        training.get("optimizer_updates") != UPDATES
        or training.get("global_batch_rows") != GLOBAL_BATCH_SIZE
        or training.get("local_rows_per_rank") != LOCAL_ROWS
        or training.get("optimizer_state_cpu_offload_enabled") is not True
        or training.get("all_168_output_gate_tensors_require_finite_nonzero_global_gradient_each_update") is not True
        or architecture.get("trainable_parameter_tensors") != 168
        or architecture.get("trainable_parameter_elements") != 21504
        or endpoint.get("rows") != len(HELDOUT_SOURCES)
        or endpoint.get("source_indices") != list(HELDOUT_SOURCES)
        or endpoint.get("source_donor_payload_sha256") != HELDOUT_PAYLOAD_SHA256
    ):
        raise ValueError("Output identity-gate causal training contract differs")
    mechanics_result = _validate_receipted_result(
        MECHANICS_RESULT,
        file_sha256=MECHANICS_RESULT_FILE_SHA256,
        receipt_sha256=MECHANICS_RESULT_RECEIPT,
    )
    if (
        mechanics_result.get("status") != "output_gate_mechanics_passed_generation_blocked"
        or mechanics_result.get("passed") is not True
        or not all(mechanics_result.get("checks", {}).values())
        or mechanics_result.get("protected_splits_opened") != []
    ):
        raise ValueError("Output identity-gate mechanics did not authorize training")
    crossfit_result = _validate_receipted_result(
        CROSSFIT_RESULT,
        file_sha256=CROSSFIT_RESULT_FILE_SHA256,
        receipt_sha256=CROSSFIT_RESULT_RECEIPT,
    )
    split = crossfit_result.get("crossfit_split", {})
    split_unsigned = dict(split)
    split_receipt = split_unsigned.pop("payload_sha256", None)
    if (
        crossfit_result.get("status")
        != "bilinear_crossfit_passed_causal_training_design_authorized"
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


def checkpointed_native_write_read(
    model: torch.nn.Module,
    batch: Any,
    *,
    dtype: torch.dtype,
) -> tuple[Mapping[str, Any], torch.Tensor]:
    def write_read(*tensors: torch.Tensor) -> torch.Tensor:
        target = _recompute_batch(batch, tensors)
        value_identity.clear(model)
        evolution._native_write(model, target, dtype=dtype)
        target_values = value_identity.capture_write_values(model)
        value_identity.set_fixed_target_values(model, target_values)
        return evolution._native_read(model, target, dtype=dtype)

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
    for _, module in causal_train.ordered_modules(model):
        occupied = module.projected_kv_occupied
        if occupied is None or occupied.ndim != 2:
            raise RuntimeError("Fixed-value checkpoint did not preserve occupancy")
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
        value_identity.clear(model)
        evolution._native_write(model, target, dtype=dtype)
        correct_state = causal_train.capture_online_state_references(modules)
        target_values = value_identity.capture_write_values(model)
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
            evolution._native_write(model, donor, dtype=dtype)
            recurrent_state = causal_train.capture_online_state_references(modules)
        audit["projected_carrier_references_fixed"] = bool(
            audit["projected_carrier_references_fixed"]
            and causal_train.install_intervened_state(
                modules,
                projected=correct_state,
                recurrent=recurrent_state,
                rotate_recurrent_layers=rotate_recurrent_layers,
            )
        )
        value_identity.set_fixed_target_values(model, target_values)
        return evolution._native_read(model, target, dtype=dtype)

    inputs = [
        target_batch.write_input_ids,
        target_batch.write_attention_mask,
        target_batch.read_input_ids,
        target_batch.read_attention_mask,
    ]
    if donor_batch is not None:
        inputs.extend((donor_batch.write_input_ids, donor_batch.write_attention_mask))
    logits = checkpoint(write_read, *inputs, use_reentrant=False)
    return logits, audit


def evaluate_condition_ce(
    model: torch.nn.Module,
    batch: Any,
    *,
    no_state: bool,
    dtype: torch.dtype,
) -> tuple[float, int]:
    if not no_state:
        return contrast.evaluate_condition_ce_original(
            model, batch, no_state=False, dtype=dtype
        )
    try:
        with torch.no_grad():
            modules = causal_train.ordered_modules(model)
            value_identity.clear(model)
            evolution._native_write(model, batch, dtype=dtype)
            correct_state = causal_train.capture_online_state_references(modules)
            target_values = value_identity.capture_write_values(model)
            fixed = causal_train.install_intervened_state(
                modules,
                projected=correct_state,
                recurrent=mechanics.zero_recurrent(correct_state),
                rotate_recurrent_layers=False,
            )
            if not fixed:
                raise RuntimeError("Zero-recurrent projected carrier changed")
            value_identity.set_fixed_target_values(model, target_values)
            logits = evolution._native_read(model, batch, dtype=dtype)
            return contrast.detached_answer_ce(logits, batch.labels)
    finally:
        value_identity.clear(model)
        reset_delta_mem_states(model)


def gradient_audit(
    named: Sequence[tuple[str, torch.nn.Parameter]],
) -> Mapping[str, Any]:
    if len(named) != 168:
        raise RuntimeError(f"Expected 168 output-gate gradients, got {len(named)}")
    local_finite_fp32 = []
    local_active = []
    for name, parameter in named:
        gradient = parameter.grad
        if "rwkv_output_identity_gate" not in name:
            raise RuntimeError(f"Non-gate parameter entered gradient audit: {name}")
        finite_fp32 = bool(
            gradient is not None
            and gradient.dtype == torch.float32
            and torch.isfinite(gradient).all().item()
        )
        local_finite_fp32.append(int(finite_fp32))
        local_active.append(
            int(finite_fp32 and bool(gradient.abs().max().gt(0).item()))
        )
    finite = torch.tensor(local_finite_fp32, device=named[0][1].device, dtype=torch.int32)
    active = torch.tensor(local_active, device=named[0][1].device, dtype=torch.int32)
    dist.all_reduce(finite, op=dist.ReduceOp.SUM)
    dist.all_reduce(active, op=dist.ReduceOp.SUM)
    return {
        "trainable_tensors": len(named),
        "global_finite_fp32_tensors": int(finite.gt(0).sum().item()),
        "global_finite_nonzero_tensors": int(active.gt(0).sum().item()),
        "passed": bool(finite.gt(0).all().item() and active.gt(0).all().item()),
    }


@contextmanager
def training_bindings() -> Iterator[None]:
    global_names = (
        "GLOBAL_BATCH_SIZE",
        "LOCAL_ROWS",
        "MIN_ACCEPTED_ROWS_PER_UPDATE",
        "MAX_TOTAL_REJECTED_ROWS",
        "OFFLOAD_OPTIMIZER_STATE_DURING_ROWS",
        "SERIALIZE_CONTROL_BRANCH_GRAPHS",
        "FIRST_UPDATE_GRADIENT_AUDITOR",
        "STEP_SCHEMA",
        "PROTOCOL_PAYLOAD_SHA256",
    )
    previous_globals = {name: getattr(causal_train, name) for name in global_names}
    previous_correct = evolution.checkpointed_native_write_read
    previous_control = causal_train.checkpointed_intervened_write_read
    previous_zero = contrast.evaluate_condition_ce
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
        contrast.evaluate_condition_ce_original = previous_zero
        contrast.evaluate_condition_ce = evaluate_condition_ce
        yield
    finally:
        contrast.evaluate_condition_ce = previous_zero
        if hasattr(contrast, "evaluate_condition_ce_original"):
            delattr(contrast, "evaluate_condition_ce_original")
        causal_train.checkpointed_intervened_write_read = previous_control
        evolution.checkpointed_native_write_read = previous_correct
        for name, value in previous_globals.items():
            setattr(causal_train, name, value)


def heldout_payload(
    rows_by_source: Mapping[int, Mapping[str, Any]],
    donor_mapping: Mapping[int, int],
) -> list[dict[str, Any]]:
    return [
        {
            "source_index": source,
            "row_sha256": rows_by_source[source]["row_sha256"],
            "donor_source_index": int(donor_mapping[source]),
            "donor_row_sha256": rows_by_source[int(donor_mapping[source])]["row_sha256"],
        }
        for source in HELDOUT_SOURCES
    ]


def _learned_identity_score(
    model: torch.nn.Module,
    labels: torch.Tensor,
) -> tuple[float, torch.Tensor]:
    query, state = crossfit.aggregate_vectors(value_identity.capture(model), labels)
    scores = torch.stack(
        [
            module.rwkv_output_identity_gate.identity.score(query[layer], state[layer])
            for layer, (_, module) in enumerate(causal_train.ordered_modules(model))
        ]
    )
    if tuple(scores.shape) != (crossfit.LAYERS,) or not bool(torch.isfinite(scores).all()):
        raise RuntimeError("Learned output-gate identity scores differ")
    return float(scores.mean().item()), query


def _evaluate_endpoint_row(
    model: torch.nn.Module,
    target: Any,
    donor: Any,
    *,
    source: int,
    donor_source: int,
    pad_token_id: int,
    device: torch.device,
) -> Mapping[str, Any]:
    modules = causal_train.ordered_modules(model)
    target_batch = evolution.collate_native_examples(
        [target], pad_token_id=pad_token_id, device=device
    )
    donor_batch = contrast.build_donor_batch(target_batch, donor, device=device)
    logits: dict[str, torch.Tensor] = {}
    fixed: dict[str, bool] = {}
    scores: dict[str, float] = {}
    queries: dict[str, torch.Tensor] = {}
    try:
        with torch.no_grad():
            value_identity.clear(model)
            evolution._native_write(model, target_batch, dtype=torch.bfloat16)
            correct_state = causal_train.capture_online_state_references(modules)
            target_values = value_identity.capture_write_values(model)

            fixed["correct"] = causal_train.install_intervened_state(
                modules, projected=correct_state, recurrent=correct_state,
                rotate_recurrent_layers=False,
            )
            value_identity.set_fixed_target_values(model, target_values)
            logits["correct"] = evolution._native_read(model, target_batch, dtype=torch.bfloat16)
            scores["correct"], queries["correct"] = _learned_identity_score(
                model, target_batch.labels
            )

            value_identity.clear(model)
            zero_state = mechanics.zero_recurrent(correct_state)
            fixed["zero"] = causal_train.install_intervened_state(
                modules, projected=correct_state, recurrent=zero_state,
                rotate_recurrent_layers=False,
            )
            value_identity.set_fixed_target_values(model, target_values)
            logits["zero"] = evolution._native_read(model, target_batch, dtype=torch.bfloat16)

            value_identity.clear(model)
            evolution._native_write(model, donor_batch, dtype=torch.bfloat16)
            donor_state = causal_train.capture_online_state_references(modules)
            fixed["donor"] = causal_train.install_intervened_state(
                modules, projected=correct_state, recurrent=donor_state,
                rotate_recurrent_layers=False,
            )
            value_identity.set_fixed_target_values(model, target_values)
            logits["donor"] = evolution._native_read(model, target_batch, dtype=torch.bfloat16)
            scores["donor"], queries["donor"] = _learned_identity_score(
                model, target_batch.labels
            )

            value_identity.clear(model)
            fixed["layer_permuted"] = causal_train.install_intervened_state(
                modules, projected=correct_state, recurrent=correct_state,
                rotate_recurrent_layers=True,
            )
            value_identity.set_fixed_target_values(model, target_values)
            logits["layer_permuted"] = evolution._native_read(
                model, target_batch, dtype=torch.bfloat16
            )
            scores["layer_permuted"], queries["layer_permuted"] = (
                _learned_identity_score(model, target_batch.labels)
            )

            value_identity.clear(model)
            fixed["projected_only_bypass"] = causal_train.install_intervened_state(
                modules, projected=correct_state, recurrent=zero_state,
                rotate_recurrent_layers=False,
            )
            value_identity.set_fixed_target_values(model, target_values)
            with mechanics.explicit_projected_only_bypass(model):
                logits["projected_only_bypass"] = evolution._native_read(
                    model, target_batch, dtype=torch.bfloat16
                )

            ce_with_tokens = {
                name: contrast.detached_answer_ce(value, target_batch.labels)
                for name, value in logits.items()
            }
        token_counts = {tokens for _, tokens in ce_with_tokens.values()}
        if len(token_counts) != 1:
            raise RuntimeError("Heldout endpoint answer token counts differ")
        ce = {name: value[0] for name, value in ce_with_tokens.items()}
        query_fixed = bool(
            torch.equal(queries["correct"], queries["donor"])
            and torch.equal(queries["correct"], queries["layer_permuted"])
        )
        return {
            "source_index": source,
            "donor_source_index": donor_source,
            "ce": ce,
            "ce_margins": {
                "zero_minus_correct": ce["zero"] - ce["correct"],
                "donor_minus_correct": ce["donor"] - ce["correct"],
                "layer_permuted_minus_correct": ce["layer_permuted"] - ce["correct"],
            },
            "learned_identity": {
                "correct_score": scores["correct"],
                "donor_score": scores["donor"],
                "layer_permuted_score": scores["layer_permuted"],
                "correct_minus_donor": scores["correct"] - scores["donor"],
                "correct_minus_layer_permuted": (
                    scores["correct"] - scores["layer_permuted"]
                ),
                "fixed_query_equal": query_fixed,
            },
            "fixed_carrier": fixed,
            "all_logits_finite": all(
                bool(torch.isfinite(value).all().item()) for value in logits.values()
            ),
            "zero_logits_byte_equal_projected_only_bypass": bool(
                torch.equal(logits["zero"], logits["projected_only_bypass"])
            ),
            "answer_tokens": next(iter(token_counts)),
        }
    finally:
        value_identity.clear(model)
        reset_delta_mem_states(model)
        del target_batch, donor_batch, logits
        evolution.release_native_row_allocator_cache(device)


def evaluate_heldout_endpoint(
    model: torch.nn.Module,
    examples: Mapping[int, Any],
    donor_mapping: Mapping[int, int],
    *,
    context: Any,
    pad_token_id: int,
    protocol: Mapping[str, Any],
) -> Mapping[str, Any]:
    model.eval()
    local_rows = []
    for index, source in enumerate(HELDOUT_SOURCES):
        if index % WORLD_SIZE != context.process_rank:
            continue
        donor_source = int(donor_mapping[source])
        local_rows.append(
            _evaluate_endpoint_row(
                model,
                examples[source],
                examples[donor_source],
                source=source,
                donor_source=donor_source,
                pad_token_id=pad_token_id,
                device=context.device,
            )
        )
    gathered = distributed.gather_objects(context, local_rows)
    rows = [row for rank_rows in gathered for row in rank_rows]
    rows.sort(key=lambda row: HELDOUT_SOURCES.index(int(row["source_index"])))
    count = len(rows)

    def mean(path: Sequence[str]) -> float:
        values = [float(row[path[0]][path[1]]) for row in rows]
        return sum(values) / max(len(values), 1)

    ce = {
        name: sum(float(row["ce"][name]) for row in rows) / max(count, 1)
        for name in ("correct", "zero", "donor", "layer_permuted", "projected_only_bypass")
    }
    ce_margins = {
        name: mean(("ce_margins", name))
        for name in (
            "zero_minus_correct",
            "donor_minus_correct",
            "layer_permuted_minus_correct",
        )
    }
    identity = {
        name: mean(("learned_identity", name))
        for name in (
            "correct_score",
            "donor_score",
            "layer_permuted_score",
            "correct_minus_donor",
            "correct_minus_layer_permuted",
        )
    }
    donor_ce_positive_fraction = sum(
        row["ce_margins"]["donor_minus_correct"] > 0.0 for row in rows
    ) / max(count, 1)
    identity["donor_pairwise_positive_fraction"] = sum(
        row["learned_identity"]["correct_minus_donor"] > 0.0 for row in rows
    ) / max(count, 1)
    identity["layer_permuted_pairwise_positive_fraction"] = sum(
        row["learned_identity"]["correct_minus_layer_permuted"] > 0.0
        for row in rows
    ) / max(count, 1)
    identity["hardest_control_positive_row_fraction"] = sum(
        min(
            row["learned_identity"]["correct_minus_donor"],
            row["learned_identity"]["correct_minus_layer_permuted"],
        ) > 0.0
        for row in rows
    ) / max(count, 1)
    endpoint_contract = protocol["heldout_causal_endpoint"]
    ce_required = endpoint_contract["required_ce_margins"]
    identity_required = endpoint_contract["required_retained_identity"]
    checks = {
        "rows_complete": count == len(HELDOUT_SOURCES),
        "sources_exact": [row["source_index"] for row in rows] == list(HELDOUT_SOURCES),
        "all_condition_logits_finite": all(row["all_logits_finite"] for row in rows),
        "projected_carrier_fixed_every_intervention": all(
            all(row["fixed_carrier"].values()) for row in rows
        ),
        "fixed_projected_query_every_identity_read": all(
            row["learned_identity"]["fixed_query_equal"] for row in rows
        ),
        "zero_equals_projected_only_bypass_every_row": all(
            row["zero_logits_byte_equal_projected_only_bypass"] for row in rows
        ),
        "zero_minus_correct_mean_positive": ce_margins["zero_minus_correct"]
        > ce_required["zero_minus_correct_mean"],
        "donor_minus_correct_mean_positive": ce_margins["donor_minus_correct"]
        > ce_required["donor_minus_correct_mean"],
        "layer_permuted_minus_correct_mean_positive": (
            ce_margins["layer_permuted_minus_correct"]
            > ce_required["layer_permuted_minus_correct_mean"]
        ),
        "donor_positive_row_fraction": donor_ce_positive_fraction
        >= ce_required["minimum_donor_positive_row_fraction"],
        "donor_identity_pairwise_fraction": identity["donor_pairwise_positive_fraction"]
        >= identity_required["donor_pairwise_positive_fraction_minimum"],
        "donor_identity_mean_margin": identity["correct_minus_donor"]
        >= identity_required["donor_mean_correct_minus_control_score_minimum"],
        "layer_permuted_identity_pairwise_fraction": (
            identity["layer_permuted_pairwise_positive_fraction"]
            >= identity_required["layer_permuted_pairwise_positive_fraction_minimum"]
        ),
    }
    return {
        "rows": count,
        "mean_ce": ce,
        "mean_ce_margins": ce_margins,
        "donor_ce_positive_row_fraction": donor_ce_positive_fraction,
        "learned_identity": identity,
        "checks": checks,
        "passed": all(checks.values()),
        "rank_rows": list(gathered),
    }


def save_output_gate_checkpoint(
    named_trainable: Sequence[tuple[str, torch.nn.Parameter]],
    path: Path,
) -> Mapping[str, Any]:
    from safetensors.torch import save_file

    state = {
        name: parameter.detach().float().cpu().contiguous()
        for name, parameter in named_trainable
    }
    if len(state) != 168:
        raise RuntimeError("Output-gate checkpoint tensor count differs")
    save_file(
        state,
        str(path),
        metadata={
            "schema": SCHEMA,
            "protocol_payload_sha256": PROTOCOL_PAYLOAD_SHA256,
        },
    )
    return {
        "path": path.name,
        "sha256": sha256_file(path),
        "tensors": len(state),
        "elements": sum(value.numel() for value in state.values()),
    }


def run(
    *,
    context: Any,
    output_dir: Path,
    base_model: Path = BASE_MODEL,
    dataset_root: Path = DATASET_ROOT,
) -> Mapping[str, Any]:
    if context.world_size != WORLD_SIZE or not hardware.four_distinct_a100s(
        context.rank_devices
    ):
        raise RuntimeError("Output identity-gate training requires exactly four distinct A100s")
    if os.environ.get("HF_ENDPOINT") != HF_MIRROR_ENDPOINT:
        raise RuntimeError(f"HF_ENDPOINT must be exactly {HF_MIRROR_ENDPOINT}")
    protocol, mechanics_result, crossfit_result = validate_protocol()
    base_model = base_model.expanduser().resolve(strict=True)
    dataset_root = dataset_root.expanduser().resolve(strict=True)
    frozen = protocol["frozen_inputs"]
    if sha256_file(base_model / "config.json") != frozen["base_config_sha256"]:
        raise ValueError("Pinned base-model config differs")
    resolved_output = output_dir.expanduser().resolve()
    fresh_error = (
        ValueError(f"Output identity-gate output must be fresh: {resolved_output}")
        if context.is_primary and resolved_output.exists()
        else None
    )
    distributed.phase_consensus(context, phase="output-gate-causal-fresh", error=fresh_error)
    create_error: BaseException | None = None
    if context.is_primary:
        try:
            resolved_output.mkdir(parents=True, exist_ok=False)
        except BaseException as error:
            create_error = error
    distributed.phase_consensus(context, phase="output-gate-causal-create", error=create_error)

    head, reconstruction = mechanics.reconstruct_crossfit_head()
    if reconstruction["reconstruction_state_sha256"] != RECONSTRUCTION_STATE_SHA256:
        raise RuntimeError("Reconstructed cross-fit head differs from signed training input")
    distributed.require_consensus(
        context,
        reconstruction["reconstruction_state_sha256"],
        description="reconstructed output identity head",
    )
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    model, tokenizer, model_audit = mechanics.base.load_model(
        base_model, device=context.device
    )
    installation = mechanics.install_reconstructed_output_gates(model, head)
    named_trainable = tuple(mechanics.selected_gates(model))
    if (
        len(named_trainable) != 168
        or sum(parameter.numel() for _, parameter in named_trainable) != 21504
        or any(parameter.dtype != torch.float32 for _, parameter in named_trainable)
    ):
        raise RuntimeError("Output identity-gate trainable isolation differs")

    training_rows = contrast.load_scene_rows(tokenizer, dataset_root)
    donor_mapping, donor_deltas, donor_payload = contrast.build_donor_mapping(training_rows)
    schedule, schedule_payload = contrast.build_schedule(
        training_rows, donor_mapping, donor_deltas
    )
    if (
        canonical_sha256(donor_payload) != frozen["training_donor_mapping_payload_sha256"]
        or canonical_sha256(schedule_payload[:UPDATES]) != TRAINING_SCHEDULE_PREFIX_SHA256
    ):
        raise RuntimeError("Output identity-gate training schedule differs")
    examples, rows_by_source, endpoint_mapping, split_payload = crossfit.authorized_examples(
        tokenizer, dataset_root
    )
    signed_split = dict(crossfit_result["crossfit_split"])
    signed_split.pop("payload_sha256")
    endpoint_payload = heldout_payload(rows_by_source, endpoint_mapping)
    if (
        split_payload != signed_split
        or canonical_sha256(split_payload) != CROSSFIT_SPLIT_PAYLOAD_SHA256
        or canonical_sha256(endpoint_payload) != HELDOUT_PAYLOAD_SHA256
    ):
        raise RuntimeError("Output identity-gate heldout endpoint binding differs")

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
        "reconstruction_state_sha256": reconstruction["reconstruction_state_sha256"],
        "seed": SEED,
        "updates": UPDATES,
        "world_size": context.world_size,
        "global_batch_size": GLOBAL_BATCH_SIZE,
        "local_rows": LOCAL_ROWS,
        "base_model": str(base_model),
        "dataset_root": str(dataset_root),
        "training_schedule_prefix_sha256": canonical_sha256(schedule_payload[:UPDATES]),
        "heldout_payload_sha256": canonical_sha256(endpoint_payload),
        "heldout_sources": list(HELDOUT_SOURCES),
        "hf_endpoint": os.environ.get("HF_ENDPOINT"),
        "rank_devices": list(context.rank_devices),
        "model_audit": model_audit,
        "installation": installation,
        "runner_sha256": sha256_file(Path(__file__)),
        "protected_splits_opened": [],
    }
    distributed.require_consensus(
        context, canonical_sha256(input_binding), description="output-gate causal input binding"
    )
    binding_error: BaseException | None = None
    if context.is_primary:
        try:
            write_json(resolved_output / "input_binding.json", input_binding)
        except BaseException as error:
            binding_error = error
    distributed.phase_consensus(
        context, phase="output-gate-causal-binding-save", error=binding_error
    )

    with training_bindings():
        training = causal_train.train(
            model,
            training_rows,
            schedule,
            updates=UPDATES,
            context=context,
            pad_token_id=int(tokenizer.pad_token_id),
            output_dir=resolved_output,
            named_trainable=named_trainable,
        )
    training_passed = bool(
        training["updates"] == UPDATES
        and training["trainable_subset_changed"] is True
        and training["recurrent_subset_changed"] is False
        and training["maximum_global_inactive_parameter_tensors"] == 0
        and training["projected_carrier_fixed_every_row"] is True
        and training["optimizer_state_cpu_offload"]["enabled"] is True
        and training["control_branch_graph_serialization"]["enabled"] is True
        and training["first_update_gradient_audit"]["passed"] is True
    )
    endpoint = evaluate_heldout_endpoint(
        model,
        examples,
        endpoint_mapping,
        context=context,
        pad_token_id=int(tokenizer.pad_token_id),
        protocol=protocol,
    )
    rank_runtime = distributed.gather_objects(
        context,
        {
            "rank": context.process_rank,
            "peak_cuda_memory_bytes": training["peak_cuda_memory_bytes"],
        },
    )
    save_error: BaseException | None = None
    checkpoint_audit: Mapping[str, Any] | None = None
    result: dict[str, Any] = {}
    if context.is_primary:
        try:
            checkpoint_audit = save_output_gate_checkpoint(
                named_trainable, resolved_output / "output_identity_gate.safetensors"
            )
            passed = bool(training_passed and endpoint["passed"])
            result = {
                "schema": SCHEMA,
                "status": PASS_STATUS if passed else FAIL_STATUS,
                "passed": passed,
                "generation_authorized": False,
                "native_generation_benchmark_authorized": False,
                "protocol_payload_sha256": PROTOCOL_PAYLOAD_SHA256,
                "seed": SEED,
                "updates": UPDATES,
                "input_binding": input_binding,
                "training": training,
                "training_passed": training_passed,
                "heldout_causal_endpoint": endpoint,
                "output_gate_checkpoint": checkpoint_audit,
                "delta_mem_adapter_saved": False,
                "rank_runtime": list(rank_runtime),
                "protected_splits_opened": [],
                "claim_policy": protocol["claim_policy"],
            }
            result["receipt"] = {
                "algorithm": "sha256",
                "payload_scope": "canonical_result_without_receipt",
                "payload_sha256": canonical_sha256(result),
            }
            write_json(resolved_output / "result.json", result)
        except BaseException as error:
            save_error = error
    distributed.phase_consensus(
        context, phase="output-gate-causal-result-save", error=save_error
    )
    del model, training_rows, examples
    gc.collect()
    torch.cuda.empty_cache()
    return result if context.is_primary else {
        "status": "worker_complete",
        "passed": bool(training_passed and endpoint["passed"]),
        "generation_authorized": False,
    }


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
        result = run(
            context=context,
            output_dir=args.output_dir,
            base_model=args.base_model,
            dataset_root=args.dataset_root,
        )
    finally:
        distributed.destroy_distributed_training(context)
    print(
        json.dumps(
            {
                "rank": context.process_rank,
                "status": result["status"],
                "passed": result["passed"],
                "generation_authorized": False,
            },
            ensure_ascii=True,
            sort_keys=True,
        ),
        flush=True,
    )
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
