#!/usr/bin/env python3

from __future__ import annotations

from contextlib import contextmanager
import json
import math
from pathlib import Path
import sys
from typing import Any, Iterator, Mapping, Sequence

import torch
from torch.utils.checkpoint import checkpoint


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[1]
for import_root in (SCRIPT_DIR, PROJECT_ROOT):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

from deltamem.core.delta import reset_delta_mem_states
from experiments.rethinking_rwkv_ms_gemma import rwkv_query_state_identity as identity
from experiments.rethinking_rwkv_ms_gemma import (
    run_natural_memory_native_rwkv_address_keyed_learned_write_causal_train as base,
)


shared = base.shared
causal_train = base.causal_train
contrast = shared.contrast
distributed = shared.distributed
evolution = shared.evolution

PROTOCOL = (
    SCRIPT_DIR
    / "natural_memory_native_rwkv_query_state_identity_causal_train_protocol_v1.json"
)
PROTOCOL_PAYLOAD_SHA256 = "5b3195da9922bbaf83171dc2a1fd8634691a1ed3ff55e4d0cfbddfb44746aff9"
SCHEMA = "rwkv_ms_natural_memory_native_query_state_identity_causal_train.v1"
STEP_SCHEMA = (
    "rwkv_ms_natural_memory_native_query_state_identity_causal_train_step.v1"
)
INPUT_SCHEMA = (
    "rwkv_ms_natural_memory_native_query_state_identity_causal_train_input.v1"
)
SEED = 110
UPDATES = 8
IDENTITY_MARGIN = 0.2
IDENTITY_WEIGHT = 1.0
HELDOUT_CANDIDATE_ROWS = 97
HELDOUT_ORDINALS = (
    522,
    657,
    1207,
    515,
    267,
    89,
    350,
    689,
    43,
    49,
    1416,
    455,
    184,
    883,
    643,
    56,
    1060,
    1326,
    996,
    1224,
    1153,
    1359,
    926,
    813,
    1292,
    1146,
    1341,
    445,
    1316,
    1135,
    1160,
    701,
)
HELDOUT_PAYLOAD_SHA256 = (
    "685ffaf479cdc9dbe6e4b48b4356335255ed2e5b90a4e6fc98fcf42ad515ce0a"
)
PASS_STATUS = "query_state_identity_heldout_passed_generation_authorized"
FAIL_STATUS = "query_state_identity_heldout_failed_generation_blocked"

_ORIGINAL_CHECKPOINTED_WRITE_READ = evolution.checkpointed_native_write_read
_ORIGINAL_BACKWARD_LOGITS = causal_train.backward_logits
_ORIGINAL_TRAIN = causal_train.train
_ORIGINAL_BUILD_DONOR_BATCH = contrast.build_donor_batch
_ORIGINAL_ENDPOINT = shared.evaluate_heldout_causal_endpoint

_pending_donor_batch: Any | None = None
_pending_identity: tuple[
    torch.nn.Module,
    torch.Tensor,
    torch.Tensor,
] | None = None
_identity_metrics: dict[str, float] = {}


def _reset_identity_metrics() -> None:
    _identity_metrics.clear()
    _identity_metrics.update(
        {
            "rows": 0.0,
            "positive_score_sum": 0.0,
            "donor_score_sum": 0.0,
            "hinge_sum": 0.0,
            "active_rows": 0.0,
            "projected_carrier_fixed_rows": 0.0,
        }
    )


def load_model(
    base_model: Path,
    *,
    device: torch.device,
    candidate: Mapping[str, Any] = base.SELECTED_CANDIDATE,
) -> tuple[torch.nn.Module, Any, Mapping[str, Any]]:
    model, tokenizer, inherited_audit = base.load_model(
        base_model,
        device=device,
        candidate=candidate,
    )
    capture_audit = identity.install(model)
    return model, tokenizer, {
        **dict(inherited_audit),
        "query_state_identity": capture_audit,
    }


def configure_trainable_parameters(
    model: torch.nn.Module,
) -> tuple[tuple[tuple[str, torch.nn.Parameter], ...], Mapping[str, Any]]:
    selected, inherited_audit = base.configure_trainable_parameters(model)
    audit = {
        **dict(inherited_audit),
        "query_state_identity_parameter_tensors": 0,
        "query_state_identity_forward_parameters": 0,
        "query_state_identity_uses_frozen_projected_address": True,
    }
    return selected, audit


def validate_protocol() -> Mapping[str, Any]:
    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    receipt = protocol.get("receipt", {})
    unsigned = dict(protocol)
    unsigned.pop("receipt", None)
    digest = base.stable.shared.canonical_sha256(unsigned)
    architecture = protocol.get("architecture", {})
    training = protocol.get("training", {})
    endpoint = protocol.get("heldout_causal_endpoint", {})
    if (
        digest != PROTOCOL_PAYLOAD_SHA256
        or receipt.get("payload_sha256") != digest
        or architecture.get("identity_score")
        != "cosine(frozen_target_projected_slot_write_address,addressed_rwkv_read)"
        or architecture.get("forward_output_changed") is not False
        or training.get("optimizer_updates") != UPDATES
        or training.get("global_batch_rows") != 4
        or training.get("identity_margin") != IDENTITY_MARGIN
        or training.get("identity_weight") != IDENTITY_WEIGHT
        or endpoint.get("candidate_rows_after_prior_endpoint_exclusions")
        != HELDOUT_CANDIDATE_ROWS
        or endpoint.get("source_ordinals") != list(HELDOUT_ORDINALS)
        or endpoint.get("source_donor_payload_sha256")
        != HELDOUT_PAYLOAD_SHA256
        or protocol.get("protected_splits_opened_by_this_protocol") != []
    ):
        raise ValueError("Query-state identity causal protocol differs")
    return protocol


def _tracked_build_donor_batch(*args: Any, **kwargs: Any) -> Any:
    global _pending_donor_batch
    donor_batch = _ORIGINAL_BUILD_DONOR_BATCH(*args, **kwargs)
    _pending_donor_batch = donor_batch
    return donor_batch


def _checkpointed_positive_identity_score(
    model: torch.nn.Module,
    target_batch: Any,
    *,
    dtype: torch.dtype,
) -> torch.Tensor:
    def score_positive(*tensors: torch.Tensor) -> torch.Tensor:
        target = evolution.NativeFullRowBatch(
            examples=target_batch.examples,
            write_input_ids=tensors[0],
            write_attention_mask=tensors[1],
            read_input_ids=tensors[2],
            read_attention_mask=tensors[3],
            labels=target_batch.labels,
        )
        identity.clear(model)
        evolution._native_write(model, target, dtype=dtype)
        target_addresses = identity.capture_write_addresses(model)
        identity.set_fixed_query_addresses(model, target_addresses)
        logits = evolution._native_read(model, target, dtype=dtype)
        score = identity.mean_score(identity.capture(model), target.labels)
        del logits
        identity.clear(model)
        reset_delta_mem_states(model)
        return score

    return checkpoint(
        score_positive,
        target_batch.write_input_ids,
        target_batch.write_attention_mask,
        target_batch.read_input_ids,
        target_batch.read_attention_mask,
        use_reentrant=False,
    )


def _checkpointed_donor_identity_score(
    model: torch.nn.Module,
    target_batch: Any,
    donor_batch: Any,
    *,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, Mapping[str, bool]]:
    modules = causal_train.ordered_modules(model)
    audit = {"projected_carrier_references_fixed": True}

    def score_donor(*tensors: torch.Tensor) -> torch.Tensor:
        target = evolution.NativeFullRowBatch(
            examples=target_batch.examples,
            write_input_ids=tensors[0],
            write_attention_mask=tensors[1],
            read_input_ids=tensors[2],
            read_attention_mask=tensors[3],
            labels=target_batch.labels,
        )
        donor = evolution.NativeFullRowBatch(
            examples=donor_batch.examples,
            write_input_ids=tensors[4],
            write_attention_mask=tensors[5],
            read_input_ids=tensors[2],
            read_attention_mask=tensors[3],
            labels=target_batch.labels,
        )
        identity.clear(model)
        evolution._native_write(model, target, dtype=dtype)
        correct_state = causal_train.capture_online_state_references(modules)
        target_addresses = identity.capture_write_addresses(model)

        evolution._native_write(model, donor, dtype=dtype)
        donor_state = causal_train.capture_online_state_references(modules)
        audit["projected_carrier_references_fixed"] = bool(
            audit["projected_carrier_references_fixed"]
            and causal_train.install_intervened_state(
                modules,
                projected=correct_state,
                recurrent=donor_state,
                rotate_recurrent_layers=False,
            )
        )
        identity.set_fixed_query_addresses(model, target_addresses)
        logits = evolution._native_read(model, target, dtype=dtype)
        score = identity.mean_score(identity.capture(model), target.labels)
        del logits
        identity.clear(model)
        reset_delta_mem_states(model)
        return score

    score = checkpoint(
        score_donor,
        target_batch.write_input_ids,
        target_batch.write_attention_mask,
        target_batch.read_input_ids,
        target_batch.read_attention_mask,
        donor_batch.write_input_ids,
        donor_batch.write_attention_mask,
        use_reentrant=False,
    )
    if audit["projected_carrier_references_fixed"] is not True:
        raise RuntimeError("Query-state identity changed the projected carrier")
    return score, audit


def _checkpointed_write_read_with_identity(
    model: torch.nn.Module,
    batch: Any,
    *,
    dtype: torch.dtype,
) -> tuple[Mapping[str, Any], torch.Tensor]:
    global _pending_donor_batch, _pending_identity
    if _pending_identity is not None:
        raise RuntimeError("Query-state identity loss was not consumed")
    result = _ORIGINAL_CHECKPOINTED_WRITE_READ(model, batch, dtype=dtype)
    donor_batch = _pending_donor_batch
    _pending_donor_batch = None
    if donor_batch is None:
        raise RuntimeError("Query-state identity donor batch is missing")
    positive_score = _checkpointed_positive_identity_score(
        model,
        batch,
        dtype=dtype,
    )
    donor_score, audit = _checkpointed_donor_identity_score(
        model,
        batch,
        donor_batch,
        dtype=dtype,
    )
    if not bool(torch.isfinite(torch.stack((positive_score, donor_score))).all()):
        raise RuntimeError("Query-state identity checkpoint produced non-finite scores")
    _pending_identity = (model, positive_score, donor_score)
    _identity_metrics["projected_carrier_fixed_rows"] += float(
        audit["projected_carrier_references_fixed"]
    )
    return result


def _backward_logits_with_identity(
    logits: torch.Tensor,
    labels: torch.Tensor,
    *,
    coefficient: float,
) -> tuple[int, int]:
    global _pending_identity
    result = _ORIGINAL_BACKWARD_LOGITS(logits, labels, coefficient=coefficient)
    pending = _pending_identity
    if pending is None:
        return result
    model, positive_score, donor_score = pending
    _pending_identity = None
    positive_value = float(positive_score.detach().item())
    donor_value = float(donor_score.detach().item())
    hinge_value = max(0.0, IDENTITY_MARGIN - positive_value + donor_value)
    scale = IDENTITY_WEIGHT / causal_train.GLOBAL_BATCH_SIZE
    identity.clear(model)
    reset_delta_mem_states(model)
    evolution.release_native_row_allocator_cache(logits.device)
    if hinge_value > 0.0:
        (-positive_score * scale).backward()
        identity.clear(model)
        reset_delta_mem_states(model)
        evolution.release_native_row_allocator_cache(logits.device)
        (donor_score * scale).backward()
    identity.clear(model)
    reset_delta_mem_states(model)
    evolution.release_native_row_allocator_cache(logits.device)
    _identity_metrics["rows"] += 1.0
    _identity_metrics["positive_score_sum"] += positive_value
    _identity_metrics["donor_score_sum"] += donor_value
    _identity_metrics["hinge_sum"] += hinge_value
    _identity_metrics["active_rows"] += float(hinge_value > 0.0)
    del positive_score, donor_score
    return result


def train_with_identity(*args: Any, **kwargs: Any) -> Mapping[str, Any]:
    _reset_identity_metrics()
    training = dict(_ORIGINAL_TRAIN(*args, **kwargs))
    context = kwargs["context"]
    gathered = distributed.gather_objects(context, dict(_identity_metrics))
    totals = {
        name: sum(float(rank_metrics[name]) for rank_metrics in gathered)
        for name in _identity_metrics
    }
    rows = int(totals["rows"])
    expected_rows = UPDATES * causal_train.GLOBAL_BATCH_SIZE
    summary = {
        "objective": "matched_donor_query_state_hinge",
        "margin": IDENTITY_MARGIN,
        "weight": IDENTITY_WEIGHT,
        "rows": rows,
        "mean_positive_score": totals["positive_score_sum"] / max(rows, 1),
        "mean_donor_score": totals["donor_score_sum"] / max(rows, 1),
        "mean_positive_minus_donor_score": (
            totals["positive_score_sum"] - totals["donor_score_sum"]
        )
        / max(rows, 1),
        "mean_hinge": totals["hinge_sum"] / max(rows, 1),
        "active_fraction": totals["active_rows"] / max(rows, 1),
        "projected_carrier_fixed_rows": int(
            totals["projected_carrier_fixed_rows"]
        ),
        "passed": bool(
            rows == expected_rows
            and totals["projected_carrier_fixed_rows"] == expected_rows
            and all(math.isfinite(value) for value in totals.values())
        ),
    }
    if summary["passed"] is not True:
        raise RuntimeError(f"Query-state identity training audit failed: {summary!r}")
    training["query_state_identity"] = summary
    return training


def _evaluate_identity_endpoint(
    model: torch.nn.Module,
    rows: Sequence[Any],
    donor_mapping: Mapping[int, int],
    *,
    context: Any,
    pad_token_id: int,
) -> Mapping[str, Any]:
    model.eval()
    modules = causal_train.ordered_modules(model)
    local_rows: list[dict[str, Any]] = []
    for endpoint_index, source_ordinal in enumerate(HELDOUT_ORDINALS):
        if endpoint_index % shared.WORLD_SIZE != context.process_rank:
            continue
        donor_ordinal = int(donor_mapping[source_ordinal])
        target_batch = evolution.collate_native_examples(
            [rows[source_ordinal].example],
            pad_token_id=pad_token_id,
            device=context.device,
        )
        donor_batch = _ORIGINAL_BUILD_DONOR_BATCH(
            target_batch,
            rows[donor_ordinal].example,
            device=context.device,
        )
        with torch.no_grad():
            identity.clear(model)
            evolution._native_write(model, target_batch, dtype=torch.bfloat16)
            correct_state = causal_train.capture_online_state_references(modules)
            target_addresses = identity.capture_write_addresses(model)
            identity.set_fixed_query_addresses(model, target_addresses)
            correct_logits = evolution._native_read(
                model,
                target_batch,
                dtype=torch.bfloat16,
            )
            positive = identity.capture(model)
            del correct_logits
            identity.clear(model)
            evolution._native_write(model, donor_batch, dtype=torch.bfloat16)
            donor_state = causal_train.capture_online_state_references(modules)
            carrier_fixed = causal_train.install_intervened_state(
                modules,
                projected=correct_state,
                recurrent=donor_state,
                rotate_recurrent_layers=False,
            )
            identity.set_fixed_query_addresses(model, target_addresses)
            donor_logits = evolution._native_read(
                model,
                target_batch,
                dtype=torch.bfloat16,
            )
            donor_read = identity.capture(model)
            del donor_logits
            positive_score, donor_score = identity.mean_scores(
                positive,
                donor_read,
                target_batch.labels,
            )
        local_rows.append(
            {
                "source_ordinal": source_ordinal,
                "donor_ordinal": donor_ordinal,
                "positive_score": float(positive_score.item()),
                "donor_score": float(donor_score.item()),
                "positive_minus_donor_score": float(
                    (positive_score - donor_score).item()
                ),
                "projected_carrier_fixed": bool(carrier_fixed),
            }
        )
        del target_batch, donor_batch, positive_score, donor_score
        identity.clear(model)
        reset_delta_mem_states(model)
        evolution.release_native_row_allocator_cache(context.device)
    gathered = distributed.gather_objects(context, local_rows)
    endpoint_rows = [row for rank_rows in gathered for row in rank_rows]
    endpoint_rows.sort(key=lambda row: HELDOUT_ORDINALS.index(row["source_ordinal"]))
    count = len(endpoint_rows)
    mean_positive = sum(row["positive_score"] for row in endpoint_rows) / max(count, 1)
    mean_donor = sum(row["donor_score"] for row in endpoint_rows) / max(count, 1)
    mean_margin = mean_positive - mean_donor
    positive_fraction = sum(
        row["positive_minus_donor_score"] > 0.0 for row in endpoint_rows
    ) / max(count, 1)
    checks = {
        "rows_complete": count == len(HELDOUT_ORDINALS),
        "projected_carrier_fixed_every_row": all(
            row["projected_carrier_fixed"] for row in endpoint_rows
        ),
        "mean_positive_minus_donor_score_positive": mean_margin > 0.0,
        "positive_row_fraction_at_least_half": positive_fraction >= 0.5,
    }
    return {
        "rows": count,
        "mean_positive_score": mean_positive,
        "mean_donor_score": mean_donor,
        "mean_positive_minus_donor_score": mean_margin,
        "positive_row_fraction": positive_fraction,
        "checks": checks,
        "passed": all(checks.values()),
        "rank_rows": list(gathered),
    }


def evaluate_heldout_causal_endpoint(
    model: torch.nn.Module,
    rows: Sequence[Any],
    donor_mapping: Mapping[int, int],
    *,
    context: Any,
    pad_token_id: int,
) -> Mapping[str, Any]:
    causal_endpoint = dict(
        _ORIGINAL_ENDPOINT(
            model,
            rows,
            donor_mapping,
            context=context,
            pad_token_id=pad_token_id,
        )
    )
    identity_endpoint = _evaluate_identity_endpoint(
        model,
        rows,
        donor_mapping,
        context=context,
        pad_token_id=pad_token_id,
    )
    checks = dict(causal_endpoint["checks"])
    checks["query_state_identity_endpoint_passed"] = identity_endpoint["passed"]
    causal_endpoint["checks"] = checks
    causal_endpoint["query_state_identity"] = identity_endpoint
    causal_endpoint["passed"] = bool(
        causal_endpoint["passed"] and identity_endpoint["passed"]
    )
    return causal_endpoint


@contextmanager
def training_bindings() -> Iterator[None]:
    global _pending_donor_batch, _pending_identity
    with base.training_bindings():
        overrides = {
            "PROTOCOL": PROTOCOL,
            "PROTOCOL_PAYLOAD_SHA256": PROTOCOL_PAYLOAD_SHA256,
            "SCHEMA": SCHEMA,
            "STEP_SCHEMA": STEP_SCHEMA,
            "INPUT_SCHEMA": INPUT_SCHEMA,
            "SEED": SEED,
            "UPDATES": UPDATES,
            "HELDOUT_ORDINALS": HELDOUT_ORDINALS,
            "HELDOUT_PAYLOAD_SHA256": HELDOUT_PAYLOAD_SHA256,
            "PASS_STATUS": PASS_STATUS,
            "FAIL_STATUS": FAIL_STATUS,
            "MODEL_LOADER": load_model,
            "TRAINABLE_CONFIGURER": configure_trainable_parameters,
            "TRAINING_FUNCTION": train_with_identity,
            "RUNNER_BINDING_PATH": Path(__file__),
            "validate_protocol": validate_protocol,
            "evaluate_heldout_causal_endpoint": evaluate_heldout_causal_endpoint,
        }
        previous = {name: getattr(shared, name) for name in overrides}
        previous_checkpointed = evolution.checkpointed_native_write_read
        previous_backward = causal_train.backward_logits
        previous_donor_builder = contrast.build_donor_batch
        _pending_donor_batch = None
        _pending_identity = None
        try:
            for name, value in overrides.items():
                setattr(shared, name, value)
            evolution.checkpointed_native_write_read = (
                _checkpointed_write_read_with_identity
            )
            causal_train.backward_logits = _backward_logits_with_identity
            contrast.build_donor_batch = _tracked_build_donor_batch
            yield
        finally:
            _pending_donor_batch = None
            _pending_identity = None
            contrast.build_donor_batch = previous_donor_builder
            causal_train.backward_logits = previous_backward
            evolution.checkpointed_native_write_read = previous_checkpointed
            for name, value in previous.items():
                setattr(shared, name, value)


def run(**kwargs: Any) -> Mapping[str, Any]:
    with training_bindings():
        return shared.run(**kwargs)


def main(argv: Sequence[str] | None = None) -> int:
    with training_bindings():
        return shared.main(argv)


if __name__ == "__main__":
    raise SystemExit(main())
