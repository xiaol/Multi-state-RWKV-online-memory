#!/usr/bin/env python3
"""Train same-space projected-value versus addressed-RWKV identity."""

from __future__ import annotations

import json
from pathlib import Path
import sys
from typing import Any, Iterator, Mapping, Sequence
from contextlib import contextmanager

import torch
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
    run_natural_memory_native_rwkv_query_state_identity_causal_train as base,
)


shared = base.shared
causal_train = base.causal_train
contrast = base.contrast
distributed = base.distributed
evolution = base.evolution
_BASE_LOAD_MODEL = base.load_model
_BASE_CONFIGURE_TRAINABLE_PARAMETERS = base.configure_trainable_parameters
SELECTED_CANDIDATE = dict(base.base.SELECTED_CANDIDATE)
_pending_target_batch: Any | None = None

PROTOCOL = SCRIPT_DIR / "natural_memory_native_rwkv_projected_value_identity_causal_train_protocol_v1.json"
PROTOCOL_PAYLOAD_SHA256 = "3dd5230bbd22a42b502e8e35ff0d2c1f65f934f23002e131f2c7554d9ab4b4b0"
SCHEMA = "rwkv_ms_natural_memory_native_projected_value_identity_causal_train.v1"
STEP_SCHEMA = "rwkv_ms_natural_memory_native_projected_value_identity_causal_train_step.v1"
INPUT_SCHEMA = "rwkv_ms_natural_memory_native_projected_value_identity_causal_train_input.v1"
SEED = 111
UPDATES = 8
IDENTITY_MARGIN = 0.2
IDENTITY_WEIGHT = 1.0
HELDOUT_CANDIDATE_ROWS = 17
HELDOUT_ORDINALS = (
    1002, 1431, 1161, 1128, 1189, 1232, 1437, 1220,
    718, 805, 1331, 546, 472, 973, 101, 1154,
)
HELDOUT_PAYLOAD_SHA256 = "94a7ce9e5c1ee6d649daa3d377e5433956ac891fc1ec13625f7e89b34f06d75b"
PASS_STATUS = "projected_value_identity_heldout_passed_generation_authorized"
FAIL_STATUS = "projected_value_identity_heldout_failed_generation_blocked"


def validate_protocol() -> Mapping[str, Any]:
    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    receipt = protocol.get("receipt", {})
    unsigned = dict(protocol)
    unsigned.pop("receipt", None)
    digest = distributed.canonical_sha256(unsigned)
    architecture = protocol.get("architecture", {})
    training = protocol.get("training", {})
    endpoint = protocol.get("heldout_causal_endpoint", {})
    if (
        digest != PROTOCOL_PAYLOAD_SHA256
        or receipt.get("payload_sha256") != digest
        or architecture.get("identity_target")
        != "detached_projected_slot_value_in_rwkv_state_read_space"
        or architecture.get("forward_output_changed") is not False
        or training.get("optimizer_updates") != UPDATES
        or training.get("global_batch_rows") != 4
        or training.get("identity_margin") != IDENTITY_MARGIN
        or endpoint.get("candidate_rows_after_prior_endpoint_exclusions")
        != HELDOUT_CANDIDATE_ROWS
        or endpoint.get("source_ordinals") != list(HELDOUT_ORDINALS)
        or endpoint.get("source_donor_payload_sha256") != HELDOUT_PAYLOAD_SHA256
        or protocol.get("protected_splits_opened_by_this_protocol") != []
    ):
        raise ValueError("Projected-value identity causal protocol differs")
    return protocol


def load_model(*args: Any, **kwargs: Any) -> tuple[torch.nn.Module, Any, Mapping[str, Any]]:
    return _BASE_LOAD_MODEL(*args, **kwargs)


def configure_trainable_parameters(
    model: torch.nn.Module,
) -> tuple[tuple[tuple[str, torch.nn.Parameter], ...], Mapping[str, Any]]:
    return _BASE_CONFIGURE_TRAINABLE_PARAMETERS(model)


def _reset_value_metrics() -> None:
    base._identity_metrics.clear()
    base._identity_metrics.update(
        {
            "rows": 0.0,
            "positive_score_sum": 0.0,
            "donor_score_sum": 0.0,
            "hinge_sum": 0.0,
            "active_rows": 0.0,
            "active_elements": 0.0,
            "elements": 0.0,
            "projected_carrier_fixed_rows": 0.0,
        }
    )


def _checkpointed_positive_value_score(
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
        value_identity.clear(model)
        evolution._native_write(model, target, dtype=dtype)
        target_values = value_identity.capture_write_values(model)
        value_identity.set_fixed_target_values(model, target_values)
        logits = evolution._native_read(model, target, dtype=dtype)
        score = value_identity.score_tensor(value_identity.capture(model), target.labels)
        del logits
        value_identity.clear(model)
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


def _checkpointed_donor_value_score(
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
        value_identity.clear(model)
        evolution._native_write(model, target, dtype=dtype)
        correct_state = causal_train.capture_online_state_references(modules)
        target_values = value_identity.capture_write_values(model)
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
        value_identity.set_fixed_target_values(model, target_values)
        logits = evolution._native_read(model, target, dtype=dtype)
        score = value_identity.score_tensor(value_identity.capture(model), target.labels)
        del logits
        value_identity.clear(model)
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
    return score, audit


def _checkpointed_write_read_with_value_identity(
    model: torch.nn.Module,
    batch: Any,
    *,
    dtype: torch.dtype,
) -> tuple[Mapping[str, Any], torch.Tensor]:
    global _pending_target_batch
    if _pending_target_batch is not None or base._pending_identity is not None:
        raise RuntimeError("Projected-value identity loss was not consumed")
    result = base._ORIGINAL_CHECKPOINTED_WRITE_READ(model, batch, dtype=dtype)
    if base._pending_donor_batch is None:
        raise RuntimeError("Projected-value identity donor batch is missing")
    # Defer both identity checkpoints until after the correct answer CE backward.
    # This keeps the per-token hinge graph from coexisting with the full-vocabulary
    # correct/control checkpoint graphs on the 40 GiB A100 budget.
    _pending_target_batch = (model, batch, dtype)
    return result


def _backward_logits_with_value_identity(
    logits: torch.Tensor,
    labels: torch.Tensor,
    *,
    coefficient: float,
) -> tuple[int, int]:
    global _pending_target_batch
    device = logits.device
    result = base._ORIGINAL_BACKWARD_LOGITS(logits, labels, coefficient=coefficient)
    # The full-vocabulary checkpoint graph is no longer needed after CE backward;
    # release its tensor reference before constructing the two identity graphs.
    del logits
    pending = _pending_target_batch
    if pending is None:
        return result
    _pending_target_batch = None
    model, target_batch, dtype = pending
    donor_batch = base._pending_donor_batch
    base._pending_donor_batch = None
    if donor_batch is None:
        raise RuntimeError("Projected-value identity donor batch is missing")
    with torch.no_grad():
        probe_positive = _checkpointed_positive_value_score(model, target_batch, dtype=dtype)
        probe_donor, audit = _checkpointed_donor_value_score(
            model, target_batch, donor_batch, dtype=dtype
        )
    if not bool(torch.isfinite(torch.cat((probe_positive.flatten(), probe_donor.flatten()))).all()):
        raise RuntimeError("Projected-value identity checkpoint produced non-finite scores")
    base._identity_metrics["projected_carrier_fixed_rows"] += float(
        audit["projected_carrier_references_fixed"]
    )
    active, hinge = value_identity.active_hinge(
        probe_positive, probe_donor, margin=IDENTITY_MARGIN
    )
    active = active.detach()
    positive_value = float(probe_positive.mean().item())
    donor_value = float(probe_donor.mean().item())
    hinge_value = float(hinge.detach().mean().item())
    score_elements = probe_positive.numel()
    scale = IDENTITY_WEIGHT / causal_train.GLOBAL_BATCH_SIZE / score_elements
    del probe_positive, probe_donor
    value_identity.clear(model)
    reset_delta_mem_states(model)
    evolution.release_native_row_allocator_cache(device)
    if bool(active.any().item()):
        positive_score = _checkpointed_positive_value_score(model, target_batch, dtype=dtype)
        (-positive_score * active).sum().mul(scale).backward()
        del positive_score
        value_identity.clear(model)
        reset_delta_mem_states(model)
        evolution.release_native_row_allocator_cache(device)
        donor_score, donor_audit = _checkpointed_donor_value_score(
            model, target_batch, donor_batch, dtype=dtype
        )
        (donor_score * active).sum().mul(scale).backward()
        del donor_score
        audit = donor_audit
    value_identity.clear(model)
    reset_delta_mem_states(model)
    evolution.release_native_row_allocator_cache(device)
    metrics = base._identity_metrics
    metrics["rows"] += 1.0
    metrics["positive_score_sum"] += positive_value
    metrics["donor_score_sum"] += donor_value
    metrics["hinge_sum"] += hinge_value
    metrics["active_rows"] += float(bool(active.any().item()))
    metrics["active_elements"] += float(active.sum().item())
    metrics["elements"] += float(active.numel())
    del active, hinge
    return result


def train_with_value_identity(*args: Any, **kwargs: Any) -> Mapping[str, Any]:
    _reset_value_metrics()
    training = dict(base._ORIGINAL_TRAIN(*args, **kwargs))
    context = kwargs["context"]
    gathered = distributed.gather_objects(context, dict(base._identity_metrics))
    totals = {
        name: sum(float(rank_metrics[name]) for rank_metrics in gathered)
        for name in base._identity_metrics
    }
    rows = int(totals["rows"])
    expected_rows = UPDATES * causal_train.GLOBAL_BATCH_SIZE
    summary = {
        "objective": "matched_donor_projected_value_identity_per_layer_token_hinge",
        "target": "detached_projected_slot_value",
        "margin": IDENTITY_MARGIN,
        "weight": IDENTITY_WEIGHT,
        "rows": rows,
        "mean_positive_score": totals["positive_score_sum"] / max(rows, 1),
        "mean_donor_score": totals["donor_score_sum"] / max(rows, 1),
        "mean_positive_minus_donor_score": (
            totals["positive_score_sum"] - totals["donor_score_sum"]
        ) / max(rows, 1),
        "mean_hinge": totals["hinge_sum"] / max(rows, 1),
        "active_fraction": totals["active_elements"] / max(totals["elements"], 1.0),
        "active_row_fraction": totals["active_rows"] / max(rows, 1),
        "projected_carrier_fixed_rows": int(totals["projected_carrier_fixed_rows"]),
        "passed": bool(
            rows == expected_rows
            and totals["projected_carrier_fixed_rows"] == expected_rows
            and all(torch.isfinite(torch.tensor(value)).item() for value in totals.values())
        ),
    }
    if summary["passed"] is not True:
        raise RuntimeError(f"Projected-value identity training audit failed: {summary!r}")
    training["projected_value_identity"] = summary
    return training


def _evaluate_value_identity_endpoint(
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
            [rows[source_ordinal].example], pad_token_id=pad_token_id, device=context.device
        )
        donor_batch = base._ORIGINAL_BUILD_DONOR_BATCH(
            target_batch, rows[donor_ordinal].example, device=context.device
        )
        with torch.no_grad():
            value_identity.clear(model)
            evolution._native_write(model, target_batch, dtype=torch.bfloat16)
            correct_state = causal_train.capture_online_state_references(modules)
            target_values = value_identity.capture_write_values(model)
            value_identity.set_fixed_target_values(model, target_values)
            correct_logits = evolution._native_read(model, target_batch, dtype=torch.bfloat16)
            del correct_logits
            positive = value_identity.capture(model)
            value_identity.clear(model)
            evolution._native_write(model, donor_batch, dtype=torch.bfloat16)
            donor_state = causal_train.capture_online_state_references(modules)
            carrier_fixed = causal_train.install_intervened_state(
                modules, projected=correct_state, recurrent=donor_state, rotate_recurrent_layers=False
            )
            value_identity.set_fixed_target_values(model, target_values)
            donor_logits = evolution._native_read(model, target_batch, dtype=torch.bfloat16)
            del donor_logits
            donor_read = value_identity.capture(model)
            positive_scores = value_identity.score_tensor(positive, target_batch.labels)
            donor_scores = value_identity.score_tensor(donor_read, target_batch.labels)
        row_margin = float((positive_scores - donor_scores).mean().item())
        local_rows.append(
            {
                "source_ordinal": source_ordinal,
                "donor_ordinal": donor_ordinal,
                "positive_score": float(positive_scores.mean().item()),
                "donor_score": float(donor_scores.mean().item()),
                "positive_minus_donor_score": row_margin,
                "positive_element_fraction": float(
                    (positive_scores > donor_scores).float().mean().item()
                ),
                "projected_carrier_fixed": bool(carrier_fixed),
            }
        )
        value_identity.clear(model)
        reset_delta_mem_states(model)
        evolution.release_native_row_allocator_cache(context.device)
    gathered = distributed.gather_objects(context, local_rows)
    endpoint_rows = [row for rank_rows in gathered for row in rank_rows]
    endpoint_rows.sort(key=lambda row: HELDOUT_ORDINALS.index(row["source_ordinal"]))
    count = len(endpoint_rows)
    mean_positive = sum(row["positive_score"] for row in endpoint_rows) / max(count, 1)
    mean_donor = sum(row["donor_score"] for row in endpoint_rows) / max(count, 1)
    mean_margin = mean_positive - mean_donor
    row_fraction = sum(row["positive_minus_donor_score"] > 0.0 for row in endpoint_rows) / max(count, 1)
    element_fraction = sum(row["positive_element_fraction"] for row in endpoint_rows) / max(count, 1)
    checks = {
        "rows_complete": count == len(HELDOUT_ORDINALS),
        "projected_carrier_fixed_every_row": all(row["projected_carrier_fixed"] for row in endpoint_rows),
        "mean_positive_minus_donor_score_positive": mean_margin > 0.0,
        "positive_row_fraction_at_least_half": row_fraction >= 0.5,
        "positive_element_fraction_at_least_half": element_fraction >= 0.5,
    }
    return {
        "rows": count,
        "mean_positive_score": mean_positive,
        "mean_donor_score": mean_donor,
        "mean_positive_minus_donor_score": mean_margin,
        "positive_row_fraction": row_fraction,
        "positive_element_fraction": element_fraction,
        "checks": checks,
        "passed": all(checks.values()),
        "rank_rows": list(gathered),
    }


def evaluate_heldout_causal_endpoint(*args: Any, **kwargs: Any) -> Mapping[str, Any]:
    causal_endpoint = dict(base._ORIGINAL_ENDPOINT(*args, **kwargs))
    identity_endpoint = _evaluate_value_identity_endpoint(*args, **kwargs)
    checks = dict(causal_endpoint["checks"])
    checks["projected_value_identity_endpoint_passed"] = identity_endpoint["passed"]
    causal_endpoint["checks"] = checks
    causal_endpoint["projected_value_identity"] = identity_endpoint
    causal_endpoint["passed"] = bool(causal_endpoint["passed"] and identity_endpoint["passed"])
    return causal_endpoint


@contextmanager
def training_bindings() -> Iterator[None]:
    global _pending_target_batch
    with base.training_bindings():
        previous = {
            "PROTOCOL": shared.PROTOCOL,
            "PROTOCOL_PAYLOAD_SHA256": shared.PROTOCOL_PAYLOAD_SHA256,
            "SCHEMA": shared.SCHEMA,
            "STEP_SCHEMA": shared.STEP_SCHEMA,
            "INPUT_SCHEMA": shared.INPUT_SCHEMA,
            "SEED": shared.SEED,
            "UPDATES": shared.UPDATES,
            "SELECTED_CANDIDATE": shared.SELECTED_CANDIDATE,
            "HELDOUT_ORDINALS": shared.HELDOUT_ORDINALS,
            "HELDOUT_PAYLOAD_SHA256": getattr(shared, "HELDOUT_PAYLOAD_SHA256", None),
            "PASS_STATUS": shared.PASS_STATUS,
            "FAIL_STATUS": shared.FAIL_STATUS,
            "MODEL_LOADER": shared.MODEL_LOADER,
            "TRAINABLE_CONFIGURER": shared.TRAINABLE_CONFIGURER,
            "TRAINING_FUNCTION": shared.TRAINING_FUNCTION,
            "RUNNER_BINDING_PATH": shared.RUNNER_BINDING_PATH,
            "validate_protocol": shared.validate_protocol,
            "evaluate_heldout_causal_endpoint": shared.evaluate_heldout_causal_endpoint,
        }
        previous_checkpointed = evolution.checkpointed_native_write_read
        previous_backward = causal_train.backward_logits
        previous_donor_builder = contrast.build_donor_batch
        base._pending_donor_batch = None
        base._pending_identity = None
        _pending_target_batch = None
        try:
            overrides = {
                "PROTOCOL": PROTOCOL,
                "PROTOCOL_PAYLOAD_SHA256": PROTOCOL_PAYLOAD_SHA256,
                "SCHEMA": SCHEMA,
                "STEP_SCHEMA": STEP_SCHEMA,
                "INPUT_SCHEMA": INPUT_SCHEMA,
                "SEED": SEED,
                "UPDATES": UPDATES,
                "SELECTED_CANDIDATE": SELECTED_CANDIDATE,
                "HELDOUT_ORDINALS": HELDOUT_ORDINALS,
                "HELDOUT_PAYLOAD_SHA256": HELDOUT_PAYLOAD_SHA256,
                "PASS_STATUS": PASS_STATUS,
                "FAIL_STATUS": FAIL_STATUS,
                "MODEL_LOADER": load_model,
                "TRAINABLE_CONFIGURER": configure_trainable_parameters,
                "TRAINING_FUNCTION": train_with_value_identity,
                "RUNNER_BINDING_PATH": Path(__file__),
                "validate_protocol": validate_protocol,
                "evaluate_heldout_causal_endpoint": evaluate_heldout_causal_endpoint,
            }
            for name, value in overrides.items():
                setattr(shared, name, value)
            evolution.checkpointed_native_write_read = _checkpointed_write_read_with_value_identity
            causal_train.backward_logits = _backward_logits_with_value_identity
            contrast.build_donor_batch = base._tracked_build_donor_batch
            yield
        finally:
            base._pending_donor_batch = None
            base._pending_identity = None
            _pending_target_batch = None
            contrast.build_donor_batch = previous_donor_builder
            causal_train.backward_logits = previous_backward
            evolution.checkpointed_native_write_read = previous_checkpointed
            for name, value in previous.items():
                if value is not None:
                    setattr(shared, name, value)


def run(**kwargs: Any) -> Mapping[str, Any]:
    with training_bindings():
        return shared.run(**kwargs)


def main(argv: Sequence[str] | None = None) -> int:
    with training_bindings():
        return shared.main(argv)


if __name__ == "__main__":
    raise SystemExit(main())
