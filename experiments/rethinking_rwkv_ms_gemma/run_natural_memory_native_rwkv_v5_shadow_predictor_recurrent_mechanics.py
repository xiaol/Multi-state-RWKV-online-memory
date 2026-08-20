#!/usr/bin/env python3
"""Run signed exact-v5 predictor cross-fit and recurrent shadow mechanics."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

import torch
import torch.distributed as dist
from torch.nn import functional as F


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[1]
for import_root in (SCRIPT_DIR, PROJECT_ROOT):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

from experiments.rethinking_rwkv_ms_gemma import (  # noqa: E402
    run_natural_memory_native_rwkv_v5_shadow_crossfit as shadow,
)
from deltamem.core.delta import reset_delta_mem_states  # noqa: E402
from experiments.rethinking_rwkv_ms_gemma import (  # noqa: E402
    rwkv_v5_shadow_certified_live_value as certified,
)


SCHEMA = "rwkv_ms_natural_memory_native_v5_shadow_predictor_recurrent_mechanics.v1"
FEATURE_SCHEMA = "rwkv_ms_natural_memory_native_v5_shadow_predictor_features.v1"
PROTOCOL = SCRIPT_DIR / "natural_memory_native_rwkv_v5_shadow_predictor_recurrent_mechanics_protocol_v1.json"
PROTOCOL_PAYLOAD_SHA256 = "c6f8393baf0b8ef778f6b9125c9b26887ac844bbd4f62213c0e2e39d7ff2630c"
WORLD_SIZE = 4
STAGE2_ROWS_PER_RANK = 11
PASSES = 8
SEED = 119
DISTRIBUTED_TIMEOUT_SECONDS = 1800
HEAD_SEED = shadow.HEAD_SEED
TRAIN_ROWS = shadow.TRAIN_ROWS
HELDOUT_ROWS = shadow.HELDOUT_ROWS
LAYERS = shadow.LAYERS
STATE_DIM = shadow.STATE_DIM
TRAIN_STEPS = shadow.TRAIN_STEPS
LEARNING_RATE = shadow.LEARNING_RATE
WEIGHT_DECAY = shadow.WEIGHT_DECAY
IDENTITY_MARGIN = shadow.IDENTITY_MARGIN
DONOR_FRACTION_GATE = 0.95
MEAN_GAP_GATE = 0.05
PERMUTED_FRACTION_GATE = 0.95
TAIL_RATIO = 0.95
MATERIAL_CONTROL_CONDITIONS = (
    "donor-shadow",
    "donor-pair",
    "zero",
    "layer-permuted",
    "row-shuffled",
    "norm-random",
    "fixed-correct-gate donor-value",
)
HF_ENDPOINT = shadow.HF_ENDPOINT
SIGNED_SOURCE_ROOT_ENV = shadow.SIGNED_SOURCE_ROOT_ENV
SIGNED_V5_COMMIT = shadow.SIGNED_V5_COMMIT
SIGNED_V5_DELTA_IMPL_SHA256 = shadow.SIGNED_V5_DELTA_IMPL_SHA256
SPLIT_SALT = shadow.SPLIT_SALT
CROSSFIT_ROOT = SCRIPT_DIR / "local_artifacts/natural_memory_native_rwkv_v5_shadow_crossfit_v1"
CROSSFIT_RESULT = CROSSFIT_ROOT / "result.json"
CROSSFIT_RESULT_SHA256 = "c3607fbc6f42b6a2ebcdfab7d5cdf399e5b8e4c8ab52a1c707e8f1d19d44108d"
CROSSFIT_RECEIPT = "4ba137387216a8f2bc2c5562a764b4f340afa795cc4dbc88d4d2cf0ea470443c"

distributed = shadow.distributed
evolution = shadow.evolution
causal_train = shadow.causal_train
contrast = shadow.contrast
endpoint = shadow.endpoint
hardware = shadow.hardware
value_identity = shadow.value_identity
bilinear = shadow.bilinear


def canonical_sha256(value: Any) -> str:
    return distributed.canonical_sha256(value)


def sha256_file(path: Path) -> str:
    return shadow.sha256_file(path)


def validate_execution_source() -> Mapping[str, Any]:
    return shadow.validate_execution_source()


def validate_protocol() -> tuple[Mapping[str, Any], Mapping[str, Any]]:
    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    unsigned = dict(protocol)
    receipt = unsigned.pop("receipt", {})
    digest = canonical_sha256(unsigned)
    frozen = protocol.get("frozen_inputs", {})
    stage1 = protocol.get("stage1_predictor_crossfit", {})
    gates = stage1.get("pass_gates", {})
    stage2 = protocol.get("stage2_recurrent_mechanics", {})
    hardware_spec = protocol.get("hardware", {})
    if (
        digest != PROTOCOL_PAYLOAD_SHA256
        or receipt.get("payload_sha256") != digest
        or frozen.get("required_source_root_environment") != SIGNED_SOURCE_ROOT_ENV
        or frozen.get("signed_v5_source_commit") != SIGNED_V5_COMMIT
        or frozen.get("signed_v5_delta_impl_sha256") != SIGNED_V5_DELTA_IMPL_SHA256
        or frozen.get("capture_seed") != SEED
        or frozen.get("head_seed") != HEAD_SEED
        or hardware_spec.get("world_size") != WORLD_SIZE
        or hardware_spec.get("stage2_rows_per_rank") != STAGE2_ROWS_PER_RANK
        or stage1.get("split_salt") != SPLIT_SALT
        or stage1.get("train_rows") != TRAIN_ROWS
        or stage1.get("heldout_rows") != HELDOUT_ROWS
        or stage1.get("fresh_capture_required") is not True
        or stage1.get("heldout_used_for_training_thresholds_or_selection") is not False
        or gates.get("donor_token_pairwise_positive_fraction_minimum") != DONOR_FRACTION_GATE
        or gates.get("donor_row_pairwise_positive_fraction_minimum") != DONOR_FRACTION_GATE
        or gates.get("donor_mean_gap_minimum") != MEAN_GAP_GATE
        or gates.get("layer_permuted_token_pairwise_positive_fraction_minimum") != PERMUTED_FRACTION_GATE
        or gates.get("layer_permuted_row_pairwise_positive_fraction_minimum") != PERMUTED_FRACTION_GATE
        or stage2.get("passes") != PASSES
        or protocol.get("model_or_adapter_training_authorized") is not False
        or protocol.get("generation_authorized") is not False
        or protocol.get("adapter_saved") is not False
        or protocol.get("protected_splits_opened_by_this_protocol") != []
    ):
        raise ValueError("Exact-v5 predictor recurrent mechanics protocol differs")
    if sha256_file(CROSSFIT_RESULT) != CROSSFIT_RESULT_SHA256:
        raise ValueError("Signed v5 shadow cross-fit result differs")
    crossfit = json.loads(CROSSFIT_RESULT.read_text(encoding="utf-8"))
    unsigned_crossfit = dict(crossfit)
    crossfit_receipt = unsigned_crossfit.pop("receipt", {})
    if (
        crossfit.get("passed") is not True
        or crossfit_receipt.get("payload_sha256") != CROSSFIT_RECEIPT
        or canonical_sha256(unsigned_crossfit) != CROSSFIT_RECEIPT
        or crossfit.get("causal_mechanics_design_authorized") is not True
        or crossfit.get("protected_splits_opened") != []
    ):
        raise ValueError("Signed v5 shadow result does not authorize mechanics")
    shadow.validate_protocol()
    return protocol, crossfit


def predictor_mask(labels: torch.Tensor) -> torch.Tensor:
    if labels.ndim != 2 or labels.size(1) < 2:
        raise ValueError("Predictor features require batched causal labels")
    mask = torch.zeros_like(labels, dtype=torch.bool)
    mask[:, :-1] = labels[:, 1:].ne(-100)
    if not bool(mask.any().item()):
        raise ValueError("Predictor features require supervised causal targets")
    return mask


def predictor_vectors(
    captured: Sequence[Any], labels: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    mask = predictor_mask(labels)
    queries: list[torch.Tensor] = []
    states: list[torch.Tensor] = []
    for read in captured:
        if tuple(read.query_address.shape[:2]) != tuple(labels.shape):
            raise ValueError("Predictor capture and labels differ")
        queries.append(read.query_address.float()[mask].detach())
        states.append(read.recurrent_read.float()[mask].detach())
    query = torch.stack(queries, dim=1)
    state = torch.stack(states, dim=1)
    expected = (int(mask.sum().item()), LAYERS, STATE_DIM)
    if tuple(query.shape) != expected or tuple(state.shape) != expected:
        raise RuntimeError("Predictor feature shape differs")
    if not bool(torch.isfinite(torch.cat((query.flatten(), state.flatten()))).all()):
        raise RuntimeError("Predictor features are non-finite")
    return query, state


def token_runtime(
    captured: Sequence[Any], labels: torch.Tensor
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    mask = predictor_mask(labels)
    queries: dict[str, torch.Tensor] = {}
    shadows: dict[str, torch.Tensor] = {}
    masks: dict[str, torch.Tensor] = {}
    for read in captured:
        expanded = mask.unsqueeze(-1)
        queries[read.module_name] = torch.where(
            expanded,
            read.query_address.detach(),
            torch.zeros_like(read.query_address),
        )
        shadows[read.module_name] = torch.where(
            expanded,
            read.recurrent_read.detach(),
            torch.zeros_like(read.recurrent_read),
        )
        masks[read.module_name] = mask.detach().clone()
    return queries, shadows, masks


def _capture_feature_condition(
    model: torch.nn.Module,
    target: Any,
    modules: Sequence[tuple[str, Any]],
    *,
    projected: Mapping[str, Mapping[str, torch.Tensor]],
    recurrent: Mapping[str, Mapping[str, torch.Tensor]],
    target_values: Mapping[str, torch.Tensor],
    rotate_recurrent_layers: bool,
) -> tuple[torch.Tensor, torch.Tensor, bool]:
    reset_delta_mem_states(model)
    fixed = causal_train.install_intervened_state(
        modules,
        projected=projected,
        recurrent=recurrent,
        rotate_recurrent_layers=rotate_recurrent_layers,
    )
    value_identity.set_fixed_target_values(model, dict(target_values))
    logits = evolution._native_read(model, target, dtype=torch.bfloat16)
    query, state = predictor_vectors(value_identity.capture(model), target.labels)
    del logits
    return query, state, bool(fixed)


def capture_predictor_row(
    model: torch.nn.Module,
    target_example: Any,
    donor_example: Any,
    *,
    pad_token_id: int,
    device: torch.device,
) -> Mapping[str, Any]:
    modules = causal_train.ordered_modules(model)
    target = evolution.collate_native_examples(
        [target_example], pad_token_id=pad_token_id, device=device
    )
    donor = contrast.build_donor_batch(target, donor_example, device=device)
    try:
        with torch.inference_mode():
            value_identity.clear(model)
            evolution._native_write(model, target, dtype=torch.bfloat16)
            target_state = shadow.clone_online_state(modules)
            target_values = {
                name: value.detach().clone()
                for name, value in value_identity.capture_write_values(model).items()
            }
            query, correct, correct_fixed = _capture_feature_condition(
                model,
                target,
                modules,
                projected=target_state,
                recurrent=target_state,
                target_values=target_values,
                rotate_recurrent_layers=False,
            )
            value_identity.clear(model)
            evolution._native_write(model, donor, dtype=torch.bfloat16)
            donor_state = shadow.clone_online_state(modules)
            donor_query, donor_shadow, donor_fixed = _capture_feature_condition(
                model,
                target,
                modules,
                projected=target_state,
                recurrent=donor_state,
                target_values=target_values,
                rotate_recurrent_layers=False,
            )
            permuted_query, permuted_shadow, permuted_fixed = _capture_feature_condition(
                model,
                target,
                modules,
                projected=target_state,
                recurrent=target_state,
                target_values=target_values,
                rotate_recurrent_layers=True,
            )
        if not (
            correct_fixed
            and donor_fixed
            and permuted_fixed
            and torch.equal(query, donor_query)
            and torch.equal(query, permuted_query)
        ):
            raise RuntimeError("Predictor query or projected carrier changed")
        return {
            "query": query.cpu().tolist(),
            "correct": correct.cpu().tolist(),
            "matched_donor": donor_shadow.cpu().tolist(),
            "layer_permuted": permuted_shadow.cpu().tolist(),
            "predictor_tokens": int(query.size(0)),
            "feature_positions": "labels[:,1:]_shifted_one_token_left",
            "projected_carrier_fixed": True,
            "state_snapshots_detached_and_cloned": True,
            "binder_or_feedback_installed": False,
        }
    finally:
        value_identity.clear(model)
        reset_delta_mem_states(model)
        evolution.release_native_row_allocator_cache(device)


LayerwiseBilinear = shadow.LayerwiseBilinear


def _feature_tensors(
    records: Sequence[Mapping[str, Any]], split: str
) -> tuple[dict[str, torch.Tensor], tuple[int, ...]]:
    selected = [
        row for row in sorted(records, key=lambda item: int(item["source_index"]))
        if row["split"] == split
    ]
    lengths = tuple(int(row["predictor_tokens"]) for row in selected)
    feature = {
        name: torch.cat(
            [torch.tensor(row[name], dtype=torch.float32) for row in selected], dim=0
        )
        for name in ("query", "correct", "matched_donor", "layer_permuted")
    }
    return feature, lengths


def _gap_metrics(gap: torch.Tensor, row_lengths: Sequence[int]) -> Mapping[str, Any]:
    token_gap = gap.mean(dim=1)
    row_values = torch.stack(
        [part.mean() for part in token_gap.split(tuple(int(value) for value in row_lengths))]
    )
    return {
        "tokens": int(token_gap.numel()),
        "rows": int(row_values.numel()),
        "mean_gap": float(row_values.mean().item()),
        "token_mean_gap": float(token_gap.mean().item()),
        "token_pairwise_positive_fraction": float(token_gap.gt(0).float().mean().item()),
        "row_pairwise_positive_fraction": float(row_values.gt(0).float().mean().item()),
        "finite": bool(torch.isfinite(gap).all().item()),
    }


def predictor_score_metrics(
    head: LayerwiseBilinear,
    query: torch.Tensor,
    correct: torch.Tensor,
    negative: torch.Tensor,
    row_lengths: Sequence[int],
) -> Mapping[str, Any]:
    with torch.no_grad():
        gap = head.score(query, correct) - head.score(query, negative)
    return _gap_metrics(gap, row_lengths)


def derive_train_only_thresholds(
    head: LayerwiseBilinear, train: Mapping[str, torch.Tensor]
) -> torch.Tensor:
    with torch.no_grad():
        correct = head.score(train["query"], train["correct"])
        donor = head.score(train["query"], train["matched_donor"])
        permuted = head.score(train["query"], train["layer_permuted"])
        negative = torch.maximum(donor, permuted)
        low_correct = torch.quantile(correct, 0.05, dim=0)
        high_negative = torch.quantile(negative, 0.95, dim=0)
        thresholds = 0.5 * (low_correct + high_negative)
    if tuple(thresholds.shape) != (LAYERS,) or not bool(torch.isfinite(thresholds).all()):
        raise RuntimeError("Train-only predictor thresholds differ")
    return thresholds


def fit_predictor_head(
    records: Sequence[Mapping[str, Any]],
) -> tuple[LayerwiseBilinear, torch.Tensor, Mapping[str, Any]]:
    train_rows = sum(row["split"] == "train" for row in records)
    heldout_rows = sum(row["split"] == "heldout" for row in records)
    if train_rows != TRAIN_ROWS or heldout_rows != HELDOUT_ROWS:
        raise RuntimeError("Predictor cross-fit row counts differ")
    train, train_lengths = _feature_tensors(records, "train")
    heldout, heldout_lengths = _feature_tensors(records, "heldout")
    torch.manual_seed(HEAD_SEED)
    head = LayerwiseBilinear()
    optimizer = torch.optim.AdamW(
        head.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )
    losses: list[float] = []
    for _ in range(TRAIN_STEPS):
        optimizer.zero_grad(set_to_none=True)
        correct = head.score(train["query"], train["correct"])
        donor = head.score(train["query"], train["matched_donor"])
        permuted = head.score(train["query"], train["layer_permuted"])
        loss = F.relu(IDENTITY_MARGIN - correct + donor).mean()
        loss = loss + F.relu(IDENTITY_MARGIN - correct + permuted).mean()
        if not bool(torch.isfinite(loss).item()):
            raise RuntimeError("Predictor head loss is non-finite")
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach().item()))
    thresholds = derive_train_only_thresholds(head, train)
    metrics = {
        split_name: {
            negative: predictor_score_metrics(
                head,
                features["query"],
                features["correct"],
                features[field],
                lengths,
            )
            for negative, field in (
                ("donor", "matched_donor"),
                ("layer_permuted", "layer_permuted"),
            )
        }
        for split_name, features, lengths in (
            ("train", train, train_lengths),
            ("heldout", heldout, heldout_lengths),
        )
    }
    donor = metrics["heldout"]["donor"]
    permuted = metrics["heldout"]["layer_permuted"]
    checks = {
        "heldout_donor_token_fraction": donor["token_pairwise_positive_fraction"] >= DONOR_FRACTION_GATE,
        "heldout_donor_row_fraction": donor["row_pairwise_positive_fraction"] >= DONOR_FRACTION_GATE,
        "heldout_donor_mean_gap": donor["mean_gap"] >= MEAN_GAP_GATE,
        "heldout_layer_permuted_token_fraction": permuted["token_pairwise_positive_fraction"] >= PERMUTED_FRACTION_GATE,
        "heldout_layer_permuted_row_fraction": permuted["row_pairwise_positive_fraction"] >= PERMUTED_FRACTION_GATE,
        "all_heldout_scores_finite": donor["finite"] and permuted["finite"],
    }
    analysis = {
        "head": bilinear.audit_payload(STATE_DIM, shadow.BOTTLENECK),
        "optimizer": {
            "name": "AdamW",
            "seed": HEAD_SEED,
            "steps": TRAIN_STEPS,
            "learning_rate": LEARNING_RATE,
            "weight_decay": WEIGHT_DECAY,
            "identity_margin": IDENTITY_MARGIN,
        },
        "loss": {"initial": losses[0], "final": losses[-1]},
        "metrics": metrics,
        "thresholds": {
            "source": "train_only",
            "method": "midpoint_q05_correct_q95_max_negative_per_layer",
            "values": thresholds.tolist(),
        },
        "checks": checks,
        "passed": all(checks.values()),
        "head_weights_saved": False,
    }
    return head, thresholds, analysis


def train_and_evaluate(records: Sequence[Mapping[str, Any]]) -> Mapping[str, Any]:
    return fit_predictor_head(records)[2]


def append_jsonl(path: Path, value: Mapping[str, Any]) -> None:
    shadow.append_jsonl(path, value)


def load_feature_records(
    output_dir: Path, split_payload: Mapping[str, Any]
) -> tuple[list[Mapping[str, Any]], list[Mapping[str, Any]]]:
    expected_by_source = {
        int(row["source_index"]): row for row in split_payload["rows"]
    }
    records: list[Mapping[str, Any]] = []
    provenance: list[Mapping[str, Any]] = []
    for rank in range(WORLD_SIZE):
        path = output_dir / f"stage1-shard-{rank}.jsonl"
        rows = [
            json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        if any(int(row["source_index"]) % WORLD_SIZE != rank for row in rows):
            raise RuntimeError("Predictor feature shard rank assignment differs")
        records.extend(rows)
        provenance.append({"path": str(path), "rows": len(rows), "sha256": sha256_file(path)})
    sources = [int(row["source_index"]) for row in records]
    if len(records) != endpoint.EVALUATION_ROWS or len(set(sources)) != len(sources):
        raise RuntimeError("Predictor feature shard coverage differs")
    for row in records:
        source_index = int(row["source_index"])
        expected = expected_by_source[source_index]
        expected_tokens = len(row["query"])
        tensors = tuple(
            torch.as_tensor(row[name], dtype=torch.float32)
            for name in ("query", "correct", "matched_donor", "layer_permuted")
        )
        if (
            row.get("schema") != FEATURE_SCHEMA
            or row.get("split") != expected["split"]
            or row.get("row_sha256") != expected["row_sha256"]
            or row.get("donor_source_index") != expected["donor_source_index"]
            or row.get("donor_row_sha256") != expected["donor_row_sha256"]
            or row.get("predictor_tokens") != expected_tokens
            or row.get("feature_positions") != "labels[:,1:]_shifted_one_token_left"
            or row.get("projected_carrier_fixed") is not True
            or row.get("state_snapshots_detached_and_cloned") is not True
            or row.get("binder_or_feedback_installed") is not False
            or any(
                tuple(tensor.shape) != (expected_tokens, LAYERS, STATE_DIM)
                for tensor in tensors
            )
            or not bool(torch.isfinite(torch.stack(tensors)).all().item())
        ):
            raise RuntimeError("Predictor feature row contract differs")
    return records, provenance


def clone_state(
    state: Mapping[str, Mapping[str, torch.Tensor]]
) -> dict[str, dict[str, torch.Tensor]]:
    return {
        name: {attribute: tensor.detach().clone() for attribute, tensor in values.items()}
        for name, values in state.items()
    }


def state_sha256(state: Mapping[str, Mapping[str, torch.Tensor]]) -> str:
    digest = hashlib.sha256()
    for name in sorted(state):
        for attribute in sorted(state[name]):
            tensor = state[name][attribute].detach().contiguous().cpu()
            digest.update(name.encode("utf-8"))
            digest.update(attribute.encode("utf-8"))
            digest.update(str(tensor.dtype).encode("ascii"))
            digest.update(json.dumps(list(tensor.shape)).encode("ascii"))
            digest.update(tensor.view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def zero_recurrent(
    state: Mapping[str, Mapping[str, torch.Tensor]]
) -> dict[str, dict[str, torch.Tensor]]:
    return {
        name: {
            attribute: (
                torch.zeros_like(values[attribute])
                if attribute == "delta_state"
                else values[attribute].detach().clone()
            )
            for attribute in causal_train.RECURRENT_ATTRIBUTES
        }
        for name, values in state.items()
    }


def row_shuffle_recurrent(
    state: Mapping[str, Mapping[str, torch.Tensor]]
) -> dict[str, dict[str, torch.Tensor]]:
    return {
        name: {
            attribute: values[attribute].detach().roll(1, dims=0).clone()
            for attribute in causal_train.RECURRENT_ATTRIBUTES
        }
        for name, values in state.items()
    }


def norm_random_recurrent(
    state: Mapping[str, Mapping[str, torch.Tensor]]
) -> dict[str, dict[str, torch.Tensor]]:
    result: dict[str, dict[str, torch.Tensor]] = {}
    for name, values in state.items():
        recurrent = values["delta_state"]
        random = torch.randn_like(recurrent.float())
        random = random * (
            recurrent.float().norm(dim=-1, keepdim=True)
            / random.norm(dim=-1, keepdim=True).clamp_min(1e-6)
        )
        result[name] = {
            "delta_state": random.to(dtype=recurrent.dtype),
            "rwkv_ms_positions": values["rwkv_ms_positions"].detach().clone(),
            "rwkv_ms_previous_source": values["rwkv_ms_previous_source"].detach().clone(),
        }
    return result


def reads_are_write_disabled(modules: Sequence[tuple[str, Any]]) -> bool:
    return all(getattr(module, "write_enabled", None) is False for _, module in modules)


def recurrent_references_fixed(
    modules: Sequence[tuple[str, Any]],
    recurrent: Mapping[str, Mapping[str, torch.Tensor]],
    *,
    rotate_recurrent_layers: bool,
) -> bool:
    names = [name for name, _ in modules]
    return all(
        getattr(module, attribute)
        is recurrent[
            names[(index + 1) % len(names)] if rotate_recurrent_layers else name
        ][attribute]
        for index, (name, module) in enumerate(modules)
        for attribute in causal_train.RECURRENT_ATTRIBUTES
    )


@torch.no_grad()
def capture_runtime_shadow(
    model: torch.nn.Module,
    target: Any,
    modules: Sequence[tuple[str, Any]],
    *,
    projected: Mapping[str, Mapping[str, torch.Tensor]],
    recurrent: Mapping[str, Mapping[str, torch.Tensor]],
    target_values: Mapping[str, torch.Tensor],
    rotate_recurrent_layers: bool = False,
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    reset_delta_mem_states(model)
    fixed = causal_train.install_intervened_state(
        modules,
        projected=projected,
        recurrent=recurrent,
        rotate_recurrent_layers=rotate_recurrent_layers,
    )
    if not fixed:
        raise RuntimeError("Stage2 projected carrier reference changed")
    value_identity.set_fixed_target_values(model, dict(target_values))
    logits = evolution._native_read(model, target, dtype=torch.bfloat16)
    if not reads_are_write_disabled(modules):
        raise RuntimeError("A stage2 shadow read left memory writes enabled")
    runtime = token_runtime(value_identity.capture(model), target.labels)
    del logits
    return runtime


def _token_ce_masks(
    logits: torch.Tensor, labels: torch.Tensor
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    selected_mask = labels[:, 1:].ne(-100)
    predictor_indices = selected_mask.any(dim=0).nonzero(as_tuple=False).flatten()
    if logits.ndim == 2:
        logits = logits.unsqueeze(0)
    if logits.size(1) == labels.size(1):
        selected_logits = logits.index_select(1, predictor_indices)
    elif logits.size(1) == predictor_indices.numel():
        selected_logits = logits
    else:
        raise ValueError("Stage2 logits do not cover causal predictors")
    selected_labels = labels.index_select(1, predictor_indices + 1)
    valid = selected_labels.ne(-100)
    token_loss = F.cross_entropy(
        selected_logits.float().reshape(-1, selected_logits.size(-1)),
        selected_labels.reshape(-1),
        ignore_index=-100,
        reduction="none",
    ).reshape_as(selected_labels)
    first = torch.zeros_like(valid)
    for row in range(valid.size(0)):
        indices = valid[row].nonzero(as_tuple=False).flatten()
        if indices.numel():
            first[row, indices[0]] = True
    later = valid & ~first

    return token_loss, {"overall": valid, "first": first, "later": later}


def _ce_parts(logits: torch.Tensor, labels: torch.Tensor) -> Mapping[str, Any]:
    token_loss, masks = _token_ce_masks(logits, labels)

    def part(mask: torch.Tensor) -> Mapping[str, Any]:
        count = int(mask.sum().item())
        total = float(token_loss.masked_select(mask).sum().item())
        return {"sum": total, "tokens": count, "mean": None if count == 0 else total / count}

    return {name: part(mask) for name, mask in masks.items()}


def _per_row_ce(logits: torch.Tensor, labels: torch.Tensor) -> Mapping[str, list[float | None]]:
    token_loss, masks = _token_ce_masks(logits, labels)
    result: dict[str, list[float | None]] = {}
    for name, mask in masks.items():
        rows: list[float | None] = []
        for row in range(mask.size(0)):
            count = int(mask[row].sum().item())
            rows.append(
                None
                if count == 0
                else float(token_loss[row].masked_select(mask[row]).mean().item())
            )
        result[name] = rows
    return result


def compare_to_correct(
    logits: torch.Tensor,
    correct_logits: torch.Tensor,
    labels: torch.Tensor,
) -> Mapping[str, Any]:
    if logits.ndim == 2:
        logits = logits.unsqueeze(0)
    if correct_logits.ndim == 2:
        correct_logits = correct_logits.unsqueeze(0)
    if tuple(logits.shape) != tuple(correct_logits.shape):
        raise ValueError("Condition and correct predictor logits differ in shape")
    predictor_indices = labels[:, 1:].ne(-100).any(dim=0).nonzero(as_tuple=False).flatten()
    valid = labels.index_select(1, predictor_indices + 1).ne(-100)
    changed = logits.ne(correct_logits).any(dim=-1) & valid
    changed_fractions = [
        float(changed[row].sum().item() / valid[row].sum().item())
        for row in range(valid.size(0))
    ]
    condition_ce = _per_row_ce(logits, labels)
    correct_ce = _per_row_ce(correct_logits, labels)
    ce_delta_by_row: dict[str, list[float | None]] = {}
    positive: dict[str, float | None] = {}
    for part in ("overall", "first", "later"):
        deltas = [
            None if condition is None or correct is None else condition - correct
            for condition, correct in zip(condition_ce[part], correct_ce[part], strict=True)
        ]
        finite = [value for value in deltas if value is not None]
        ce_delta_by_row[part] = deltas
        positive[part] = (
            None
            if not finite
            else sum(value > 0.0 for value in finite) / len(finite)
        )
    return {
        "predictor_logit_changed_fraction_by_row": changed_fractions,
        "mean_predictor_logit_changed_fraction": sum(changed_fractions) / len(changed_fractions),
        "rows_at_least_095_changed_fraction": sum(
            value >= 0.95 for value in changed_fractions
        )
        / len(changed_fractions),
        "ce_delta_by_row": ce_delta_by_row,
        "ce_positive_row_fraction": positive,
    }


def _last_values(
    modules: Sequence[tuple[str, Any]],
) -> dict[str, torch.Tensor]:
    values: dict[str, torch.Tensor] = {}
    for name, module in modules:
        value = module.rwkv_v5_shadow_last_value
        if value is None:
            raise RuntimeError(f"Stage2 live value missing for {name}")
        values[name] = value.detach().clone()
    return values


def _captured_live_values(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {
        read.module_name: read.recurrent_read.detach().clone()
        for read in value_identity.capture(model)
    }


def _last_gates(modules: Sequence[tuple[str, Any]]) -> dict[str, torch.Tensor]:
    gates: dict[str, torch.Tensor] = {}
    for name, module in modules:
        gate = module.rwkv_v5_shadow_last_gate
        if gate is None:
            raise RuntimeError(f"Stage2 gate missing for {name}")
        gates[name] = gate.detach().clone()
    return gates


def _live_delta(
    previous: Mapping[str, torch.Tensor],
    current: Mapping[str, torch.Tensor],
    masks: Mapping[str, torch.Tensor],
) -> Mapping[str, Any]:
    norms = []
    for name in sorted(current):
        difference = (current[name].float() - previous[name].float()).norm(dim=-1)
        norms.append(difference.masked_select(masks[name]))
    values = torch.cat(norms)
    return {
        "sum": float(values.sum().item()),
        "vectors": int(values.numel()),
        "mean": float(values.mean().item()),
    }


def _predictor_only_values(
    values: Mapping[str, torch.Tensor], masks: Mapping[str, torch.Tensor]
) -> dict[str, torch.Tensor]:
    return {
        name: torch.where(
            masks[name].unsqueeze(-1),
            value.detach(),
            torch.zeros_like(value),
        )
        for name, value in values.items()
    }


@torch.no_grad()
def run_condition_passes(
    model: torch.nn.Module,
    target: Any,
    modules: Sequence[tuple[str, Any]],
    *,
    projected: Mapping[str, Mapping[str, torch.Tensor]],
    recurrent: Mapping[str, Mapping[str, torch.Tensor]],
    target_values: Mapping[str, torch.Tensor],
    queries: Mapping[str, torch.Tensor],
    shadows: Mapping[str, torch.Tensor],
    masks: Mapping[str, torch.Tensor],
    rotate_recurrent_layers: bool = False,
    feedback: bool = True,
    binder_enabled: bool = True,
    gate_overrides_by_pass: Sequence[Mapping[str, torch.Tensor]] | None = None,
    live_value_overrides: Mapping[str, torch.Tensor] | None = None,
) -> Mapping[str, Any]:
    passes: list[Mapping[str, Any]] = []
    logits_by_pass: list[torch.Tensor] = []
    values_by_pass: list[dict[str, torch.Tensor]] = []
    gates_by_pass: list[dict[str, torch.Tensor]] = []
    for pass_index in range(PASSES):
        reset_delta_mem_states(model)
        fixed = causal_train.install_intervened_state(
            modules,
            projected=projected,
            recurrent=recurrent,
            rotate_recurrent_layers=rotate_recurrent_layers,
        )
        if not fixed:
            raise RuntimeError("Stage2 projected carrier changed during a read")
        recurrent_fixed = recurrent_references_fixed(
            modules,
            recurrent,
            rotate_recurrent_layers=rotate_recurrent_layers,
        )
        if not recurrent_fixed:
            raise RuntimeError("Stage2 recurrent state reference changed before a read")
        value_identity.set_fixed_target_values(model, dict(target_values))
        if binder_enabled:
            certified.set_runtime(
                modules,
                queries=queries,
                shadows=shadows,
                seq_len=int(target.read_input_ids.size(1)),
                gate_overrides=(
                    None if gate_overrides_by_pass is None
                    else gate_overrides_by_pass[pass_index]
                ),
                correction_masks=masks,
                live_value_overrides=live_value_overrides,
            )
        else:
            certified.clear_runtime(modules)
        if feedback and pass_index > 0:
            certified.set_shifted_feedback(
                modules,
                _predictor_only_values(values_by_pass[-1], masks),
            )
        else:
            certified.clear_shifted_feedback(modules)
        logits = evolution._native_read(model, target, dtype=torch.bfloat16).detach().clone()
        write_disabled = reads_are_write_disabled(modules)
        if not write_disabled:
            raise RuntimeError("A stage2 condition read left memory writes enabled")
        values = (
            _last_values(modules)
            if binder_enabled
            else _captured_live_values(model)
        )
        gates = _last_gates(modules) if binder_enabled else {}
        delta = None if not values_by_pass else _live_delta(values_by_pass[-1], values, masks)
        passes.append(
            {
                "pass": pass_index + 1,
                "ce": _ce_parts(logits, target.labels),
                "live_value_delta_from_previous": delta,
                "logits_finite": bool(torch.isfinite(logits).all().item()),
                "write_enabled_after_read": False,
                "recurrent_state_references_fixed_before_read": True,
            }
        )
        logits_by_pass.append(logits)
        values_by_pass.append(values)
        gates_by_pass.append(gates)
        certified.clear_runtime(modules)
        certified.clear_shifted_feedback(modules)
    delta_means = [
        item["live_value_delta_from_previous"]["mean"]
        for item in passes[1:]
    ]
    tail = [
        delta_means[index + 1] <= TAIL_RATIO * delta_means[index]
        for index in range(len(delta_means) - 4, len(delta_means) - 1)
    ]
    pass2_logits_differ = not torch.equal(logits_by_pass[1], logits_by_pass[0])
    pass2_live_value_delta_positive = delta_means[0] > 0.0
    return {
        "passes": passes,
        "_logits_by_pass": logits_by_pass,
        "gates_by_pass": gates_by_pass,
        "pass2_logits_differ": pass2_logits_differ,
        "pass2_live_value_delta_positive": pass2_live_value_delta_positive,
        "pass2_differs_pass1": (
            pass2_logits_differ and pass2_live_value_delta_positive
        ),
        "disabled_exact_collapse": all(
            torch.equal(logits_by_pass[0], logits) for logits in logits_by_pass[1:]
        ),
        "tail_contraction_checks": tail,
        "tail_contracted": all(tail),
    }


def _donor_batch(target: Any, donor_examples: Sequence[Any], *, pad_token_id: int, device: torch.device) -> Any:
    donor_writes = evolution.collate_native_examples(
        donor_examples, pad_token_id=pad_token_id, device=device
    )
    return evolution.NativeFullRowBatch(
        examples=target.examples,
        write_input_ids=donor_writes.write_input_ids,
        write_attention_mask=donor_writes.write_attention_mask,
        read_input_ids=target.read_input_ids,
        read_attention_mask=target.read_attention_mask,
        labels=target.labels,
    )


def run_stage2(
    model: torch.nn.Module,
    target_examples: Sequence[Any],
    donor_examples: Sequence[Any],
    head: LayerwiseBilinear,
    thresholds: torch.Tensor,
    *,
    pad_token_id: int,
    device: torch.device,
) -> Mapping[str, Any]:
    if len(target_examples) != STAGE2_ROWS_PER_RANK or len(donor_examples) != STAGE2_ROWS_PER_RANK:
        raise RuntimeError("Stage2 requires exactly eleven heldout rows per rank")
    modules = causal_train.ordered_modules(model)
    target = evolution.collate_native_examples(
        target_examples, pad_token_id=pad_token_id, device=device
    )
    donor = _donor_batch(
        target, donor_examples, pad_token_id=pad_token_id, device=device
    )
    value_identity.clear(model)
    reset_delta_mem_states(model)
    with torch.inference_mode():
        evolution._native_write(model, target, dtype=torch.bfloat16)
        target_state = shadow.clone_online_state(modules)
        target_values = {
            name: value.detach().clone()
            for name, value in value_identity.capture_write_values(model).items()
        }
        value_identity.clear(model)
        evolution._native_write(model, donor, dtype=torch.bfloat16)
        donor_state = shadow.clone_online_state(modules)
    target_hash_before = state_sha256(target_state)
    donor_hash_before = state_sha256(donor_state)
    zero_state = zero_recurrent(target_state)
    shuffled_state = row_shuffle_recurrent(target_state)
    torch.manual_seed(SEED + int(device.index or 0))
    random_state = norm_random_recurrent(target_state)
    correct_query, correct_shadow, masks = capture_runtime_shadow(
        model, target, modules, projected=target_state, recurrent=target_state,
        target_values=target_values,
    )
    donor_query, donor_shadow, donor_masks = capture_runtime_shadow(
        model, target, modules, projected=target_state, recurrent=donor_state,
        target_values=target_values,
    )
    _, zero_shadow, _ = capture_runtime_shadow(
        model, target, modules, projected=target_state, recurrent=zero_state,
        target_values=target_values,
    )
    _, permuted_shadow, _ = capture_runtime_shadow(
        model, target, modules, projected=target_state, recurrent=target_state,
        target_values=target_values, rotate_recurrent_layers=True,
    )
    _, shuffled_shadow, _ = capture_runtime_shadow(
        model, target, modules, projected=target_state, recurrent=shuffled_state,
        target_values=target_values,
    )
    _, random_shadow, _ = capture_runtime_shadow(
        model, target, modules, projected=target_state, recurrent=random_state,
        target_values=target_values,
    )
    if any(
        not torch.equal(correct_query[name], donor_query[name])
        or not torch.equal(masks[name], donor_masks[name])
        for name in correct_query
    ):
        raise RuntimeError("Stage2 token-causal target query changed with shadow state")
    binder_audit = certified.install(model, head, thresholds=thresholds)
    feedback_audit = certified.install_shifted_feedback(model)
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    common = {
        "model": model,
        "target": target,
        "modules": modules,
        "projected": target_state,
        "target_values": target_values,
        "queries": correct_query,
        "masks": masks,
    }
    conditions: dict[str, Mapping[str, Any]] = {}
    conditions["correct"] = run_condition_passes(
        **common, recurrent=target_state, shadows=correct_shadow
    )
    correct_gates = conditions["correct"]["gates_by_pass"]
    conditions["donor-shadow"] = run_condition_passes(
        **common, recurrent=target_state, shadows=donor_shadow
    )
    conditions["donor-pair"] = run_condition_passes(
        **common, recurrent=donor_state, shadows=donor_shadow
    )
    conditions["zero"] = run_condition_passes(
        **common, recurrent=zero_state, shadows=zero_shadow
    )
    conditions["layer-permuted"] = run_condition_passes(
        **common,
        recurrent=target_state,
        shadows=permuted_shadow,
        rotate_recurrent_layers=True,
    )
    conditions["row-shuffled"] = run_condition_passes(
        **common, recurrent=target_state, shadows=shuffled_shadow
    )
    conditions["norm-random"] = run_condition_passes(
        **common, recurrent=target_state, shadows=random_shadow
    )
    conditions["fixed-correct-gate donor-value"] = run_condition_passes(
        **common,
        recurrent=target_state,
        shadows=correct_shadow,
        gate_overrides_by_pass=correct_gates,
        live_value_overrides=donor_shadow,
    )
    conditions["correct-no-feedback"] = run_condition_passes(
        **common,
        recurrent=target_state,
        shadows=correct_shadow,
        feedback=False,
    )
    conditions["disabled"] = run_condition_passes(
        **common,
        recurrent=target_state,
        shadows=correct_shadow,
        feedback=False,
        binder_enabled=False,
    )
    correct_logits = conditions["correct"]["_logits_by_pass"]
    for name, condition in conditions.items():
        if name != "correct":
            condition["comparisons_to_correct"] = {
                "pass1": compare_to_correct(
                    condition["_logits_by_pass"][0],
                    correct_logits[0],
                    target.labels,
                ),
                "pass8": compare_to_correct(
                    condition["_logits_by_pass"][-1],
                    correct_logits[-1],
                    target.labels,
                ),
            }
        condition.pop("_logits_by_pass", None)
        condition.pop("gates_by_pass", None)
    target_hash_after = state_sha256(target_state)
    donor_hash_after = state_sha256(donor_state)
    return {
        "rows": len(target_examples),
        "writes": {
            "target": 1,
            "donor": 1,
            "during_reads": 0,
            "write_disabled_verified_after_every_read": True,
        },
        "installation": {
            "binder": binder_audit,
            "feedback": feedback_audit,
            "all_model_and_helper_parameters_frozen": True,
            "live_value_map_identity_frozen": True,
        },
        "conditions": conditions,
        "state_hashes": {
            "target_before": target_hash_before,
            "target_after": target_hash_after,
            "donor_before": donor_hash_before,
            "donor_after": donor_hash_after,
        },
        "checks": {
            "correct_pass2_differs_pass1": conditions["correct"]["pass2_differs_pass1"],
            "correct_tail_contracted": conditions["correct"]["tail_contracted"],
            "disabled_exact_collapse": conditions["disabled"]["disabled_exact_collapse"],
            "immutable_target_state": target_hash_before == target_hash_after,
            "immutable_donor_state": donor_hash_before == donor_hash_after,
            "all_logits_finite": all(
                item["logits_finite"]
                for condition in conditions.values()
                for item in condition["passes"]
            ),
            "material_controls_change_predictor_logits": all(
                conditions[name]["comparisons_to_correct"][pass_name][
                    "rows_at_least_095_changed_fraction"
                ]
                >= 0.95
                for name in MATERIAL_CONTROL_CONDITIONS
                for pass_name in ("pass1", "pass8")
            ),
            "fixed_gate_donor_value_original_fusion_uses_target_state": all(
                item["recurrent_state_references_fixed_before_read"] is True
                for item in conditions["fixed-correct-gate donor-value"]["passes"]
            ),
            "no_feedback_pass1_exact_correct": all(
                value == 0.0
                for value in conditions["correct-no-feedback"][
                    "comparisons_to_correct"
                ]["pass1"]["predictor_logit_changed_fraction_by_row"]
            ),
            "no_feedback_pass8_material": conditions["correct-no-feedback"][
                "comparisons_to_correct"
            ]["pass8"]["rows_at_least_095_changed_fraction"] >= 0.95,
            "disabled_correction_material": all(
                conditions["disabled"]["comparisons_to_correct"][pass_name][
                    "rows_at_least_095_changed_fraction"
                ]
                >= 0.95
                for pass_name in ("pass1", "pass8")
            ),
        },
    }


def _broadcast_stage1(
    context: Any, head: LayerwiseBilinear | None, thresholds: torch.Tensor | None,
    analysis: Mapping[str, Any] | None,
) -> tuple[LayerwiseBilinear, torch.Tensor, Mapping[str, Any]]:
    payload: list[Any] = [None]
    if context.is_primary:
        if head is None or thresholds is None or analysis is None:
            raise RuntimeError("Primary stage1 payload is incomplete")
        payload[0] = {
            "head": {name: value.detach().cpu() for name, value in head.state_dict().items()},
            "thresholds": thresholds.cpu(),
            "analysis": dict(analysis),
        }
    dist.broadcast_object_list(payload, src=0, group=context.control_group)
    if payload[0] is None:
        raise RuntimeError("Stage1 broadcast returned no payload")
    restored = LayerwiseBilinear()
    restored.load_state_dict(payload[0]["head"])
    return restored, payload[0]["thresholds"], payload[0]["analysis"]


def aggregate_stage2(rows: Sequence[Mapping[str, Any]]) -> Mapping[str, Any]:
    if len(rows) != WORLD_SIZE or sum(int(row["rows"]) for row in rows) != HELDOUT_ROWS:
        raise RuntimeError("Stage2 gathered row coverage differs")
    condition_names = tuple(rows[0]["conditions"])
    aggregate: dict[str, Any] = {}
    for name in condition_names:
        passes = []
        for pass_index in range(PASSES):
            ce: dict[str, Any] = {}
            for part in ("overall", "first", "later"):
                total = sum(
                    float(row["conditions"][name]["passes"][pass_index]["ce"][part]["sum"])
                    for row in rows
                )
                tokens = sum(
                    int(row["conditions"][name]["passes"][pass_index]["ce"][part]["tokens"])
                    for row in rows
                )
                ce[part] = {"sum": total, "tokens": tokens, "mean": None if tokens == 0 else total / tokens}
            local_delta = [
                row["conditions"][name]["passes"][pass_index]["live_value_delta_from_previous"]
                for row in rows
            ]
            if pass_index == 0:
                delta = None
            else:
                vectors = sum(int(item["vectors"]) for item in local_delta)
                total = sum(float(item["sum"]) for item in local_delta)
                delta = {"sum": total, "vectors": vectors, "mean": total / vectors}
            passes.append({"pass": pass_index + 1, "ce": ce, "live_value_delta_from_previous": delta})
        deltas = [item["live_value_delta_from_previous"]["mean"] for item in passes[1:]]
        tail = [
            deltas[index + 1] <= TAIL_RATIO * deltas[index]
            for index in range(len(deltas) - 4, len(deltas) - 1)
        ]
        aggregate[name] = {
            "passes": passes,
            "pass2_differs_pass1_all_ranks": all(row["conditions"][name]["pass2_differs_pass1"] for row in rows),
            "disabled_exact_collapse_all_ranks": all(row["conditions"][name]["disabled_exact_collapse"] for row in rows),
            "tail_contraction_checks": tail,
            "tail_contracted": all(tail),
        }
        if name != "correct":
            comparisons: dict[str, Any] = {}
            for pass_name in ("pass1", "pass8"):
                changed = [
                    value
                    for row in rows
                    for value in row["conditions"][name]["comparisons_to_correct"][
                        pass_name
                    ]["predictor_logit_changed_fraction_by_row"]
                ]
                ce_delta_by_row = {
                    part: [
                        value
                        for row in rows
                        for value in row["conditions"][name][
                            "comparisons_to_correct"
                        ][pass_name]["ce_delta_by_row"][part]
                    ]
                    for part in ("overall", "first", "later")
                }
                comparisons[pass_name] = {
                    "predictor_logit_changed_fraction_by_row": changed,
                    "mean_predictor_logit_changed_fraction": sum(changed) / len(changed),
                    "rows_at_least_095_changed_fraction": sum(
                        value >= 0.95 for value in changed
                    )
                    / len(changed),
                    "ce_delta_by_row": ce_delta_by_row,
                    "ce_positive_row_fraction": {
                        part: (
                            None
                            if not [value for value in values if value is not None]
                            else sum(
                                value > 0.0
                                for value in values
                                if value is not None
                            )
                            / len([value for value in values if value is not None])
                        )
                        for part, values in ce_delta_by_row.items()
                    },
                }
            aggregate[name]["comparisons_to_correct"] = comparisons
    for name, condition in aggregate.items():
        if name == "correct":
            continue
        for pass_name, pass_index in (("pass1", 0), ("pass8", PASSES - 1)):
            condition["comparisons_to_correct"][pass_name][
                "condition_minus_correct_ce"
            ] = {
                part: (
                    condition["passes"][pass_index]["ce"][part]["mean"]
                    - aggregate["correct"]["passes"][pass_index]["ce"][part]["mean"]
                )
                for part in ("overall", "first", "later")
                if condition["passes"][pass_index]["ce"][part]["mean"] is not None
                and aggregate["correct"]["passes"][pass_index]["ce"][part]["mean"] is not None
            }
    checks = {
        "exactly_44_heldout_rows_balanced_11_per_rank": all(int(row["rows"]) == STAGE2_ROWS_PER_RANK for row in rows),
        "correct_pass2_differs_pass1": aggregate["correct"]["pass2_differs_pass1_all_ranks"],
        "correct_tail_contracted": aggregate["correct"]["tail_contracted"],
        "disabled_exact_collapse": aggregate["disabled"]["disabled_exact_collapse_all_ranks"],
        "immutable_state_hashes": all(
            row["state_hashes"]["target_before"] == row["state_hashes"]["target_after"]
            and row["state_hashes"]["donor_before"] == row["state_hashes"]["donor_after"]
            for row in rows
        ),
        "no_writes_during_reads": all(
            row["writes"]["during_reads"] == 0
            and row["writes"]["write_disabled_verified_after_every_read"] is True
            and all(
                item["write_enabled_after_read"] is False
                for condition in row["conditions"].values()
                for item in condition["passes"]
            )
            for row in rows
        ),
        "all_logits_finite": all(row["checks"]["all_logits_finite"] for row in rows),
        "material_controls_change_predictor_logits": all(
            aggregate[name]["comparisons_to_correct"][pass_name][
                "rows_at_least_095_changed_fraction"
            ]
            >= 0.95
            for name in MATERIAL_CONTROL_CONDITIONS
            for pass_name in ("pass1", "pass8")
        ),
        "shadow_gate_swap_material": all(
            aggregate["donor-shadow"]["comparisons_to_correct"][pass_name][
                "rows_at_least_095_changed_fraction"
            ]
            >= 0.95
            for pass_name in ("pass1", "pass8")
        ),
        "isolated_live_value_swap_material": all(
            aggregate["fixed-correct-gate donor-value"]["comparisons_to_correct"][
                pass_name
            ]["rows_at_least_095_changed_fraction"]
            >= 0.95
            for pass_name in ("pass1", "pass8")
        ),
        "fixed_gate_donor_value_original_fusion_target_state": all(
            row["checks"]["fixed_gate_donor_value_original_fusion_uses_target_state"]
            for row in rows
        ),
        "no_feedback_pass1_exact_correct": all(
            value == 0.0
            for value in aggregate["correct-no-feedback"]["comparisons_to_correct"][
                "pass1"
            ]["predictor_logit_changed_fraction_by_row"]
        ),
        "no_feedback_pass8_material": aggregate["correct-no-feedback"][
            "comparisons_to_correct"
        ]["pass8"]["rows_at_least_095_changed_fraction"] >= 0.95,
        "disabled_correction_material_pass1_pass8": all(
            aggregate["disabled"]["comparisons_to_correct"][pass_name][
                "rows_at_least_095_changed_fraction"
            ]
            >= 0.95
            for pass_name in ("pass1", "pass8")
        ),
    }
    return {"conditions": aggregate, "rank_rows": list(rows), "checks": checks, "passed": all(checks.values())}


def run(
    *,
    base_model: Path,
    dataset_root: Path,
    output_dir: Path,
    resume_complete_stage1_shards: bool = False,
) -> Mapping[str, Any]:
    context = distributed.initialize_distributed_training(
        "cuda", timeout_seconds=DISTRIBUTED_TIMEOUT_SECONDS
    )
    if context is None:
        raise RuntimeError("Run with torchrun --nproc_per_node=4")
    try:
        protocol, signed_crossfit = validate_protocol()
        source_audit = validate_execution_source()
        if context.world_size != WORLD_SIZE or not hardware.four_distinct_a100s(context.rank_devices):
            raise RuntimeError("Predictor recurrent mechanics requires four distinct A100s")
        if os.environ.get("HF_ENDPOINT") != HF_ENDPOINT:
            raise RuntimeError(f"HF_ENDPOINT must be exactly {HF_ENDPOINT}")
        output_error: BaseException | None = None
        if context.is_primary:
            if resume_complete_stage1_shards:
                expected = {
                    f"stage1-shard-{rank}.jsonl" for rank in range(WORLD_SIZE)
                }
                observed = (
                    {path.name for path in output_dir.iterdir()}
                    if output_dir.is_dir()
                    else set()
                )
                if not output_dir.is_dir() or observed != expected:
                    output_error = ValueError(
                        "Predictor recurrent resume requires exactly four stage1 shards "
                        f"and no result: {output_dir}"
                    )
            elif output_dir.exists():
                output_error = ValueError(
                    f"Predictor recurrent output must be fresh: {output_dir}"
                )
            else:
                try:
                    output_dir.mkdir(parents=True, exist_ok=False)
                except BaseException as error:
                    output_error = error
        distributed.phase_consensus(
            context, phase="predictor-recurrent-output", error=output_error
        )
        torch.manual_seed(SEED)
        torch.cuda.manual_seed_all(SEED)
        model, tokenizer, model_audit = shadow.load_exact_v5_model(base_model, device=context.device)
        capture_audit = value_identity.install(model)
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        examples, rows, mapping, split_payload = shadow.authorized_examples(tokenizer, dataset_root)
        if (
            split_payload["train_sources"]
            != signed_crossfit["crossfit_split"]["train_sources"]
            or split_payload["heldout_sources"]
            != signed_crossfit["crossfit_split"]["heldout_sources"]
        ):
            raise RuntimeError("Predictor stage does not reproduce the signed 176/44 split")
        split = {int(row["source_index"]): str(row["split"]) for row in split_payload["rows"]}
        records: list[Mapping[str, Any]] | None = None
        provenance: list[Mapping[str, Any]] = []
        resume_error: BaseException | None = None
        if resume_complete_stage1_shards:
            if context.is_primary:
                try:
                    records, provenance = load_feature_records(output_dir, split_payload)
                except BaseException as error:
                    resume_error = error
            distributed.phase_consensus(
                context,
                phase="predictor-recurrent-resume-stage1-validation",
                error=resume_error,
            )
        else:
            shard_path = output_dir / f"stage1-shard-{context.process_rank}.jsonl"
            shard_sources = [
                source
                for source in sorted(examples)
                if source % WORLD_SIZE == context.process_rank
            ]
            for ordinal, source_index in enumerate(shard_sources, start=1):
                donor_index = mapping[source_index]
                feature = capture_predictor_row(
                    model,
                    examples[source_index],
                    examples[donor_index],
                    pad_token_id=int(tokenizer.pad_token_id),
                    device=context.device,
                )
                append_jsonl(
                    shard_path,
                    {
                        "schema": FEATURE_SCHEMA,
                        "source_index": source_index,
                        "row_sha256": rows[source_index]["row_sha256"],
                        "donor_source_index": donor_index,
                        "donor_row_sha256": rows[donor_index]["row_sha256"],
                        "split": split[source_index],
                        **feature,
                    },
                )
                print(
                    f"V5_PREDICTOR_STAGE1 rank={context.process_rank} row={source_index} ordinal={ordinal}/{len(shard_sources)}",
                    flush=True,
                )
        dist.barrier()
        head: LayerwiseBilinear | None = None
        thresholds: torch.Tensor | None = None
        stage1: Mapping[str, Any] | None = None
        if context.is_primary:
            if records is None:
                records, provenance = load_feature_records(output_dir, split_payload)
            head, thresholds, stage1 = fit_predictor_head(records)
        head, thresholds, stage1 = _broadcast_stage1(context, head, thresholds, stage1)
        stage2_local: Mapping[str, Any] | None = None
        stage2: Mapping[str, Any] | None = None
        if stage1["passed"]:
            heldout = sorted(
                int(row["source_index"])
                for row in split_payload["rows"]
                if row["split"] == "heldout"
            )
            assigned = heldout[context.process_rank::WORLD_SIZE]
            if len(assigned) != STAGE2_ROWS_PER_RANK:
                raise RuntimeError("Heldout ordinal sharding is not balanced 11/GPU")
            stage2_local = run_stage2(
                model,
                [examples[source] for source in assigned],
                [examples[mapping[source]] for source in assigned],
                head,
                thresholds.to(context.device),
                pad_token_id=int(tokenizer.pad_token_id),
                device=context.device,
            )
            stage2_local = {
                **dict(stage2_local),
                "rank": context.process_rank,
                "source_indices": assigned,
                "donor_source_indices": [mapping[source] for source in assigned],
            }
            gathered = distributed.gather_objects(context, stage2_local)
            if context.is_primary:
                stage2 = aggregate_stage2(gathered)
        result: dict[str, Any] = {}
        if context.is_primary:
            passed = bool(stage1["passed"] and stage2 is not None and stage2["passed"])
            result = {
                "schema": SCHEMA,
                "status": (
                    "predictor_recurrent_mechanics_passed_training_generation_blocked"
                    if passed
                    else (
                        "predictor_crossfit_failed_stage2_not_run"
                        if not stage1["passed"]
                        else "predictor_recurrent_mechanics_failed_training_generation_blocked"
                    )
                ),
                "passed": passed,
                "stage1": stage1,
                "stage2_executed": bool(stage1["passed"]),
                "stage2": stage2,
                "crossfit_split": split_payload,
                "feature_provenance": provenance,
                "stage1_capture": {
                    "fresh_capture_required": True,
                    "resumed_complete_shards_after_control_timeout": bool(
                        resume_complete_stage1_shards
                    ),
                    "recaptured_rows_during_resume": 0
                    if resume_complete_stage1_shards
                    else endpoint.EVALUATION_ROWS,
                },
                "source_audit": source_audit,
                "model_audit": {
                    **dict(model_audit),
                    "capture": capture_audit,
                    "stage1_output_changed_by_capture": False,
                },
                "hardware": {"world_size": WORLD_SIZE, "rank_devices": list(context.rank_devices)},
                "protocol_payload_sha256": PROTOCOL_PAYLOAD_SHA256,
                "protocol_objective": protocol["objective"],
                "model_or_adapter_training_authorized": False,
                "generation_authorized": False,
                "no_adapter_weights_saved": True,
                "protected_splits_opened": [],
                "code_bindings": {
                    "runner_sha256": sha256_file(Path(__file__)),
                    "helper_sha256": sha256_file(Path(certified.__file__)),
                    "source_crossfit_runner_sha256": sha256_file(Path(shadow.__file__)),
                },
            }
            result["receipt"] = {
                "algorithm": "sha256",
                "payload_scope": "canonical_result_without_receipt",
                "payload_sha256": canonical_sha256(result),
            }
            (output_dir / "result.json").write_text(
                json.dumps(result, ensure_ascii=True, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
        dist.barrier()
        return result
    finally:
        distributed.destroy_distributed_training(context)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-model", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--resume-complete-stage1-shards", action="store_true")
    args = parser.parse_args(argv)
    result = run(
        base_model=args.base_model.expanduser().resolve(strict=True),
        dataset_root=args.dataset_root.expanduser().resolve(strict=True),
        output_dir=args.output_dir.expanduser().resolve(),
        resume_complete_stage1_shards=args.resume_complete_stage1_shards,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
