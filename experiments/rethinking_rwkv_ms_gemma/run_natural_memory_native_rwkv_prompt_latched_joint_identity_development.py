#!/usr/bin/env python3
"""Train a prompt-latched address/receptance identity gate on open rows."""

from __future__ import annotations

import json
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[1]
for import_root in (SCRIPT_DIR, PROJECT_ROOT):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

import torch

from deltamem.core.cumulative_rwkv_residual import (
    SourceBoundJointIdentityFFN,
    SourceCumulativeResidualRouter,
)
from experiments.rethinking_rwkv_ms_gemma import (
    run_natural_memory_native_rwkv_source_bound_outer_ffn_development_train as base,
)


SCHEMA = "rwkv_ms_natural_memory_native_rwkv_prompt_latched_joint_identity_development.v1"
STEP_SCHEMA = f"{SCHEMA}.step"
INPUT_SCHEMA = f"{SCHEMA}.input"
PROTOCOL_SCHEMA = f"{SCHEMA}.protocol"
PROTOCOL = SCRIPT_DIR / (
    "natural_memory_native_rwkv_prompt_latched_joint_identity_development_protocol_v1.json"
)
PROTOCOL_FILE_SHA256 = (
    "fe1536b354cf8970f5851d92034d548fd2c72f8272a7efa8b9f7d3ee724e21fb"
)
PROTOCOL_PAYLOAD_SHA256 = (
    "769be278338ecd1059ff2a4023300e45fdef6ade27fe6b99220d8f758e70eb94"
)
DEFAULT_OUTPUT = SCRIPT_DIR / (
    "local_artifacts/"
    "natural_memory_native_rwkv_prompt_latched_joint_identity_development_v1"
)
PRIOR_RESULT = SCRIPT_DIR / (
    "local_artifacts/"
    "natural_memory_native_rwkv_source_bound_outer_ffn_development_train_v3/"
    "result.json"
)
PRIOR_RESULT_SHA256 = (
    "ecd0d14d0b1ea2f295777a349862b885d435fec02545ec7edc04edeec37f86ab"
)
PRIOR_RESULT_RECEIPT = (
    "e367f7560044ed8f3a53e19d218690de0389d73f63024c2953a58a398df8dd5d"
)
SEED = 20260826
IDENTITY_FEATURE_DIM = 2 * len(base.ANCHORS) * base.NATIVE_READ_DIM
TRAINABLE_ELEMENTS = (
    base.NATIVE_READ_DIM * base.BOTTLENECK_DIM
    + IDENTITY_FEATURE_DIM * base.BOTTLENECK_DIM
    + base.BOTTLENECK_DIM * base.HIDDEN_DIM
)
DISCRIMINATIVE_TARGET_PAYLOAD_SHA256 = (
    "a6c1a04f43a1726d2d2d2fee0f1297000f9b559933d286bc4fa0fc6027cebcb5"
)

_ORIGINAL_ROUTED_PREDICTOR_LOGITS = base.routed_predictor_logits


def validate_protocol() -> Mapping[str, Any]:
    if base.sha256_file(PROTOCOL) != PROTOCOL_FILE_SHA256:
        raise ValueError("Prompt-latched joint identity protocol file hash differs")
    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    if protocol.get("schema") != PROTOCOL_SCHEMA:
        raise ValueError("Prompt-latched joint identity protocol schema differs")
    base.validate_receipt(
        protocol,
        scope="canonical_protocol_without_receipt",
        description="Prompt-latched joint identity protocol",
    )
    if protocol["receipt"]["payload_sha256"] != PROTOCOL_PAYLOAD_SHA256:
        raise ValueError("Prompt-latched joint identity protocol receipt differs")
    if (
        protocol.get("open_development_only") is not True
        or protocol.get("protected_mechanics_authorized") is not False
        or protocol.get("protected_causal_authorized") is not False
        or protocol.get("native_benchmark_authorized") is not False
    ):
        raise ValueError("Prompt-latched joint identity access policy differs")
    if protocol.get("authorization_basis") != {
        "prior_result": str(PRIOR_RESULT.relative_to(SCRIPT_DIR)),
        "prior_result_receipt": PRIOR_RESULT_RECEIPT,
        "prior_result_sha256": PRIOR_RESULT_SHA256,
        "prior_status": "open_heldout_failed_not_promoted",
    }:
        raise ValueError("Prompt-latched joint identity authorization differs")
    if base.sha256_file(PRIOR_RESULT) != PRIOR_RESULT_SHA256:
        raise ValueError("Prompt-latched joint identity prior result differs")
    prior = json.loads(PRIOR_RESULT.read_text(encoding="utf-8"))
    base.validate_receipt(
        prior,
        scope="canonical_result_without_receipt",
        description="Prompt-latched joint identity prior result",
    )
    if (
        prior.get("status") != "open_heldout_failed_not_promoted"
        or prior["receipt"]["payload_sha256"] != PRIOR_RESULT_RECEIPT
    ):
        raise ValueError("Prompt-latched joint identity prior decision differs")
    expected_architecture = {
        "anchor_layers": list(base.ANCHORS),
        "bottleneck_dim": base.BOTTLENECK_DIM,
        "compatibility_scale": base.COMPATIBILITY_SCALE,
        "exact_zero_state_path": True,
        "hidden_dim": base.HIDDEN_DIM,
        "identity_feature_dim": IDENTITY_FEATURE_DIM,
        "identity_features": (
            "per-anchor concat(normalized_receptance * mapped_address, "
            "abs(normalized_receptance - mapped_address))"
        ),
        "native_read_dim": base.NATIVE_READ_DIM,
        "prompt_boundary_source_latched": True,
        "query_only_hidden_only_or_native_hidden_read_bypass": False,
        "residual_gain": base.RESIDUAL_GAIN,
        "selected_native_rwkv_read_is_only_value": True,
        "trainable_parameter_elements": TRAINABLE_ELEMENTS,
        "trainable_parameter_tensors": base.TRAINABLE_TENSORS,
    }
    if protocol.get("architecture") != expected_architecture:
        raise ValueError("Prompt-latched joint identity architecture differs")
    expected_training = {
        "contrast_temperature": base.CONTRAST_TEMPERATURE,
        "correct_ce_weight": base.CORRECT_CE_WEIGHT,
        "donor_contrast_weight": base.DONOR_CONTRAST_WEIGHT,
        "donor_margin": base.DONOR_MARGIN,
        "first_update_gradient_contract": {
            "outer_ffn.output_up.weight": True,
            "outer_ffn.query_gate.weight": False,
            "outer_ffn.state_down.weight": False,
        },
        "global_batch_rows": base.GLOBAL_BATCH_ROWS,
        "gradient_clip": base.GRADIENT_CLIP,
        "heldout_pairs": base.HELDOUT_PAIRS,
        "layer_contrast_weight": base.LAYER_CONTRAST_WEIGHT,
        "layer_margin": base.LAYER_MARGIN,
        "learning_rate": base.LEARNING_RATE,
        "local_batch_rows_per_rank": base.LOCAL_BATCH_ROWS,
        "optimizer": "fused AdamW with rank-averaged gradients",
        "single_ce_weight": base.SINGLE_CE_WEIGHT,
        "subsequent_update_gradient_contract": "all three trainable tensors active",
        "target_mode": base.TRAINING_TARGET_MODE,
        "target_payload_sha256": DISCRIMINATIVE_TARGET_PAYLOAD_SHA256,
        "train_controls": list(base.TRAIN_CONTROLS),
        "train_pairs": base.TRAIN_PAIRS,
        "updates": base.UPDATES,
        "weight_decay": base.WEIGHT_DECAY,
    }
    if protocol.get("training") != expected_training:
        raise ValueError("Prompt-latched joint identity training differs")
    expected_gate = {
        "correct_gain_vs_provider_off_mean_minimum": (
            base.HELDOUT_CORRECT_GAIN_MINIMUM
        ),
        "donor_both_minus_target_mean_minimum": base.HELDOUT_DONOR_MEAN_MINIMUM,
        "donor_both_positive_row_fraction_minimum": (
            base.HELDOUT_DONOR_POSITIVE_MINIMUM
        ),
        "layer_both_positive_row_fraction_minimum": (
            base.HELDOUT_LAYER_POSITIVE_MINIMUM
        ),
        "mechanics_must_pass": True,
        "prompt_latched_target_source_fraction_minimum": 0.75,
    }
    if (
        protocol.get("heldout_gate") != expected_gate
        or protocol.get("discriminative_heldout_gate") != expected_gate
    ):
        raise ValueError("Prompt-latched joint identity gate differs")
    if protocol.get("split", {}).get("payload_sha256") != base.canonical_sha256(
        prior["input_binding"]["split"]
    ):
        raise ValueError("Prompt-latched joint identity split differs")
    return protocol


def make_router(
    maps: Mapping[int, Any], device: torch.device
) -> SourceCumulativeResidualRouter:
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    outer_ffn = SourceBoundJointIdentityFFN(
        state_dim=base.NATIVE_READ_DIM,
        hidden_dim=base.HIDDEN_DIM,
        anchor_count=len(base.ANCHORS),
        bottleneck_dim=base.BOTTLENECK_DIM,
    )
    return SourceCumulativeResidualRouter(
        maps=maps,
        anchor_layers=base.ANCHORS,
        compatibility_scale=base.COMPATIBILITY_SCALE,
        residual_gain=base.RESIDUAL_GAIN,
        required_receptance_calls=2,
        outer_ffn=outer_ffn,
    ).to(device)


def latch_banks(
    banks: tuple[Mapping[int, torch.Tensor], ...],
    latched_source_ids: torch.Tensor,
) -> tuple[dict[int, torch.Tensor], ...]:
    if latched_source_ids.ndim != 1 or latched_source_ids.dtype.is_floating_point:
        raise ValueError("Prompt latch source IDs must be an integer batch vector")
    states, addresses, occupied, source_ids = banks
    latched_occupied = {}
    for layer in base.ANCHORS:
        if source_ids[layer].size(0) != latched_source_ids.size(0):
            raise ValueError("Prompt latch batch geometry differs")
        mask = source_ids[layer].eq(latched_source_ids[:, None])
        latched_occupied[layer] = occupied[layer] & mask
        if bool(latched_occupied[layer].sum(dim=1).gt(1).any().item()):
            raise RuntimeError("Prompt latch retained more than one source")
    return (
        {layer: value for layer, value in states.items()},
        {layer: value for layer, value in addresses.items()},
        latched_occupied,
        {layer: value for layer, value in source_ids.items()},
    )


def prompt_latched_routed_predictor_logits(
    model: torch.nn.Module,
    batch: Any,
    modules: Sequence[tuple[str, Any]],
    modules_by_layer: Mapping[int, Any],
    target_state: Mapping[str, Mapping[str, torch.Tensor]],
    *,
    router: SourceCumulativeResidualRouter,
    banks: tuple[Mapping[int, torch.Tensor], ...],
    predictor: int,
) -> tuple[torch.Tensor, tuple[Mapping[str, Any], ...]]:
    _, prompt_predictor = base.screen.retrieval.first_prompt_boundary(batch.labels)
    if predictor <= prompt_predictor:
        return _ORIGINAL_ROUTED_PREDICTOR_LOGITS(
            model,
            batch,
            modules,
            modules_by_layer,
            target_state,
            router=router,
            banks=banks,
            predictor=predictor,
        )
    with torch.no_grad():
        _, prompt_diagnostics = _ORIGINAL_ROUTED_PREDICTOR_LOGITS(
            model,
            batch,
            modules,
            modules_by_layer,
            target_state,
            router=router,
            banks=banks,
            predictor=prompt_predictor,
        )
    prompt_terminal = prompt_diagnostics[-1]
    selected_slots = prompt_terminal["selected_slot"][:, 0]
    safe_slots = selected_slots.clamp_min(0)
    latched_source_ids = prompt_terminal["source_ids"].gather(
        1, safe_slots[:, None]
    )[:, 0]
    latched_source_ids = torch.where(
        selected_slots.ge(0),
        latched_source_ids,
        torch.full_like(latched_source_ids, -1),
    )
    latched = latch_banks(banks, latched_source_ids)
    logits, diagnostics = _ORIGINAL_ROUTED_PREDICTOR_LOGITS(
        model,
        batch,
        modules,
        modules_by_layer,
        target_state,
        router=router,
        banks=latched,
        predictor=predictor,
    )
    enriched = [dict(item) for item in diagnostics]
    enriched[-1]["prompt_latched_source_ids"] = latched_source_ids.detach().clone()
    enriched[-1]["prompt_predictor_index"] = int(prompt_predictor)
    return logits, tuple(enriched)


def configure() -> None:
    base.SCHEMA = SCHEMA
    base.STEP_SCHEMA = STEP_SCHEMA
    base.INPUT_SCHEMA = INPUT_SCHEMA
    base.PROTOCOL_SCHEMA = PROTOCOL_SCHEMA
    base.PROTOCOL = PROTOCOL
    base.PROTOCOL_FILE_SHA256 = PROTOCOL_FILE_SHA256
    base.PROTOCOL_PAYLOAD_SHA256 = PROTOCOL_PAYLOAD_SHA256
    base.DEFAULT_OUTPUT = DEFAULT_OUTPUT
    base.SEED = SEED
    base.TRAINABLE_ELEMENTS = TRAINABLE_ELEMENTS
    base.DISCRIMINATIVE_TARGET_PAYLOAD_SHA256 = (
        DISCRIMINATIVE_TARGET_PAYLOAD_SHA256
    )
    base.validate_protocol = validate_protocol
    base.make_router = make_router
    base.routed_predictor_logits = prompt_latched_routed_predictor_logits
    base.__file__ = str(Path(__file__).resolve())


def main(argv: Sequence[str] | None = None) -> int:
    configure()
    return base.main(argv)


if __name__ == "__main__":
    raise SystemExit(main())
