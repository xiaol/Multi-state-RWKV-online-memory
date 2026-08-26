#!/usr/bin/env python3
"""Train the direct address-modulated RWKV value path on open narrative rows."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

import torch

from deltamem.core.cumulative_rwkv_residual import (
    SourceBoundAddressModulatedFeedbackFFN,
    SourceCumulativeResidualRouter,
)
from experiments.rethinking_rwkv_ms_gemma import (
    run_natural_memory_native_rwkv_address_keyed_feedback_development_train as previous,
)


SCRIPT_DIR = Path(__file__).resolve().parent
base = previous.base
SCHEMA = "rwkv_ms_natural_memory_native_rwkv_address_modulated_feedback_development.v2"
STEP_SCHEMA = f"{SCHEMA}.step"
INPUT_SCHEMA = f"{SCHEMA}.input"
SPLIT_SCHEMA = f"{SCHEMA}.split"
PROTOCOL_SCHEMA = f"{SCHEMA}.protocol"
PROTOCOL = SCRIPT_DIR / "natural_memory_native_rwkv_address_modulated_feedback_development_protocol_v2.json"
PROTOCOL_FILE_SHA256 = "091f797c4e376c4c863aca8fad68ad42d9e1728ddf69397fc8100bcd2647265d"
PROTOCOL_PAYLOAD_SHA256 = "e91e7d1222d4e712577cd74fe521a83ba9df3a563c240f8a5c99f8ce178cd4d8"
DEFAULT_MATERIALIZATION = previous.DEFAULT_MATERIALIZATION
DEFAULT_OUTPUT = SCRIPT_DIR / (
    "local_artifacts/natural_memory_native_rwkv_address_modulated_feedback_development_train_v2"
)

ANCHORS = previous.ANCHORS
TERMINAL_ANCHOR = previous.TERMINAL_ANCHOR
COMPATIBILITY_SCALE = previous.COMPATIBILITY_SCALE
RESIDUAL_GAIN = previous.RESIDUAL_GAIN
NATIVE_READ_DIM = previous.NATIVE_READ_DIM
HIDDEN_DIM = previous.HIDDEN_DIM
BOTTLENECK_DIM = previous.BOTTLENECK_DIM
INITIAL_DECAY = previous.INITIAL_DECAY
SEED = 20260828
SPLIT_SALT = previous.SPLIT_SALT
TRAIN_PAIRS = previous.TRAIN_PAIRS
HELDOUT_PAIRS = previous.HELDOUT_PAIRS
TRAIN_ROWS = previous.TRAIN_ROWS
HELDOUT_ROWS = previous.HELDOUT_ROWS
UPDATES = previous.UPDATES
TRAINABLE_TENSORS = previous.TRAINABLE_TENSORS
TRAINABLE_ELEMENTS = previous.TRAINABLE_ELEMENTS
DISCRIMINATIVE_TARGET_PAYLOAD_SHA256 = "7f8604b9eb9d5ec9df6aeed0eeaf4b5a21b6e3277b2038abd34f97184962b94e"
TARGET_SELECTED_FRACTION_MINIMUM = previous.TARGET_SELECTED_FRACTION_MINIMUM
CORRECT_POSITIVE_FRACTION_MINIMUM = previous.CORRECT_POSITIVE_FRACTION_MINIMUM
DONOR_MEAN_MINIMUM = previous.DONOR_MEAN_MINIMUM
DONOR_POSITIVE_MINIMUM = previous.DONOR_POSITIVE_MINIMUM
LAYER_MEAN_MINIMUM = previous.LAYER_MEAN_MINIMUM
LAYER_POSITIVE_MINIMUM = previous.LAYER_POSITIVE_MINIMUM
MANIFEST_SHA256 = previous.MANIFEST_SHA256
SPLIT_PAYLOAD_SHA256 = "a0bdb4abdc0e2a2c5ed26963965cfb0d3ce321f7f80ae4dcea9382fdd245e38f"


def validate_protocol() -> Mapping[str, Any]:
    if base.sha256_file(PROTOCOL) != PROTOCOL_FILE_SHA256:
        raise ValueError("Address-modulated feedback protocol file hash differs")
    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    if protocol.get("schema") != PROTOCOL_SCHEMA:
        raise ValueError("Address-modulated feedback protocol schema differs")
    base.validate_receipt(
        protocol,
        scope="canonical_protocol_without_receipt",
        description="Address-modulated feedback protocol",
    )
    if protocol["receipt"]["payload_sha256"] != PROTOCOL_PAYLOAD_SHA256:
        raise ValueError("Address-modulated feedback protocol receipt differs")
    if (
        protocol.get("open_development_only") is not True
        or protocol.get("protected_mechanics_authorized") is not False
        or protocol.get("protected_causal_authorized") is not False
        or protocol.get("native_benchmark_authorized") is not False
    ):
        raise ValueError("Address-modulated feedback access policy differs")
    expected_architecture = {
        "anchor_layers": list(ANCHORS),
        "address_value_is_direct_value_path_modulation": True,
        "anchor_native_reads_are_blockwise_rms_normalized": True,
        "bottleneck_dim": BOTTLENECK_DIM,
        "bundle_dim": len(ANCHORS) * NATIVE_READ_DIM,
        "compatibility_scale": COMPATIBILITY_SCALE,
        "causal_recurrent_scan": True,
        "decay_initial_value": INITIAL_DECAY,
        "exact_zero_state_or_address_path": True,
        "hidden_dim": HIDDEN_DIM,
        "hidden_query_is_gate_only": True,
        "native_read_dim_per_anchor": NATIVE_READ_DIM,
        "one_canonical_source_route_shared_by_all_anchors": True,
        "query_only_hidden_only_or_projected_value_bypass": False,
        "residual_gain": RESIDUAL_GAIN,
        "selected_native_rwkv_read_is_mandatory_value": True,
        "state_value_times_address_value_gate_times_query_gate": True,
        "trainable_parameter_elements": TRAINABLE_ELEMENTS,
        "trainable_parameter_tensors": TRAINABLE_TENSORS,
    }
    if protocol.get("architecture") != expected_architecture:
        raise ValueError("Address-modulated feedback architecture differs")
    if protocol.get("training") != {
        "contrast_temperature": base.CONTRAST_TEMPERATURE,
        "correct_ce_weight": base.CORRECT_CE_WEIGHT,
        "donor_contrast_weight": base.DONOR_CONTRAST_WEIGHT,
        "donor_margin": base.DONOR_MARGIN,
        "first_update_gradient_contract": {
            "outer_ffn.address_down.weight": False,
            "outer_ffn.decay_logit": False,
            "outer_ffn.hidden_gate.weight": False,
            "outer_ffn.output_up.weight": True,
            "outer_ffn.query_gate.weight": False,
            "outer_ffn.state_down.weight": False,
        },
        "global_batch_rows": base.GLOBAL_BATCH_ROWS,
        "gradient_clip": base.GRADIENT_CLIP,
        "heldout_pairs": HELDOUT_PAIRS,
        "layer_contrast_weight": base.LAYER_CONTRAST_WEIGHT,
        "layer_margin": base.LAYER_MARGIN,
        "learning_rate": base.LEARNING_RATE,
        "local_batch_rows_per_rank": base.LOCAL_BATCH_ROWS,
        "optimizer": "fused AdamW with rank-averaged gradients",
        "single_ce_weight": base.SINGLE_CE_WEIGHT,
        "subsequent_update_gradient_contract": "all six trainable tensors active",
        "target_mode": base.TRAINING_TARGET_MODE,
        "target_payload_sha256": DISCRIMINATIVE_TARGET_PAYLOAD_SHA256,
        "train_controls": list(base.TRAIN_CONTROLS),
        "train_pairs": TRAIN_PAIRS,
        "updates": UPDATES,
        "weight_decay": base.WEIGHT_DECAY,
    }:
        raise ValueError("Address-modulated feedback training differs")
    gate = {
        "correct_gain_vs_provider_off_mean_minimum": base.HELDOUT_CORRECT_GAIN_MINIMUM,
        "correct_gain_vs_provider_off_positive_row_fraction_minimum": CORRECT_POSITIVE_FRACTION_MINIMUM,
        "donor_both_minus_target_mean_minimum": DONOR_MEAN_MINIMUM,
        "donor_both_positive_row_fraction_minimum": DONOR_POSITIVE_MINIMUM,
        "layer_both_minus_target_mean_minimum": LAYER_MEAN_MINIMUM,
        "layer_both_positive_row_fraction_minimum": LAYER_POSITIVE_MINIMUM,
        "mechanics_must_pass": True,
        "prompt_source_and_confidence_fixed_across_interventions": True,
        "target_selected_fraction_minimum": TARGET_SELECTED_FRACTION_MINIMUM,
        "zero_controls_exact_provider_off": True,
    }
    if protocol.get("heldout_gate") != gate or protocol.get("discriminative_heldout_gate") != gate:
        raise ValueError("Address-modulated feedback heldout gate differs")
    if protocol.get("split") != {
        "heldout_pairs": HELDOUT_PAIRS,
        "manifest_sha256": MANIFEST_SHA256,
        "payload_sha256": SPLIT_PAYLOAD_SHA256,
        "train_pairs": TRAIN_PAIRS,
    }:
        raise ValueError("Address-modulated feedback split differs")
    return protocol


def make_router(maps: Mapping[int, Any], device: torch.device) -> SourceCumulativeResidualRouter:
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    outer_ffn = SourceBoundAddressModulatedFeedbackFFN(
        state_dim=NATIVE_READ_DIM,
        hidden_dim=HIDDEN_DIM,
        anchor_count=len(ANCHORS),
        bottleneck_dim=BOTTLENECK_DIM,
        initial_decay=INITIAL_DECAY,
    )
    return SourceCumulativeResidualRouter(
        maps=maps,
        anchor_layers=ANCHORS,
        compatibility_scale=COMPATIBILITY_SCALE,
        residual_gain=RESIDUAL_GAIN,
        required_receptance_calls=2,
        outer_ffn=outer_ffn,
    ).to(device)


def configure() -> None:
    previous.configure()
    base.SCHEMA = SCHEMA
    base.STEP_SCHEMA = STEP_SCHEMA
    base.INPUT_SCHEMA = INPUT_SCHEMA
    base.SPLIT_SCHEMA = SPLIT_SCHEMA
    base.PROTOCOL_SCHEMA = PROTOCOL_SCHEMA
    base.PROTOCOL = PROTOCOL
    base.PROTOCOL_FILE_SHA256 = PROTOCOL_FILE_SHA256
    base.PROTOCOL_PAYLOAD_SHA256 = PROTOCOL_PAYLOAD_SHA256
    base.DEFAULT_MATERIALIZATION = DEFAULT_MATERIALIZATION
    base.DEFAULT_OUTPUT = DEFAULT_OUTPUT
    base.SEED = SEED
    base.TRAINABLE_ELEMENTS = TRAINABLE_ELEMENTS
    base.TRAINABLE_TENSORS = TRAINABLE_TENSORS
    base.DISCRIMINATIVE_TARGET_PAYLOAD_SHA256 = DISCRIMINATIVE_TARGET_PAYLOAD_SHA256
    base.validate_protocol = validate_protocol
    base.make_router = make_router
    base.__file__ = str(Path(__file__).resolve())


def main(argv: Any = None) -> int:
    configure()
    return base.main(argv)


if __name__ == "__main__":
    raise SystemExit(main())
