#!/usr/bin/env python3
"""Train the address-keyed RWKV feedback cell on the fresh narrative split."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys
from typing import Any, Mapping

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[1]
for import_root in (SCRIPT_DIR, PROJECT_ROOT):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

import torch
import torch.distributed as dist

from deltamem.core.cumulative_rwkv_residual import (
    SourceBoundAddressKeyedFeedbackFFN,
    SourceCumulativeResidualRouter,
)
from experiments.rethinking_rwkv_ms_gemma import (
    materialize_natural_memory_native_rwkv_narrative_identity as development_materializer,
)
from experiments.rethinking_rwkv_ms_gemma import (
    run_natural_memory_native_rwkv_source_bound_outer_ffn_development_train as base,
)
from experiments.rethinking_rwkv_ms_gemma import (
    run_natural_memory_native_rwkv_source_multi_anchor_bundle_development_train as inherited,
)


SCHEMA = "rwkv_ms_natural_memory_native_rwkv_address_keyed_feedback_development.v1"
STEP_SCHEMA = f"{SCHEMA}.step"
INPUT_SCHEMA = f"{SCHEMA}.input"
SPLIT_SCHEMA = f"{SCHEMA}.split"
PROTOCOL_SCHEMA = f"{SCHEMA}.protocol"
PROTOCOL = SCRIPT_DIR / "natural_memory_native_rwkv_address_keyed_feedback_development_protocol_v1.json"
PROTOCOL_FILE_SHA256 = "b7580c3c06bfd2076aa510aa897316b522c090fab8a37abbe2ff2fe88dfa0300"
PROTOCOL_PAYLOAD_SHA256 = "832b2f7b317fefea88f405d83bd28d6e796d424a386933998f2bca13bef32f29"
DEFAULT_MATERIALIZATION = SCRIPT_DIR / (
    "local_artifacts/natural_memory_native_rwkv_narrative_identity_v1"
)
DEFAULT_OUTPUT = SCRIPT_DIR / (
    "local_artifacts/natural_memory_native_rwkv_address_keyed_feedback_development_train_v1"
)

ANCHORS = (5, 11, 17)
TERMINAL_ANCHOR = ANCHORS[-1]
COMPATIBILITY_SCALE = 1.0
RESIDUAL_GAIN = 1.0 / 32.0
NATIVE_READ_DIM = 32
HIDDEN_DIM = 2560
BOTTLENECK_DIM = 32
INITIAL_DECAY = 0.5
SEED = 20260827
SPLIT_SALT = "rwkv-address-keyed-feedback-narrative-v1:"
TRAIN_PAIRS = 16
HELDOUT_PAIRS = 16
TRAIN_ROWS = 32
HELDOUT_ROWS = 32
UPDATES = 32
TRAINABLE_TENSORS = 6
TRAINABLE_ELEMENTS = (
    2 * (len(ANCHORS) * NATIVE_READ_DIM * BOTTLENECK_DIM)
    + 3 * (HIDDEN_DIM * BOTTLENECK_DIM)
    + BOTTLENECK_DIM
)
DISCRIMINATIVE_TARGET_PAYLOAD_SHA256 = (
    "29cee3b87c859a2c353e116a33606290aaf8e689e308418eed3100143acbddff"
)
TARGET_SELECTED_FRACTION_MINIMUM = 0.875
CORRECT_POSITIVE_FRACTION_MINIMUM = 0.625
DONOR_MEAN_MINIMUM = 0.02
DONOR_POSITIVE_MINIMUM = 0.75
LAYER_MEAN_MINIMUM = 0.01
LAYER_POSITIVE_MINIMUM = 0.75
MANIFEST_SHA256 = "22fd6e8ffa180b2d1ea10cfcf25665858f5eb4d28be90f99f213fbd910cc1396"
MANIFEST_RECEIPT = "4085cfb50eba027759ce8ae5d73556a608884dd1420cf47c6350adb52c6fe4cc"
SPLIT_PAYLOAD_SHA256 = "e4193d91703c18855578a00c8e3f05cf74fbafe885e7dc8fd9501fef971add08"


def validate_protocol() -> Mapping[str, Any]:
    if base.sha256_file(PROTOCOL) != PROTOCOL_FILE_SHA256:
        raise ValueError("Address-keyed feedback protocol file hash differs")
    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    if protocol.get("schema") != PROTOCOL_SCHEMA:
        raise ValueError("Address-keyed feedback protocol schema differs")
    base.validate_receipt(
        protocol,
        scope="canonical_protocol_without_receipt",
        description="Address-keyed feedback protocol",
    )
    if protocol["receipt"]["payload_sha256"] != PROTOCOL_PAYLOAD_SHA256:
        raise ValueError("Address-keyed feedback protocol receipt differs")
    if (
        protocol.get("open_development_only") is not True
        or protocol.get("protected_mechanics_authorized") is not False
        or protocol.get("protected_causal_authorized") is not False
        or protocol.get("native_benchmark_authorized") is not False
    ):
        raise ValueError("Address-keyed feedback access policy differs")
    expected_architecture = {
        "anchor_layers": list(ANCHORS),
        "address_value_is_selected_mapped_address": True,
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
        "state_value_times_address_query_gate": True,
        "trainable_parameter_elements": TRAINABLE_ELEMENTS,
        "trainable_parameter_tensors": TRAINABLE_TENSORS,
    }
    if protocol.get("architecture") != expected_architecture:
        raise ValueError("Address-keyed feedback architecture differs")
    expected_first = {
        "outer_ffn.address_down.weight": False,
        "outer_ffn.decay_logit": False,
        "outer_ffn.hidden_gate.weight": False,
        "outer_ffn.output_up.weight": True,
        "outer_ffn.query_gate.weight": False,
        "outer_ffn.state_down.weight": False,
    }
    expected_training = {
        "contrast_temperature": base.CONTRAST_TEMPERATURE,
        "correct_ce_weight": base.CORRECT_CE_WEIGHT,
        "donor_contrast_weight": base.DONOR_CONTRAST_WEIGHT,
        "donor_margin": base.DONOR_MARGIN,
        "first_update_gradient_contract": expected_first,
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
    }
    if protocol.get("training") != expected_training:
        raise ValueError("Address-keyed feedback training differs")
    expected_gate = {
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
    if protocol.get("heldout_gate") != expected_gate or protocol.get("discriminative_heldout_gate") != expected_gate:
        raise ValueError("Address-keyed feedback heldout gate differs")
    if protocol.get("split") != {
        "heldout_pairs": HELDOUT_PAIRS,
        "manifest_sha256": MANIFEST_SHA256,
        "payload_sha256": SPLIT_PAYLOAD_SHA256,
        "train_pairs": TRAIN_PAIRS,
    }:
        raise ValueError("Address-keyed feedback split differs")
    if protocol.get("paper_basis", {}).get("review_sha256") != base.sha256_file(SCRIPT_DIR / "FULL_BANDWIDTH_RWKV_REVIEW.md"):
        raise ValueError("Address-keyed feedback paper review differs")
    return protocol


def make_router(maps: Mapping[int, Any], device: torch.device) -> SourceCumulativeResidualRouter:
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    outer_ffn = SourceBoundAddressKeyedFeedbackFFN(
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


def synchronize_gradients(*args: Any, **kwargs: Any) -> Mapping[str, Any]:
    named = args[0] if args else kwargs["named"]
    update = int(kwargs["update"])
    world_size = int(kwargs["world_size"])
    finite = True
    maximum = 0.0
    activity: dict[str, bool] = {}
    for name, parameter in named:
        if parameter.grad is None:
            raise RuntimeError("Address-keyed trainable parameter has no gradient")
        dist.all_reduce(parameter.grad, op=dist.ReduceOp.SUM)
        parameter.grad.div_(float(world_size))
        finite = finite and bool(torch.isfinite(parameter.grad).all().item())
        local_maximum = float(parameter.grad.abs().max().item())
        activity[name] = local_maximum > 0.0
        maximum = max(maximum, local_maximum)
    expected = (
        {name: name == "outer_ffn.output_up.weight" for name, _ in named}
        if update == 0
        else {name: True for name, _ in named}
    )
    passed = activity == expected
    if not finite or not passed:
        raise RuntimeError("Address-keyed synchronized gradients are invalid")
    return {
        "all_trainable_gradients_finite": finite,
        "tensor_activity": activity,
        "expected_tensor_activity": expected,
        "gradient_contract_passed": passed,
        "maximum_absolute_gradient": maximum,
    }


def configure() -> None:
    development_materializer.SEALED_MANIFEST_SHA256 = MANIFEST_SHA256
    inherited.ANCHORS = ANCHORS
    inherited.TERMINAL_ANCHOR = TERMINAL_ANCHOR
    inherited.HELDOUT_ROWS = HELDOUT_ROWS
    inherited.TRAIN_ROWS = TRAIN_ROWS
    inherited.TRAIN_PAIRS = TRAIN_PAIRS
    inherited.HELDOUT_PAIRS = HELDOUT_PAIRS
    inherited.UPDATES = UPDATES
    inherited.TARGET_SELECTED_FRACTION_MINIMUM = TARGET_SELECTED_FRACTION_MINIMUM
    inherited.CORRECT_POSITIVE_FRACTION_MINIMUM = CORRECT_POSITIVE_FRACTION_MINIMUM
    inherited.LAYER_MEAN_MINIMUM = LAYER_MEAN_MINIMUM
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
    base.ANCHORS = ANCHORS
    base.TERMINAL_ANCHOR = TERMINAL_ANCHOR
    base.COMPATIBILITY_SCALE = COMPATIBILITY_SCALE
    base.RESIDUAL_GAIN = RESIDUAL_GAIN
    base.NATIVE_READ_DIM = NATIVE_READ_DIM
    base.HIDDEN_DIM = HIDDEN_DIM
    base.BOTTLENECK_DIM = BOTTLENECK_DIM
    base.SEED = SEED
    base.SPLIT_SALT = SPLIT_SALT
    base.TRAIN_PAIRS = TRAIN_PAIRS
    base.HELDOUT_PAIRS = HELDOUT_PAIRS
    base.TRAIN_ROWS = TRAIN_ROWS
    base.HELDOUT_ROWS = HELDOUT_ROWS
    base.UPDATES = UPDATES
    base.TRAINABLE_TENSORS = TRAINABLE_TENSORS
    base.TRAINABLE_ELEMENTS = TRAINABLE_ELEMENTS
    base.DISCRIMINATIVE_TARGET_PAYLOAD_SHA256 = DISCRIMINATIVE_TARGET_PAYLOAD_SHA256
    base.development_materializer = development_materializer
    base.validate_protocol = validate_protocol
    base.make_router = make_router
    base.synchronize_gradients = synchronize_gradients
    base.routed_predictor_logits = inherited.prompt_latched_routed_predictor_logits
    base.training_loss = inherited.conditional_training_loss
    base.train_outer_ffn = inherited.train_outer_ffn
    base.evaluate_heldout = inherited.evaluate_heldout
    base.evaluate_discriminative_heldout = inherited.evaluate_discriminative_heldout
    base.__file__ = str(Path(__file__).resolve())


def main(argv: Any = None) -> int:
    configure()
    return base.main(argv)


if __name__ == "__main__":
    raise SystemExit(main())
