#!/usr/bin/env python3
"""Train the address/query-conditioned low-rank RWKV read on fresh open rows."""

from __future__ import annotations

import json
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

import torch
import torch.distributed as dist

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[1]
for import_root in (SCRIPT_DIR, PROJECT_ROOT):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

from deltamem.core.cumulative_rwkv_residual import (  # noqa: E402
    SourceBoundLowRankQueryMultiAnchorFFN,
)
from experiments.rethinking_rwkv_ms_gemma import (  # noqa: E402
    materialize_natural_memory_native_rwkv_low_rank_query_development as development_materializer,
)
from experiments.rethinking_rwkv_ms_gemma import (  # noqa: E402
    run_natural_memory_native_rwkv_source_multi_anchor_bundle_development_train as multi,
)


SCHEMA = "rwkv_ms_natural_memory_native_rwkv_low_rank_query_development.v1"
STEP_SCHEMA = f"{SCHEMA}.step"
INPUT_SCHEMA = f"{SCHEMA}.input"
SPLIT_SCHEMA = f"{SCHEMA}.split"
PROTOCOL_SCHEMA = f"{SCHEMA}.protocol"
PROTOCOL = SCRIPT_DIR / "natural_memory_native_rwkv_low_rank_query_development_protocol_v1.json"
PROTOCOL_FILE_SHA256 = "a8fcc923d24b0bec5b17f1c792d9ec2e36f5627406d363a57429942554f5217a"
PROTOCOL_PAYLOAD_SHA256 = "b7743b3f9b7981c669aae9eac1383bd308b11239a791f16548c541e294f11ab5"
DEFAULT_MATERIALIZATION = SCRIPT_DIR / (
    "local_artifacts/natural_memory_native_rwkv_low_rank_query_development_v1"
)
DEFAULT_OUTPUT = SCRIPT_DIR / (
    "local_artifacts/natural_memory_native_rwkv_low_rank_query_development_train_v1"
)

SEED = 20260830
SPLIT_SALT = "rwkv-source-low-rank-query-open-v1:"
TRAIN_PAIRS = 24
HELDOUT_PAIRS = 16
TRAIN_ROWS = 48
HELDOUT_ROWS = 32
UPDATES = 48
TRAINABLE_TENSORS = 6
TRAINABLE_ELEMENTS = 198400
SPLIT_PAYLOAD_SHA256 = "deed23dcb3509bf35cecdd841f1948d4e34ecd2c92a92f1bf575db1e95301418"
DISCRIMINATIVE_TARGET_PAYLOAD_SHA256 = "dd6cecd199c49684555c5a3214f294795a49d8bc61675705b02186bc2aa103cd"


def validate_protocol() -> Mapping[str, Any]:
    if multi.base.sha256_file(PROTOCOL) != PROTOCOL_FILE_SHA256:
        raise ValueError("Low-rank query protocol file hash differs")
    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    if protocol.get("schema") != PROTOCOL_SCHEMA:
        raise ValueError("Low-rank query protocol schema differs")
    multi.base.validate_receipt(
        protocol,
        scope="canonical_protocol_without_receipt",
        description="Low-rank query protocol",
    )
    if protocol["receipt"]["payload_sha256"] != PROTOCOL_PAYLOAD_SHA256:
        raise ValueError("Low-rank query protocol receipt differs")
    if (
        protocol.get("open_development_only") is not True
        or protocol.get("protected_mechanics_authorized") is not False
        or protocol.get("protected_causal_authorized") is not False
        or protocol.get("native_benchmark_authorized") is not False
    ):
        raise ValueError("Low-rank query access policy differs")
    architecture = protocol.get("architecture", {})
    expected_architecture = {
        "anchor_layers": list(multi.base.ANCHORS),
        "anchor_native_reads_are_blockwise_rms_normalized": True,
        "bottleneck_dim": multi.base.BOTTLENECK_DIM,
        "bundle_dim": multi.STATE_BUNDLE_DIM,
        "conditioned_receptance": architecture.get("conditioned_receptance"),
        "exact_zero_state_path": True,
        "hidden_dim": multi.base.HIDDEN_DIM,
        "hidden_query_role": "low_rank_pair_conditioner_and_bundle_gate_only",
        "low_rank": 4,
        "native_read_dim_per_anchor": multi.base.NATIVE_READ_DIM,
        "one_canonical_source_route_shared_by_all_anchors": True,
        "prompt_boundary_confidence_latched_across_interventions": True,
        "prompt_boundary_source_latched_across_interventions": True,
        "query_only_hidden_only_or_projected_value_bypass": False,
        "residual_gain": multi.base.RESIDUAL_GAIN,
        "rwkv_matrix_query_before_each_anchor_readout": True,
        "state_value_times_hidden_gate": True,
        "trainable_parameter_elements": TRAINABLE_ELEMENTS,
        "trainable_parameter_tensors": TRAINABLE_TENSORS,
    }
    if architecture != expected_architecture or not isinstance(
        architecture["conditioned_receptance"], str
    ):
        raise ValueError("Low-rank query architecture contract differs")
    training = protocol.get("training", {})
    if (
        training.get("train_pairs") != TRAIN_PAIRS
        or training.get("heldout_pairs") != HELDOUT_PAIRS
        or training.get("updates") != UPDATES
        or training.get("local_batch_rows_per_rank") != multi.base.LOCAL_BATCH_ROWS
        or training.get("global_batch_rows") != multi.base.GLOBAL_BATCH_ROWS
        or training.get("first_update_gradient_contract")
        != {
            "outer_ffn.address_down": False,
            "outer_ffn.query_down": False,
            "outer_ffn.pair_up": False,
            "outer_ffn.state_down.weight": False,
            "outer_ffn.query_gate.weight": False,
            "outer_ffn.output_up.weight": True,
        }
        or training.get("subsequent_update_gradient_contract")
        != "update 2 pair_up state_down query_gate output_up active; update 3 onward all six active"
    ):
        raise ValueError("Low-rank query training contract differs")
    expected_gate = {
        "all_logits_and_residuals_finite": True,
        "correct_gain_vs_provider_off_mean_minimum": multi.base.HELDOUT_CORRECT_GAIN_MINIMUM,
        "correct_gain_vs_provider_off_positive_row_fraction_minimum": 0.625,
        "donor_both_minus_target_mean_minimum": multi.base.HELDOUT_DONOR_MEAN_MINIMUM,
        "donor_both_positive_row_fraction_minimum": multi.base.HELDOUT_DONOR_POSITIVE_MINIMUM,
        "layer_both_minus_target_mean_minimum": 0.01,
        "layer_both_positive_row_fraction_minimum": multi.base.HELDOUT_LAYER_POSITIVE_MINIMUM,
        "mechanics_must_pass": True,
        "prompt_source_and_confidence_fixed_across_interventions": True,
        "target_selected_fraction_minimum": 0.875,
        "zero_controls_exact_provider_off": True,
    }
    if protocol.get("heldout_gate") != expected_gate or protocol.get(
        "discriminative_heldout_gate"
    ) != expected_gate:
        raise ValueError("Low-rank query heldout gate differs")
    if protocol.get("split") != {
        "heldout_pairs": HELDOUT_PAIRS,
        "manifest_sha256": development_materializer.SEALED_MANIFEST_SHA256,
        "payload_sha256": SPLIT_PAYLOAD_SHA256,
        "train_pairs": TRAIN_PAIRS,
    }:
        raise ValueError("Low-rank query split contract differs")
    return protocol


def make_router(maps: Mapping[int, Any], device: Any):
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    outer_ffn = SourceBoundLowRankQueryMultiAnchorFFN(
        state_dim=multi.base.NATIVE_READ_DIM,
        hidden_dim=multi.base.HIDDEN_DIM,
        anchor_count=len(multi.base.ANCHORS),
        bottleneck_dim=multi.base.BOTTLENECK_DIM,
        rank=4,
    )
    return multi.base.SourceCumulativeResidualRouter(
        maps=maps,
        anchor_layers=multi.base.ANCHORS,
        compatibility_scale=multi.base.COMPATIBILITY_SCALE,
        residual_gain=multi.base.RESIDUAL_GAIN,
        required_receptance_calls=2,
        outer_ffn=outer_ffn,
    ).to(device)


def synchronize_gradients(
    named: Sequence[tuple[str, torch.nn.Parameter]],
    *,
    world_size: int,
    update: int,
) -> Mapping[str, Any]:
    finite = True
    maximum = 0.0
    activity = {}
    for name, parameter in named:
        if parameter.grad is None:
            raise RuntimeError("Low-rank query trainable parameter has no gradient")
        dist.all_reduce(parameter.grad, op=dist.ReduceOp.SUM)
        parameter.grad.div_(float(world_size))
        finite = finite and bool(torch.isfinite(parameter.grad).all().item())
        local_maximum = float(parameter.grad.abs().max().item())
        activity[name] = local_maximum > 0.0
        maximum = max(maximum, local_maximum)
    if update == 0:
        expected_activity = {
            name: name == "outer_ffn.output_up.weight" for name, _ in named
        }
    elif update == 1:
        expected_activity = {
            name: name
            in {
                "outer_ffn.pair_up",
                "outer_ffn.state_down.weight",
                "outer_ffn.query_gate.weight",
                "outer_ffn.output_up.weight",
            }
            for name, _ in named
        }
    else:
        expected_activity = {name: True for name, _ in named}
    gradient_contract_passed = activity == expected_activity
    if not finite or not gradient_contract_passed:
        raise RuntimeError(
            "Low-rank query synchronized gradients are invalid: "
            f"activity={activity} expected={expected_activity} "
            f"update={update}"
        )
    return {
        "all_trainable_gradients_finite": finite,
        "tensor_activity": activity,
        "expected_tensor_activity": expected_activity,
        "gradient_contract_passed": gradient_contract_passed,
        "maximum_absolute_gradient": maximum,
    }


def configure() -> None:
    multi.SCHEMA = SCHEMA
    multi.STEP_SCHEMA = STEP_SCHEMA
    multi.INPUT_SCHEMA = INPUT_SCHEMA
    multi.SPLIT_SCHEMA = SPLIT_SCHEMA
    multi.PROTOCOL_SCHEMA = PROTOCOL_SCHEMA
    multi.PROTOCOL = PROTOCOL
    multi.PROTOCOL_FILE_SHA256 = PROTOCOL_FILE_SHA256
    multi.PROTOCOL_PAYLOAD_SHA256 = PROTOCOL_PAYLOAD_SHA256
    multi.DEFAULT_MATERIALIZATION = DEFAULT_MATERIALIZATION
    multi.DEFAULT_OUTPUT = DEFAULT_OUTPUT
    multi.SEED = SEED
    multi.SPLIT_SALT = SPLIT_SALT
    multi.TRAIN_PAIRS = TRAIN_PAIRS
    multi.HELDOUT_PAIRS = HELDOUT_PAIRS
    multi.TRAIN_ROWS = TRAIN_ROWS
    multi.HELDOUT_ROWS = HELDOUT_ROWS
    multi.UPDATES = UPDATES
    multi.TRAINABLE_TENSORS = TRAINABLE_TENSORS
    multi.TRAINABLE_ELEMENTS = TRAINABLE_ELEMENTS
    multi.SPLIT_PAYLOAD_SHA256 = SPLIT_PAYLOAD_SHA256
    multi.DISCRIMINATIVE_TARGET_PAYLOAD_SHA256 = (
        DISCRIMINATIVE_TARGET_PAYLOAD_SHA256
    )
    multi.development_materializer = development_materializer
    multi.configure()
    multi.base.SCHEMA = SCHEMA
    multi.base.STEP_SCHEMA = STEP_SCHEMA
    multi.base.INPUT_SCHEMA = INPUT_SCHEMA
    multi.base.SPLIT_SCHEMA = SPLIT_SCHEMA
    multi.base.PROTOCOL_SCHEMA = PROTOCOL_SCHEMA
    multi.base.PROTOCOL = PROTOCOL
    multi.base.PROTOCOL_FILE_SHA256 = PROTOCOL_FILE_SHA256
    multi.base.PROTOCOL_PAYLOAD_SHA256 = PROTOCOL_PAYLOAD_SHA256
    multi.base.DEFAULT_MATERIALIZATION = DEFAULT_MATERIALIZATION
    multi.base.DEFAULT_OUTPUT = DEFAULT_OUTPUT
    multi.base.SEED = SEED
    multi.base.SPLIT_SALT = SPLIT_SALT
    multi.base.TRAIN_PAIRS = TRAIN_PAIRS
    multi.base.HELDOUT_PAIRS = HELDOUT_PAIRS
    multi.base.TRAIN_ROWS = TRAIN_ROWS
    multi.base.HELDOUT_ROWS = HELDOUT_ROWS
    multi.base.UPDATES = UPDATES
    multi.base.TRAINABLE_TENSORS = TRAINABLE_TENSORS
    multi.base.TRAINABLE_ELEMENTS = TRAINABLE_ELEMENTS
    multi.base.DISCRIMINATIVE_TARGET_PAYLOAD_SHA256 = (
        DISCRIMINATIVE_TARGET_PAYLOAD_SHA256
    )
    multi.base.development_materializer = development_materializer
    multi.base.validate_protocol = validate_protocol
    multi.base.make_router = make_router
    multi.base.synchronize_gradients = synchronize_gradients
    multi.base.__file__ = str(Path(__file__).resolve())


def main(argv: Sequence[str] | None = None) -> int:
    configure()
    return multi.base.main(argv)


if __name__ == "__main__":
    raise SystemExit(main())
