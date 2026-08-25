#!/usr/bin/env python3
"""Train the prompt-latched multi-anchor RWKV value bundle on fresh open rows."""

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
    SourceBoundMultiAnchorBundleFFN,
    SourceCumulativeResidualRouter,
)
from experiments.rethinking_rwkv_ms_gemma import (
    materialize_natural_memory_native_rwkv_source_multi_anchor_bundle_development as development_materializer,
)
from experiments.rethinking_rwkv_ms_gemma import (
    run_natural_memory_native_rwkv_source_bound_outer_ffn_development_train as base,
)


SCHEMA = "rwkv_ms_natural_memory_native_rwkv_source_multi_anchor_bundle_development.v1"
STEP_SCHEMA = f"{SCHEMA}.step"
INPUT_SCHEMA = f"{SCHEMA}.input"
SPLIT_SCHEMA = f"{SCHEMA}.split"
PROTOCOL_SCHEMA = f"{SCHEMA}.protocol"
PROTOCOL = SCRIPT_DIR / (
    "natural_memory_native_rwkv_source_multi_anchor_bundle_development_protocol_v1.json"
)
PROTOCOL_FILE_SHA256 = (
    "c8473441dc9ddb8a52dc202df034e0f48522ddc8ae62d64705878155386bc9e0"
)
PROTOCOL_PAYLOAD_SHA256 = (
    "6aae9d443dc7199438371a291c1cc615165bef863d3c9c8a1a57a6881e0c8a94"
)
DEFAULT_MATERIALIZATION = SCRIPT_DIR / (
    "local_artifacts/"
    "natural_memory_native_rwkv_source_multi_anchor_bundle_development_v1"
)
DEFAULT_OUTPUT = SCRIPT_DIR / (
    "local_artifacts/"
    "natural_memory_native_rwkv_source_multi_anchor_bundle_development_train_v1"
)
PRIOR_RESULT = SCRIPT_DIR / (
    "local_artifacts/"
    "natural_memory_native_rwkv_prompt_latched_joint_identity_development_v2/"
    "result.json"
)
PRIOR_RESULT_SHA256 = (
    "819cb586acbbc4f391256048cf1bd38a774237d46ae3398fbd9c204983cb5746"
)
PRIOR_RESULT_RECEIPT = (
    "25fc989427c5acb56f3c78f681b3ee508dc3d2f24268c7f8b464648915548563"
)
PAPER_REVIEW = SCRIPT_DIR / "FULL_BANDWIDTH_RWKV_REVIEW.md"
PAPER_REVIEW_SHA256 = (
    "14ad0b37c5dfc4b3ae003a830cedfb8245f9338e5b6e79238d70bc5f7cf5085a"
)

SEED = 20260827
SPLIT_SALT = "rwkv-source-multi-anchor-bundle-open-pair-split-v1:"
TRAIN_PAIRS = 24
HELDOUT_PAIRS = 16
TRAIN_ROWS = 48
HELDOUT_ROWS = 32
UPDATES = 48
STATE_BUNDLE_DIM = len(base.ANCHORS) * base.NATIVE_READ_DIM
TRAINABLE_ELEMENTS = (
    STATE_BUNDLE_DIM * base.BOTTLENECK_DIM
    + base.HIDDEN_DIM * base.BOTTLENECK_DIM
    + base.BOTTLENECK_DIM * base.HIDDEN_DIM
)
SPLIT_PAYLOAD_SHA256 = (
    "08c7c51e8d30ff3d62ca38635a3ad978b3c328ef92570792f5badc578af7b673"
)
DISCRIMINATIVE_TARGET_PAYLOAD_SHA256 = (
    "cd199224f374f6e0eb71c5b5fc14f36d68d546e064e85e5f30458be729b7330f"
)
TARGET_SELECTED_FRACTION_MINIMUM = 0.875
CORRECT_POSITIVE_FRACTION_MINIMUM = 0.625
LAYER_MEAN_MINIMUM = 0.01

_ORIGINAL_ROUTED_PREDICTOR_LOGITS = base.routed_predictor_logits
_ORIGINAL_EVALUATE_HELDOUT = base.evaluate_heldout
_ORIGINAL_EVALUATE_DISCRIMINATIVE_HELDOUT = base.evaluate_discriminative_heldout
_ORIGINAL_SCREEN_PREDICTOR_PASS = base.screen.predictor_pass
_PROMPT_LATCH_AUDITS: list[Mapping[str, Any]] = []
_FIRST_TOKEN_PROMPT_LATCH_AUDITS: list[Mapping[str, Any]] = []


def validate_protocol() -> Mapping[str, Any]:
    if base.sha256_file(PROTOCOL) != PROTOCOL_FILE_SHA256:
        raise ValueError("Multi-anchor bundle protocol file hash differs")
    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    if protocol.get("schema") != PROTOCOL_SCHEMA:
        raise ValueError("Multi-anchor bundle protocol schema differs")
    base.validate_receipt(
        protocol,
        scope="canonical_protocol_without_receipt",
        description="Multi-anchor bundle protocol",
    )
    if protocol["receipt"]["payload_sha256"] != PROTOCOL_PAYLOAD_SHA256:
        raise ValueError("Multi-anchor bundle protocol receipt differs")
    if (
        protocol.get("open_development_only") is not True
        or protocol.get("protected_mechanics_authorized") is not False
        or protocol.get("protected_causal_authorized") is not False
        or protocol.get("native_benchmark_authorized") is not False
    ):
        raise ValueError("Multi-anchor bundle access policy differs")
    if protocol.get("authorization_basis") != {
        "prior_result": str(PRIOR_RESULT.relative_to(SCRIPT_DIR)),
        "prior_result_receipt": PRIOR_RESULT_RECEIPT,
        "prior_result_sha256": PRIOR_RESULT_SHA256,
        "prior_status": "open_heldout_failed_not_promoted",
    }:
        raise ValueError("Multi-anchor bundle authorization differs")
    if base.sha256_file(PRIOR_RESULT) != PRIOR_RESULT_SHA256:
        raise ValueError("Multi-anchor bundle prior result differs")
    prior = json.loads(PRIOR_RESULT.read_text(encoding="utf-8"))
    base.validate_receipt(
        prior,
        scope="canonical_result_without_receipt",
        description="Multi-anchor bundle prior result",
    )
    if (
        prior.get("status") != "open_heldout_failed_not_promoted"
        or prior["receipt"]["payload_sha256"] != PRIOR_RESULT_RECEIPT
    ):
        raise ValueError("Multi-anchor bundle prior decision differs")
    if (
        base.sha256_file(PAPER_REVIEW) != PAPER_REVIEW_SHA256
        or protocol.get("paper_basis", {}).get("review_sha256")
        != PAPER_REVIEW_SHA256
    ):
        raise ValueError("Multi-anchor bundle paper review differs")
    expected_architecture = {
        "anchor_layers": list(base.ANCHORS),
        "anchor_native_reads_are_blockwise_rms_normalized": True,
        "bottleneck_dim": base.BOTTLENECK_DIM,
        "bundle_dim": STATE_BUNDLE_DIM,
        "compatibility_scale": base.COMPATIBILITY_SCALE,
        "exact_zero_bundle_path": True,
        "hidden_dim": base.HIDDEN_DIM,
        "hidden_query_is_gate_only": True,
        "native_read_dim_per_anchor": base.NATIVE_READ_DIM,
        "one_canonical_source_route_shared_by_all_anchors": True,
        "prompt_boundary_confidence_latched_across_interventions": True,
        "prompt_boundary_source_latched_across_interventions": True,
        "query_only_hidden_only_or_projected_value_bypass": False,
        "residual_gain": base.RESIDUAL_GAIN,
        "state_value_times_hidden_gate": True,
        "trainable_parameter_elements": TRAINABLE_ELEMENTS,
        "trainable_parameter_tensors": base.TRAINABLE_TENSORS,
    }
    if protocol.get("architecture") != expected_architecture:
        raise ValueError("Multi-anchor bundle architecture differs")
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
        "heldout_pairs": HELDOUT_PAIRS,
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
        "train_pairs": TRAIN_PAIRS,
        "updates": UPDATES,
        "weight_decay": base.WEIGHT_DECAY,
    }
    if protocol.get("training") != expected_training:
        raise ValueError("Multi-anchor bundle training differs")
    expected_gate = {
        "correct_gain_vs_provider_off_mean_minimum": (
            base.HELDOUT_CORRECT_GAIN_MINIMUM
        ),
        "correct_gain_vs_provider_off_positive_row_fraction_minimum": (
            CORRECT_POSITIVE_FRACTION_MINIMUM
        ),
        "donor_both_minus_target_mean_minimum": base.HELDOUT_DONOR_MEAN_MINIMUM,
        "donor_both_positive_row_fraction_minimum": (
            base.HELDOUT_DONOR_POSITIVE_MINIMUM
        ),
        "layer_both_minus_target_mean_minimum": LAYER_MEAN_MINIMUM,
        "layer_both_positive_row_fraction_minimum": (
            base.HELDOUT_LAYER_POSITIVE_MINIMUM
        ),
        "mechanics_must_pass": True,
        "prompt_source_and_confidence_fixed_across_interventions": True,
        "target_selected_fraction_minimum": TARGET_SELECTED_FRACTION_MINIMUM,
        "zero_controls_exact_provider_off": True,
    }
    if (
        protocol.get("heldout_gate") != expected_gate
        or protocol.get("discriminative_heldout_gate") != expected_gate
    ):
        raise ValueError("Multi-anchor bundle heldout gate differs")
    if protocol.get("split") != {
        "heldout_pairs": HELDOUT_PAIRS,
        "manifest_sha256": development_materializer.SEALED_MANIFEST_SHA256,
        "payload_sha256": SPLIT_PAYLOAD_SHA256,
        "train_pairs": TRAIN_PAIRS,
    }:
        raise ValueError("Multi-anchor bundle split differs")
    return protocol


def make_router(
    maps: Mapping[int, Any], device: torch.device
) -> SourceCumulativeResidualRouter:
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    outer_ffn = SourceBoundMultiAnchorBundleFFN(
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
        raise ValueError("Multi-anchor prompt latch requires integer batch IDs")
    states, addresses, occupied, source_ids = banks
    latched_occupied = {}
    for layer in base.ANCHORS:
        if source_ids[layer].size(0) != latched_source_ids.size(0):
            raise ValueError("Multi-anchor prompt latch batch geometry differs")
        mask = source_ids[layer].eq(latched_source_ids[:, None])
        latched_occupied[layer] = occupied[layer] & mask
        if bool(latched_occupied[layer].sum(dim=1).gt(1).any().item()):
            raise RuntimeError("Multi-anchor prompt latch retained multiple sources")
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
    memory_mass_override: torch.Tensor | None = None,
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
            memory_mass_override=memory_mass_override,
        )
    if memory_mass_override is not None:
        raise ValueError("Multi-anchor prompt latch owns the confidence override")
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
    selected_slot = prompt_terminal["selected_slot"][0, 0]
    safe_slot = selected_slot.clamp_min(0)
    reference_source = prompt_terminal["source_ids"][0, safe_slot]
    reference_source = torch.where(
        selected_slot.ge(0), reference_source, torch.full_like(reference_source, -1)
    )
    batch_size = int(next(iter(banks[0].values())).size(0))
    latched_source_ids = reference_source.reshape(1).expand(batch_size).clone()
    prompt_mass = (
        prompt_terminal["memory_mass"][0:1].detach().clone().expand(batch_size, -1, -1)
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
        memory_mass_override=prompt_mass,
    )
    enriched = [dict(item) for item in diagnostics]
    effective_mass = enriched[-1]["memory_mass"]
    mass_exact = torch.equal(effective_mass, prompt_mass)
    source_shared = bool(latched_source_ids.eq(reference_source).all().item())
    mass_shared = torch.equal(prompt_mass, prompt_mass[0:1].expand_as(prompt_mass))
    enriched[-1]["prompt_latched_source_ids"] = latched_source_ids.detach().clone()
    enriched[-1]["prompt_latched_memory_mass"] = prompt_mass.detach().clone()
    enriched[-1]["prompt_predictor_index"] = int(prompt_predictor)
    _PROMPT_LATCH_AUDITS.append(
        {
            "effective_mass_exact": mass_exact,
            "prompt_mass_shared_across_interventions": mass_shared,
            "prompt_source_shared_across_interventions": source_shared,
            "reference_source_id": int(reference_source.item()),
        }
    )
    return logits, tuple(enriched)


def prompt_fixed_mechanics_predictor_pass(
    model: torch.nn.Module,
    batch: Any,
    modules: Sequence[tuple[str, Any]],
    modules_by_layer: Mapping[int, Any],
    target_state: Mapping[str, Mapping[str, torch.Tensor]],
    *,
    batch_size: int,
    router: SourceCumulativeResidualRouter | None,
    banks: tuple[Mapping[int, torch.Tensor], ...] | None,
    compatibility_maps: Mapping[int, Any] | None = None,
    memory_mass_override: torch.Tensor | None = None,
) -> Mapping[str, Any]:
    if router is None:
        return _ORIGINAL_SCREEN_PREDICTOR_PASS(
            model,
            batch,
            modules,
            modules_by_layer,
            target_state,
            batch_size=batch_size,
            router=None,
            banks=None,
            compatibility_maps=compatibility_maps,
            memory_mass_override=memory_mass_override,
        )
    if banks is None or compatibility_maps is None:
        raise ValueError("Prompt-fixed mechanics pass requires banks and maps")
    if memory_mass_override is not None:
        raise ValueError("Prompt-fixed mechanics pass owns the confidence override")
    correct_banks = base.select_controls(banks, ("correct_four_way",))
    pilot = _ORIGINAL_SCREEN_PREDICTOR_PASS(
        model,
        batch,
        modules,
        modules_by_layer,
        target_state,
        batch_size=1,
        router=router,
        banks=correct_banks,
        compatibility_maps=compatibility_maps,
    )
    terminal = pilot["diagnostics"][-1]
    selected_slot = terminal["selected_slot"][0, 0]
    safe_slot = selected_slot.clamp_min(0)
    reference_source = terminal["source_ids"][0, safe_slot]
    reference_source = torch.where(
        selected_slot.ge(0), reference_source, torch.full_like(reference_source, -1)
    )
    latched_source_ids = reference_source.reshape(1).expand(batch_size).clone()
    prompt_mass = terminal["memory_mass"][0:1].detach().clone().expand(
        batch_size, -1, -1
    )
    latched = latch_banks(banks, latched_source_ids)
    routed = _ORIGINAL_SCREEN_PREDICTOR_PASS(
        model,
        batch,
        modules,
        modules_by_layer,
        target_state,
        batch_size=batch_size,
        router=router,
        banks=latched,
        compatibility_maps=compatibility_maps,
        memory_mass_override=prompt_mass,
    )
    effective_mass = routed["diagnostics"][-1]["memory_mass"]
    _FIRST_TOKEN_PROMPT_LATCH_AUDITS.append(
        {
            "effective_mass_exact": torch.equal(effective_mass, prompt_mass),
            "prompt_mass_shared_across_interventions": torch.equal(
                prompt_mass, prompt_mass[0:1].expand_as(prompt_mass)
            ),
            "prompt_source_shared_across_interventions": bool(
                latched_source_ids.eq(reference_source).all().item()
            ),
            "reference_source_id": int(reference_source.item()),
        }
    )
    return routed


def _strengthen_gate(result: Mapping[str, Any], *, original_view: bool) -> Mapping[str, Any]:
    strengthened = dict(result)
    checks = dict(result["checks"])
    margins = (
        result["aggregate"]["target_ce_margins"]
        if original_view
        else result["target_ce_margins"]
    )
    selected_fraction = (
        result["aggregate"]["terminal_target_selected_fraction"]
        if original_view
        else result["target_selected_fraction"]
    )
    checks.update(
        {
            "target_selected_fraction_strict": (
                selected_fraction >= TARGET_SELECTED_FRACTION_MINIMUM
            ),
            "correct_gain_positive_row_fraction": (
                margins["gain_vs_provider_off"]["positive_fraction"]
                >= CORRECT_POSITIVE_FRACTION_MINIMUM
            ),
            "layer_both_mean_margin": (
                margins["layer_both_minus_target"]["mean"]
                >= LAYER_MEAN_MINIMUM
            ),
        }
    )
    strengthened["checks"] = checks
    strengthened["passed"] = all(checks.values())
    return strengthened


def evaluate_heldout(*args: Any, **kwargs: Any) -> Mapping[str, Any]:
    _FIRST_TOKEN_PROMPT_LATCH_AUDITS.clear()
    previous = base.screen.predictor_pass
    base.screen.predictor_pass = prompt_fixed_mechanics_predictor_pass
    try:
        result = _strengthen_gate(
            _ORIGINAL_EVALUATE_HELDOUT(*args, **kwargs), original_view=True
        )
    finally:
        base.screen.predictor_pass = previous
    context = kwargs["context"]
    gathered = base.distributed.gather_objects(
        context, tuple(_FIRST_TOKEN_PROMPT_LATCH_AUDITS)
    )
    audits = [audit for rank_audits in gathered for audit in rank_audits]
    audit_checks = {
        "prompt_latch_rows_complete": len(audits) == HELDOUT_ROWS,
        "prompt_confidence_override_exact": all(
            audit["effective_mass_exact"] for audit in audits
        ),
        "prompt_confidence_shared_across_interventions": all(
            audit["prompt_mass_shared_across_interventions"] for audit in audits
        ),
        "prompt_source_shared_across_interventions": all(
            audit["prompt_source_shared_across_interventions"] for audit in audits
        ),
    }
    strengthened = dict(result)
    strengthened["prompt_latch"] = {
        "rows": audits,
        "checks": audit_checks,
        "passed": all(audit_checks.values()),
    }
    strengthened_checks = {**result["checks"], **audit_checks}
    strengthened["checks"] = strengthened_checks
    strengthened["passed"] = all(strengthened_checks.values())
    return strengthened


def evaluate_discriminative_heldout(
    *args: Any, **kwargs: Any
) -> Mapping[str, Any]:
    _PROMPT_LATCH_AUDITS.clear()
    result = _strengthen_gate(
        _ORIGINAL_EVALUATE_DISCRIMINATIVE_HELDOUT(*args, **kwargs),
        original_view=False,
    )
    context = kwargs["context"]
    gathered = base.distributed.gather_objects(context, tuple(_PROMPT_LATCH_AUDITS))
    audits = [audit for rank_audits in gathered for audit in rank_audits]
    audit_checks = {
        "prompt_latch_rows_complete": len(audits) == HELDOUT_ROWS,
        "prompt_confidence_override_exact": all(
            audit["effective_mass_exact"] for audit in audits
        ),
        "prompt_confidence_shared_across_interventions": all(
            audit["prompt_mass_shared_across_interventions"] for audit in audits
        ),
        "prompt_source_shared_across_interventions": all(
            audit["prompt_source_shared_across_interventions"] for audit in audits
        ),
    }
    strengthened = dict(result)
    strengthened["prompt_latch"] = {
        "rows": audits,
        "checks": audit_checks,
        "passed": all(audit_checks.values()),
    }
    strengthened_checks = {**result["checks"], **audit_checks}
    strengthened["checks"] = strengthened_checks
    strengthened["passed"] = all(strengthened_checks.values())
    return strengthened


def configure() -> None:
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
    base.SPLIT_SALT = SPLIT_SALT
    base.TRAIN_PAIRS = TRAIN_PAIRS
    base.HELDOUT_PAIRS = HELDOUT_PAIRS
    base.TRAIN_ROWS = TRAIN_ROWS
    base.HELDOUT_ROWS = HELDOUT_ROWS
    base.UPDATES = UPDATES
    base.TRAINABLE_ELEMENTS = TRAINABLE_ELEMENTS
    base.DISCRIMINATIVE_TARGET_PAYLOAD_SHA256 = (
        DISCRIMINATIVE_TARGET_PAYLOAD_SHA256
    )
    base.development_materializer = development_materializer
    base.validate_protocol = validate_protocol
    base.make_router = make_router
    base.routed_predictor_logits = prompt_latched_routed_predictor_logits
    base.evaluate_heldout = evaluate_heldout
    base.evaluate_discriminative_heldout = evaluate_discriminative_heldout
    base.__file__ = str(Path(__file__).resolve())


def main(argv: Sequence[str] | None = None) -> int:
    configure()
    return base.main(argv)


if __name__ == "__main__":
    raise SystemExit(main())
