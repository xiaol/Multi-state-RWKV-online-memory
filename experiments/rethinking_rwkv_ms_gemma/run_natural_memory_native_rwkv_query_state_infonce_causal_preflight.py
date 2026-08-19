#!/usr/bin/env python3
"""One-update causal mechanics preflight for the InfoNCE compatibility head.

It is intentionally bounded: the native model, write path, and answer
readout stay frozen; the only optimizer update is the compatibility projector.
Every causal control is a serialized no-grad diagnostic.  This file never
saves adapters and never evaluates any held-out or generation split.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

import torch
import torch.distributed as dist

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[1]
for import_root in (SCRIPT_DIR, PROJECT_ROOT):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

from deltamem.core.delta import reset_delta_mem_states
from experiments.rethinking_rwkv_ms_gemma import rwkv_projected_value_identity as value_identity
from experiments.rethinking_rwkv_ms_gemma import (
    run_natural_memory_native_rwkv_address_keyed_learned_write_causal_train as base,
)
from experiments.rethinking_rwkv_ms_gemma import (
    run_natural_memory_native_rwkv_query_state_infonce_screen as screen,
)


SCHEMA = "rwkv_ms_natural_memory_native_query_state_infonce_causal_preflight.v1"
PROTOCOL = SCRIPT_DIR / "natural_memory_native_rwkv_query_state_infonce_causal_preflight_protocol_v1.json"
PROTOCOL_PAYLOAD_SHA256 = "2155cc03c5941b95237104b2f81e5ab0c9832265e5f9df837e4633deda68c661"
SEED = 113

shared = base.shared
distributed = shared.distributed
evolution = shared.evolution
contrast = shared.contrast
causal_train = base.causal_train


def _validate_protocol() -> Mapping[str, Any]:
    payload = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    unsigned = dict(payload)
    receipt = unsigned.pop("receipt", {})
    if (
        distributed.canonical_sha256(unsigned) != PROTOCOL_PAYLOAD_SHA256
        or receipt.get("payload_sha256") != PROTOCOL_PAYLOAD_SHA256
        or payload.get("protected_splits_opened_by_this_protocol") != []
    ):
        raise ValueError("InfoNCE causal preflight protocol differs")
    return payload


def _save_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.write_text(json.dumps(payload, sort_keys=True, indent=2) + "\n", encoding="utf-8")


def _ce(logits: torch.Tensor, labels: torch.Tensor) -> tuple[float, int]:
    value, tokens = contrast.detached_answer_ce(logits, labels)
    if not isinstance(value, float) or not torch.isfinite(logits).all():
        raise RuntimeError("Causal control logits are non-finite")
    return value, tokens


def _zero_recurrent_state(
    state: Mapping[str, Mapping[str, torch.Tensor]],
) -> dict[str, dict[str, torch.Tensor]]:
    """Keep the target projected carrier but replace only recurrent tensors."""
    return {
        module_name: {
            attribute: torch.zeros_like(values[attribute])
            for attribute in causal_train.RECURRENT_ATTRIBUTES
        }
        for module_name, values in state.items()
    }


def _local_controls_and_vectors(
    model: torch.nn.Module,
    target_example: Any,
    donor_example: Any,
    *,
    pad_token_id: int,
    device: torch.device,
) -> tuple[Mapping[str, Any], torch.Tensor, torch.Tensor]:
    target = evolution.collate_native_examples([target_example], pad_token_id=pad_token_id, device=device)
    donor = contrast.build_donor_batch(target, donor_example, device=device)
    modules = causal_train.ordered_modules(model)
    try:
        with torch.no_grad():
            value_identity.clear(model)
            evolution._native_write(model, target, dtype=torch.bfloat16)
            correct_state = causal_train.capture_online_state_references(modules)
            fixed_values = value_identity.capture_write_values(model)
            value_identity.set_fixed_target_values(model, fixed_values)
            correct_logits = evolution._native_read(model, target, dtype=torch.bfloat16)
            correct_ce, tokens = _ce(correct_logits, target.labels)
            queries, states = screen._aggregate_answer_vectors(
                value_identity.capture(model), target.labels
            )
            del correct_logits
            value_identity.clear(model)
            reset_delta_mem_states(model)

            # Serialized zero control: preserve the captured target carrier and
            # install explicit zero recurrent tensors before the target read.
            zero_state = _zero_recurrent_state(correct_state)
            zero_fixed = causal_train.install_intervened_state(
                modules, projected=correct_state, recurrent=zero_state, rotate_recurrent_layers=False
            )
            zero_logits = evolution._native_read(model, target, dtype=torch.bfloat16)
            zero_ce, zero_tokens = _ce(zero_logits, target.labels)
            del zero_logits
            reset_delta_mem_states(model)

            # Serialized donor control with target projected carrier references.
            evolution._native_write(model, donor, dtype=torch.bfloat16)
            donor_state = causal_train.capture_online_state_references(modules)
            donor_fixed = causal_train.install_intervened_state(
                modules, projected=correct_state, recurrent=donor_state, rotate_recurrent_layers=False
            )
            donor_logits = evolution._native_read(model, target, dtype=torch.bfloat16)
            donor_ce, donor_tokens = _ce(donor_logits, target.labels)
            del donor_logits
            reset_delta_mem_states(model)

            # Serialized layer permutation with the same target projected carrier.
            permuted_fixed = causal_train.install_intervened_state(
                modules, projected=correct_state, recurrent=correct_state, rotate_recurrent_layers=True
            )
            permuted_logits = evolution._native_read(model, target, dtype=torch.bfloat16)
            permuted_ce, permuted_tokens = _ce(permuted_logits, target.labels)
            del permuted_logits
        if len({tokens, zero_tokens, donor_tokens, permuted_tokens}) != 1:
            raise RuntimeError("Causal control answer token counts differ")
        return {
            "correct_ce": correct_ce,
            "zero_ce": zero_ce,
            "donor_ce": donor_ce,
            "layer_permuted_ce": permuted_ce,
            "answer_target_tokens": tokens,
            "all_control_logits_finite": True,
            "projected_carrier_fixed": bool(zero_fixed and donor_fixed and permuted_fixed),
            "zero_projected_carrier_fixed": bool(zero_fixed),
        }, queries, states
    finally:
        value_identity.clear(model)
        reset_delta_mem_states(model)
        evolution.release_native_row_allocator_cache(device)


def run(*, base_model: Path, dataset_root: Path, output_dir: Path) -> Mapping[str, Any]:
    context = distributed.initialize_distributed_training("cuda")
    if context is None:
        raise RuntimeError("Run this preflight with torchrun --nproc_per_node=4")
    try:
        protocol = _validate_protocol()
        if os.environ.get("HF_ENDPOINT") != "https://hf-mirror.com":
            raise ValueError("HF_ENDPOINT must be https://hf-mirror.com")
        if not screen.hardware_screen.four_distinct_a100s(context.rank_devices):
            raise RuntimeError("Preflight requires four distinct A100 ranks")
        freshness_error: BaseException | None = None
        if context.is_primary and output_dir.exists():
            freshness_error = ValueError(f"Output must be fresh: {output_dir}")
        distributed.phase_consensus(context, phase="infonce-causal-preflight-output-fresh", error=freshness_error)
        creation_error: BaseException | None = None
        if context.is_primary:
            try:
                output_dir.mkdir(parents=True, exist_ok=False)
            except BaseException as error:
                creation_error = error
        distributed.phase_consensus(context, phase="infonce-causal-preflight-output-create", error=creation_error)
        torch.manual_seed(SEED)
        torch.cuda.manual_seed_all(SEED)
        model, tokenizer, inherited_audit = base.load_model(base_model, device=context.device)
        install_audit = screen._install_projectors(model)
        named = screen._freeze_except_projectors(model)
        model.eval()
        rows = contrast.load_scene_rows(tokenizer, dataset_root)
        donors, deltas, _ = contrast.build_donor_mapping(rows)
        schedule, _ = contrast.build_schedule(rows, donors, deltas)
        source = int(schedule[0].source_ordinals[context.process_rank])
        donor = int(schedule[0].donor_ordinals[context.process_rank])
        controls, queries, states = _local_controls_and_vectors(
            model, rows[source].example, rows[donor].example,
            pad_token_id=int(tokenizer.pad_token_id), device=context.device,
        )
        all_states = screen._gather_states(states, context.world_size).detach()
        model.train()
        optimizer = torch.optim.AdamW([parameter for _, parameter in named], lr=0.05, weight_decay=0.0, fused=True)
        optimizer.zero_grad(set_to_none=True)
        loss, margin = screen._local_infonce(model, queries, all_states, process_rank=context.process_rank)
        if not bool(torch.isfinite(loss).item() and torch.isfinite(margin).item()):
            raise RuntimeError("Preflight InfoNCE loss is non-finite")
        loss.backward()
        gradient = screen._gradient_audit(named)
        if gradient["passed"] is not True:
            raise RuntimeError(f"Preflight compatibility gradients failed: {gradient!r}")
        for _, parameter in named:
            dist.all_reduce(parameter.grad, op=dist.ReduceOp.SUM)
            parameter.grad.div_(context.world_size)
        optimizer.step()
        local = {
            "rank": context.process_rank, "source_ordinal": source, "donor_ordinal": donor,
            **controls, "infonce_loss": float(loss.detach().item()), "infonce_margin": float(margin.detach().item()),
        }
        rank_rows = distributed.gather_objects(context, local)
        ce_names = ("correct_ce", "zero_ce", "donor_ce", "layer_permuted_ce")
        mean_ce = {name: sum(row[name] for row in rank_rows) / context.world_size for name in ce_names}
        checks = {
            "four_distinct_a100_ranks": True,
            "all_control_logits_finite": all(row["all_control_logits_finite"] for row in rank_rows),
            "projected_carrier_fixed_every_row": all(row["projected_carrier_fixed"] for row in rank_rows),
            "infonce_loss_finite": all(torch.isfinite(torch.tensor(row["infonce_loss"])).item() for row in rank_rows),
            "gradient_audit_passed": gradient["passed"] is True,
            "all_four_open_fit_rows_complete": len(rank_rows) == 4,
        }
        result: dict[str, Any] = {
            "schema": SCHEMA, "status": "causal_preflight_passed_generation_blocked" if all(checks.values()) else "causal_preflight_failed_generation_blocked",
            "passed": all(checks.values()), "generation_authorized": False, "next_causal_train_authorized": False,
            "protocol_payload_sha256": PROTOCOL_PAYLOAD_SHA256, "protocol_objective": protocol["objective"],
            "seed": SEED, "updates": 1, "hf_endpoint": os.environ.get("HF_ENDPOINT"),
            "protected_splits_opened": [], "no_adapter_weights_saved": True,
            "model_audit": {**dict(inherited_audit), "infonce": dict(install_audit)}, "gradient": gradient,
            "control_branch_graph_serialization": {"enabled": True, "control_autograd_graphs": 0, "infonce_autograd_graphs": 1},
            "ce_controls_depend_on_compatibility_head": False,
            "ce_control_interpretation": "Mechanics-only fixed-carrier diagnostics; the frozen answer CE controls do not include the compatibility head.",
            "mean_condition_ce": mean_ce,
            "mean_ce_margins": {"zero_minus_correct": mean_ce["zero_ce"] - mean_ce["correct_ce"], "donor_minus_correct": mean_ce["donor_ce"] - mean_ce["correct_ce"], "layer_permuted_minus_correct": mean_ce["layer_permuted_ce"] - mean_ce["correct_ce"]},
            "mean_infonce_loss": sum(row["infonce_loss"] for row in rank_rows) / context.world_size,
            "mean_infonce_margin": sum(row["infonce_margin"] for row in rank_rows) / context.world_size,
            "checks": checks, "rank_rows": list(rank_rows), "rank_devices": list(context.rank_devices),
        }
        if context.is_primary:
            result["receipt"] = {"algorithm": "sha256", "payload_scope": "canonical_result_without_receipt", "payload_sha256": distributed.canonical_sha256(result)}
            _save_json(output_dir / "result.json", result)
        dist.barrier()
        return result
    finally:
        distributed.destroy_distributed_training(context)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-model", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    run(base_model=args.base_model.expanduser().resolve(strict=True), dataset_root=args.dataset_root.expanduser().resolve(strict=True), output_dir=args.output_dir.expanduser().resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
