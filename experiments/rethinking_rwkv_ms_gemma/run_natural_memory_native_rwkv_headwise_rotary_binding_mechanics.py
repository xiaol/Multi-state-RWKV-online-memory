#!/usr/bin/env python3
"""Four-A100 mechanics gate for headwise rotary RWKV binding.

This run changes no model weights, saves no adapter, and does not evaluate a
protected split.  It compares the exact unbound RWKV read against the
correct-address bound/read-inverse path and exercises donor, zero, and
layer-permuted recurrent controls.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import hashlib
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
for root in (SCRIPT_DIR, PROJECT_ROOT):
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

from deltamem.core.delta import reset_delta_mem_states
from experiments.rethinking_rwkv_ms_gemma import rwkv_headwise_rotary_integration as rotary
from experiments.rethinking_rwkv_ms_gemma import run_natural_memory_native_rwkv_address_keyed_learned_write_causal_train as base
from experiments.rethinking_rwkv_ms_gemma import run_natural_memory_native_rwkv_query_state_infonce_screen as geometry


SCHEMA = "rwkv_ms_natural_memory_native_headwise_rotary_binding_mechanics.v1"
PASS_STATUS = "headwise_rotary_binding_mechanics_passed_causal_endpoint_authorized"
FAIL_STATUS = "headwise_rotary_binding_mechanics_failed_causal_endpoint_blocked"
PROTOCOL = SCRIPT_DIR / "natural_memory_native_rwkv_headwise_rotary_binding_mechanics_protocol_v1.json"
PROTOCOL_PAYLOAD_SHA256 = "4f12a66cd6852697e3f2695e9e1ec231807ddfa3b50a296c7d28f2996c30c27c"
SEED = 115
CORRECT_MAX_ABS_TOLERANCE = 1e-5

shared = base.shared
distributed = shared.distributed
evolution = shared.evolution
contrast = shared.contrast
causal_train = base.causal_train


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_protocol() -> Mapping[str, Any]:
    payload = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    unsigned = dict(payload)
    receipt = unsigned.pop("receipt", {})
    if (
        distributed.canonical_sha256(unsigned) != PROTOCOL_PAYLOAD_SHA256
        or receipt.get("payload_sha256") != PROTOCOL_PAYLOAD_SHA256
        or payload.get("generation_authorized") is not False
        or payload.get("causal_endpoint_authorized") is not False
        or payload.get("adapter_saved") is not False
        or payload.get("protected_splits_opened_by_this_protocol") != []
    ):
        raise ValueError("Headwise rotary mechanics protocol differs")
    return payload


def zero_recurrent(
    state: Mapping[str, Mapping[str, torch.Tensor]],
) -> dict[str, dict[str, torch.Tensor]]:
    return {
        name: {
            attribute: torch.zeros_like(values[attribute])
            for attribute in causal_train.RECURRENT_ATTRIBUTES
        }
        for name, values in state.items()
    }


def _ordered_modules(model: torch.nn.Module) -> tuple[tuple[str, Any], ...]:
    return causal_train.ordered_modules(model)


def _snapshot_captures(model: torch.nn.Module) -> dict[str, Mapping[str, Any]]:
    result: dict[str, Mapping[str, Any]] = {}
    for name, module in _ordered_modules(model):
        captures = module.rwkv_rotary_read_captures
        if set(captures) != {"addressed", "global"}:
            raise RuntimeError(f"Rotary read captures are incomplete for {name}: {set(captures)}")
        result[name] = {
            kind: {key: value.clone() for key, value in payload.items()}
            for kind, payload in captures.items()
        }
    if not result:
        raise RuntimeError("Rotary mechanics captured no layers")
    return result


def _write_audit(model: torch.nn.Module, mask: torch.Tensor) -> Mapping[str, Any]:
    max_address_spread = 0.0
    all_stable = True
    all_finite = True
    rows = 0
    for name, module in _ordered_modules(model):
        address = module.rwkv_rotary_write_address
        routes = module.last_write_routes
        if address is None or routes is None:
            raise RuntimeError(f"Rotary write audit missing for {name}")
        all_finite = all_finite and bool(torch.isfinite(address).all().item())
        all_finite = all_finite and bool(torch.isfinite(routes).all().item())
        valid = mask.to(device=address.device, dtype=torch.bool)
        if tuple(address.shape[:2]) != tuple(valid.shape):
            raise RuntimeError(f"Rotary write/address mask shape differs for {name}")
        first = address.masked_fill(~valid.unsqueeze(-1), 0.0)
        counts = valid.sum(dim=1, keepdim=True).clamp_min(1).to(dtype=address.dtype)
        mean = first.sum(dim=1, keepdim=True) / counts.unsqueeze(-1)
        spread = (address - mean).abs().masked_fill(~valid.unsqueeze(-1), 0.0).amax().item()
        max_address_spread = max(max_address_spread, float(spread))
        all_stable = all_stable and spread <= 1e-6
        route_mass = routes.float().abs().sum(dim=-1)
        all_stable = all_stable and bool(route_mass.gt(0).all().item())
        rows += int(valid.sum().item())
    return {
        "all_values_finite": all_finite,
        "stable_address_per_routed_slot": all_stable,
        "max_write_address_spread": max_address_spread,
        "valid_write_tokens": rows,
    }


def _capture_delta_metrics(
    baseline: Mapping[str, Mapping[str, Any]],
    rotary_capture: Mapping[str, Mapping[str, Any]],
    mask: torch.Tensor,
) -> Mapping[str, Any]:
    max_abs = 0.0
    finite = True
    shape_match = True
    code_match = True
    write_slot_code_match = True
    shape_mismatches: list[Mapping[str, Any]] = []
    for name in baseline:
        for kind in ("addressed", "global"):
            raw = baseline[name][kind]["raw"]
            decoded = rotary_capture[name][kind]["decoded"]
            finite = finite and bool(torch.isfinite(decoded).all().item())
            baseline_codes = baseline[name][kind].get("write_codes")
            rotary_codes = rotary_capture[name][kind].get("write_codes")
            if baseline_codes is not None and rotary_codes is not None:
                code_match = code_match and bool(torch.equal(baseline_codes, rotary_codes))
            slot_codes = rotary_capture[name][kind].get("slot_codes")
            write_codes = rotary_capture[name][kind].get("write_codes")
            if slot_codes is not None and write_codes is not None and slot_codes.ndim == 4 and write_codes.ndim == 3 and slot_codes.shape[:2] == write_codes.shape[:2]:
                matches = slot_codes.eq(write_codes.unsqueeze(2)).all(dim=-1).any(dim=2)
                write_slot_code_match = write_slot_code_match and bool(matches.all().item())
            if raw.shape != decoded.shape:
                shape_match = False
                shape_mismatches.append(
                    {
                        "module": name,
                        "kind": kind,
                        "baseline": tuple(raw.shape),
                        "rotary": tuple(decoded.shape),
                    }
                )
                continue
            valid_mask = mask[:, : raw.shape[1]]
            diff = (decoded - raw).abs().masked_fill(
                ~valid_mask.unsqueeze(-1).to(dtype=torch.bool), 0.0
            )
            max_abs = max(max_abs, float(diff.max().item()))
    return {
        "finite": finite,
        "shape_match": shape_match,
        "write_code_match": code_match,
        "write_slot_code_match": write_slot_code_match,
        "shape_mismatches": shape_mismatches,
        "correct_decoded_minus_unbound_raw_max_abs": max_abs,
    }


def _donor_distortion(
    capture: Mapping[str, Mapping[str, Any]],
    mask: torch.Tensor,
) -> Mapping[str, Any]:
    changes: list[torch.Tensor] = []
    for layer in capture.values():
        for payload in layer.values():
            raw = payload["raw"]
            decoded = payload["decoded"]
            norm = raw.float().norm(dim=-1).clamp_min(1e-6)
            relative = (decoded.float() - raw.float()).norm(dim=-1) / norm
            valid_mask = mask[:, : relative.shape[1]]
            changes.append(
                relative.masked_select(
                    valid_mask.to(device=relative.device, dtype=torch.bool)
                )
            )
    values = torch.cat(changes)
    if not bool(torch.isfinite(values).all().item()):
        raise RuntimeError("Rotary donor distortion is non-finite")
    return {
        "decoded_change_row_fraction": float(values.ge(0.05).float().mean().item()),
        "decoded_mean_normalized_l2_change": float(values.mean().item()),
        "decoded_change_rows": int(values.numel()),
    }


def _captures_finite(captures: Mapping[str, Mapping[str, Any]]) -> bool:
    return all(
        bool(torch.isfinite(value).all().item())
        for layer in captures.values()
        for payload in layer.values()
        for value in payload.values()
    )


@torch.no_grad()
def _run_native_write(
    model: torch.nn.Module,
    batch: Any,
) -> Mapping[str, Mapping[str, torch.Tensor]]:
    evolution._native_write(model, batch, dtype=torch.bfloat16)
    return causal_train.capture_online_state_references(_ordered_modules(model))


@contextmanager
def _explicit_projected_only_bypass(model: torch.nn.Module):
    modules = _ordered_modules(model)
    saved = [(module, module.memory_readout_mode, module.rwkv_ms_hybrid_mode) for _, module in modules]
    for module, _, _ in saved:
        module.memory_readout_mode = "projected_kv_slots"
        module.rwkv_ms_hybrid_mode = "addressed_moe_controller"
    try:
        yield
    finally:
        for module, readout_mode, hybrid_mode in saved:
            module.memory_readout_mode = readout_mode
            module.rwkv_ms_hybrid_mode = hybrid_mode


@torch.no_grad()
def _read_branch(
    model: torch.nn.Module,
    target: Any,
    donor: Any,
    kind: str,
) -> tuple[torch.Tensor, Mapping[str, Any], Mapping[str, Mapping[str, Any]]]:
    modules = _ordered_modules(model)
    rotary.clear_transient(model)
    reset_delta_mem_states(model)
    target_state = _run_native_write(model, target)
    if kind == "correct":
        recurrent = target_state
    elif kind == "zero":
        recurrent = zero_recurrent(target_state)
    elif kind == "donor":
        donor_state = _run_native_write(model, donor)
        recurrent = donor_state
    elif kind == "permuted":
        recurrent = target_state
    elif kind == "bypass":
        recurrent = target_state
    else:
        raise ValueError(f"Unknown rotary mechanics branch: {kind}")
    fixed = causal_train.install_intervened_state(
        modules,
        projected=target_state,
        recurrent=recurrent,
        rotate_recurrent_layers=kind == "permuted",
    )
    rotary.clear_transient(model)
    if kind == "bypass":
        with _explicit_projected_only_bypass(model):
            logits = evolution._native_read(model, target, dtype=torch.bfloat16)
    else:
        logits = evolution._native_read(model, target, dtype=torch.bfloat16)
    captures = _snapshot_captures(model) if kind != "bypass" else {}
    return logits, {"projected_carrier_fixed": bool(fixed)}, captures


def run(*, base_model: Path, dataset_root: Path, output_dir: Path) -> Mapping[str, Any]:
    context = distributed.initialize_distributed_training("cuda")
    if context is None:
        raise RuntimeError("Run with torchrun --nproc_per_node=4")
    try:
        protocol = validate_protocol()
        if os.environ.get("HF_ENDPOINT") != "https://hf-mirror.com":
            raise RuntimeError("Requires HF_ENDPOINT=https://hf-mirror.com")
        if not geometry.hardware_screen.four_distinct_a100s(context.rank_devices):
            raise RuntimeError("Requires exactly four distinct A100 GPUs")
        error = ValueError(f"Output must be fresh: {output_dir}") if context.is_primary and output_dir.exists() else None
        distributed.phase_consensus(context, phase="rotary-mechanics-fresh", error=error)
        if context.is_primary:
            try:
                output_dir.mkdir(parents=True, exist_ok=False)
            except BaseException as exc:
                error = exc
        distributed.phase_consensus(context, phase="rotary-mechanics-create", error=error)

        torch.manual_seed(SEED)
        torch.cuda.manual_seed_all(SEED)
        model, tokenizer, audit = base.load_model(base_model, device=context.device)
        installation = rotary.install(model, state_dim=32, head_size=32, trainable_projection=True)
        rotary.set_capture(model, True)
        rows = contrast.load_scene_rows(tokenizer, dataset_root)
        mapping, deltas, _ = contrast.build_donor_mapping(rows)
        schedule, _ = contrast.build_schedule(rows, mapping, deltas)
        source = int(schedule[0].source_ordinals[context.process_rank])
        donor_source = int(schedule[0].donor_ordinals[context.process_rank])
        target = evolution.collate_native_examples(
            [rows[source].example], pad_token_id=int(tokenizer.pad_token_id), device=context.device
        )
        donor = contrast.build_donor_batch(target, rows[donor_source].example, device=context.device)
        write_mask = target.write_attention_mask.to(device=context.device, dtype=torch.bool)
        read_mask = target.read_attention_mask.to(device=context.device, dtype=torch.bool)
        model.eval()
        rotary.set_enabled(model, False)
        baseline_logits, _, baseline_capture = _read_branch(model, target, donor, "correct")
        rotary.set_enabled(model, True)
        correct_logits, correct_audit, correct_capture = _read_branch(model, target, donor, "correct")
        zero_logits, zero_audit, zero_capture = _read_branch(model, target, donor, "zero")
        donor_logits, donor_audit, donor_capture = _read_branch(model, target, donor, "donor")
        permuted_logits, permuted_audit, permuted_capture = _read_branch(model, target, donor, "permuted")
        bypass_logits, bypass_audit, _ = _read_branch(model, target, donor, "bypass")
        rotary.clear_transient(model)
        reset_delta_mem_states(model)
        _run_native_write(model, target)
        write_audit = _write_audit(model, write_mask)
        correct_delta = _capture_delta_metrics(baseline_capture, correct_capture, read_mask)
        donor_delta = _donor_distortion(donor_capture, read_mask)
        finite_logits = all(
            bool(torch.isfinite(logits).all().item())
            for logits in (baseline_logits, correct_logits, zero_logits, donor_logits, permuted_logits, bypass_logits)
        )
        zero_equal = bool(torch.equal(zero_logits, bypass_logits))
        checks = {
            "all_control_logits_finite": finite_logits,
            "projected_carrier_fixed_every_intervention": all(
                row["projected_carrier_fixed"] for row in (correct_audit, zero_audit, donor_audit, permuted_audit, bypass_audit)
            ),
            "correct_capture_shapes_match": correct_delta["shape_match"],
            "correct_write_code_matches_unbound": correct_delta.get("write_code_match", False),
            "correct_write_code_matches_read_slot": correct_delta.get("write_slot_code_match", False),
            "correct_decoded_minus_unbound_raw_read_within_tolerance": correct_delta["correct_decoded_minus_unbound_raw_max_abs"] <= CORRECT_MAX_ABS_TOLERANCE,
            "zero_recurrent_logits_byte_equal_explicit_projected_only_bypass": zero_equal,
            "matched_donor_decoded_change_row_fraction": donor_delta["decoded_change_row_fraction"] >= 0.95,
            "matched_donor_decoded_mean_normalized_l2_change": donor_delta["decoded_mean_normalized_l2_change"] >= 0.05,
            "stable_write_address_per_routed_slot": write_audit["stable_address_per_routed_slot"],
            "all_values_finite": write_audit["all_values_finite"] and correct_delta["finite"] and _captures_finite(correct_capture) and _captures_finite(zero_capture) and _captures_finite(donor_capture) and _captures_finite(permuted_capture),
        }
        passed = all(checks.values())
        local = {
            "rank": context.process_rank,
            "source_ordinal": source,
            "donor_ordinal": donor_source,
            "write_audit": write_audit,
            "correct_delta": correct_delta,
            "donor_delta": donor_delta,
            "logit_max_abs_correct_vs_unbound": float((correct_logits - baseline_logits).abs().max().item()),
            "logit_max_abs_zero_vs_bypass": float((zero_logits - bypass_logits).abs().max().item()),
            "checks": checks,
        }
        gathered = distributed.gather_objects(context, local)
        global_checks = {
            key: all(bool(row["checks"][key]) for row in gathered)
            for key in checks
        }
        passed = all(global_checks.values())
        result: dict[str, Any] = {
            "schema": SCHEMA,
            "status": PASS_STATUS if passed else FAIL_STATUS,
            "passed": passed,
            "generation_authorized": False,
            "causal_endpoint_authorized": passed,
            "protocol_payload_sha256": PROTOCOL_PAYLOAD_SHA256,
            "protocol_objective": protocol["objective"],
            "installation": installation,
            "model_audit": audit,
            "global_checks": global_checks,
            "rank_rows": list(gathered),
            "no_model_updates": True,
            "no_adapter_weights_saved": True,
            "protected_splits_opened": [],
            "hf_endpoint": os.environ.get("HF_ENDPOINT"),
        }
        if context.is_primary:
            result["receipt"] = {
                "algorithm": "sha256",
                "payload_scope": "canonical_result_without_receipt",
                "payload_sha256": distributed.canonical_sha256(result),
            }
            (output_dir / "result.json").write_text(
                json.dumps(result, ensure_ascii=True, sort_keys=True, indent=2) + "\n",
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
    args = parser.parse_args(argv)
    run(
        base_model=args.base_model.expanduser().resolve(strict=True),
        dataset_root=args.dataset_root.expanduser().resolve(strict=True),
        output_dir=args.output_dir.expanduser().resolve(),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
