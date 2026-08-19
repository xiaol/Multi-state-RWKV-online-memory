#!/usr/bin/env python3
"""One-update, four-A100 mechanics check for the output-coupled identity gate.

The passed cross-fit deliberately saved feature shards but no head checkpoint.
This runner therefore reconstructs the exact bilinear head deterministically
from the signed shards, installs those weights into an output-coupled gate,
and performs one open-fit mechanics update.  It saves no adapter and does not
run a causal endpoint or generation.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
import sys
from types import MethodType
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
from experiments.rethinking_rwkv_ms_gemma import rwkv_projected_value_identity as value_identity
from experiments.rethinking_rwkv_ms_gemma import rwkv_output_identity_gate as output_gate
from experiments.rethinking_rwkv_ms_gemma import run_natural_memory_native_rwkv_address_keyed_learned_write_causal_train as base
from experiments.rethinking_rwkv_ms_gemma import run_natural_memory_native_rwkv_query_state_bilinear_crossfit as crossfit
from experiments.rethinking_rwkv_ms_gemma import run_natural_memory_native_rwkv_query_state_infonce_screen as geometry

SCHEMA = "rwkv_ms_natural_memory_native_output_identity_gate_mechanics.v1"
PROTOCOL = SCRIPT_DIR / "natural_memory_native_rwkv_output_identity_gate_mechanics_protocol_v1.json"
PROTOCOL_PAYLOAD_SHA256 = "10e8f4d82f6c42a1691513be02c3643ac9e62be72522df70b129f1933e8a638a"
SEED = 115
CROSSFIT_ROOT = SCRIPT_DIR / "local_artifacts/natural_memory_native_rwkv_query_state_bilinear_crossfit_v1"
CROSSFIT_RESULT_SHA256 = "5e41c4569273fd5841381fcb6c5738b26212dd326b4f7cf56b589528df346ba3"
CROSSFIT_RECEIPT = "89392eaeffa50c0bed9109fd8db3d33a5625eb4ff7117f81d710cf9b5be93945"
SHARD_HASHES = (
    "5ae1827fa71d9ef0a76fec6b4d96853bf40d55667d70cd8a930a3bf59bb39ada",
    "981b36516248086f739229e71d4877d078db26ca1ead9ffd90fab7b1e1ae8136",
    "410be4be336639060d9b8056e68cbefa91e99aa5331db255b7e97d4790755650",
    "165ab30ccd44d0a14ed85fa23477591e01e9951e982fca0af75091cd411bf7eb",
)

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
        or payload.get("protected_splits_opened_by_this_protocol") != []
        or payload.get("generation_authorized") is not False
        or payload.get("adapter_saved") is not False
    ):
        raise ValueError("Output identity-gate mechanics protocol differs")
    if sha256_file(CROSSFIT_ROOT / "result.json") != CROSSFIT_RESULT_SHA256:
        raise ValueError("Signed cross-fit result file differs")
    result = json.loads((CROSSFIT_ROOT / "result.json").read_text(encoding="utf-8"))
    if result.get("receipt", {}).get("payload_sha256") != CROSSFIT_RECEIPT or result.get("passed") is not True:
        raise ValueError("Cross-fit receipt does not authorize mechanics")
    for index, expected in enumerate(SHARD_HASHES):
        if sha256_file(CROSSFIT_ROOT / f"shard-{index}.jsonl") != expected:
            raise ValueError(f"Cross-fit shard {index} differs")
    return payload


def reconstruct_crossfit_head() -> tuple[crossfit.LayerwiseBilinear, Mapping[str, Any]]:
    """Replay the signed CPU training exactly; no missing checkpoint is assumed."""
    result = json.loads((CROSSFIT_ROOT / "result.json").read_text(encoding="utf-8"))
    records, provenance = crossfit.load_feature_records(
        CROSSFIT_ROOT, result["crossfit_split"]
    )
    ordered = sorted(records, key=lambda row: int(row["source_index"]))
    feature = {
        name: torch.tensor([row[name] for row in ordered], dtype=torch.float32)
        for name in ("query", "correct", "matched_donor", "layer_permuted")
    }
    train_index = torch.tensor(
        [index for index, row in enumerate(ordered) if row["split"] == "train"],
        dtype=torch.long,
    )
    torch.manual_seed(crossfit.SEED)
    head = crossfit.LayerwiseBilinear()
    train = {name: value.index_select(0, train_index) for name, value in feature.items()}
    optimizer = torch.optim.AdamW(head.parameters(), lr=crossfit.LEARNING_RATE, weight_decay=crossfit.WEIGHT_DECAY)
    for _ in range(crossfit.TRAIN_STEPS):
        optimizer.zero_grad(set_to_none=True)
        correct = head.score(train["query"], train["correct"])
        donor = head.score(train["query"], train["matched_donor"])
        permuted = head.score(train["query"], train["layer_permuted"])
        loss = torch.relu(crossfit.IDENTITY_MARGIN - correct + donor).mean()
        loss = loss + torch.relu(crossfit.IDENTITY_MARGIN - correct + permuted).mean()
        loss.backward()
        optimizer.step()
    analysis = crossfit.train_and_evaluate(records)
    if analysis != result["analysis"]:
        raise RuntimeError("Deterministic cross-fit reconstruction differs from signed analysis")
    digest = distributed.canonical_sha256({name: value.detach().tolist() for name, value in head.state_dict().items()})
    return head, {"feature_provenance": provenance, "reconstruction_state_sha256": digest, "analysis_reproduced": True}


def _output_fuse(module: Any, projected: torch.Tensor, recurrent: torch.Tensor, route_agreement: torch.Tensor | None = None, query_state_gate: torch.Tensor | None = None, global_recurrent_reads: torch.Tensor | None = None, hidden_states: torch.Tensor | None = None) -> torch.Tensor:
    fused = module.rwkv_output_identity_gate_original_fuse(projected, recurrent, route_agreement, query_state_gate, global_recurrent_reads, hidden_states)
    query = module.rwkv_query_state_identity_query_address
    if query is None:
        raise RuntimeError("Output identity gate has no fixed projected-value query")
    gated, values = module.rwkv_output_identity_gate(
        projected.float(),
        fused.float() - projected.float(),
        query,
        recurrent,
    )
    module.rwkv_output_identity_gate_values = values
    return gated.to(dtype=projected.dtype)


def install_reconstructed_output_gates(model: torch.nn.Module, head: crossfit.LayerwiseBilinear) -> Mapping[str, Any]:
    capture = value_identity.install(model)
    modules = causal_train.ordered_modules(model)
    if len(modules) != crossfit.LAYERS:
        raise RuntimeError("Output gate layer count differs")
    for layer, (_, module) in enumerate(modules):
        gate = output_gate.BoundedOutputIdentityGate(crossfit.STATE_DIM, bottleneck=crossfit.BOTTLENECK).to(next(module.parameters()).device)
        source = head.heads[layer].state_dict()
        gate.identity.load_state_dict(source)
        module.add_module("rwkv_output_identity_gate", gate)
        module.rwkv_output_identity_gate_original_fuse = module._fuse_projected_rwkv_reads
        module.rwkv_output_identity_gate_values = None
        module._fuse_projected_rwkv_reads = MethodType(_output_fuse, module)
    return {"capture": capture, "layers": len(modules), "output_gate": output_gate.architecture_payload(), "reconstructed_weights_loaded": True}


def selected_gates(model: torch.nn.Module) -> list[tuple[str, torch.nn.Parameter]]:
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    named = [(name, parameter) for name, parameter in model.named_parameters() if "rwkv_output_identity_gate" in name]
    for _, parameter in named:
        parameter.requires_grad_(True)
    if len(named) != 168:
        raise RuntimeError(f"Expected 168 output gate tensors, got {len(named)}")
    return sorted(named)


def zero_recurrent(state: Mapping[str, Mapping[str, torch.Tensor]]) -> dict[str, dict[str, torch.Tensor]]:
    return {name: {attribute: torch.zeros_like(values[attribute]) for attribute in causal_train.RECURRENT_ATTRIBUTES} for name, values in state.items()}


@contextmanager
def explicit_projected_only_bypass(model: torch.nn.Module):
    """Use the established projected-only mode without starving DeepEmbed hooks."""
    modules = tuple(causal_train.ordered_modules(model))
    saved = [
        (module, module.memory_readout_mode, module.rwkv_ms_hybrid_mode)
        for _, module in modules
    ]
    for module, _, _ in saved:
        module.memory_readout_mode = "projected_kv_slots"
        module.rwkv_ms_hybrid_mode = "addressed_moe_controller"
    try:
        yield
    finally:
        for module, readout_mode, hybrid_mode in saved:
            module.memory_readout_mode = readout_mode
            module.rwkv_ms_hybrid_mode = hybrid_mode


def _write_references(model: torch.nn.Module, batch: Any) -> tuple[Any, Mapping[str, Mapping[str, torch.Tensor]], Mapping[str, torch.Tensor]]:
    evolution._native_write(model, batch, dtype=torch.bfloat16)
    refs = causal_train.capture_online_state_references(causal_train.ordered_modules(model))
    values = value_identity.capture_write_values(model)
    return batch, refs, values


def controls(model: torch.nn.Module, target: Any, donor: Any) -> Mapping[str, Any]:
    modules = causal_train.ordered_modules(model)
    def read_condition(kind: str) -> tuple[torch.Tensor, bool]:
        value_identity.clear(model); reset_delta_mem_states(model)
        _, correct, values = _write_references(model, target)
        fixed = True
        if kind == "zero":
            fixed = causal_train.install_intervened_state(modules, projected=correct, recurrent=zero_recurrent(correct), rotate_recurrent_layers=False)
        elif kind == "donor":
            evolution._native_write(model, donor, dtype=torch.bfloat16)
            donor_refs = causal_train.capture_online_state_references(modules)
            fixed = causal_train.install_intervened_state(modules, projected=correct, recurrent=donor_refs, rotate_recurrent_layers=False)
        elif kind == "permuted":
            fixed = causal_train.install_intervened_state(modules, projected=correct, recurrent=correct, rotate_recurrent_layers=True)
        value_identity.set_fixed_target_values(model, values)
        if kind == "bypass":
            with explicit_projected_only_bypass(model):
                logits = evolution._native_read(model, target, dtype=torch.bfloat16)
        else:
            logits = evolution._native_read(model, target, dtype=torch.bfloat16)
        return logits, bool(fixed)
    with torch.no_grad():
        correct, _ = read_condition("correct")
        zero, zero_fixed = read_condition("zero")
        donor_logits, donor_fixed = read_condition("donor")
        permuted, permuted_fixed = read_condition("permuted")
        bypass, bypass_fixed = read_condition("bypass")
        ce = {name: contrast.detached_answer_ce(logits, target.labels)[0] for name, logits in {"correct": correct, "zero": zero, "donor": donor_logits, "layer_permuted": permuted, "projected_only_bypass": bypass}.items()}
        finite = all(bool(torch.isfinite(logits).all()) for logits in (correct, zero, donor_logits, permuted, bypass))
        zero_equal = bool(torch.equal(zero, bypass))
    return {"ce": ce, "all_logits_finite": finite, "carrier_fixed": bool(zero_fixed and donor_fixed and permuted_fixed and bypass_fixed), "zero_logits_byte_equal_projected_only_bypass": zero_equal}


def gradient_audit(named: Sequence[tuple[str, torch.nn.Parameter]]) -> Mapping[str, Any]:
    activity = torch.tensor([int(parameter.grad is not None and torch.isfinite(parameter.grad).all() and parameter.grad.abs().max().gt(0)) for _, parameter in named], device=named[0][1].device, dtype=torch.int32)
    dist.all_reduce(activity)
    return {"trainable_tensors": len(named), "global_finite_nonzero_tensors": int(activity.gt(0).sum()), "passed": bool(activity.gt(0).all())}


def run(*, base_model: Path, dataset_root: Path, output_dir: Path) -> Mapping[str, Any]:
    context = distributed.initialize_distributed_training("cuda")
    if context is None: raise RuntimeError("Run with torchrun --nproc_per_node=4")
    try:
        protocol = validate_protocol()
        if os.environ.get("HF_ENDPOINT") != "https://hf-mirror.com" or not geometry.hardware_screen.four_distinct_a100s(context.rank_devices): raise RuntimeError("Requires HF mirror and four distinct A100s")
        error = ValueError(f"Output must be fresh: {output_dir}") if context.is_primary and output_dir.exists() else None
        distributed.phase_consensus(context, phase="output-gate-fresh", error=error)
        error = None
        if context.is_primary:
            try: output_dir.mkdir(parents=True, exist_ok=False)
            except BaseException as exc: error = exc
        distributed.phase_consensus(context, phase="output-gate-create", error=error)
        head, reconstruction = reconstruct_crossfit_head()
        distributed.require_consensus(context, reconstruction["reconstruction_state_sha256"], description="reconstructed cross-fit head")
        torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)
        model, tokenizer, audit = base.load_model(base_model, device=context.device)
        installation = install_reconstructed_output_gates(model, head)
        named = selected_gates(model)
        rows = contrast.load_scene_rows(tokenizer, dataset_root); mapping, deltas, _ = contrast.build_donor_mapping(rows); schedule, _ = contrast.build_schedule(rows, mapping, deltas)
        source = int(schedule[0].source_ordinals[context.process_rank]); donor_source = int(schedule[0].donor_ordinals[context.process_rank])
        target = evolution.collate_native_examples([rows[source].example], pad_token_id=int(tokenizer.pad_token_id), device=context.device); donor = contrast.build_donor_batch(target, rows[donor_source].example, device=context.device)
        model.eval(); diagnostics = controls(model, target, donor)
        model.train(); value_identity.clear(model); reset_delta_mem_states(model); _, _, values = _write_references(model, target); value_identity.set_fixed_target_values(model, values); logits = evolution._native_read(model, target, dtype=torch.bfloat16)
        loss_sum, tokens, _ = evolution.checkpointed_native_answer_loss_sum_and_count(logits, target.labels, chunk_tokens=contrast.CE_CHUNK_TOKENS); loss = loss_sum / tokens
        if not bool(torch.isfinite(loss)): raise RuntimeError("Output-gate update loss is non-finite")
        optimizer = torch.optim.AdamW([parameter for _, parameter in named], lr=1e-4, weight_decay=0.0, fused=True); optimizer.zero_grad(set_to_none=True); loss.backward(); gradients = gradient_audit(named)
        if not gradients["passed"]: raise RuntimeError(f"Output-gate gradients inactive: {gradients!r}")
        for _, parameter in named: dist.all_reduce(parameter.grad); parameter.grad.div_(context.world_size)
        optimizer.step()
        local = {"rank": context.process_rank, "source_ordinal": source, "donor_ordinal": donor_source, "update_loss": float(loss.detach()), **diagnostics}
        gathered = distributed.gather_objects(context, local)
        checks = {"four_a100_ranks": True, "all_control_logits_finite": all(row["all_logits_finite"] for row in gathered), "carrier_fixed_all_conditions": all(row["carrier_fixed"] for row in gathered), "zero_equals_projected_only_bypass": all(row["zero_logits_byte_equal_projected_only_bypass"] for row in gathered), "all_gate_gradients_active": gradients["passed"]}
        result = {"schema": SCHEMA, "status": "output_gate_mechanics_passed_generation_blocked" if all(checks.values()) else "output_gate_mechanics_failed_generation_blocked", "passed": all(checks.values()), "generation_authorized": False, "causal_endpoint_authorized": False, "protocol_payload_sha256": PROTOCOL_PAYLOAD_SHA256, "protocol_objective": protocol["objective"], "crossfit_provenance": reconstruction, "installation": installation, "model_audit": audit, "gradient": gradients, "checks": checks, "rank_rows": list(gathered), "no_adapter_weights_saved": True, "protected_splits_opened": [], "hf_endpoint": os.environ.get("HF_ENDPOINT")}
        if context.is_primary:
            result["receipt"] = {"algorithm":"sha256", "payload_scope":"canonical_result_without_receipt", "payload_sha256":distributed.canonical_sha256(result)}
            (output_dir / "result.json").write_text(json.dumps(result, ensure_ascii=True, sort_keys=True, indent=2)+"\n", encoding="utf-8")
        dist.barrier(); return result
    finally: distributed.destroy_distributed_training(context)


def main(argv: Sequence[str] | None = None) -> int:
    parser=argparse.ArgumentParser(description=__doc__); parser.add_argument("--base-model",type=Path,required=True); parser.add_argument("--dataset-root",type=Path,required=True); parser.add_argument("--output-dir",type=Path,required=True); args=parser.parse_args(argv); run(base_model=args.base_model.expanduser().resolve(strict=True),dataset_root=args.dataset_root.expanduser().resolve(strict=True),output_dir=args.output_dir.expanduser().resolve()); return 0
if __name__ == "__main__": raise SystemExit(main())
