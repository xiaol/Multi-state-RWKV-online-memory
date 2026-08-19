#!/usr/bin/env python3
"""Cheap four-rank screen for learned projected-value/RWKV state compatibility.

This is deliberately *not* a causal benchmark or a generation run.  It reads
only the already-open scene fit rows used by the inherited training schedule,
captures the frozen projected value and addressed recurrent read once per
rank, and learns the zero-noop query compatibility maps against in-batch
negative states.  No adapter is saved.
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
import torch.nn.functional as F


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[1]
for import_root in (SCRIPT_DIR, PROJECT_ROOT):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

from deltamem.core.delta import reset_delta_mem_states
from experiments.rethinking_rwkv_ms_gemma import rwkv_projected_value_identity as value_identity
from experiments.rethinking_rwkv_ms_gemma import rwkv_query_state_identity as capture_identity
from experiments.rethinking_rwkv_ms_gemma import rwkv_query_state_infonce as infonce
from experiments.rethinking_rwkv_ms_gemma import (
    run_natural_memory_native_rwkv_address_keyed_learned_write_causal_train as base,
)
from experiments.rethinking_rwkv_ms_gemma import (
    run_natural_memory_native_rwkv_addressed_value_screen as hardware_screen,
)


SCHEMA = "rwkv_ms_natural_memory_native_query_state_infonce_screen.v1"
PROTOCOL = SCRIPT_DIR / "natural_memory_native_rwkv_query_state_infonce_screen_protocol_v1.json"
PROTOCOL_PAYLOAD_SHA256 = "d7e2205ab3a03918cb8c3deeeefd3c438ec9cfcfeb616602a0de3e5f3e386a71"
SEED = 112
RANK = 4
TEMPERATURE = 0.07
DEFAULT_STEPS = 32

shared = base.shared
distributed = shared.distributed
evolution = shared.evolution
contrast = shared.contrast
causal_train = base.causal_train


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=True, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _validate_protocol() -> Mapping[str, Any]:
    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    unsigned = dict(protocol)
    receipt = unsigned.pop("receipt", {})
    digest = distributed.canonical_sha256(unsigned)
    if (
        digest != PROTOCOL_PAYLOAD_SHA256
        or receipt.get("payload_sha256") != digest
        or protocol.get("protected_splits_opened_by_this_protocol") != []
        or protocol.get("training", {}).get("global_rows") != 4
        or protocol.get("training", {}).get("steps") != DEFAULT_STEPS
    ):
        raise ValueError("InfoNCE screen protocol differs")
    return protocol


def _install_projectors(model: torch.nn.Module) -> Mapping[str, Any]:
    capture_audit = capture_identity.install(model)
    modules = tuple(causal_train.ordered_modules(model))
    installed: list[str] = []
    initial_noop = True
    for module_name, module in modules:
        if hasattr(module, "rwkv_query_state_infonce_projector"):
            raise RuntimeError(f"InfoNCE projector already installed on {module_name}")
        projector = infonce.LowRankQueryProjector(
            int(module.state_read_dim), rank=RANK
        ).to(device=next(module.parameters()).device)
        module.add_module("rwkv_query_state_infonce_projector", projector)
        # The screen isolates compatibility geometry.  The native writer and
        # recurrent read are frozen; a later causal preflight may opt into
        # joint write training only after this screen passes.
        projector.requires_grad_(True)
        probe = torch.randn(2, int(module.state_read_dim), device=next(module.parameters()).device)
        initial_noop = bool(initial_noop and projector.audit(probe)["initialized_exact_noop"])
        installed.append(module_name)
    if len(installed) != 42:
        raise RuntimeError(f"Expected 42 Delta-Mem modules, found {len(installed)}")
    return {
        "capture": dict(capture_audit),
        "projector_modules": len(installed),
        "projector_parameter_tensors": len(installed) * 2,
        "projector_parameter_elements": len(installed) * 2 * 32 * RANK,
        "initialized_exact_noop": initial_noop,
        "forward_output_changed": False,
    }


def _freeze_except_projectors(model: torch.nn.Module) -> list[tuple[str, torch.nn.Parameter]]:
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    selected: list[tuple[str, torch.nn.Parameter]] = []
    for name, parameter in model.named_parameters():
        if "rwkv_query_state_infonce_projector" in name:
            parameter.requires_grad_(True)
            selected.append((name, parameter))
    selected.sort(key=lambda item: item[0])
    if len(selected) != 84 or any(parameter.dtype != torch.float32 for _, parameter in selected):
        raise RuntimeError("InfoNCE screen trainable projector isolation failed")
    return selected


def _aggregate_answer_vectors(
    captured: Sequence[capture_identity.CapturedQueryStateRead], labels: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    valid = labels.ne(-100)
    if labels.ndim != 2 or not bool(valid.any().item()):
        raise RuntimeError("InfoNCE screen requires answer target positions")
    queries: list[torch.Tensor] = []
    states: list[torch.Tensor] = []
    for read in captured:
        if tuple(read.query_address.shape[:2]) != tuple(labels.shape):
            raise RuntimeError("InfoNCE captured read and labels differ")
        queries.append(read.query_address.float()[valid].mean(dim=0).detach())
        states.append(read.recurrent_read.float()[valid].mean(dim=0).detach())
    query_tensor = torch.stack(queries)
    state_tensor = torch.stack(states)
    if not bool(torch.isfinite(torch.cat((query_tensor.flatten(), state_tensor.flatten()))).all()):
        raise RuntimeError("InfoNCE captured vectors are non-finite")
    return query_tensor, state_tensor


def _capture_local_vectors(
    model: torch.nn.Module,
    example: Any,
    *,
    pad_token_id: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    batch = evolution.collate_native_examples([example], pad_token_id=pad_token_id, device=device)
    try:
        with torch.no_grad():
            value_identity.clear(model)
            evolution._native_write(model, batch, dtype=torch.bfloat16)
            values = value_identity.capture_write_values(model)
            value_identity.set_fixed_target_values(model, values)
            logits = evolution._native_read(model, batch, dtype=torch.bfloat16)
            del logits
            vectors = _aggregate_answer_vectors(capture_identity.capture(model), batch.labels)
        return vectors
    finally:
        value_identity.clear(model)
        reset_delta_mem_states(model)
        evolution.release_native_row_allocator_cache(device)


def _gather_states(local_states: torch.Tensor, world_size: int) -> torch.Tensor:
    gathered = [torch.empty_like(local_states) for _ in range(world_size)]
    dist.all_gather(gathered, local_states.contiguous())
    return torch.stack(gathered, dim=1)  # [layers, global_rows, state_dim]


def _local_infonce(
    model: torch.nn.Module,
    frozen_queries: torch.Tensor,
    all_states: torch.Tensor,
    *,
    process_rank: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    logits: list[torch.Tensor] = []
    for index, (_, module) in enumerate(causal_train.ordered_modules(model)):
        projector = module.rwkv_query_state_infonce_projector
        query = F.normalize(projector(frozen_queries[index]), dim=-1, eps=1e-6)
        states = F.normalize(all_states[index], dim=-1, eps=1e-6)
        logits.append(query @ states.transpose(0, 1) / TEMPERATURE)
    table = torch.stack(logits)
    labels = torch.full((table.shape[0],), process_rank, device=table.device, dtype=torch.long)
    loss = F.cross_entropy(table, labels)
    positive = table[torch.arange(table.shape[0], device=table.device), labels]
    negative = table.clone()
    negative[torch.arange(table.shape[0], device=table.device), labels] = float("-inf")
    margin = (positive - negative.max(dim=1).values).mean()
    return loss, margin


def _gradient_audit(
    named: Sequence[tuple[str, torch.nn.Parameter]]) -> Mapping[str, Any]:
    local = []
    for name, parameter in named:
        gradient = parameter.grad
        local.append(
            0 if gradient is None else int(bool(torch.isfinite(gradient).all().item()) and bool(gradient.abs().max().gt(0).item()))
        )
    activity = torch.tensor(local, device=named[0][1].device, dtype=torch.int32)
    dist.all_reduce(activity, op=dist.ReduceOp.SUM)
    up = [int(value) for (name, _), value in zip(named, activity.tolist()) if name.endswith(".up")]
    down = [int(value) for (name, _), value in zip(named, activity.tolist()) if name.endswith(".down")]
    return {
        "global_finite_nonzero_up_tensors": sum(value > 0 for value in up),
        "global_finite_nonzero_down_tensors": sum(value > 0 for value in down),
        "expected_cold_start_down_tensors": len(down),
        "passed": len(up) == 42 and all(value > 0 for value in up) and all(value == 0 for value in down),
    }


def run(
    *, base_model: Path, dataset_root: Path, output_dir: Path, steps: int
) -> Mapping[str, Any]:
    if steps < 1:
        raise ValueError("steps must be positive")
    context = distributed.initialize_distributed_training("cuda")
    if context is None:
        raise RuntimeError("Run this screen with torchrun --nproc_per_node=4")
    try:
        protocol = _validate_protocol()
        if not hardware_screen.four_distinct_a100s(context.rank_devices):
            raise RuntimeError("InfoNCE screen requires four distinct A100 ranks")
        freshness_error: BaseException | None = None
        if context.is_primary and output_dir.exists():
            freshness_error = ValueError(f"InfoNCE screen output must be fresh: {output_dir}")
        distributed.phase_consensus(
            context, phase="infonce-output-fresh", error=freshness_error
        )
        creation_error: BaseException | None = None
        if context.is_primary:
            try:
                output_dir.mkdir(parents=True, exist_ok=False)
            except BaseException as error:
                creation_error = error
        distributed.phase_consensus(
            context, phase="infonce-output-create", error=creation_error
        )
        torch.manual_seed(SEED)
        torch.cuda.manual_seed_all(SEED)
        model, tokenizer, inherited_audit = base.load_model(base_model, device=context.device)
        install_audit = _install_projectors(model)
        named = _freeze_except_projectors(model)
        rows = contrast.load_scene_rows(tokenizer, dataset_root)
        donor_mapping, donor_deltas, schedule_payload = contrast.build_donor_mapping(rows)
        schedule, _ = contrast.build_schedule(rows, donor_mapping, donor_deltas)
        sources = schedule[0].source_ordinals
        if len(sources) < context.world_size:
            raise RuntimeError("InfoNCE screen first open-fit schedule has too few rows")
        # The inherited open-fit schedule uses two rows per rank.  This screen
        # takes its first four rows as one explicit global contrast batch.
        source_ordinal = int(sources[context.process_rank])
        frozen_queries, local_states = _capture_local_vectors(
            model, rows[source_ordinal].example, pad_token_id=int(tokenizer.pad_token_id), device=context.device
        )
        all_states = _gather_states(local_states, context.world_size).detach()
        initial_loss, initial_margin = _local_infonce(
            model, frozen_queries, all_states, process_rank=context.process_rank
        )
        optimizer = torch.optim.AdamW([parameter for _, parameter in named], lr=0.05, weight_decay=0.0, fused=True)
        losses = [float(initial_loss.detach().item())]
        margins = [float(initial_margin.detach().item())]
        gradient = None
        for step in range(steps):
            optimizer.zero_grad(set_to_none=True)
            loss, margin = _local_infonce(
                model, frozen_queries, all_states, process_rank=context.process_rank
            )
            if not bool(torch.isfinite(loss).item() and torch.isfinite(margin).item()):
                raise RuntimeError("InfoNCE screen objective is non-finite")
            loss.backward()
            if step == 0:
                gradient = _gradient_audit(named)
                if gradient["passed"] is not True:
                    raise RuntimeError(f"InfoNCE screen cold-start gradient audit failed: {gradient!r}")
            # The objective is a distributed average of local-anchor losses.
            for _, parameter in named:
                dist.all_reduce(parameter.grad, op=dist.ReduceOp.SUM)
                parameter.grad.div_(context.world_size)
            optimizer.step()
            losses.append(float(loss.detach().item()))
            margins.append(float(margin.detach().item()))
        local = {
            "rank": context.process_rank,
            "source_ordinal": source_ordinal,
            "initial_loss": losses[0],
            "final_loss": losses[-1],
            "initial_margin": margins[0],
            "final_margin": margins[-1],
        }
        rank_rows = distributed.gather_objects(context, local)
        mean_initial_loss = sum(row["initial_loss"] for row in rank_rows) / context.world_size
        mean_final_loss = sum(row["final_loss"] for row in rank_rows) / context.world_size
        mean_initial_margin = sum(row["initial_margin"] for row in rank_rows) / context.world_size
        mean_final_margin = sum(row["final_margin"] for row in rank_rows) / context.world_size
        result = {
            "schema": SCHEMA,
            "protocol_payload_sha256": PROTOCOL_PAYLOAD_SHA256,
            "protocol_objective": protocol["objective"],
            "status": "screen_passed_causal_preflight_not_authorized" if mean_final_loss < mean_initial_loss and mean_final_margin > 0.0 else "screen_failed_causal_preflight_blocked",
            "passed": bool(mean_final_loss < mean_initial_loss and mean_final_margin > 0.0),
            "generation_authorized": False,
            "causal_preflight_authorized": False,
            "reason": "This geometry-only screen never evaluates donor/zero/layer-permuted causal CE controls.",
            "seed": SEED,
            "steps": steps,
            "temperature": TEMPERATURE,
            "base_model": str(base_model),
            "dataset_root": str(dataset_root),
            "hf_endpoint": os.environ.get("HF_ENDPOINT"),
            "protected_splits_opened": [],
            "model_audit": {**dict(inherited_audit), "infonce": dict(install_audit)},
            "gradient": gradient,
            "mean_initial_loss": mean_initial_loss,
            "mean_final_loss": mean_final_loss,
            "mean_initial_margin": mean_initial_margin,
            "mean_final_margin": mean_final_margin,
            "rank_rows": list(rank_rows),
            "rank_devices": list(context.rank_devices),
            "no_adapter_weights_saved": True,
        }
        if context.is_primary:
            result["receipt"] = {
                "algorithm": "sha256",
                "payload_scope": "canonical_result_without_receipt",
                "payload_sha256": distributed.canonical_sha256(result),
            }
            _write_json(output_dir / "result.json", result)
        dist.barrier()
        return result
    finally:
        distributed.destroy_distributed_training(context)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-model", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=DEFAULT_STEPS)
    args = parser.parse_args(argv)
    run(
        base_model=args.base_model.expanduser().resolve(strict=True),
        dataset_root=args.dataset_root.expanduser().resolve(strict=True),
        output_dir=args.output_dir.expanduser().resolve(),
        steps=args.steps,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
