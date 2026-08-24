#!/usr/bin/env python3
"""Run the open-split four-way virtual-KV identity screen."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
from types import MethodType
from typing import Any, Mapping, Sequence

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

import torch


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[1]
for import_root in (SCRIPT_DIR, PROJECT_ROOT):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

from deltamem.core.delta import (  # noqa: E402
    reset_delta_mem_states,
    set_delta_mem_projected_kv_read_query_mask,
    set_delta_mem_write_enabled,
)
from deltamem.core.virtual_kv import ExplicitRWKVVirtualKV, VirtualKVShape  # noqa: E402
from experiments.rethinking_rwkv_ms_gemma import (  # noqa: E402
    materialize_natural_memory_native_rwkv_continuous_write_open_fit as materializer,
)
from experiments.rethinking_rwkv_ms_gemma import (  # noqa: E402
    natural_memory_distributed as distributed,
)
from experiments.rethinking_rwkv_ms_gemma import (  # noqa: E402
    run_natural_memory_native_evolution as evolution,
)
from experiments.rethinking_rwkv_ms_gemma import (  # noqa: E402
    run_natural_memory_native_rwkv_address_decoded_reconstruction as address_decode,
)
from experiments.rethinking_rwkv_ms_gemma import (  # noqa: E402
    run_natural_memory_native_rwkv_continuous_write_mechanics as mechanics,
)
from experiments.rethinking_rwkv_ms_gemma import (  # noqa: E402
    run_natural_memory_native_rwkv_v5_shadow_crossfit as exact_v5,
)
from experiments.rethinking_rwkv_ms_gemma import (  # noqa: E402
    rwkv_continuous_write_integration as integration,
)


SCHEMA = "rwkv_ms_natural_memory_native_rwkv_virtual_kv_identity.v1"
PROTOCOL = SCRIPT_DIR / "natural_memory_native_rwkv_virtual_kv_identity_protocol_v1.json"
MATERIALIZATION = SCRIPT_DIR / "local_artifacts/natural_memory_native_rwkv_continuous_write_open_fit_v1"
DEFAULT_BASE_MODEL = Path("/root/X/.cache/hf/gemma-4-E4B-it-a4c2d58")
DEFAULT_OUTPUT = SCRIPT_DIR / "local_artifacts/natural_memory_native_rwkv_virtual_kv_identity_v1"
WORLD_SIZE = 4
SEED = 211
ANCHOR_LAYERS = (5, 11, 17, 23)
ADDRESS_DIM = 64
STATE_RANK = 32
SLOTS = 4
FIT_ROWS = 64
RETRIEVAL_ROWS = 32
RIDGE = 1.0
HF_ENDPOINT = "https://hf-mirror.com"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_sha256(value: Any) -> str:
    return distributed.canonical_sha256(value)


def _signed_json(path: Path, value: Mapping[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(
        json.dumps(value, ensure_ascii=True, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _validate_protocol() -> Mapping[str, Any]:
    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    unsigned = dict(protocol)
    receipt = unsigned.pop("receipt", None)
    expected = {
        "algorithm": "sha256",
        "payload_scope": "canonical_protocol_without_receipt",
        "payload_sha256": canonical_sha256(unsigned),
    }
    if receipt != expected:
        raise ValueError("virtual-KV identity protocol receipt differs")
    if protocol.get("schema") != "rwkv_ms_natural_memory_native_rwkv_virtual_kv_identity_protocol.v1":
        raise ValueError("virtual-KV identity protocol schema differs")
    architecture = protocol["architecture"]
    if (
        architecture.get("anchor_layers") != list(ANCHOR_LAYERS)
        or architecture.get("virtual_slots") != SLOTS
        or architecture.get("query_length") != 1
        or architecture.get("attention_implementation") != "eager only"
        or architecture.get("model_parameters_updated") is not False
        or architecture.get("full_bandwidth_feedback_installed") is not False
    ):
        raise ValueError("virtual-KV architecture contract differs")
    lifecycle = protocol["data_lifecycle"]
    if lifecycle.get("already_open_bundles_only") != ["fit", "retrieval"]:
        raise ValueError("virtual-KV split contract differs")
    if lifecycle.get("protected_bytes_tokenized_or_forwarded") is not False:
        raise ValueError("virtual-KV protected-byte contract differs")
    execution = protocol["execution"]
    if execution.get("world_size") != WORLD_SIZE or execution.get("hf_endpoint") != HF_ENDPOINT:
        raise ValueError("virtual-KV execution contract differs")
    return protocol


def _load_rows(manifest: Mapping[str, Any], split: str) -> list[dict[str, Any]]:
    if split not in {"fit", "retrieval"}:
        raise PermissionError("virtual-KV screen may open only FIT and retrieval")
    rows = materializer._read_bundle(MATERIALIZATION, manifest, split)
    expected = FIT_ROWS if split == "fit" else RETRIEVAL_ROWS
    if len(rows) != expected:
        raise ValueError(f"virtual-KV {split} row count differs")
    return sorted(rows, key=lambda row: int(row["source_index"]))


def _feature_tensors(
    state: Mapping[str, Mapping[str, torch.Tensor]],
    modules: Sequence[tuple[str, Any]],
) -> dict[str, torch.Tensor]:
    tensors = {
        "state": torch.stack([state[name]["delta_state"].float() for name, _ in modules]),
        "positions": torch.stack([state[name]["rwkv_ms_positions"].long() for name, _ in modules]),
        "previous_source": torch.stack([state[name]["rwkv_ms_previous_source"].float() for name, _ in modules]),
        "keys": torch.stack([state[name]["projected_kv_keys"].float() for name, _ in modules]),
        "values": torch.stack([state[name]["projected_kv_values"].float() for name, _ in modules]),
        "occupied": torch.stack([state[name]["projected_kv_occupied"].bool() for name, _ in modules]),
    }
    if (
        tuple(tensors["state"].shape) != (len(modules), 1, 1, SLOTS, STATE_RANK, STATE_RANK)
        or tuple(tensors["keys"].shape) != (len(modules), 1, SLOTS, ADDRESS_DIM)
        or tuple(tensors["values"].shape) != (len(modules), 1, SLOTS, STATE_RANK)
        or tuple(tensors["occupied"].shape) != (len(modules), 1, SLOTS)
    ):
        raise ValueError("virtual-KV captured feature geometry differs")
    return {name: value.cpu().contiguous() for name, value in tensors.items()}


def _active_vector(tensor: torch.Tensor, occupied: torch.Tensor) -> torch.Tensor:
    active = occupied.nonzero(as_tuple=False)
    if active.size(0) != 1:
        raise ValueError("virtual-KV rows must have exactly one active slot")
    return tensor[active[0, 0]]


def _install_state(
    model: torch.nn.Module,
    modules: Sequence[tuple[str, Any]],
    feature: Mapping[str, torch.Tensor],
) -> None:
    state = {
        name: {
            "delta_state": feature["state"][index].to(next(module.parameters()).device),
            "rwkv_ms_positions": feature["positions"][index].to(next(module.parameters()).device),
            "rwkv_ms_previous_source": feature["previous_source"][index].to(next(module.parameters()).device),
            "projected_kv_keys": feature["keys"][index].to(next(module.parameters()).device),
            "projected_kv_values": feature["values"][index].to(next(module.parameters()).device),
            "projected_kv_occupied": feature["occupied"][index].to(next(module.parameters()).device),
            "projected_kv_surprise": torch.ones_like(
                feature["occupied"][index], dtype=torch.float32, device=next(module.parameters()).device
            ),
        }
        for index, (name, module) in enumerate(modules)
    }
    recurrent = {
        name: {
            "delta_state": values["delta_state"],
            "rwkv_ms_positions": values["rwkv_ms_positions"],
            "rwkv_ms_previous_source": values["rwkv_ms_previous_source"],
        }
        for name, values in state.items()
    }
    projected = {
        name: {
            attribute: values[attribute]
            for attribute in ("projected_kv_keys", "projected_kv_values", "projected_kv_occupied", "projected_kv_surprise")
        }
        for name, values in state.items()
    }
    if not exact_v5.causal_train.install_intervened_state(
        modules, projected=projected, recurrent=recurrent, rotate_recurrent_layers=False
    ):
        raise RuntimeError("virtual-KV state installation failed")


def _capture_attention_geometry(
    model: torch.nn.Module,
    modules: Sequence[tuple[str, Any]],
    feature: Mapping[str, torch.Tensor],
    batch: Any,
) -> dict[int, Mapping[str, torch.Tensor]]:
    _install_state(model, modules, feature)
    set_delta_mem_write_enabled(model, False)
    set_delta_mem_projected_kv_read_query_mask(model, None)
    _, predictor_index = address_decode.retrieval.first_prompt_boundary(batch.labels)
    captures: dict[int, Mapping[str, torch.Tensor]] = {}
    originals = {}
    try:
        for _, module in modules:
            layer = int(module.layer_idx)
            if layer not in ANCHOR_LAYERS:
                continue
            original = module._append_rwkv_virtual_kv
            originals[layer] = original

            def observer(
                wrapped: Any,
                query_states: torch.Tensor,
                key_states: torch.Tensor,
                value_states: torch.Tensor,
                attention_mask: torch.Tensor | None,
                *,
                position_embeddings: tuple[torch.Tensor, torch.Tensor],
                layer: int = layer,
                original: Any = original,
            ):
                if not 0 <= predictor_index < query_states.size(2):
                    raise ValueError("virtual-KV predictor index is outside attention query")
                query = query_states[:, :, predictor_index : predictor_index + 1].float()
                repeated_keys = key_states.float().repeat_interleave(
                    wrapped.num_key_value_groups, dim=1
                )
                logits = torch.einsum(
                    "bhqd,bhkd->bhqk", query, repeated_keys
                ) * float(wrapped.scaling)
                if attention_mask is not None:
                    mask_row = attention_mask[
                        :, :, predictor_index : predictor_index + 1, : key_states.size(2)
                    ]
                    if mask_row.dtype == torch.bool:
                        logits = logits.masked_fill(~mask_row, -torch.inf)
                    else:
                        logits = logits + mask_row.float()
                captures[layer] = {
                    "query": query.detach().cpu(),
                    "real_logsumexp": torch.logsumexp(logits, dim=-1).detach().cpu(),
                }
                return original(
                    query_states,
                    key_states,
                    value_states,
                    attention_mask,
                    position_embeddings=position_embeddings,
                )

            module._append_rwkv_virtual_kv = MethodType(observer, module)
        evolution._native_read(model, batch, dtype=torch.bfloat16)
    finally:
        for _, module in modules:
            layer = int(module.layer_idx)
            if layer in originals:
                module._append_rwkv_virtual_kv = originals[layer]
    if set(captures) != set(ANCHOR_LAYERS):
        raise RuntimeError("virtual-KV causal attention capture missed an anchor")
    return captures


def _fit_key_weights(
    records: Sequence[Mapping[str, Any]],
    module_indices: Mapping[int, int],
) -> dict[int, torch.Tensor]:
    result: dict[int, torch.Tensor] = {}
    for layer in ANCHOR_LAYERS:
        index = module_indices[layer]
        addresses = []
        queries = []
        for record in records:
            keys = record["features"]["keys"][index, 0]
            occupied = record["features"]["occupied"][index, 0]
            addresses.append(_active_vector(keys, occupied))
            query = record["attention"][layer]["query"]
            module = record["module_shapes"][layer]
            groups = int(module["num_query_heads"] // module["num_kv_heads"])
            queries.append(
                query.reshape(
                    module["num_kv_heads"], groups, module["head_dim"]
                ).mean(dim=1)
            )
        address = torch.stack(addresses).float()
        address = address / address.square().mean(dim=-1, keepdim=True).add(1e-12).sqrt()
        query = torch.stack(queries).float()
        query = query / query.square().mean(dim=-1, keepdim=True).add(1e-12).sqrt()
        gram = address.T @ address + RIDGE * torch.eye(ADDRESS_DIM)
        weights = []
        for head in range(query.size(1)):
            solved = torch.linalg.solve(gram, address.T @ query[:, head])
            weights.append(solved.T)
        result[layer] = torch.cat(weights, dim=0).contiguous()
    return result


def _candidate_sources(rows: Sequence[Mapping[str, Any]]) -> dict[int, tuple[int, int, int, int]]:
    by_source = {int(row["source_index"]): row for row in rows}
    groups: dict[frozenset[int], list[int]] = {}
    for source, row in by_source.items():
        group = frozenset((source, int(row["donor_source_index"])))
        groups.setdefault(group, []).append(source)
    ordered_groups = sorted(groups.values(), key=lambda values: min(values))
    result = {}
    for source, row in by_source.items():
        own = frozenset((source, int(row["donor_source_index"])))
        others = [values[0] for values in ordered_groups if frozenset(values) != own]
        if len(others) < 2:
            raise ValueError("virtual-KV distractor pool is too small")
        result[source] = (source, int(row["donor_source_index"]), others[0], others[1])
    return result


def _bank(
    records: Mapping[int, Mapping[str, Any]],
    sources: Sequence[int],
    module_index: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    state = torch.zeros(1, 1, SLOTS, STATE_RANK, STATE_RANK, device=device)
    keys = torch.zeros(1, SLOTS, ADDRESS_DIM, device=device)
    values = torch.zeros(1, SLOTS, STATE_RANK, device=device)
    occupied = torch.ones(1, SLOTS, dtype=torch.bool, device=device)
    for slot, source in enumerate(sources):
        feature = records[int(source)]["features"]
        state[0, 0, slot] = _active_vector(feature["state"][module_index, 0, 0].to(device), feature["occupied"][module_index, 0].to(device))
        keys[0, slot] = _active_vector(feature["keys"][module_index, 0].to(device), feature["occupied"][module_index, 0].to(device))
        values[0, slot] = _active_vector(feature["values"][module_index, 0].to(device), feature["occupied"][module_index, 0].to(device))
    return state, keys, values, occupied


def _make_builder(module: Any, weight: torch.Tensor, *, seed: int) -> ExplicitRWKVVirtualKV:
    builder = ExplicitRWKVVirtualKV(
        VirtualKVShape(
            key_dim=ADDRESS_DIM,
            state_heads=1,
            rank=STATE_RANK,
            slots=SLOTS,
            kv_heads=int(module.num_key_value_heads),
            head_dim=int(module.head_dim),
            seed=seed,
        )
    ).to(next(module.parameters()).device)
    with torch.no_grad():
        builder.key_proj.copy_(weight.to(builder.key_proj))
    builder.eval()
    return builder


def _capture_shard(
    context: Any,
    rows: Sequence[Mapping[str, Any]],
    model: torch.nn.Module,
    tokenizer: Any,
    modules: Sequence[tuple[str, Any]],
    output_dir: Path,
    split: str,
) -> None:
    examples = address_decode.retrieval._encode_rows(tokenizer, rows)
    local_rows = list(rows[context.process_rank :: WORLD_SIZE])
    records = []
    anchor_modules = {int(module.layer_idx): module for _, module in modules if int(module.layer_idx) in ANCHOR_LAYERS}
    for row in local_rows:
        source = int(row["source_index"])
        batch = evolution.collate_native_examples([examples[source]], pad_token_id=int(tokenizer.pad_token_id), device=context.device)
        state, audit, _ = mechanics.capture_write_condition(model, batch, modules, mode=integration.CONTINUOUS_MODE, override=None, reference_mode="none")
        if audit.get("formula_byte_exact_all_modules") is not True or audit.get("all_state_tensors_finite") is not True:
            raise RuntimeError("virtual-KV capture write audit differs")
        feature = _feature_tensors(state, modules)
        attention = _capture_attention_geometry(model, modules, feature, batch)
        records.append(
            {
                "source_index": source,
                "donor_source_index": int(row["donor_source_index"]),
                "row_sha256": row["row_sha256"],
                "features": feature,
                "attention": attention,
                "module_shapes": {
                    layer: {
                        "num_query_heads": int(module.base.config.num_attention_heads),
                        "num_kv_heads": int(module.num_key_value_heads),
                        "head_dim": int(module.head_dim),
                    }
                    for layer, module in anchor_modules.items()
                },
            }
        )
        reset_delta_mem_states(model)
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.save({"schema": f"{SCHEMA}.shard", "split": split, "rank": context.process_rank, "world_size": WORLD_SIZE, "records": records}, output_dir / f"{split}-shard-{context.process_rank}.pt")


def _load_shards(output_dir: Path, split: str) -> dict[int, Mapping[str, Any]]:
    merged: dict[int, Mapping[str, Any]] = {}
    for rank in range(WORLD_SIZE):
        payload = torch.load(output_dir / f"{split}-shard-{rank}.pt", map_location="cpu", weights_only=False)
        if payload.get("schema") != f"{SCHEMA}.shard" or payload.get("split") != split or payload.get("world_size") != WORLD_SIZE:
            raise ValueError("virtual-KV capture shard contract differs")
        for record in payload.get("records", []):
            source = int(record["source_index"])
            if source in merged:
                raise ValueError("virtual-KV capture source repeated")
            merged[source] = record
    return merged


def _score_retrieval_row(
    modules: Sequence[tuple[str, Any]],
    target: Mapping[str, Any],
    records: Mapping[int, Mapping[str, Any]],
    candidate_sources: Sequence[int],
    key_weights: Mapping[int, torch.Tensor],
) -> dict[int, Mapping[str, float]]:
    module_by_layer = {int(module.layer_idx): (index, module) for index, (_, module) in enumerate(modules)}
    scores: dict[int, Mapping[str, float]] = {}
    for layer in ANCHOR_LAYERS:
        index, module = module_by_layer[layer]
        device = next(module.parameters()).device
        bank_state, bank_keys, bank_values, bank_occupied = _bank(
            records, candidate_sources, index, device
        )
        builder = _make_builder(module, key_weights[layer], seed=SEED + layer)
        query = target["attention"][layer]["query"].to(device)
        real_logsumexp = target["attention"][layer]["real_logsumexp"].to(device)
        dummy_keys = torch.zeros(
            1, module.num_key_value_heads, 1, module.head_dim, device=device
        )
        built = builder(
            state=bank_state,
            address_keys=bank_keys,
            occupied=bank_occupied,
            query_states=query,
            real_keys=dummy_keys,
            real_values=dummy_keys,
            attention_mask=None,
        )
        if built is None:
            raise RuntimeError("virtual-KV active bank was unexpectedly disabled")
        virtual_keys, _, _ = built
        expanded_virtual = virtual_keys.float().repeat_interleave(
            module.num_key_value_groups, dim=1
        )
        virtual_logits = torch.einsum(
            "bhqd,bhkd->bhqk", query.float(), expanded_virtual
        ) * float(module.scaling)
        virtual_logsumexp = torch.logsumexp(virtual_logits, dim=-1)
        denominator = torch.logaddexp(real_logsumexp.float(), virtual_logsumexp)
        mass = torch.exp(virtual_logits - denominator.unsqueeze(-1)).mean(
            dim=(0, 1, 2)
        ).detach().cpu()
        correct = float(mass[0].item())
        wrong = mass[1:]
        strict_top1 = float(correct > float(wrong.max().item()))
        scores[layer] = {
            "correct_mass": correct,
            "wrong_max_mass": float(wrong.max().item()),
            "margin": correct - float(wrong.max().item()),
            "top1": strict_top1,
            "virtual_mass": float(mass.sum().item()),
            "zero_state_disabled": float(
                builder(
                    state=torch.zeros_like(bank_state),
                    address_keys=bank_keys,
                    occupied=bank_occupied,
                    query_states=query,
                    real_keys=dummy_keys,
                    real_values=dummy_keys,
                    attention_mask=None,
                )
                is None
            ),
        }
    return scores


def run(*, base_model: Path, output_dir: Path) -> Mapping[str, Any]:
    context = distributed.initialize_distributed_training("cuda", required_world_size=WORLD_SIZE, timeout_seconds=1800)
    if context is None:
        raise RuntimeError("Run with torchrun --nproc_per_node=4")
    try:
        protocol = _validate_protocol()
        if os.environ.get("HF_ENDPOINT") != HF_ENDPOINT or not exact_v5.hardware.four_distinct_a100s(context.rank_devices):
            raise RuntimeError("virtual-KV identity requires four A100s and HF mirror")
        manifest = address_decode._load_manifest_only(MATERIALIZATION)
        fit_rows = _load_rows(manifest, "fit")
        retrieval_rows = _load_rows(manifest, "retrieval")
        torch.manual_seed(SEED)
        torch.cuda.manual_seed_all(SEED)
        model, tokenizer, model_audit = exact_v5.load_exact_v5_model(base_model, device=context.device)
        model.eval()
        modules = exact_v5.causal_train.ordered_modules(model)
        module_names = tuple(name for name, _ in modules)
        maps = mechanics.load_frozen_maps(module_names)
        integration.install(model, rank=mechanics.MAP_RANK, seed=SEED, trainable_map=False)
        for name, module in modules:
            module.rwkv_continuous_write_conditioner.load_frozen_map(maps[name].down, maps[name].up)
            module.base.config._attn_implementation = "eager"
        if hasattr(model.config, "text_config"):
            model.config.text_config._attn_implementation = "eager"
        integration.set_mode(model, integration.CONTINUOUS_MODE)
        integration.set_capture(model, True)
        mechanics.install_feature_observer(modules)
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        if context.is_primary:
            if output_dir.exists():
                raise ValueError(f"virtual-KV output already exists: {output_dir}")
            output_dir.mkdir(parents=True, exist_ok=False)
        context.control_group and torch.distributed.barrier(group=context.control_group)
        _capture_shard(context, fit_rows, model, tokenizer, modules, output_dir, "fit")
        context.control_group and torch.distributed.barrier(group=context.control_group)
        fit_records = _load_shards(output_dir, "fit")
        if len(fit_records) != FIT_ROWS:
            raise RuntimeError("virtual-KV FIT coverage differs")
        module_indices = {int(module.layer_idx): index for index, (_, module) in enumerate(modules)}
        key_weights = _fit_key_weights(list(fit_records.values()), module_indices)
        context.control_group and torch.distributed.barrier(group=context.control_group)
        _capture_shard(context, retrieval_rows, model, tokenizer, modules, output_dir, "retrieval")
        context.control_group and torch.distributed.barrier(group=context.control_group)
        retrieval_records = _load_shards(output_dir, "retrieval")
        if len(retrieval_records) != RETRIEVAL_ROWS:
            raise RuntimeError("virtual-KV retrieval coverage differs")
        candidates = _candidate_sources(retrieval_rows)
        local_metrics = {}
        for row in retrieval_rows[context.process_rank :: WORLD_SIZE]:
            source = int(row["source_index"])
            target = retrieval_records[source]
            local_metrics[source] = _score_retrieval_row(
                modules,
                target,
                retrieval_records,
                candidates[source],
                key_weights,
            )
        gathered = [None] * WORLD_SIZE
        torch.distributed.all_gather_object(gathered, local_metrics, group=context.control_group)
        metrics: dict[int, Mapping[str, Mapping[str, float]]] = {}
        for payload in gathered:
            metrics.update(payload)
        per_anchor = {}
        for layer in ANCHOR_LAYERS:
            rows = [metrics[source][layer] for source in sorted(metrics)]
            margins = torch.tensor([row["margin"] for row in rows])
            top1 = torch.tensor([row["top1"] for row in rows])
            mass = torch.tensor([row["virtual_mass"] for row in rows])
            correct_mass = torch.tensor([row["correct_mass"] for row in rows])
            wrong_mass = torch.tensor([row["wrong_max_mass"] for row in rows])
            zero_disabled = torch.tensor([row["zero_state_disabled"] for row in rows])
            per_anchor[str(layer)] = {
                "within_bank_top1": float(top1.mean().item()),
                "mean_margin": float(margins.mean().item()),
                "positive_margin_row_fraction": float(margins.gt(0).float().mean().item()),
                "virtual_mass_mean": float(mass.mean().item()),
                "virtual_mass_nonzero_fraction": float(mass.gt(0).float().mean().item()),
                "correct_mass_exceeds_wrong_fraction": float(correct_mass.gt(wrong_mass).float().mean().item()),
                "zero_state_disabled_fraction": float(zero_disabled.mean().item()),
            }
        checks = {
            layer: (
                values["within_bank_top1"] >= 0.75
                and values["mean_margin"] >= 0.05
                and values["positive_margin_row_fraction"] >= 0.75
                and values["virtual_mass_nonzero_fraction"] >= 0.95
                and values["correct_mass_exceeds_wrong_fraction"] >= 0.75
            )
            for layer, values in per_anchor.items()
        }
        passed = sum(checks.values()) >= 3
        result = {
            "schema": SCHEMA,
            "status": "virtual_kv_identity_passed_open_split" if passed else "virtual_kv_identity_failed_open_split",
            "passed": passed,
            "protocol_receipt": protocol["receipt"]["payload_sha256"],
            "hardware": {"world_size": context.world_size, "rank_devices": list(context.rank_devices), "backend": context.backend, "control_backend": context.control_backend},
            "model_audit": model_audit,
            "fit_rows": FIT_ROWS,
            "retrieval_rows": RETRIEVAL_ROWS,
            "anchor_layers": list(ANCHOR_LAYERS),
            "candidate_bank_size": SLOTS,
            "fit_adapter": "bias-free per-anchor ridge address-to-query-key map, frozen before retrieval capture",
            "per_anchor": per_anchor,
            "anchor_checks": checks,
            "fit_and_retrieval_shards": [
                {"path": f"{split}-shard-{rank}.pt", "sha256": sha256_file(output_dir / f"{split}-shard-{rank}.pt")}
                for split in ("fit", "retrieval")
                for rank in range(WORLD_SIZE)
            ],
            "already_open_bundles_read": ["fit", "retrieval"],
            "mechanics_causal_generation_or_native_bytes_opened": False,
            "model_parameters_updated": False,
            "full_bandwidth_feedback_installed": False,
            "native_gain_claimed": False,
            "sota_claimed": False,
        }
        result["receipt"] = {"algorithm": "sha256", "payload_scope": "canonical_result_without_receipt", "payload_sha256": canonical_sha256(result)}
        if context.is_primary:
            _signed_json(output_dir / "result.json", result)
        context.control_group and torch.distributed.barrier(group=context.control_group)
        return result
    finally:
        distributed.destroy_distributed_training(context)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-model", type=Path, default=DEFAULT_BASE_MODEL)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args(argv)


if __name__ == "__main__":
    arguments = parse_args()
    outcome = run(base_model=arguments.base_model, output_dir=arguments.output_dir)
    print(json.dumps(outcome, ensure_ascii=True, sort_keys=True))
    raise SystemExit(0 if outcome.get("passed") else 1)
