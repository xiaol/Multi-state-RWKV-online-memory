#!/usr/bin/env python3
"""Run the structural and gradient gate for the projected-KV canary."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed

from deltamem.core.delta import (
    HFDeltaMemConfig,
    attach_delta_mem,
    freeze_non_delta_mem_params,
    get_delta_mem_online_state,
    iter_delta_mem_modules,
    load_delta_mem_online_state,
    reset_delta_mem_states,
    set_delta_mem_read_context_mask,
    set_delta_mem_write_enabled,
    set_delta_mem_write_message_ids,
    set_delta_mem_write_sentence_ids,
)
from deltamem.train.delta_sft_experimental import (
    EpisodeCausalLMCollator,
    _disable_training_cache,
    _promote_trainable_parameters_to_fp32,
    build_episode_training_examples,
    checkpoint_frozen_mlp_activations,
)
from experiments.rethinking_rwkv_ms_gemma import (
    prepare_synthetic_associative_retrieval_canary as canary,
)
from experiments.rethinking_rwkv_ms_gemma import (
    run_synthetic_associative_retrieval_gate0 as gate0,
)


RECEIPT_SCHEMA = "rwkv_ms_synthetic_associative_retrieval_preflight.v1"
HF_MIRROR_ENDPOINT = "https://hf-mirror.com"
TARGET_LAYERS = tuple(range(42))
EXPECTED_QUERY_SLOTS = (0, 0, 1, 1)


def build_delta_config() -> HFDeltaMemConfig:
    return HFDeltaMemConfig(
        rank=4,
        alpha=8,
        memory_backend="rwkv_ms",
        rwkv_ms_num_states=canary.RWKV_MS_NUM_STATES,
        rwkv_ms_chunk_size=128,
        rwkv_ms_boundary_mode="fixed_chunk",
        rwkv_ms_write_mode="recurrent",
        rwkv_ms_erase_gate=1.0,
        rwkv_ms_read_top_k=0,
        rwkv_ms_output_init_scale=0.02,
        rwkv_ms_semantics_version=2,
        num_state_heads=1,
        beta_bias_init=0.0,
        couple_lambda=True,
        state_update_mode="standard",
        rankwise_gates=True,
        output_init="base_slice_fixed",
        base_slice_ref_width=8,
        delta_heads=("q", "o"),
        delta_o_rmsnorm=False,
        memory_fusion_mode="add",
        memory_fusion_placement="attention_output",
        memory_fusion_residual_scale=1.0,
        memory_fusion_residual_scale_max=1.0,
        trainable_delta_scale=True,
        delta_scale_init=0.1,
        delta_scale_max=0.5,
        delta_scale_granularity="head",
        delta_scale_parameterization="alpha_over_rank",
        online_gain=0.2,
        target_layers=TARGET_LAYERS,
        memory_readout_mode="projected_kv_slots",
        projected_kv_key_dim=canary.PROJECTED_KV_KEY_DIM,
        projected_kv_temperature=16.0,
        projected_kv_update_cosine_threshold=1.0,
        memory_write_source="learned_hidden",
        memory_write_granularity="message_mean",
    )


def _read_context_mask(batch: dict[str, torch.Tensor]) -> torch.Tensor:
    labels = batch["labels"]
    attention_mask = batch["attention_mask"]
    valid_tokens = attention_mask.ne(0)
    mask = labels.eq(-100) & valid_tokens
    mask[:, :-1] |= (
        labels[:, 1:].ne(-100)
        & valid_tokens[:, :-1]
        & valid_tokens[:, 1:]
    )
    return mask


def _tensor_sha256(value: torch.Tensor) -> str:
    contiguous = value.detach().float().cpu().contiguous().numpy().tobytes()
    return hashlib.sha256(contiguous).hexdigest()


def _prepare_batch(tokenizer: Any, rows: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
    examples = [
        build_episode_training_examples(
            tokenizer,
            row["messages"],
            canary.MAX_READ_LENGTH,
            assistant_loss_mode="final_assistant_only",
            episode_recent_messages=1,
            max_write_length=canary.MAX_WRITE_LENGTH,
            include_sentence_ids=True,
            require_scene_state_semantic_mask=True,
        )[0]
        for row in rows
    ]
    return EpisodeCausalLMCollator(tokenizer)(examples)


def _move_batch(
    batch: dict[str, torch.Tensor],
    device: torch.device,
) -> dict[str, torch.Tensor]:
    return {name: value.to(device) for name, value in batch.items()}


def _prime_correct_state(
    model: torch.nn.Module,
    batch: dict[str, torch.Tensor],
) -> None:
    reset_delta_mem_states(model)
    set_delta_mem_read_context_mask(model, None)
    set_delta_mem_write_message_ids(model, batch["write_message_ids"])
    set_delta_mem_write_sentence_ids(model, batch["write_sentence_ids"])
    set_delta_mem_write_enabled(model, True)
    model(
        input_ids=batch["write_input_ids"],
        attention_mask=batch["write_attention_mask"],
        use_cache=False,
        return_dict=True,
        logits_to_keep=1,
    )
    set_delta_mem_write_message_ids(model, None)
    set_delta_mem_write_sentence_ids(model, None)


def _audit_write_state(model: torch.nn.Module) -> tuple[list[dict[str, Any]], int]:
    audits: list[dict[str, Any]] = []
    modules = list(iter_delta_mem_modules(model))
    if len(modules) != len(TARGET_LAYERS):
        raise RuntimeError(
            f"Expected {len(TARGET_LAYERS)} projected-KV layers; got {len(modules)}"
        )
    for name, module in modules:
        keys = module.projected_kv_keys
        values = module.projected_kv_values
        occupied = module.projected_kv_occupied
        routes = module.last_write_routes
        if keys is None or values is None or occupied is None or routes is None:
            raise RuntimeError(f"Projected-KV write state is absent at {name}")
        if tuple(keys.shape) != (4, 2, canary.PROJECTED_KV_KEY_DIM):
            raise RuntimeError(f"Projected-KV key shape differs at {name}: {keys.shape}")
        if tuple(occupied.shape) != (4, 2) or tuple(routes.shape) != (4, 2, 2):
            raise RuntimeError(f"Projected-KV occupancy/routes differ at {name}")
        occupancy_counts = occupied.sum(dim=-1)
        route_counts = routes.float().sum(dim=-1)
        if not bool(occupancy_counts.eq(2).all()):
            raise RuntimeError(
                f"Two message writes did not occupy two slots at {name}: "
                f"{occupancy_counts.detach().cpu().tolist()}"
            )
        if not bool(route_counts.eq(1).all()):
            raise RuntimeError(f"Projected-KV write routes are not one-hot at {name}")
        route_indices = routes.argmax(dim=-1)
        expected_write_routes = torch.tensor(
            [[0, 1]] * 4,
            device=route_indices.device,
            dtype=route_indices.dtype,
        )
        if not torch.equal(route_indices, expected_write_routes):
            raise RuntimeError(
                f"Projected-KV initialization routes differ at {name}: "
                f"{route_indices.detach().cpu().tolist()}"
            )
        key_norms = keys.float().norm(dim=-1)
        if not bool(torch.isfinite(keys).all()) or not torch.allclose(
            key_norms,
            torch.ones_like(key_norms),
            rtol=1e-3,
            atol=1e-3,
        ):
            raise RuntimeError(f"Projected-KV keys are not finite unit vectors at {name}")
        if not bool(torch.isfinite(values).all()):
            raise RuntimeError(f"Projected-KV values are not finite at {name}")
        keys.retain_grad()
        values.retain_grad()
        slot_cosine = (keys[:, 0].float() * keys[:, 1].float()).sum(dim=-1)
        audits.append(
            {
                "module": name,
                "layer": int(module.layer_idx),
                "occupied_per_row": occupancy_counts.detach().cpu().tolist(),
                "write_slot_indices": route_indices.detach().cpu().tolist(),
                "record_key_cosine": slot_cosine.detach().cpu().tolist(),
                "key_norm_min": float(key_norms.min().detach().cpu().item()),
                "key_norm_max": float(key_norms.max().detach().cpu().item()),
            }
        )
    return audits, len(modules)


def _target_metadata(source: dict[str, Any]) -> tuple[list[int], list[int], list[int]]:
    row_records = source["row_records"]
    positions = [int(row["token_metadata"]["target_label_position"]) for row in row_records]
    targets = [int(row["token_metadata"]["target_token_id"]) for row in row_records]
    donors = [int(row["token_metadata"]["donor_target_token_id"]) for row in row_records]
    return positions, targets, donors


def _selected_loss(
    logits: torch.Tensor,
    positions: list[int],
    target_ids: list[int],
) -> torch.Tensor:
    row_indices = torch.arange(len(positions), device=logits.device)
    predictor_positions = torch.tensor(positions, device=logits.device) - 1
    targets = torch.tensor(target_ids, device=logits.device)
    selected_logits = logits[row_indices, predictor_positions]
    return F.cross_entropy(selected_logits.float(), targets)


def _audit_read_routes(
    model: torch.nn.Module,
    positions: list[int],
) -> tuple[list[dict[str, Any]], float]:
    expected = torch.tensor(EXPECTED_QUERY_SLOTS)
    audits: list[dict[str, Any]] = []
    matches = 0
    total = 0
    for name, module in iter_delta_mem_modules(model):
        routes = module.last_read_routes
        if routes is None or tuple(routes.shape[:1]) != (4,) or routes.size(-1) != 2:
            raise RuntimeError(f"Projected-KV read routes are absent at {name}")
        row_indices = torch.arange(4, device=routes.device)
        predictor_positions = torch.tensor(positions, device=routes.device) - 1
        selected = routes[row_indices, predictor_positions]
        if not bool(torch.isfinite(selected).all()):
            raise RuntimeError(f"Projected-KV read routes are non-finite at {name}")
        hard_indices = selected.argmax(dim=-1).detach().cpu()
        layer_matches = int(hard_indices.eq(expected).sum().item())
        matches += layer_matches
        total += 4
        audits.append(
            {
                "module": name,
                "layer": int(module.layer_idx),
                "target_predictor_slot_indices": hard_indices.tolist(),
                "intended_slot_match_count": layer_matches,
            }
        )
    return audits, matches / total


def _gradient_audit(model: torch.nn.Module) -> list[dict[str, Any]]:
    audits: list[dict[str, Any]] = []
    for name, module in iter_delta_mem_modules(model):
        key_grad = module.projected_kv_key_proj.grad
        stored_key_grad = (
            None if module.projected_kv_keys is None else module.projected_kv_keys.grad
        )
        stored_value_grad = (
            None if module.projected_kv_values is None else module.projected_kv_values.grad
        )
        if key_grad is None or stored_key_grad is None or stored_value_grad is None:
            raise RuntimeError(f"Projected-KV gradient path is absent at {name}")
        norms = {
            "key_projection_grad_norm": float(key_grad.float().norm().item()),
            "stored_key_grad_norm": float(stored_key_grad.float().norm().item()),
            "stored_value_grad_norm": float(stored_value_grad.float().norm().item()),
        }
        if any(not math.isfinite(value) or value <= 0.0 for value in norms.values()):
            raise RuntimeError(f"Projected-KV gradient path is zero/non-finite at {name}: {norms}")
        audits.append({"module": name, "layer": int(module.layer_idx), **norms})
    return audits


def _permuted_state(
    state: dict[str, torch.Tensor],
    indices: tuple[int, ...],
) -> dict[str, torch.Tensor]:
    order = torch.tensor(indices, dtype=torch.long)
    return {name: value.index_select(0, order) for name, value in state.items()}


def _wrong_slot_state(state: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    result = {name: value.clone() for name, value in state.items()}
    value_names = [
        name for name in result if name.endswith(".__projected_kv_values")
    ]
    if len(value_names) != len(TARGET_LAYERS):
        raise RuntimeError("Projected-KV state does not contain one value tensor per layer")
    for name in value_names:
        result[name] = result[name].flip(1)
    return result


def _condition_scores(
    model: torch.nn.Module,
    batch: dict[str, torch.Tensor],
    state: dict[str, torch.Tensor] | None,
    positions: list[int],
    target_ids: list[int],
    donor_ids: list[int],
) -> tuple[dict[str, Any], torch.Tensor]:
    reset_delta_mem_states(model)
    if state is not None:
        load_delta_mem_online_state(model, state)
    set_delta_mem_write_enabled(model, False)
    set_delta_mem_read_context_mask(model, _read_context_mask(batch))
    outputs = model(
        input_ids=batch["input_ids"],
        attention_mask=batch["attention_mask"],
        use_cache=False,
        return_dict=True,
    )
    logits = outputs.logits
    row_indices = torch.arange(4, device=logits.device)
    predictors = torch.tensor(positions, device=logits.device) - 1
    predictor_logits = logits[row_indices, predictors].float().detach().cpu()
    target_tensor = torch.tensor(target_ids).unsqueeze(1)
    donor_tensor = torch.tensor(donor_ids).unsqueeze(1)
    target_scores = predictor_logits.gather(1, target_tensor).squeeze(1)
    donor_scores = predictor_logits.gather(1, donor_tensor).squeeze(1)
    return (
        {
            "target_logits": target_scores.tolist(),
            "donor_logits": donor_scores.tolist(),
            "target_minus_donor_margins": (target_scores - donor_scores).tolist(),
            "predictor_logits_sha256": _tensor_sha256(predictor_logits),
        },
        predictor_logits,
    )


def run_preflight(
    source_manifest: Path,
    gate0_receipt: Path,
    model_path: Path,
    output: Path,
    *,
    device_name: str,
    dtype_name: str,
    attn_implementation: str,
) -> dict[str, Any]:
    if os.environ.get("HF_ENDPOINT") != HF_MIRROR_ENDPOINT:
        raise ValueError(f"HF_ENDPOINT must be exactly {HF_MIRROR_ENDPOINT}")
    output = output.expanduser().resolve()
    if output.exists() or output.is_symlink():
        raise ValueError(f"Preflight output must be fresh: {output}")
    source = canary.load_source_bundle(
        source_manifest,
        model_path=model_path,
        verify_model_hashes=True,
    )
    gate_receipt = gate0.validate_receipt(
        gate0_receipt,
        source_manifest,
        model_path,
        verify_model_hashes=False,
    )
    if source["manifest"]["spec"]["memory_topology"] != {
        "backend": "rwkv_ms",
        "readout": "projected_kv_slots",
        "write_granularity": "message_mean",
        "num_states": 2,
        "projected_kv_key_dim": 32,
        "expected_record_proposals": 2,
    }:
        raise ValueError("Associative source memory topology is not the exact two-slot gate")

    dtype = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }[dtype_name]
    device = torch.device(device_name)
    set_seed(42)
    tokenizer = AutoTokenizer.from_pretrained(
        source["model"]["path"],
        local_files_only=True,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    batch = _move_batch(_prepare_batch(tokenizer, source["rows"]), device)
    positions, target_ids, donor_ids = _target_metadata(source)

    model = AutoModelForCausalLM.from_pretrained(
        source["model"]["path"],
        torch_dtype=dtype,
        attn_implementation=attn_implementation,
        local_files_only=True,
        low_cpu_mem_usage=True,
    ).to(device)
    _disable_training_cache(model)
    config = build_delta_config()
    replaced = attach_delta_mem(model, config)
    trainable_names = freeze_non_delta_mem_params(model)
    _promote_trainable_parameters_to_fp32(model)
    checkpointed_mlps = checkpoint_frozen_mlp_activations(model)
    model.train()
    model.zero_grad(set_to_none=True)

    _prime_correct_state(model, batch)
    write_audit, module_count = _audit_write_state(model)
    correct_state = get_delta_mem_online_state(model)
    set_delta_mem_write_enabled(model, False)
    set_delta_mem_read_context_mask(model, _read_context_mask(batch))
    outputs = model(
        input_ids=batch["input_ids"],
        attention_mask=batch["attention_mask"],
        use_cache=False,
        return_dict=True,
    )
    read_audit, intended_route_fraction = _audit_read_routes(model, positions)
    loss = _selected_loss(outputs.logits, positions, target_ids)
    if not bool(torch.isfinite(loss)):
        raise RuntimeError("Projected-KV preflight loss is non-finite")
    loss.backward()
    gradient_audit = _gradient_audit(model)

    model.eval()
    condition_states = {
        "correct": correct_state,
        "donor": _permuted_state(correct_state, canary.DONOR_INDICES),
        "wrong_slot": _wrong_slot_state(correct_state),
        "no_write": None,
    }
    condition_receipts: dict[str, dict[str, Any]] = {}
    condition_logits: dict[str, torch.Tensor] = {}
    with torch.no_grad():
        for condition, state in condition_states.items():
            receipt, predictor_logits = _condition_scores(
                model,
                batch,
                state,
                positions,
                target_ids,
                donor_ids,
            )
            condition_receipts[condition] = receipt
            condition_logits[condition] = predictor_logits
    distinguishability: dict[str, dict[str, float]] = {}
    correct_logits = condition_logits["correct"]
    for condition in ("donor", "wrong_slot", "no_write"):
        delta = (correct_logits - condition_logits[condition]).abs()
        metrics = {
            "maximum_absolute_logit_delta": float(delta.max().item()),
            "mean_absolute_logit_delta": float(delta.mean().item()),
        }
        if (
            not math.isfinite(metrics["maximum_absolute_logit_delta"])
            or metrics["maximum_absolute_logit_delta"] <= 0.0
        ):
            raise RuntimeError(f"Correct and {condition} conditions are indistinguishable")
        distinguishability[f"correct_vs_{condition}"] = metrics

    decision = {
        "passed": True,
        "criteria": {
            "exactly_42_target_layers": module_count == 42,
            "two_occupied_slots_every_row_every_layer": True,
            "write_routes_0_then_1_every_row_every_layer": True,
            "finite_nonzero_key_projection_grad_every_layer": True,
            "finite_nonzero_stored_key_grad_every_layer": True,
            "finite_nonzero_stored_value_grad_every_layer": True,
            "correct_donor_wrong_slot_no_write_are_distinguishable": True,
        },
    }
    receipt: dict[str, Any] = {
        "schema": RECEIPT_SCHEMA,
        "source": {
            "manifest_path": str(source["manifest_path"]),
            "manifest_file_sha256": source["manifest_file_sha256"],
            "manifest_sha256": source["manifest_sha256"],
            "train_path": str(source["train_path"]),
            "train_sha256": source["train_sha256"],
            "rows_path": str(source["rows_path"]),
            "rows_sha256": source["rows_sha256"],
        },
        "gate0": gate_receipt,
        "model": source["model"],
        "runtime": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "device": str(device),
            "dtype": dtype_name,
            "attn_implementation": attn_implementation,
            "hf_endpoint": os.environ.get("HF_ENDPOINT"),
            "seed": 42,
        },
        "delta_config": config.to_dict(),
        "delta_config_sha256": canary.canonical_sha256(config.to_dict()),
        "replaced_modules": replaced,
        "trainable_parameter_names": trainable_names,
        "checkpointed_frozen_mlps": checkpointed_mlps,
        "target_label_positions": positions,
        "target_token_ids": target_ids,
        "donor_target_token_ids": donor_ids,
        "selected_target_loss": float(loss.detach().float().cpu().item()),
        "write_audit": write_audit,
        "read_initialization_audit": {
            "expected_query_slot_indices": list(EXPECTED_QUERY_SLOTS),
            "intended_route_fraction": intended_route_fraction,
            "layers": read_audit,
        },
        "gradient_audit": gradient_audit,
        "conditions": condition_receipts,
        "distinguishability": distinguishability,
        "gate": decision,
    }
    receipt["receipt_sha256"] = canary.canonical_sha256(receipt)
    canary.atomic_write(
        output,
        json.dumps(receipt, ensure_ascii=True, indent=2, sort_keys=True).encode("utf-8")
        + b"\n",
    )
    receipt["receipt_path"] = str(output)
    receipt["receipt_file_sha256"] = canary.sha256_file(output)
    return receipt


def validate_receipt(
    receipt_path: Path,
    source_manifest: Path,
    gate0_receipt: Path,
    model_path: Path,
    *,
    verify_model_hashes: bool,
) -> dict[str, Any]:
    path = receipt_path.expanduser().resolve()
    if not path.is_file() or path.is_symlink():
        raise ValueError(f"Projected-KV preflight receipt is invalid: {path}")
    receipt = json.loads(path.read_text(encoding="utf-8"))
    unsigned = dict(receipt)
    declared_hash = unsigned.pop("receipt_sha256", None)
    if declared_hash != canary.canonical_sha256(unsigned):
        raise ValueError("Projected-KV preflight receipt SHA-256 differs")
    source = canary.load_source_bundle(
        source_manifest,
        model_path=model_path,
        verify_model_hashes=verify_model_hashes,
    )
    gate_receipt = gate0.validate_receipt(
        gate0_receipt,
        source_manifest,
        model_path,
        verify_model_hashes=False,
    )
    expected_config = json.loads(
        json.dumps(build_delta_config().to_dict(), ensure_ascii=True)
    )
    expected_source = {
        "manifest_path": str(source["manifest_path"]),
        "manifest_file_sha256": source["manifest_file_sha256"],
        "manifest_sha256": source["manifest_sha256"],
        "train_path": str(source["train_path"]),
        "train_sha256": source["train_sha256"],
        "rows_path": str(source["rows_path"]),
        "rows_sha256": source["rows_sha256"],
    }
    runtime = receipt.get("runtime")
    if (
        receipt.get("schema") != RECEIPT_SCHEMA
        or receipt.get("source") != expected_source
        or receipt.get("gate0") != gate_receipt
        or receipt.get("model") != source["model"]
        or receipt.get("delta_config") != expected_config
        or receipt.get("delta_config_sha256")
        != canary.canonical_sha256(expected_config)
        or not isinstance(runtime, dict)
        or runtime.get("hf_endpoint") != HF_MIRROR_ENDPOINT
        or runtime.get("seed") != 42
        or receipt.get("gate", {}).get("passed") is not True
    ):
        raise ValueError("Projected-KV preflight receipt binding differs")
    return {
        "valid": True,
        "receipt_path": str(path),
        "receipt_file_sha256": canary.sha256_file(path),
        "receipt_sha256": declared_hash,
        "gate": receipt["gate"],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--gate0-receipt", type=Path, required=True)
    parser.add_argument("--model-path", type=Path, required=True)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--output", type=Path)
    group.add_argument("--validate-receipt", type=Path)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--dtype",
        choices=("bfloat16", "float16", "float32"),
        default="bfloat16",
    )
    parser.add_argument("--attn-implementation", default="sdpa")
    parser.add_argument("--verify-model-hashes", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.validate_receipt is not None:
        result = validate_receipt(
            args.validate_receipt,
            args.source_manifest,
            args.gate0_receipt,
            args.model_path,
            verify_model_hashes=args.verify_model_hashes,
        )
    else:
        result = run_preflight(
            args.source_manifest,
            args.gate0_receipt,
            args.model_path,
            args.output,
            device_name=args.device,
            dtype_name=args.dtype,
            attn_implementation=args.attn_implementation,
        )
    print(json.dumps(result, ensure_ascii=True, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
