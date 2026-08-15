#!/usr/bin/env python3
"""Prove that the proposed native recurrent RWKV-MS path is active."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import sys
from typing import Any, Mapping, Sequence

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = Path(__file__).resolve().parents[2]
for import_root in (SCRIPT_DIR, PROJECT_ROOT):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

from deltamem.chat_templates import apply_chat_template  # noqa: E402
from deltamem.core.delta import (  # noqa: E402
    HFDeltaMemConfig,
    attach_delta_mem,
    collect_delta_mem_output_ratio_stats,
    get_delta_mem_online_state,
    iter_delta_mem_modules,
    load_delta_mem_online_state,
    reset_delta_mem_states,
    set_delta_mem_write_enabled,
)


SCHEMA = "rwkv_ms_natural_memory_native_recurrent_preflight.v1"
HF_MIRROR_ENDPOINT = "https://hf-mirror.com"
PROTOCOL = SCRIPT_DIR / "natural_memory_native_recurrent_rwkv_protocol_v1.json"
PROTOCOL_PAYLOAD_SHA256 = (
    "2bf65e14682b416b39ea7d21aa8d25e6d49000a870b4b70a4129aa13e7253f29"
)
EXPECTED_BASE_CONFIG_SHA256 = (
    "33b10c02df3c2e8536cf323d29d53262aaa2f4d11dbe19bc729373fbe90295d4"
)
EXPECTED_LAYERS = 42
MIN_WRITE_TOKENS = 385
MIN_LOGIT_DELTA = 1e-6
PREFLIGHT_SEED = 57
PROJECTED_STATE_SUFFIXES = (
    ".__projected_kv_keys",
    ".__projected_kv_values",
    ".__projected_kv_occupied",
    ".__projected_kv_surprise",
)


def canonical_sha256(value: Any) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_protocol() -> Mapping[str, Any]:
    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    receipt = protocol.get("receipt")
    if not isinstance(receipt, Mapping):
        raise ValueError("Recurrent protocol receipt is missing")
    unsigned = dict(protocol)
    unsigned.pop("receipt")
    digest = canonical_sha256(unsigned)
    if (
        digest != PROTOCOL_PAYLOAD_SHA256
        or receipt.get("payload_sha256") != digest
    ):
        raise ValueError("Recurrent protocol payload hash differs")
    candidate = protocol.get("paired_architectures", {}).get(
        "recurrent_candidate", {}
    )
    required = {
        "memory_backend": "rwkv_ms",
        "memory_readout_mode": "delta",
        "rwkv_ms_num_states": 4,
        "rwkv_ms_chunk_size": 128,
        "rwkv_ms_write_mode": "recurrent",
        "rwkv_ms_semantics_version": 2,
    }
    mismatches = [
        key for key, expected in required.items() if candidate.get(key) != expected
    ]
    if mismatches:
        raise ValueError(
            "Recurrent protocol architecture differs: " + ", ".join(mismatches)
        )
    if protocol.get("protected_splits_opened_by_this_protocol") != []:
        raise ValueError("Recurrent preflight may not authorize protected data")
    return protocol


def build_config() -> HFDeltaMemConfig:
    return HFDeltaMemConfig(
        rank=32,
        alpha=64,
        memory_backend="rwkv_ms",
        normalize_qk=True,
        couple_lambda=True,
        state_update_mode="standard",
        rankwise_gates=True,
        output_init="base_slice_fixed",
        base_slice_ref_width=8,
        online_gain=0.2,
        num_state_heads=1,
        num_memory_partitions=1,
        memory_readout_mode="delta",
        memory_write_source="learned_hidden",
        memory_write_granularity="token",
        target_modules=("self_attn",),
        target_layers=tuple(range(EXPECTED_LAYERS)),
        delta_heads=("q", "o"),
        memory_fusion_mode="content_gated_qo_add",
        memory_fusion_gate_init=0.1,
        memory_fusion_placement="attention_output",
        memory_fusion_residual_scale=1.0,
        memory_fusion_residual_scale_max=1.0,
        trainable_delta_scale=True,
        delta_scale_init=0.1,
        delta_scale_max=0.5,
        delta_scale_granularity="head",
        delta_scale_parameterization="alpha_over_rank",
        rwkv_ms_num_states=4,
        rwkv_ms_chunk_size=128,
        rwkv_ms_boundary_mode="fixed_chunk",
        rwkv_ms_write_mode="recurrent",
        rwkv_ms_erase_gate=1.0,
        rwkv_ms_read_top_k=0,
        rwkv_ms_mask_empty_slots=False,
        rwkv_ms_output_init_scale=0.02,
        rwkv_ms_semantics_version=2,
    )


def layer_ordinal(name: str) -> int:
    match = re.search(r"\.layers\.(\d+)\.", name)
    if match is None:
        raise ValueError(f"Cannot resolve layer ordinal from {name}")
    return int(match.group(1))


def permute_recurrent_state(
    state: Mapping[str, torch.Tensor],
    module_names: Sequence[str],
) -> dict[str, torch.Tensor]:
    ordered = tuple(sorted(module_names, key=layer_ordinal))
    suffixes = ("", ".__rwkv_ms_positions", ".__rwkv_ms_previous_source")
    expected = {
        f"{module_name}{suffix}" for module_name in ordered for suffix in suffixes
    }
    if set(state) != expected:
        raise ValueError("Recurrent online-state bundle contains unexpected keys")
    permuted: dict[str, torch.Tensor] = {}
    for index, target in enumerate(ordered):
        source = ordered[(index + 1) % len(ordered)]
        for suffix in suffixes:
            target_key = f"{target}{suffix}"
            source_key = f"{source}{suffix}"
            if state[target_key].shape != state[source_key].shape:
                raise ValueError("Recurrent layer-state shapes differ")
            permuted[target_key] = state[source_key].detach().cpu().clone()
    return permuted


def encode(tokenizer, messages: Sequence[Mapping[str, str]]) -> Mapping[str, torch.Tensor]:
    rendered = apply_chat_template(
        tokenizer,
        list(messages),
        tokenize=False,
        add_generation_prompt=True,
    )
    return tokenizer(rendered, return_tensors="pt", add_special_tokens=False)


def write_messages(model, encoded, *, device: str) -> dict[str, torch.Tensor]:
    reset_delta_mem_states(model)
    set_delta_mem_write_enabled(model, True)
    with torch.inference_mode():
        model(
            input_ids=encoded.input_ids.to(device),
            attention_mask=encoded.attention_mask.to(device),
            use_cache=False,
            return_dict=True,
            logits_to_keep=1,
        )
    return get_delta_mem_online_state(model)


def read_logits(model, encoded, *, device: str) -> torch.Tensor:
    set_delta_mem_write_enabled(model, False)
    with torch.inference_mode():
        outputs = model(
            input_ids=encoded.input_ids.to(device),
            attention_mask=encoded.attention_mask.to(device),
            use_cache=False,
            return_dict=True,
            logits_to_keep=1,
        )
    return outputs.logits.detach().float().cpu()


def state_audit(
    state: Mapping[str, torch.Tensor],
    module_names: Sequence[str],
    *,
    write_tokens: int,
) -> Mapping[str, Any]:
    layer_rows = []
    projected_names = sorted(
        name for name in state if name.endswith(PROJECTED_STATE_SUFFIXES)
    )
    for name in sorted(module_names, key=layer_ordinal):
        matrix = state[name].float()
        positions = state[f"{name}.__rwkv_ms_positions"]
        previous = state[f"{name}.__rwkv_ms_previous_source"].float()
        slot_norms = matrix.norm(dim=(-1, -2)).reshape(-1).tolist()
        layer_rows.append(
            {
                "layer": layer_ordinal(name),
                "position": int(positions.reshape(-1)[0].item()),
                "slot_norms": [float(value) for value in slot_norms],
                "previous_source_norm": float(previous.norm().item()),
            }
        )
    checks = {
        "projected_kv_sidecars_absent": not projected_names,
        "all_layer_position_counters_advance": all(
            row["position"] == write_tokens for row in layer_rows
        ),
        "all_layer_recurrent_matrix_states_nonzero_after_write": all(
            all(norm > 0.0 for norm in row["slot_norms"]) for row in layer_rows
        ),
        "all_layer_previous_sources_nonzero_after_write": all(
            row["previous_source_norm"] > 0.0 for row in layer_rows
        ),
    }
    return {
        "checks": checks,
        "projected_state_names": projected_names,
        "layers": layer_rows,
    }


def run(args: argparse.Namespace) -> Mapping[str, Any]:
    if os.environ.get("HF_ENDPOINT") != HF_MIRROR_ENDPOINT:
        raise ValueError(f"HF_ENDPOINT must be exactly {HF_MIRROR_ENDPOINT}")
    protocol = validate_protocol()
    set_seed(PREFLIGHT_SEED)
    base_model = args.base_model.expanduser().resolve(strict=True)
    if sha256_file(base_model / "config.json") != EXPECTED_BASE_CONFIG_SHA256:
        raise ValueError("Pinned Gemma base config differs")
    tokenizer = AutoTokenizer.from_pretrained(base_model, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        dtype=torch.float32,
        device_map={"": args.device},
        attn_implementation="sdpa",
        local_files_only=True,
    ).eval()
    replaced = attach_delta_mem(model, build_config())
    modules = tuple(iter_delta_mem_modules(model))
    module_names = tuple(name for name, _ in modules)
    static_checks = {
        "all_42_wrappers_use_delta_readout": (
            len(replaced) == EXPECTED_LAYERS
            and len(modules) == EXPECTED_LAYERS
            and all(
                module.memory_backend == "rwkv_ms"
                and module.memory_readout_mode == "delta"
                for _, module in modules
            )
        ),
        "write_phase_uses_no_kv_cache": True,
        "read_phase_disables_memory_writes": True,
    }

    write_text = " ".join(
        f"memory-record-{index} has value {index % 17}."
        for index in range(36)
    )
    write_encoded = encode(
        tokenizer,
        (
            {"role": "system", "content": "Store the supplied records."},
            {"role": "user", "content": write_text},
        ),
    )
    write_tokens = int(write_encoded.input_ids.size(1))
    if write_tokens < MIN_WRITE_TOKENS:
        raise ValueError(
            f"Preflight prompt has {write_tokens} tokens; need {MIN_WRITE_TOKENS}"
        )
    correct_state = write_messages(model, write_encoded, device=args.device)
    audit = state_audit(correct_state, module_names, write_tokens=write_tokens)

    read_encoded = encode(
        tokenizer,
        (
            {"role": "system", "content": "Answer from the stored records."},
            {"role": "user", "content": "What value belongs to memory-record-27?"},
        ),
    )
    correct_logits = read_logits(model, read_encoded, device=args.device)
    correct_output_stats = collect_delta_mem_output_ratio_stats(model)

    reset_delta_mem_states(model)
    zero_logits = read_logits(model, read_encoded, device=args.device)
    zero_output_stats = collect_delta_mem_output_ratio_stats(model)

    permuted_state = permute_recurrent_state(correct_state, module_names)
    reset_delta_mem_states(model)
    load_delta_mem_online_state(model, permuted_state)
    permuted_logits = read_logits(model, read_encoded, device=args.device)
    permuted_output_stats = collect_delta_mem_output_ratio_stats(model)

    correct_zero_delta = float((correct_logits - zero_logits).abs().max().item())
    correct_permuted_delta = float(
        (correct_logits - permuted_logits).abs().max().item()
    )
    behavioral_checks = {
        "correct_state_read_changes_logits_vs_zero_state": (
            correct_zero_delta > MIN_LOGIT_DELTA
        ),
        "correct_state_read_changes_logits_vs_layer_permuted_state": (
            correct_permuted_delta > MIN_LOGIT_DELTA
        ),
    }
    checks = {**static_checks, **audit["checks"], **behavioral_checks}
    passed = all(checks.values())
    result = {
        "schema": SCHEMA,
        "passed": passed,
        "protocol_payload_sha256": PROTOCOL_PAYLOAD_SHA256,
        "protocol_objective": protocol["objective"],
        "base_model": str(base_model),
        "base_config_sha256": EXPECTED_BASE_CONFIG_SHA256,
        "hf_endpoint": HF_MIRROR_ENDPOINT,
        "device": args.device,
        "seed": PREFLIGHT_SEED,
        "dtype": "float32",
        "attn_implementation": "sdpa",
        "write_tokens": write_tokens,
        "wrapped_layers": len(modules),
        "config": build_config().to_dict(),
        "checks": checks,
        "logit_deltas": {
            "correct_vs_zero_max_abs": correct_zero_delta,
            "correct_vs_layer_permuted_max_abs": correct_permuted_delta,
            "minimum": MIN_LOGIT_DELTA,
        },
        "state_audit": audit,
        "condition_output_stats": {
            "correct_state": correct_output_stats,
            "zero_state": zero_output_stats,
            "layer_permuted_state": permuted_output_stats,
        },
        "protected_splits_opened": [],
        "runner_sha256": sha256_file(Path(__file__)),
    }
    unsigned_digest = canonical_sha256(result)
    result["receipt"] = {
        "algorithm": "sha256",
        "payload_scope": "canonical_result_without_receipt",
        "payload_sha256": unsigned_digest,
    }
    return result


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--base-model",
        type=Path,
        default=Path("/root/X/.cache/hf/gemma-4-E4B-it-a4c2d58"),
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    result = run(args)
    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    rendered = json.dumps(result, indent=2, sort_keys=True, ensure_ascii=True) + "\n"
    if output.exists() and output.read_text(encoding="utf-8") != rendered:
        raise ValueError(f"Existing recurrent preflight result differs: {output}")
    output.write_text(rendered, encoding="utf-8")
    status = "PASS" if result["passed"] else "FAIL"
    print(
        f"RECURRENT_RWKV_PREFLIGHT_{status} "
        f"layers={result['wrapped_layers']} write_tokens={result['write_tokens']} "
        f"zero_delta={result['logit_deltas']['correct_vs_zero_max_abs']:.6g} "
        "permuted_delta="
        f"{result['logit_deltas']['correct_vs_layer_permuted_max_abs']:.6g}",
        flush=True,
    )
    if not result["passed"]:
        failed = [name for name, value in result["checks"].items() if not value]
        print("failed_checks=" + ",".join(failed), flush=True)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
