#!/usr/bin/env python3
"""Run the sealed cumulative virtual-KV mechanics gate on four A100s."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any, Callable, Mapping, Sequence

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

import torch
import torch.distributed as dist
import torch.nn.functional as F


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
from deltamem.core.virtual_kv import (  # noqa: E402
    CumulativeRWKVCompatibilityRouter,
    ExplicitRWKVVirtualKV,
    VirtualKVShape,
)
from experiments.rethinking_rwkv_ms_gemma import (  # noqa: E402
    natural_memory_distributed as distributed,
)
from experiments.rethinking_rwkv_ms_gemma import (  # noqa: E402
    run_natural_memory_native_rwkv_continuous_write_mechanics as mechanics,
)
from experiments.rethinking_rwkv_ms_gemma import (  # noqa: E402
    run_natural_memory_native_rwkv_continuous_write_retrieval as retrieval,
)
from experiments.rethinking_rwkv_ms_gemma import (  # noqa: E402
    run_natural_memory_native_rwkv_v5_shadow_crossfit as exact_v5,
)
from experiments.rethinking_rwkv_ms_gemma import (  # noqa: E402
    rwkv_continuous_write_integration as integration,
)


SCHEMA = "rwkv_ms_natural_memory_native_rwkv_cumulative_virtual_kv_mechanics.v1"
SHARD_SCHEMA = f"{SCHEMA}.shard"
PROTOCOL = SCRIPT_DIR / (
    "natural_memory_native_rwkv_cumulative_virtual_kv_mechanics_protocol_v1.json"
)
LAUNCH_BINDING = SCRIPT_DIR / (
    "natural_memory_native_rwkv_cumulative_virtual_kv_mechanics_launch_v1.json"
)
CUMULATIVE_PROTOCOL = SCRIPT_DIR / (
    "natural_memory_native_rwkv_cumulative_compatibility_bias_protocol_v1.json"
)
CUMULATIVE_RESULT = SCRIPT_DIR / (
    "local_artifacts/natural_memory_native_rwkv_cumulative_compatibility_bias_v1/result.json"
)
DEFAULT_BASE_MODEL = Path("/root/X/.cache/hf/gemma-4-E4B-it-a4c2d58")
DEFAULT_MATERIALIZATION = SCRIPT_DIR / (
    "local_artifacts/natural_memory_native_rwkv_continuous_write_open_fit_v1"
)
DEFAULT_OUTPUT = SCRIPT_DIR / (
    "local_artifacts/natural_memory_native_rwkv_cumulative_virtual_kv_mechanics_v1"
)
RUNTIME_PARENT_COMMIT = "969a320a44149eab1a801a1413b10f5cd2db1618"
RUNTIME_PARENT_DELTA_IMPL_SHA256 = (
    "34921ecf14adb6ce6fb42bdb2513259ea66f194af118aed084cc66bcd4c05b7f"
)
RUNTIME_PARENT_VIRTUAL_KV_SHA256 = (
    "3f2a2049addfddc397c46c9c02deae59861d0a6a0a62b6dd304e7a5ccd52c889"
)
WORLD_SIZE = 4
ROWS = 32
ROWS_PER_RANK = 8
MODULES = 42
ANCHORS = (5, 11, 17, 23)
ADDRESS_DIM = 64
STATE_DIM = 32
SLOTS = 4
COMPATIBILITY_SCALE = 32.0
VALUE_PROBE_RANK = 8
VALUE_HIDDEN = 128
VALUE_SEED_BASE = 211
HF_ENDPOINT = "https://hf-mirror.com"
TIMEOUT_SECONDS = 1800
SLOT_PERMUTATION = (2, 0, 3, 1)
CONTROL_NAMES = (
    "correct_four_way",
    "joint_slot_permutation",
    "single_target",
    "matched_donor_state_only",
    "matched_donor_address_only",
    "matched_donor_address_and_state",
    "layer_rolled_state_only",
    "layer_rolled_address_only",
    "layer_rolled_address_and_state",
    "zero_state",
    "zero_address",
    "zero_state_and_address",
)
CONTROL_INDEX = {name: index for index, name in enumerate(CONTROL_NAMES)}
MATERIAL_LOGIT_DELTA = 1e-3
FULL_CACHED_DIAGNOSTIC_ATOL = 1e-4
CACHED_NULL_REPLAY_ATOL = 0.0
PERMUTATION_ATOL = 1e-5


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_sha256(value: Any) -> str:
    return distributed.canonical_sha256(value)


def signed_json(path: Path, value: Mapping[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(
        json.dumps(value, ensure_ascii=True, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def validate_receipt(
    value: Mapping[str, Any], *, scope: str, description: str
) -> None:
    unsigned = dict(value)
    receipt = unsigned.pop("receipt", None)
    expected = {
        "algorithm": "sha256",
        "payload_scope": scope,
        "payload_sha256": canonical_sha256(unsigned),
    }
    if receipt != expected:
        raise ValueError(f"{description} receipt differs")


def git(*arguments: str) -> str:
    return subprocess.check_output(
        ["git", *arguments], cwd=PROJECT_ROOT, text=True
    ).strip()


def dependency_paths() -> Mapping[str, Path]:
    return {
        "runner": Path(__file__).resolve(),
        "virtual_kv_runtime": PROJECT_ROOT / "deltamem/core/virtual_kv.py",
        "delta_mem_runtime": PROJECT_ROOT / "deltamem/core/delta_impl.py",
        "continuous_mechanics_helpers": Path(mechanics.__file__).resolve(),
        "continuous_retrieval_firewall": Path(retrieval.__file__).resolve(),
        "continuous_write_integration": Path(integration.__file__).resolve(),
        "distributed_runtime": Path(distributed.__file__).resolve(),
        "exact_adapter_loader": Path(exact_v5.__file__).resolve(),
        "native_runtime": Path(mechanics.evolution.__file__).resolve(),
        "materializer": Path(mechanics.materializer.__file__).resolve(),
    }


def dependency_bindings() -> Mapping[str, str]:
    return {
        role: sha256_file(path) for role, path in sorted(dependency_paths().items())
    }


def git_blob_sha256(commit: str, relative_path: str) -> str:
    payload = subprocess.check_output(
        ["git", "show", f"{commit}:{relative_path}"], cwd=PROJECT_ROOT
    )
    return hashlib.sha256(payload).hexdigest()


def validate_protocol_contract(protocol: Mapping[str, Any]) -> None:
    authorization = protocol.get("authorization_basis", {})
    frozen = protocol.get("frozen_inputs", {})
    architecture = protocol.get("architecture", {})
    builder = protocol.get("value_builder", {})
    controls = protocol.get("candidate_and_control_bank", {})
    gates = protocol.get("required_gates", {})
    identity = gates.get("identity_per_anchor", {})
    execution = protocol.get("execution", {})
    lifecycle = protocol.get("data_lifecycle", {})
    launch = protocol.get("launch_binding", {})
    expected_map_path = str(mechanics.MAP_FILE.relative_to(SCRIPT_DIR))
    if (
        protocol.get("schema")
        != "rwkv_ms_natural_memory_native_rwkv_cumulative_virtual_kv_mechanics_protocol.v1"
        or authorization.get("runtime_parent_commit") != RUNTIME_PARENT_COMMIT
        or authorization.get("runtime_parent_delta_impl_sha256")
        != RUNTIME_PARENT_DELTA_IMPL_SHA256
        or authorization.get("runtime_parent_virtual_kv_sha256")
        != RUNTIME_PARENT_VIRTUAL_KV_SHA256
        or authorization.get("continuous_retrieval_result_file_sha256")
        != mechanics.RETRIEVAL_RESULT_SHA256
        or authorization.get("continuous_retrieval_result_receipt")
        != mechanics.RETRIEVAL_RESULT_RECEIPT
        or authorization.get("continuous_write_manifest_file_sha256")
        != mechanics.MANIFEST_FILE_SHA256
        or authorization.get("continuous_write_manifest_receipt")
        != mechanics.MANIFEST_RECEIPT
        or authorization.get("continuous_write_split_schema")
        != "rwkv_ms_continuous_write_fit_split.v2"
        or authorization.get("continuous_write_split_receipt")
        != "d9ad640c208aae6983ce603f5d1918b06ab4ba9e93ed935d9cbfe1ac25f4801a"
        or frozen.get("base_model") != "google/gemma-4-E4B-it"
        or frozen.get("base_model_revision")
        != "a4c2d58be94dda072b918d9db64ee85c8ed34e3f"
        or frozen.get("adapter_weights_sha256")
        != exact_v5.V5_ADAPTER_WEIGHTS_SHA256
        or frozen.get("adapter_config_sha256")
        != exact_v5.V5_ADAPTER_CONFIG_SHA256
        or frozen.get("memory_backend") != "rwkv_ms"
        or frozen.get("hybrid_mode") != "address_keyed_moe_deepembed_ffn"
        or frozen.get("state_read_dim") != STATE_DIM
        or frozen.get("rwkv_memory_rank") != STATE_DIM
        or frozen.get("rwkv_num_states") != SLOTS
        or frozen.get("projected_kv_key_dim") != ADDRESS_DIM
        or frozen.get("frozen_map_file") != expected_map_path
        or frozen.get("frozen_map_file_sha256") != mechanics.MAP_FILE_SHA256
        or frozen.get("frozen_map_digest") != mechanics.MAP_DIGEST
        or frozen.get("map_rank") != mechanics.MAP_RANK
        or frozen.get("map_ridge") != mechanics.RIDGE
        or frozen.get("conditioning_seed") != mechanics.SEED
        or frozen.get("conditioning_k_gain") != mechanics.K_GAIN
        or frozen.get("conditioning_a_gain") != mechanics.A_GAIN
        or frozen.get("conditioning_b_gain") != mechanics.B_GAIN
        or frozen.get("conditioning_trainable_map") is not False
        or frozen.get("conditioning_features") != ["k", "a", "b"]
        or frozen.get("conditioning_value_feature_unchanged") is not True
        or frozen.get("model_or_map_training") is not False
        or architecture.get("anchor_layers") != list(ANCHORS)
        or architecture.get("anchor_types") != ["full_attention"] * len(ANCHORS)
        or architecture.get("compatibility_scale") != COMPATIBILITY_SCALE
        or architecture.get("compatibility_clamp_or_temperature") is not False
        or architecture.get("virtual_slots") != SLOTS
        or architecture.get("virtual_keys")
        != "exact all-zero tensors; identity enters only through the additive virtual-mask suffix"
        or architecture.get("attention_implementation") != "eager"
        or architecture.get("query_length") != 1
        or architecture.get("provider_attached_during_prefill") is not False
        or architecture.get("native_target_state_or_projected_carrier_replaced_by_controls")
        is not False
        or architecture.get("full_bandwidth_feedback_installed") is not False
        or builder.get("state_rank") != STATE_DIM
        or builder.get("state_heads") != 1
        or builder.get("probe_rank") != VALUE_PROBE_RANK
        or builder.get("hidden_width") != VALUE_HIDDEN
        or builder.get("kv_heads") != 2
        or builder.get("head_dim") != 512
        or builder.get("key_radius") != 1.0
        or builder.get("value_radius") != 1.0
        or builder.get("seed_by_layer")
        != {str(layer): VALUE_SEED_BASE + layer for layer in ANCHORS}
        or controls.get("candidate_order")
        != ["target", "matched_donor", "distractor_1", "distractor_2"]
        or controls.get("four_slot_conditions")
        != ["correct_four_way", "joint_slot_permutation"]
        or controls.get("joint_slot_permutation") != list(SLOT_PERMUTATION)
        or controls.get("single_active_slot_conditions")
        != list(CONTROL_NAMES[2:])
        or identity.get("strict_target_top1_fraction") != 0.75
        or identity.get("mean_target_over_strongest_wrong_margin") != 0.05
        or identity.get("matched_donor_positive_fraction") != 0.75
        or identity.get("live_layer_roll_positive_fraction") != 0.75
        or identity.get("nonzero_virtual_mass_fraction") != 0.95
        or gates.get("minimum_anchor_layers_passing") != 3
        or gates.get("material_predictor_change_fraction") != 0.95
        or gates.get("material_predictor_max_absolute_logit_delta")
        != MATERIAL_LOGIT_DELTA
        or gates.get("cached_null_replay_max_absolute_logit_tolerance")
        != CACHED_NULL_REPLAY_ATOL
        or gates.get("full_vs_cached_null_diagnostic_tolerance")
        != FULL_CACHED_DIAGNOSTIC_ATOL
        or gates.get("full_vs_cached_null_is_diagnostic_only") is not True
        or gates.get("joint_slot_permutation_final_logit_tolerance")
        != PERMUTATION_ATOL
        or execution.get("world_size") != WORLD_SIZE
        or execution.get("rows_per_rank") != ROWS_PER_RANK
        or execution.get("required_gpu_substring") != "A100"
        or execution.get("distinct_gpu_uuids") is not True
        or execution.get("hf_endpoint") != HF_ENDPOINT
        or execution.get("model_or_adapter_parameter_updates") != 0
        or execution.get("mechanics_evaluations") != 1
        or execution.get("protected_bundle_byte_opens") != 1
        or lifecycle.get("mechanics_bundle_byte_read_authorized_by_this_protocol")
        is not True
        or lifecycle.get("causal_bundle_byte_read_authorized") is not False
        or lifecycle.get("generation_or_native_benchmark_authorized") is not False
        or lifecycle.get("full_bandwidth_feedback_authorized") is not False
        or launch.get("path") != LAUNCH_BINDING.name
        or launch.get("schema")
        != "rwkv_ms_natural_memory_native_rwkv_cumulative_virtual_kv_mechanics_launch.v1"
        or launch.get("validated_before_protected_access") is not True
        or protocol.get("mechanics_bundle_byte_read_authorized_by_this_protocol")
        is not True
        or protocol.get("causal_bundle_byte_read_authorized") is not False
    ):
        raise ValueError("Cumulative virtual-KV mechanics protocol contract differs")


def validate_protocol(
    base_model: Path, *, validate_large_weights: bool = True
) -> Mapping[str, Any]:
    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    validate_receipt(
        protocol,
        scope="canonical_protocol_without_receipt",
        description="Cumulative virtual-KV mechanics protocol",
    )
    validate_protocol_contract(protocol)
    delta_path = PROJECT_ROOT / "deltamem/core/delta_impl.py"
    virtual_kv_path = PROJECT_ROOT / "deltamem/core/virtual_kv.py"
    if (
        sha256_file(delta_path) != RUNTIME_PARENT_DELTA_IMPL_SHA256
        or sha256_file(virtual_kv_path) != RUNTIME_PARENT_VIRTUAL_KV_SHA256
        or git_blob_sha256(RUNTIME_PARENT_COMMIT, "deltamem/core/delta_impl.py")
        != RUNTIME_PARENT_DELTA_IMPL_SHA256
        or git_blob_sha256(RUNTIME_PARENT_COMMIT, "deltamem/core/virtual_kv.py")
        != RUNTIME_PARENT_VIRTUAL_KV_SHA256
    ):
        raise ValueError("Parent-bound runtime source differs")
    frozen = protocol["frozen_inputs"]
    expected_files = {"config.json": frozen["base_config_sha256"]}
    if validate_large_weights:
        expected_files["model.safetensors"] = frozen["base_weights_sha256"]
    for name, expected in expected_files.items():
        if sha256_file(base_model / name) != expected:
            raise ValueError(f"Base-model file differs: {name}")
    adapter = mechanics.SCRIPT_DIR / frozen["adapter_directory"]
    if (
        sha256_file(adapter / "delta_mem_adapter.pt")
        != frozen["adapter_weights_sha256"]
        or sha256_file(adapter / "delta_mem_config.json")
        != frozen["adapter_config_sha256"]
        or sha256_file(CUMULATIVE_PROTOCOL)
        != protocol["authorization_basis"]["cumulative_protocol_file_sha256"]
        or sha256_file(CUMULATIVE_RESULT)
        != protocol["authorization_basis"]["cumulative_result_file_sha256"]
    ):
        raise ValueError("Frozen adapter or cumulative authorization differs")
    cumulative = json.loads(CUMULATIVE_RESULT.read_text(encoding="utf-8"))
    if (
        cumulative.get("receipt", {}).get("payload_sha256")
        != protocol["authorization_basis"]["cumulative_result_receipt"]
        or cumulative.get("passed") is not True
        or cumulative.get("status")
        != "cumulative_compatibility_bias_passed_live_mechanics_protocol_authorized"
    ):
        raise ValueError("Cumulative result does not authorize live mechanics")
    exact_v5.validate_protocol()
    return protocol


def validate_launch_contract(
    protocol: Mapping[str, Any],
    launch: Mapping[str, Any],
    *,
    head_parent: str,
    code_parent: str,
) -> None:
    code_commit = str(launch.get("authorized_code_commit", ""))
    if (
        launch.get("schema")
        != "rwkv_ms_natural_memory_native_rwkv_cumulative_virtual_kv_mechanics_launch.v1"
        or protocol.get("launch_binding", {}).get("path") != LAUNCH_BINDING.name
        or protocol.get("launch_binding", {}).get("schema") != launch.get("schema")
        or protocol.get("launch_binding", {}).get("validated_before_protected_access")
        is not True
        or not code_commit
        or head_parent != code_commit
        or code_parent != RUNTIME_PARENT_COMMIT
        or launch.get("runtime_parent_commit") != RUNTIME_PARENT_COMMIT
        or launch.get("runtime_parent_commit")
        != protocol["authorization_basis"]["runtime_parent_commit"]
        or launch.get("protocol_file_sha256") != sha256_file(PROTOCOL)
        or launch.get("protocol_receipt")
        != protocol["receipt"]["payload_sha256"]
        or launch.get("dependency_bindings") != dependency_bindings()
        or launch.get("dependency_digest")
        != canonical_sha256(dependency_bindings())
    ):
        raise ValueError("Cumulative virtual-KV mechanics launch binding differs")


def validate_launch_binding(protocol: Mapping[str, Any]) -> Mapping[str, Any]:
    launch = json.loads(LAUNCH_BINDING.read_text(encoding="utf-8"))
    validate_receipt(
        launch,
        scope="canonical_launch_without_receipt",
        description="Cumulative virtual-KV mechanics launch",
    )
    head = git("rev-parse", "HEAD")
    head_parent = git("rev-parse", "HEAD^")
    code_commit = str(launch.get("authorized_code_commit", ""))
    validate_launch_contract(
        protocol,
        launch,
        head_parent=head_parent,
        code_parent=git("rev-parse", f"{code_commit}^"),
    )
    changed = git("diff", "--name-only", code_commit, head).splitlines()
    launch_relative = str(LAUNCH_BINDING.relative_to(PROJECT_ROOT))
    if changed != [launch_relative]:
        raise RuntimeError("Launch commit must change only the launch binding")
    tracked_dirty = git("status", "--porcelain", "--untracked-files=no")
    if tracked_dirty:
        raise RuntimeError("Tracked worktree must be clean before mechanics access")
    for path in (Path(__file__).resolve(), PROTOCOL):
        relative = str(path.relative_to(PROJECT_ROOT))
        committed = subprocess.check_output(
            ["git", "show", f"{code_commit}:{relative}"], cwd=PROJECT_ROOT
        )
        if committed != path.read_bytes():
            raise RuntimeError(f"Authorized code object differs: {relative}")
    committed_launch = subprocess.check_output(
        ["git", "show", f"HEAD:{launch_relative}"], cwd=PROJECT_ROOT
    )
    if committed_launch != LAUNCH_BINDING.read_bytes():
        raise RuntimeError("Launch binding differs from the committed HEAD object")
    return {**launch, "launch_head": head}


def tensor_digest(tensor: torch.Tensor) -> str:
    value = tensor.detach().cpu().float().contiguous()
    digest = hashlib.sha256()
    digest.update(str(tuple(value.shape)).encode("ascii"))
    digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def builder_digest(
    builders: Mapping[int, ExplicitRWKVVirtualKV],
) -> tuple[str, Mapping[str, str]]:
    aggregate = hashlib.sha256()
    per_layer: dict[str, str] = {}
    for layer in ANCHORS:
        local = hashlib.sha256()
        for name, tensor in sorted(builders[layer].state_dict().items()):
            value = tensor.detach().cpu().float().contiguous()
            payloads = (
                name.encode("utf-8"),
                str(tuple(value.shape)).encode("ascii"),
                value.numpy().tobytes(),
            )
            for payload in payloads:
                local.update(payload)
                aggregate.update(str(layer).encode("ascii") + b"\0" + payload)
        per_layer[str(layer)] = local.hexdigest()
    return aggregate.hexdigest(), per_layer


def make_router(
    modules_by_layer: Mapping[int, Any],
    maps: Mapping[str, Any],
    names_by_layer: Mapping[int, str],
) -> tuple[CumulativeRWKVCompatibilityRouter, Mapping[str, Any]]:
    builders = {}
    for layer in ANCHORS:
        module = modules_by_layer[layer]
        builder = ExplicitRWKVVirtualKV(
            VirtualKVShape(
                key_dim=ADDRESS_DIM,
                state_heads=1,
                rank=STATE_DIM,
                slots=SLOTS,
                kv_heads=int(module.num_key_value_heads),
                head_dim=int(module.head_dim),
                probe_rank=VALUE_PROBE_RANK,
                value_hidden=VALUE_HIDDEN,
                seed=VALUE_SEED_BASE + layer,
            )
        ).to(next(module.parameters()).device)
        builder.eval()
        for parameter in builder.parameters():
            parameter.requires_grad_(False)
        builders[layer] = builder
    aggregate, per_layer = builder_digest(builders)
    router = CumulativeRWKVCompatibilityRouter(
        builders=builders,
        maps={layer: maps[names_by_layer[layer]] for layer in ANCHORS},
        anchor_layers=ANCHORS,
        compatibility_scale=COMPATIBILITY_SCALE,
        required_receptance_calls=2,
    ).to(next(modules_by_layer[ANCHORS[0]].parameters()).device)
    router.eval()
    return router, {"aggregate_sha256": aggregate, "per_layer_sha256": per_layer}


def candidate_sources(rows: Sequence[Mapping[str, Any]]) -> Mapping[int, tuple[int, ...]]:
    by_source = {int(row["source_index"]): row for row in rows}
    pairs = sorted(
        {
            tuple(sorted((source, int(row["donor_source_index"]))))
            for source, row in by_source.items()
        }
    )
    result = {}
    for source, row in by_source.items():
        donor = int(row["donor_source_index"])
        own = tuple(sorted((source, donor)))
        distractors = [pair[0] for pair in pairs if pair != own]
        if len(distractors) < 2:
            raise ValueError("Cumulative mechanics distractor pool is too small")
        result[source] = (source, donor, distractors[0], distractors[1])
    return result


def active_slot(
    values: torch.Tensor, occupied: torch.Tensor, *, slot_axis: int
) -> torch.Tensor:
    indices = occupied[0].nonzero(as_tuple=False).flatten()
    if indices.numel() != 1:
        raise ValueError("Cumulative mechanics source must have one occupied slot")
    return values.select(slot_axis, int(indices[0].item()))


def source_state_component(
    natural_cache: Mapping[int, Any],
    source: int,
    name: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    state = natural_cache[source]["state"][name]
    occupied = state["projected_kv_occupied"]
    recurrent = active_slot(state["delta_state"], occupied, slot_axis=2)[0]
    address = active_slot(state["projected_kv_keys"], occupied, slot_axis=1)[0]
    if tuple(recurrent.shape) != (1, STATE_DIM, STATE_DIM):
        raise ValueError("Cumulative mechanics recurrent component geometry differs")
    if tuple(address.shape) != (ADDRESS_DIM,):
        raise ValueError("Cumulative mechanics address component geometry differs")
    return recurrent, address


def control_banks(
    natural_cache: Mapping[int, Any],
    sources: Sequence[int],
    names_by_layer: Mapping[int, str],
    ordered_names: Sequence[str],
    device: torch.device,
) -> tuple[
    Mapping[int, torch.Tensor],
    Mapping[int, torch.Tensor],
    Mapping[int, torch.Tensor],
    Mapping[int, torch.Tensor],
]:
    batch_size = len(CONTROL_NAMES)
    states = {
        layer: torch.zeros(
            batch_size, 1, SLOTS, STATE_DIM, STATE_DIM, device=device
        )
        for layer in ANCHORS
    }
    addresses = {
        layer: torch.zeros(batch_size, SLOTS, ADDRESS_DIM, device=device)
        for layer in ANCHORS
    }
    occupied = {
        layer: torch.zeros(batch_size, SLOTS, dtype=torch.bool, device=device)
        for layer in ANCHORS
    }
    source_ids = {
        layer: torch.tensor(sources, dtype=torch.long, device=device)
        .unsqueeze(0)
        .expand(batch_size, -1)
        .clone()
        for layer in ANCHORS
    }
    permutation = torch.tensor(SLOT_PERMUTATION, dtype=torch.long)
    for layer in ANCHORS:
        name = names_by_layer[layer]
        rolled_name = ordered_names[(layer - 1) % len(ordered_names)]
        source_components = [
            source_state_component(natural_cache, int(source), name)
            for source in sources
        ]
        rolled_components = [
            source_state_component(natural_cache, int(source), rolled_name)
            for source in sources
        ]
        correct_state = torch.stack([item[0] for item in source_components], dim=1).to(device)
        correct_address = torch.stack([item[1] for item in source_components], dim=0).to(device)
        rolled_state = torch.stack([item[0] for item in rolled_components], dim=1).to(device)
        rolled_address = torch.stack([item[1] for item in rolled_components], dim=0).to(device)
        states[layer][CONTROL_INDEX["correct_four_way"]] = correct_state
        addresses[layer][CONTROL_INDEX["correct_four_way"]] = correct_address
        occupied[layer][CONTROL_INDEX["correct_four_way"]] = True
        states[layer][CONTROL_INDEX["joint_slot_permutation"]] = correct_state.index_select(
            1, permutation.to(device)
        )
        addresses[layer][CONTROL_INDEX["joint_slot_permutation"]] = (
            correct_address.index_select(0, permutation.to(device))
        )
        occupied[layer][CONTROL_INDEX["joint_slot_permutation"]] = True
        source_ids[layer][CONTROL_INDEX["joint_slot_permutation"]] = source_ids[
            layer
        ][CONTROL_INDEX["joint_slot_permutation"]].index_select(
            0, permutation.to(device)
        )

        target_state, target_address = source_components[0]
        donor_state, donor_address = source_components[1]
        rolled_target_state, rolled_target_address = rolled_components[0]
        isolated = {
            "single_target": (target_state, target_address),
            "matched_donor_state_only": (donor_state, target_address),
            "matched_donor_address_only": (target_state, donor_address),
            "matched_donor_address_and_state": (donor_state, donor_address),
            "layer_rolled_state_only": (rolled_target_state, target_address),
            "layer_rolled_address_only": (target_state, rolled_target_address),
            "layer_rolled_address_and_state": (
                rolled_target_state,
                rolled_target_address,
            ),
            "zero_state": (torch.zeros_like(target_state), target_address),
            "zero_address": (target_state, torch.zeros_like(target_address)),
            "zero_state_and_address": (
                torch.zeros_like(target_state),
                torch.zeros_like(target_address),
            ),
        }
        for control, (control_state, control_address) in isolated.items():
            row = CONTROL_INDEX[control]
            states[layer][row, :, 0] = control_state.to(device)
            addresses[layer][row, 0] = control_address.to(device)
            occupied[layer][row, 0] = True
    return states, addresses, occupied, source_ids


def repeated_online_state(
    source_state: Mapping[str, Mapping[str, torch.Tensor]],
    batch_size: int,
    modules: Sequence[tuple[str, Any]],
) -> tuple[Mapping[str, Any], Mapping[str, Any]]:
    projected: dict[str, dict[str, torch.Tensor]] = {}
    recurrent: dict[str, dict[str, torch.Tensor]] = {}
    for name, module in modules:
        device = next(module.parameters()).device
        values = source_state[name]
        projected[name] = {
            attribute: values[attribute].to(device).repeat(
                batch_size, *([1] * (values[attribute].ndim - 1))
            )
            for attribute in mechanics.PROJECTED_ATTRIBUTES
        }
        recurrent[name] = {
            attribute: values[attribute].to(device).repeat(
                batch_size, *([1] * (values[attribute].ndim - 1))
            )
            for attribute in mechanics.RECURRENT_ATTRIBUTES
        }
    return projected, recurrent


def install_target_state(
    model: torch.nn.Module,
    modules: Sequence[tuple[str, Any]],
    source_state: Mapping[str, Mapping[str, torch.Tensor]],
    batch_size: int,
) -> tuple[Mapping[str, Any], Mapping[str, Any]]:
    reset_delta_mem_states(model)
    projected, recurrent = repeated_online_state(source_state, batch_size, modules)
    fixed = mechanics.causal_train.install_intervened_state(
        modules,
        projected=projected,
        recurrent=recurrent,
        rotate_recurrent_layers=False,
    )
    if not fixed or not mechanics._module_references_exact(
        modules, projected, recurrent
    ):
        raise RuntimeError("Cumulative mechanics target state installation failed")
    set_delta_mem_write_enabled(model, False)
    set_delta_mem_projected_kv_read_query_mask(model, None)
    return projected, recurrent


def cache_snapshot(cache: Any) -> Mapping[int, tuple[torch.Tensor, torch.Tensor]]:
    snapshots = {}
    for layer in ANCHORS:
        cache_layer = cache.layers[layer]
        snapshots[layer] = (
            cache_layer.keys.detach().clone(),
            cache_layer.values.detach().clone(),
        )
    return snapshots


def cache_audit(
    cache: Any,
    before: Mapping[int, tuple[torch.Tensor, torch.Tensor]],
    *,
    prefix_length: int,
) -> Mapping[str, Any]:
    lengths = {str(layer): int(cache.get_seq_length(layer)) for layer in ANCHORS}
    prefix_exact = {}
    for layer in ANCHORS:
        keys, values = before[layer]
        after_layer = cache.layers[layer]
        prefix_exact[str(layer)] = bool(
            torch.equal(after_layer.keys[..., :prefix_length, :], keys)
            and torch.equal(after_layer.values[..., :prefix_length, :], values)
        )
    return {
        "prefill_lengths": {str(layer): prefix_length for layer in ANCHORS},
        "post_q1_lengths": lengths,
        "prefix_cache_bytes_unchanged": prefix_exact,
        "one_real_position_appended_no_virtual_slots": all(
            value == prefix_length + 1 for value in lengths.values()
        ),
    }


def independent_router_equation(
    *,
    compatibility_map: Any,
    receptance: torch.Tensor,
    state: torch.Tensor,
    addresses: torch.Tensor,
    occupied: torch.Tensor,
    running: dict[str, Any],
    compatibility_scale: float,
) -> Mapping[str, torch.Tensor | int]:
    flattened_receptance = receptance[:, 0].float().reshape(receptance.size(0), -1)

    def rms_normalize(value: torch.Tensor) -> torch.Tensor:
        square_mean = value.float().square().mean(dim=-1, keepdim=True)
        normalized = value.float() / square_mean.clamp_min(1e-12).sqrt()
        return torch.where(square_mean.gt(0.0), normalized, torch.zeros_like(normalized))

    normalized_addresses = rms_normalize(addresses)
    latent = F.linear(
        normalized_addresses,
        compatibility_map.down.to(device=addresses.device),
    )
    mapped_addresses = rms_normalize(
        F.linear(latent, compatibility_map.up.to(device=addresses.device))
    )
    local_scores = (
        mapped_addresses * rms_normalize(flattened_receptance).unsqueeze(1)
    ).mean(dim=-1)
    local_active = (
        occupied
        & state.float().square().sum(dim=(-1, -2)).sum(dim=1).gt(0.0)
        & addresses.float().square().sum(dim=-1).gt(0.0)
    )
    count = int(running["count"]) + 1
    score_sum = running["score_sum"] + local_scores
    active = running["active"] & local_active
    accumulated_scores = score_sum / float(count)
    attention_bias = float(compatibility_scale) * accumulated_scores
    running["count"] = count
    running["score_sum"] = score_sum
    running["active"] = active
    return {
        "count": count,
        "local_scores": local_scores,
        "accumulated_scores": accumulated_scores,
        "attention_bias": attention_bias,
        "active": active,
    }


def independent_equation_checks(
    diagnostic: Mapping[str, Any], expected: Mapping[str, Any]
) -> Mapping[str, bool]:
    return {
        "count_exact": int(diagnostic["count"]) == int(expected["count"]),
        "local_scores_exact": bool(
            torch.equal(diagnostic["local_scores"], expected["local_scores"])
        ),
        "accumulated_scores_exact": bool(
            torch.equal(
                diagnostic["accumulated_scores"],
                expected["accumulated_scores"],
            )
        ),
        "attention_bias_exact": bool(
            torch.equal(diagnostic["attention_bias"], expected["attention_bias"])
        ),
        "active_exact": bool(torch.equal(diagnostic["active"], expected["active"])),
    }


def provider_observer(
    router: CumulativeRWKVCompatibilityRouter,
    layer: int,
    observations: dict[int, Mapping[str, Any]],
    banks: tuple[Mapping[int, torch.Tensor], ...],
    compatibility_maps: Mapping[int, Any],
    independent_running: dict[str, Any],
):
    base_provider = router.provider_for(layer)

    def provider(**kwargs):
        independent = independent_router_equation(
            compatibility_map=compatibility_maps[layer],
            receptance=kwargs["module"].rwkv_virtual_router_receptance,
            state=banks[0][layer],
            addresses=banks[1][layer],
            occupied=banks[2][layer],
            running=independent_running,
            compatibility_scale=COMPATIBILITY_SCALE,
        )
        result = base_provider(**kwargs)
        diagnostic = router.diagnostics[-1]
        active = diagnostic["active"]
        equation_checks = independent_equation_checks(diagnostic, independent)
        observation: dict[str, Any] = {
            "active": active.detach().cpu(),
            "attention_bias": diagnostic["attention_bias"].detach().cpu(),
            "local_scores": diagnostic["local_scores"].detach().cpu(),
            "accumulated_scores": diagnostic["accumulated_scores"].detach().cpu(),
            "source_ids": diagnostic["source_ids"].detach().cpu(),
            "receptance": kwargs["module"].rwkv_virtual_router_receptance.detach().cpu(),
            "receptance_calls": int(
                kwargs["module"].rwkv_virtual_router_receptance_calls
            ),
            "disabled": result is None,
            "independent_count": int(independent["count"]),
            "independent_local_scores": independent["local_scores"].detach().cpu(),
            "independent_accumulated_scores": independent[
                "accumulated_scores"
            ].detach().cpu(),
            "independent_attention_bias": independent["attention_bias"].detach().cpu(),
            "independent_active": independent["active"].detach().cpu(),
            "independent_equation_checks": equation_checks,
        }
        if result is not None:
            virtual_keys, virtual_values, mask = result
            real_keys = kwargs["key_states"]
            query = kwargs["query_states"]
            groups = int(kwargs["module"].num_key_value_groups)
            combined_keys = torch.cat((real_keys, virtual_keys), dim=2)
            expanded_keys = combined_keys.repeat_interleave(groups, dim=1)
            logits = torch.einsum("bhqd,bhkd->bhqk", query.float(), expanded_keys.float())
            logits = logits * float(kwargs["module"].scaling)
            if mask.dtype == torch.bool:
                logits = logits.masked_fill(~mask, -torch.inf)
            else:
                logits = logits + mask.float()
            real_length = real_keys.size(2)
            mass = torch.softmax(logits, dim=-1)[..., real_length:].sum(dim=-1)
            suffix = mask[..., real_length:]
            expected_suffix = torch.where(
                independent["active"][:, None, None, :].to(suffix.device),
                independent["attention_bias"][:, None, None, :].to(suffix),
                torch.full((), torch.finfo(suffix.dtype).min, device=suffix.device),
            )
            input_mask = kwargs["attention_mask"]
            if input_mask is None:
                real_mask_exact = bool(mask[..., :real_length].eq(0.0).all().item())
            else:
                real_mask_exact = bool(
                    mask[..., :real_length].dtype == input_mask.dtype
                    and torch.equal(mask[..., :real_length], input_mask)
                )
            observation.update(
                {
                    "virtual_keys_exact_zero": bool(
                        torch.equal(virtual_keys, torch.zeros_like(virtual_keys))
                    ),
                    "virtual_values": virtual_values.detach().cpu(),
                    "suffix_bias_exact": bool(torch.equal(suffix, expected_suffix)),
                    "real_mask_prefix_exact": real_mask_exact,
                    "virtual_attention_mass": mass.detach().cpu(),
                    "real_cache_length": real_length,
                    "attention_width_with_virtual": real_length + SLOTS,
                }
            )
        observations[layer] = observation
        return result

    return provider


def clear_providers(modules_by_layer: Mapping[int, Any]) -> None:
    for layer in ANCHORS:
        modules_by_layer[layer].clear_virtual_kv_provider()


@torch.no_grad()
def predictor_pass(
    model: torch.nn.Module,
    batch: Any,
    modules: Sequence[tuple[str, Any]],
    modules_by_layer: Mapping[int, Any],
    target_state: Mapping[str, Mapping[str, torch.Tensor]],
    *,
    batch_size: int,
    router: CumulativeRWKVCompatibilityRouter | None,
    banks: tuple[Mapping[int, torch.Tensor], ...] | None,
    compatibility_maps: Mapping[int, Any] | None = None,
) -> Mapping[str, Any]:
    first_label, predictor = retrieval.first_prompt_boundary(batch.labels)
    if predictor < 1:
        raise ValueError("Cumulative mechanics predictor requires a nonempty prefill")
    projected, recurrent = install_target_state(
        model, modules, target_state, batch_size
    )
    projected_before = mechanics.state_sha256(
        projected, mechanics.PROJECTED_ATTRIBUTES
    )
    recurrent_before = mechanics.state_sha256(
        recurrent, mechanics.RECURRENT_ATTRIBUTES
    )
    input_ids = batch.read_input_ids.repeat(batch_size, 1)
    attention_mask = batch.read_attention_mask.repeat(batch_size, 1)
    prefix_ids = input_ids[:, :predictor]
    prefix_mask = attention_mask[:, :predictor]
    prefix_positions = torch.arange(
        predictor, device=input_ids.device, dtype=torch.long
    ).unsqueeze(0).expand(batch_size, -1)
    query_positions = torch.full(
        (batch_size, 1), predictor, device=input_ids.device, dtype=torch.long
    )
    clear_providers(modules_by_layer)
    observations: dict[int, Mapping[str, Any]] = {}
    diagnostics: tuple[Mapping[str, Any], ...] = ()
    try:
        with mechanics.evolution.runtime._autocast_context(
            input_ids.device, torch.bfloat16
        ):
            prefill = model(
                input_ids=prefix_ids,
                attention_mask=prefix_mask,
                position_ids=prefix_positions,
                use_cache=True,
                return_dict=True,
                logits_to_keep=1,
            )
        cache = prefill.past_key_values
        if any(int(cache.get_seq_length(layer)) != predictor for layer in ANCHORS):
            raise RuntimeError("Cumulative mechanics prefill cache length differs")
        before_cache = cache_snapshot(cache)
        if router is not None:
            if banks is None or compatibility_maps is None:
                raise ValueError(
                    "Routed cumulative mechanics pass requires banks and frozen maps"
                )
            independent_running = {
                "count": 0,
                "score_sum": torch.zeros_like(banks[2][ANCHORS[0]], dtype=torch.float32),
                "active": banks[2][ANCHORS[0]].detach().clone(),
            }
            router.begin_forward(
                states=banks[0],
                address_keys=banks[1],
                occupied=banks[2],
                source_ids=banks[3],
            )
            for layer in ANCHORS:
                modules_by_layer[layer].set_virtual_kv_provider(
                    provider_observer(
                        router,
                        layer,
                        observations,
                        banks,
                        compatibility_maps,
                        independent_running,
                    )
                )
        with mechanics.evolution.runtime._autocast_context(
            input_ids.device, torch.bfloat16
        ):
            output = model(
                input_ids=input_ids[:, predictor : predictor + 1],
                attention_mask=attention_mask[:, : predictor + 1],
                position_ids=query_positions,
                past_key_values=cache,
                use_cache=True,
                return_dict=True,
                logits_to_keep=1,
            )
        if router is not None:
            diagnostics = router.end_forward()
            if tuple(item["layer"] for item in diagnostics) != ANCHORS:
                raise RuntimeError("Cumulative mechanics router lifecycle differs")
        audit = cache_audit(cache, before_cache, prefix_length=predictor)
    finally:
        clear_providers(modules_by_layer)
        if router is not None and (router.active or router.completed):
            router.abort_forward()
    projected_after = mechanics.state_sha256(
        mechanics.clone_online_state_cpu(modules), mechanics.PROJECTED_ATTRIBUTES
    )
    recurrent_after = mechanics.state_sha256(
        mechanics.clone_online_state_cpu(modules), mechanics.RECURRENT_ATTRIBUTES
    )
    audit = {
        **audit,
        "projected_carrier_bytes_unchanged": projected_before == projected_after,
        "rwkv_state_bytes_unchanged": recurrent_before == recurrent_after,
        "first_label": first_label,
        "predictor": predictor,
    }
    return {
        "logits": output.logits[:, -1].detach().cpu().float(),
        "diagnostics": diagnostics,
        "observations": observations,
        "audit": audit,
    }


@torch.no_grad()
def full_null_predictor(
    model: torch.nn.Module,
    batch: Any,
    modules: Sequence[tuple[str, Any]],
    modules_by_layer: Mapping[int, Any],
    target_state: Mapping[str, Mapping[str, torch.Tensor]],
) -> torch.Tensor:
    _, predictor = retrieval.first_prompt_boundary(batch.labels)
    install_target_state(model, modules, target_state, 1)
    clear_providers(modules_by_layer)
    positions = torch.arange(
        predictor + 1, device=batch.read_input_ids.device, dtype=torch.long
    ).unsqueeze(0)
    with mechanics.evolution.runtime._autocast_context(
        batch.read_input_ids.device, torch.bfloat16
    ):
        output = model(
            input_ids=batch.read_input_ids[:, : predictor + 1],
            attention_mask=batch.read_attention_mask[:, : predictor + 1],
            position_ids=positions,
            use_cache=False,
            return_dict=True,
            logits_to_keep=1,
        )
    return output.logits[:, -1].detach().cpu().float()


def logit_comparison(left: torch.Tensor, right: torch.Tensor) -> Mapping[str, Any]:
    if tuple(left.shape) != tuple(right.shape):
        raise ValueError("Cumulative mechanics logit shapes differ")
    difference = left.float() - right.float()
    return {
        "byte_exact": bool(torch.equal(left, right)),
        "maximum_absolute_delta": float(difference.abs().max().item()),
        "normalized_l2": float(
            (difference.norm() / right.float().norm().clamp_min(1e-12)).item()
        ),
        "material": bool(difference.abs().max().item() >= MATERIAL_LOGIT_DELTA),
    }


def row_result(
    source: int,
    sources: Sequence[int],
    routed: Mapping[str, Any],
    baseline: Mapping[str, Any],
    baseline_replay: Mapping[str, Any],
    full_null: torch.Tensor,
) -> Mapping[str, Any]:
    routed_logits = routed["logits"]
    baseline_logits = baseline["logits"]
    baseline_replay_logits = baseline_replay["logits"]
    observations = routed["observations"]
    identity = {}
    path_checks = {}
    correct_index = CONTROL_INDEX["correct_four_way"]
    permuted_index = CONTROL_INDEX["joint_slot_permutation"]
    single_index = CONTROL_INDEX["single_target"]
    donor_state_index = CONTROL_INDEX["matched_donor_state_only"]
    donor_address_index = CONTROL_INDEX["matched_donor_address_only"]
    layer_state_index = CONTROL_INDEX["layer_rolled_state_only"]
    layer_address_index = CONTROL_INDEX["layer_rolled_address_only"]
    zero_indices = [
        CONTROL_INDEX["zero_state"],
        CONTROL_INDEX["zero_address"],
        CONTROL_INDEX["zero_state_and_address"],
    ]
    permutation = torch.tensor(SLOT_PERMUTATION, dtype=torch.long)
    for layer in ANCHORS:
        observation = observations[layer]
        scores = observation["accumulated_scores"][correct_index]
        correct = scores[0]
        strongest_wrong = scores[1:].max()
        rolled_score = observation["accumulated_scores"][layer_address_index, 0]
        identity[str(layer)] = {
            "scores": [float(value) for value in scores],
            "strict_target_top1": bool(correct > strongest_wrong),
            "target_over_strongest_wrong_margin": float(correct - strongest_wrong),
            "target_over_matched_donor_margin": float(correct - scores[1]),
            "target_over_live_layer_roll_margin": float(correct - rolled_score),
            "correct_virtual_mass": float(
                observation["virtual_attention_mass"][correct_index].mean().item()
            ),
        }
        correct_values = observation["virtual_values"][correct_index]
        permuted_values = observation["virtual_values"][permuted_index]
        single_values = observation["virtual_values"][single_index, :, 0]
        donor_state_values = observation["virtual_values"][donor_state_index, :, 0]
        donor_address_values = observation["virtual_values"][donor_address_index, :, 0]
        path_checks[str(layer)] = {
            "virtual_keys_exact_zero": observation["virtual_keys_exact_zero"],
            "suffix_bias_exact": observation["suffix_bias_exact"],
            "real_mask_prefix_exact": observation["real_mask_prefix_exact"],
            "receptance_calls_exactly_two": observation["receptance_calls"] == 2,
            "independent_router_equation_exact": all(
                observation["independent_equation_checks"].values()
            ),
            "candidate_permutation_bias_close": bool(
                torch.allclose(
                    observation["attention_bias"][permuted_index],
                    observation["attention_bias"][correct_index].index_select(
                        0, permutation
                    ),
                    atol=1e-6,
                    rtol=1e-6,
                )
            ),
            "candidate_permutation_values_close": bool(
                torch.allclose(
                    permuted_values,
                    correct_values.index_select(1, permutation),
                    atol=1e-6,
                    rtol=1e-6,
                )
            ),
            "address_only_values_exact": bool(
                torch.equal(single_values, donor_address_values)
            ),
            "state_only_values_changed": bool(
                not torch.equal(single_values, donor_state_values)
            ),
            "state_only_layer5_bias_exact": (
                True
                if layer != ANCHORS[0]
                else bool(
                    torch.equal(
                        observation["attention_bias"][single_index],
                        observation["attention_bias"][donor_state_index],
                    )
                )
            ),
            "address_only_bias_changed": bool(
                not torch.equal(
                    observation["attention_bias"][single_index],
                    observation["attention_bias"][donor_address_index],
                )
            ),
            "zero_state_inactive": bool(
                not observation["active"][CONTROL_INDEX["zero_state"]].any()
            ),
            "zero_address_inactive_and_zero_bias": bool(
                not observation["active"][CONTROL_INDEX["zero_address"]].any()
                and observation["attention_bias"][
                    CONTROL_INDEX["zero_address"]
                ].eq(0.0).all()
            ),
            "zero_both_inactive_and_zero_bias": bool(
                not observation["active"][
                    CONTROL_INDEX["zero_state_and_address"]
                ].any()
                and observation["attention_bias"][
                    CONTROL_INDEX["zero_state_and_address"]
                ].eq(0.0).all()
            ),
        }
    comparisons = {
        "correct_vs_provider_off": logit_comparison(
            routed_logits[correct_index], baseline_logits[correct_index]
        ),
        "donor_state_only_vs_single_target": logit_comparison(
            routed_logits[donor_state_index], routed_logits[single_index]
        ),
        "donor_address_only_vs_single_target": logit_comparison(
            routed_logits[donor_address_index], routed_logits[single_index]
        ),
        "donor_both_vs_single_target": logit_comparison(
            routed_logits[CONTROL_INDEX["matched_donor_address_and_state"]],
            routed_logits[single_index],
        ),
        "layer_state_only_vs_single_target": logit_comparison(
            routed_logits[layer_state_index], routed_logits[single_index]
        ),
        "layer_address_only_vs_single_target": logit_comparison(
            routed_logits[layer_address_index], routed_logits[single_index]
        ),
        "layer_both_vs_single_target": logit_comparison(
            routed_logits[CONTROL_INDEX["layer_rolled_address_and_state"]],
            routed_logits[single_index],
        ),
        "joint_slot_permutation_vs_correct": logit_comparison(
            routed_logits[permuted_index], routed_logits[correct_index]
        ),
        "full_null_vs_cached_null": logit_comparison(
            full_null[0], baseline_logits[0]
        ),
        "cached_null_replay_vs_cached_null": logit_comparison(
            baseline_replay_logits, baseline_logits
        ),
    }
    zero_parity = {
        CONTROL_NAMES[index]: bool(
            torch.equal(routed_logits[index], baseline_logits[index])
        )
        for index in zero_indices
    }
    all_path_checks = all(
        value is True
        for layer_checks in path_checks.values()
        for value in layer_checks.values()
    )
    audit = {
        "routed": dict(routed["audit"]),
        "provider_off": dict(baseline["audit"]),
        "provider_off_replay": dict(baseline_replay["audit"]),
        "candidate_source_order": list(sources),
        "router_path_checks": path_checks,
        "all_router_path_checks": all_path_checks,
        "zero_controls_byte_exact_provider_off": zero_parity,
        "joint_slot_permutation_final_logits_close": (
            comparisons["joint_slot_permutation_vs_correct"][
                "maximum_absolute_delta"
            ]
            <= PERMUTATION_ATOL
        ),
        "cached_null_replay_close": (
            comparisons["cached_null_replay_vs_cached_null"][
                "maximum_absolute_delta"
            ]
            <= CACHED_NULL_REPLAY_ATOL
        ),
        "full_cached_null_diagnostic_close": (
            comparisons["full_null_vs_cached_null"]["maximum_absolute_delta"]
            <= FULL_CACHED_DIAGNOSTIC_ATOL
        ),
    }
    return {
        "source_index": source,
        "donor_source_index": int(sources[1]),
        "candidate_sources": list(sources),
        "identity": identity,
        "comparisons": comparisons,
        "audit": audit,
        "routed_logits_sha256": tensor_digest(routed_logits),
        "provider_off_logits_sha256": tensor_digest(baseline_logits),
        "provider_off_replay_logits_sha256": tensor_digest(
            baseline_replay_logits
        ),
    }


def aggregate(rows: Sequence[Mapping[str, Any]], protocol: Mapping[str, Any]) -> Mapping[str, Any]:
    per_anchor = {}
    anchor_passes = {}
    for layer in ANCHORS:
        metrics = [row["identity"][str(layer)] for row in rows]
        top1 = sum(value["strict_target_top1"] for value in metrics) / len(metrics)
        margins = [value["target_over_strongest_wrong_margin"] for value in metrics]
        donor = [value["target_over_matched_donor_margin"] for value in metrics]
        rolled = [value["target_over_live_layer_roll_margin"] for value in metrics]
        mass = [value["correct_virtual_mass"] for value in metrics]
        per_anchor[str(layer)] = {
            "strict_target_top1_fraction": top1,
            "mean_target_over_strongest_wrong_margin": sum(margins) / len(margins),
            "matched_donor_positive_fraction": sum(value > 0 for value in donor)
            / len(donor),
            "live_layer_roll_positive_fraction": sum(value > 0 for value in rolled)
            / len(rolled),
            "nonzero_virtual_mass_fraction": sum(value > 0 for value in mass)
            / len(mass),
            "mean_virtual_mass": sum(mass) / len(mass),
        }
        gate = protocol["required_gates"]["identity_per_anchor"]
        anchor_passes[str(layer)] = (
            top1 >= gate["strict_target_top1_fraction"]
            and per_anchor[str(layer)]["mean_target_over_strongest_wrong_margin"]
            >= gate["mean_target_over_strongest_wrong_margin"]
            and per_anchor[str(layer)]["matched_donor_positive_fraction"]
            >= gate["matched_donor_positive_fraction"]
            and per_anchor[str(layer)]["live_layer_roll_positive_fraction"]
            >= gate["live_layer_roll_positive_fraction"]
            and per_anchor[str(layer)]["nonzero_virtual_mass_fraction"]
            >= gate["nonzero_virtual_mass_fraction"]
        )
    material_names = (
        "correct_vs_provider_off",
        "donor_state_only_vs_single_target",
        "donor_address_only_vs_single_target",
        "donor_both_vs_single_target",
        "layer_state_only_vs_single_target",
        "layer_address_only_vs_single_target",
        "layer_both_vs_single_target",
    )
    material = {
        name: sum(row["comparisons"][name]["material"] for row in rows) / len(rows)
        for name in material_names
    }
    invariants = {
        "all_router_path_checks": all(
            row["audit"]["all_router_path_checks"] for row in rows
        ),
        "all_zero_controls_byte_exact_provider_off": all(
            all(row["audit"]["zero_controls_byte_exact_provider_off"].values())
            for row in rows
        ),
        "all_joint_permutation_logits_close": all(
            row["audit"]["joint_slot_permutation_final_logits_close"] for row in rows
        ),
        "all_cached_null_replays_close": all(
            row["audit"]["cached_null_replay_close"] for row in rows
        ),
        "all_cache_state_invariants": all(
            all(
                branch["one_real_position_appended_no_virtual_slots"]
                and all(branch["prefix_cache_bytes_unchanged"].values())
                and branch["projected_carrier_bytes_unchanged"]
                and branch["rwkv_state_bytes_unchanged"]
                for branch in (
                    row["audit"]["routed"],
                    row["audit"]["provider_off"],
                    row["audit"]["provider_off_replay"],
                )
            )
            for row in rows
        ),
    }
    diagnostics = {
        "all_full_cached_null_close_at_diagnostic_tolerance": all(
            row["audit"]["full_cached_null_diagnostic_close"] for row in rows
        ),
        "maximum_full_cached_null_logit_delta": max(
            row["comparisons"]["full_null_vs_cached_null"][
                "maximum_absolute_delta"
            ]
            for row in rows
        ),
    }
    material_gate = protocol["required_gates"][
        "material_predictor_change_fraction"
    ]
    passed = (
        sum(anchor_passes.values())
        >= protocol["required_gates"]["minimum_anchor_layers_passing"]
        and all(value >= material_gate for value in material.values())
        and all(invariants.values())
    )
    return {
        "passed": passed,
        "per_anchor": per_anchor,
        "anchor_passes": anchor_passes,
        "passing_anchor_count": sum(anchor_passes.values()),
        "material_predictor_change_fraction": material,
        "invariants": invariants,
        "diagnostics": diagnostics,
    }


def parameter_versions(model: torch.nn.Module) -> tuple[tuple[str, int], ...]:
    return tuple(
        (name, int(parameter._version)) for name, parameter in model.named_parameters()
    )


def consensual_operation(
    context: Any, *, phase: str, operation: Callable[[], Any]
) -> Any:
    result: Any = None
    error: BaseException | None = None
    try:
        result = operation()
    except BaseException as caught:
        error = caught
    distributed.phase_consensus(context, phase=phase, error=error)
    if error is not None:
        raise error
    return result


def prepare_output(context: Any, output_dir: Path) -> None:
    if not context.is_primary:
        return
    if output_dir.exists():
        raise ValueError(f"Cumulative mechanics output must be fresh: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=False)


def validate_immediate_protected_access(
    *,
    context: Any,
    base_model: Path,
    materialization_root: Path,
    output_dir: Path,
    protocol: Mapping[str, Any],
    launch: Mapping[str, Any],
    retrieval_result: Mapping[str, Any],
    manifest: Mapping[str, Any],
    maps: Mapping[str, Any],
    ordered_names: Sequence[str],
    builder_audit: Mapping[str, Any],
    model: torch.nn.Module,
    versions_before: tuple[tuple[str, int], ...],
) -> Mapping[str, Any]:
    refreshed_protocol = validate_protocol(base_model, validate_large_weights=False)
    refreshed_launch = validate_launch_binding(refreshed_protocol)
    refreshed_retrieval = mechanics.validate_retrieval_authorization()
    refreshed_manifest = retrieval._load_manifest_only(materialization_root, protocol)
    if (
        refreshed_protocol["receipt"] != protocol["receipt"]
        or refreshed_launch["receipt"] != launch["receipt"]
        or refreshed_launch["launch_head"] != launch["launch_head"]
        or refreshed_retrieval["receipt"] != retrieval_result["receipt"]
        or refreshed_manifest["receipt"] != manifest["receipt"]
        or sha256_file(mechanics.MAP_FILE) != mechanics.MAP_FILE_SHA256
        or retrieval.map_digest(maps, ordered_names) != mechanics.MAP_DIGEST
        or builder_audit != protocol["value_builder"]["frozen_tensor_digests"]
        or parameter_versions(model) != versions_before
        or os.environ.get("HF_ENDPOINT") != HF_ENDPOINT
        or context.world_size != WORLD_SIZE
        or not mechanics.hardware.four_distinct_a100s(context.rank_devices)
        or not output_dir.is_dir()
        or any(output_dir.iterdir())
    ):
        raise RuntimeError("Immediate cumulative mechanics access binding differs")
    binding = {
        "head": refreshed_launch["launch_head"],
        "protocol_file_sha256": sha256_file(PROTOCOL),
        "protocol_receipt": protocol["receipt"]["payload_sha256"],
        "launch_file_sha256": sha256_file(LAUNCH_BINDING),
        "launch_receipt": launch["receipt"]["payload_sha256"],
        "dependency_digest": canonical_sha256(dependency_bindings()),
        "retrieval_receipt": retrieval_result["receipt"]["payload_sha256"],
        "manifest_receipt": manifest["receipt"]["payload_sha256"],
        "map_file_sha256": mechanics.MAP_FILE_SHA256,
        "map_digest": mechanics.MAP_DIGEST,
        "builder_audit": dict(builder_audit),
        "parameter_versions_sha256": canonical_sha256(list(versions_before)),
        "output_dir": str(output_dir.resolve()),
        "hardware": list(context.rank_devices),
        "hf_endpoint": HF_ENDPOINT,
    }
    return {**binding, "digest": canonical_sha256(binding)}


def run(
    *,
    base_model: Path,
    materialization_root: Path,
    output_dir: Path,
    preflight_only: bool,
) -> Mapping[str, Any]:
    context = distributed.initialize_distributed_training(
        "cuda", required_world_size=WORLD_SIZE, timeout_seconds=TIMEOUT_SECONDS
    )
    if context is None:
        raise RuntimeError("Run with torchrun --nproc_per_node=4")
    try:
        if (
            context.world_size != WORLD_SIZE
            or context.backend != "nccl"
            or context.control_backend != "gloo"
            or not mechanics.hardware.four_distinct_a100s(context.rank_devices)
        ):
            raise RuntimeError("Cumulative mechanics requires four distinct A100s")
        if os.environ.get("HF_ENDPOINT") != HF_ENDPOINT:
            raise RuntimeError(f"HF_ENDPOINT must be exactly {HF_ENDPOINT}")
        def preflight():
            protocol = validate_protocol(
                base_model, validate_large_weights=context.is_primary
            )
            launch = validate_launch_binding(protocol)
            retrieval_result = mechanics.validate_retrieval_authorization()
            manifest = retrieval._load_manifest_only(materialization_root, protocol)
            return protocol, launch, retrieval_result, manifest

        protocol, launch, retrieval_result, manifest = consensual_operation(
            context,
            phase="cumulative-virtual-kv-preflight-without-protected-access",
            operation=preflight,
        )
        preflight_digest = canonical_sha256(
            {
                "protocol": protocol["receipt"]["payload_sha256"],
                "launch": launch["receipt"]["payload_sha256"],
                "retrieval": retrieval_result["receipt"]["payload_sha256"],
                "manifest": manifest["receipt"]["payload_sha256"],
                "dependencies": dependency_bindings(),
                "head": launch["launch_head"],
            }
        )
        distributed.require_consensus(
            context, preflight_digest, description="cumulative mechanics preflight"
        )

        torch.manual_seed(VALUE_SEED_BASE)
        torch.cuda.manual_seed_all(VALUE_SEED_BASE)
        model, tokenizer, model_audit = exact_v5.load_exact_v5_model(
            base_model, device=context.device
        )
        model.eval()
        modules = mechanics.causal_train.ordered_modules(model)
        ordered_names = tuple(name for name, _ in modules)
        if len(modules) != MODULES:
            raise RuntimeError("Cumulative mechanics module inventory differs")
        maps = mechanics.load_frozen_maps(ordered_names)
        integration.install(
            model,
            rank=mechanics.MAP_RANK,
            seed=mechanics.SEED,
            k_gain=mechanics.K_GAIN,
            a_gain=mechanics.A_GAIN,
            b_gain=mechanics.B_GAIN,
            trainable_map=False,
        )
        for name, module in modules:
            module.rwkv_continuous_write_conditioner.load_frozen_map(
                maps[name].down, maps[name].up
            )
            module.base.config._attn_implementation = "eager"
        if hasattr(model.config, "text_config"):
            model.config.text_config._attn_implementation = "eager"
        integration.set_mode(model, integration.CONTINUOUS_MODE)
        integration.set_capture(model, True)
        mechanics.install_feature_observer(modules)
        modules_by_layer = {int(module.layer_idx): module for _, module in modules}
        names_by_layer = {int(module.layer_idx): name for name, module in modules}
        if any(
            modules_by_layer[layer].is_kv_shared_layer
            or modules_by_layer[layer].layer_type != "full_attention"
            for layer in ANCHORS
        ):
            raise RuntimeError("Cumulative mechanics anchor attention type differs")
        router, builder_audit = make_router(
            modules_by_layer, maps, names_by_layer
        )
        compatibility_maps = {
            layer: maps[names_by_layer[layer]] for layer in ANCHORS
        }
        if builder_audit != protocol["value_builder"]["frozen_tensor_digests"]:
            raise RuntimeError("Cumulative mechanics value-builder digest differs")
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        versions_before = parameter_versions(model)
        preflight_result = {
            "schema": f"{SCHEMA}.preflight",
            "passed": True,
            "protected_mechanics_bytes_opened": False,
            "protocol_receipt": protocol["receipt"]["payload_sha256"],
            "launch_receipt": launch["receipt"]["payload_sha256"],
            "preflight_digest": preflight_digest,
            "builder_audit": builder_audit,
            "model_audit": model_audit,
            "hardware": list(context.rank_devices),
        }
        if preflight_only:
            return preflight_result

        consensual_operation(
            context,
            phase="fresh-cumulative-mechanics-output",
            operation=lambda: prepare_output(context, output_dir),
        )
        protected_access = consensual_operation(
            context,
            phase="immediate-cumulative-mechanics-protected-access-binding",
            operation=lambda: validate_immediate_protected_access(
                context=context,
                base_model=base_model,
                materialization_root=materialization_root,
                output_dir=output_dir,
                protocol=protocol,
                launch=launch,
                retrieval_result=retrieval_result,
                manifest=manifest,
                maps=maps,
                ordered_names=ordered_names,
                builder_audit=builder_audit,
                model=model,
                versions_before=versions_before,
            ),
        )
        distributed.require_consensus(
            context,
            protected_access["digest"],
            description="cumulative mechanics immediate protected-access binding",
        )
        primary_rows = consensual_operation(
            context,
            phase="primary-only-protected-cumulative-mechanics-read",
            operation=lambda: (
                mechanics._load_authorized_mechanics_bundle(
                    materialization_root, manifest, protocol
                )
                if context.is_primary
                else None
            ),
        )
        mechanics_rows = retrieval._broadcast_primary_object(context, primary_rows)
        if (
            not isinstance(mechanics_rows, list)
            or len(mechanics_rows) != ROWS
            or len({int(row["source_index"]) for row in mechanics_rows}) != ROWS
        ):
            raise RuntimeError("Broadcast cumulative mechanics row coverage differs")
        mechanics_binding = canonical_sha256(
            [
                {
                    "source_index": row["source_index"],
                    "donor_source_index": row["donor_source_index"],
                    "row_sha256": row["row_sha256"],
                    "donor_row_sha256": row["donor_row_sha256"],
                }
                for row in mechanics_rows
            ]
        )
        distributed.require_consensus(
            context,
            mechanics_binding,
            description="cumulative mechanics protected rows",
        )
        examples = retrieval._encode_rows(tokenizer, mechanics_rows)
        assigned_rows = mechanics_rows[context.process_rank :: WORLD_SIZE]
        if len(assigned_rows) != ROWS_PER_RANK:
            raise RuntimeError("Cumulative mechanics rank assignment differs")
        local_cache = {}
        for ordinal, row in enumerate(assigned_rows, start=1):
            source = int(row["source_index"])
            batch = mechanics.evolution.collate_native_examples(
                [examples[source]],
                pad_token_id=int(tokenizer.pad_token_id),
                device=context.device,
            )
            state, audit, address = mechanics.capture_write_condition(
                model,
                batch,
                modules,
                mode=integration.CONTINUOUS_MODE,
                override=None,
                reference_mode="none",
            )
            mechanics._clear_feature_references(modules)
            if (
                audit.get("formula_byte_exact_all_modules") is not True
                or audit.get("all_state_tensors_finite") is not True
            ):
                raise RuntimeError("Cumulative mechanics natural write audit differs")
            local_cache[source] = {"state": state, "address": address}
            print(
                f"CUMULATIVE_VKV_WRITE rank={context.process_rank} "
                f"row={source} ordinal={ordinal}/{ROWS_PER_RANK}",
                flush=True,
            )
        natural_cache = mechanics._gather_natural_cache(context, local_cache)
        candidates = candidate_sources(mechanics_rows)
        local_results = []
        for ordinal, row in enumerate(assigned_rows, start=1):
            source = int(row["source_index"])
            batch = mechanics.evolution.collate_native_examples(
                [examples[source]],
                pad_token_id=int(tokenizer.pad_token_id),
                device=context.device,
            )
            banks = control_banks(
                natural_cache,
                candidates[source],
                names_by_layer,
                ordered_names,
                context.device,
            )
            full_null = full_null_predictor(
                model,
                batch,
                modules,
                modules_by_layer,
                natural_cache[source]["state"],
            )
            baseline = predictor_pass(
                model,
                batch,
                modules,
                modules_by_layer,
                natural_cache[source]["state"],
                batch_size=len(CONTROL_NAMES),
                router=None,
                banks=None,
            )
            baseline_replay = predictor_pass(
                model,
                batch,
                modules,
                modules_by_layer,
                natural_cache[source]["state"],
                batch_size=len(CONTROL_NAMES),
                router=None,
                banks=None,
            )
            routed = predictor_pass(
                model,
                batch,
                modules,
                modules_by_layer,
                natural_cache[source]["state"],
                batch_size=len(CONTROL_NAMES),
                router=router,
                banks=banks,
                compatibility_maps=compatibility_maps,
            )
            local_results.append(
                row_result(
                    source,
                    candidates[source],
                    routed,
                    baseline,
                    baseline_replay,
                    full_null,
                )
            )
            reset_delta_mem_states(model)
            mechanics.evolution.release_native_row_allocator_cache(context.device)
            print(
                f"CUMULATIVE_VKV_READ rank={context.process_rank} "
                f"row={source} ordinal={ordinal}/{ROWS_PER_RANK}",
                flush=True,
            )
        shard = {
            "schema": SHARD_SCHEMA,
            "rank": context.process_rank,
            "world_size": WORLD_SIZE,
            "assignment": "sorted_rows_rank_stride_4",
            "protocol_receipt": protocol["receipt"]["payload_sha256"],
            "launch_receipt": launch["receipt"]["payload_sha256"],
            "preflight_digest": preflight_digest,
            "protected_access_digest": protected_access["digest"],
            "mechanics_binding": mechanics_binding,
            "rows": local_results,
        }
        shard["receipt"] = {
            "algorithm": "sha256",
            "payload_scope": "canonical_shard_without_receipt",
            "payload_sha256": canonical_sha256(shard),
        }
        signed_json(output_dir / f"shard-{context.process_rank}.json", shard)
        dist.barrier(group=context.control_group)
        gathered = distributed.gather_objects(context, local_results)
        rows = sorted(
            [item for rank_rows in gathered for item in rank_rows],
            key=lambda item: int(item["source_index"]),
        )
        if len(rows) != ROWS or len({row["source_index"] for row in rows}) != ROWS:
            raise RuntimeError("Cumulative mechanics result coverage differs")
        analysis = aggregate(rows, protocol)
        versions_after = parameter_versions(model)
        if versions_after != versions_before:
            raise RuntimeError("Cumulative mechanics model parameter versions changed")
        if dependency_bindings() != launch["dependency_bindings"]:
            raise RuntimeError("Cumulative mechanics dependency changed during execution")
        if git("rev-parse", "HEAD") != launch["launch_head"]:
            raise RuntimeError("Cumulative mechanics HEAD changed during execution")
        result = {
            "schema": SCHEMA,
            "status": (
                "cumulative_virtual_kv_mechanics_passed_causal_protocol_draft_authorized"
                if analysis["passed"]
                else "cumulative_virtual_kv_mechanics_failed_family_retired"
            ),
            "passed": analysis["passed"],
            "protocol_file_sha256": sha256_file(PROTOCOL),
            "protocol_receipt": protocol["receipt"]["payload_sha256"],
            "launch_file_sha256": sha256_file(LAUNCH_BINDING),
            "launch_receipt": launch["receipt"]["payload_sha256"],
            "preflight_digest": preflight_digest,
            "protected_access_digest": protected_access["digest"],
            "mechanics_binding": mechanics_binding,
            "hardware": {
                "world_size": context.world_size,
                "devices": list(context.rank_devices),
                "four_distinct_a100s": True,
                "hf_endpoint": os.environ["HF_ENDPOINT"],
            },
            "model_audit": model_audit,
            "builder_audit": builder_audit,
            "analysis": analysis,
            "rows": rows,
            "shards": [
                {
                    "rank": rank,
                    "path": f"shard-{rank}.json",
                    "sha256": sha256_file(output_dir / f"shard-{rank}.json"),
                }
                for rank in range(WORLD_SIZE)
            ],
            "mechanics_rows_opened": ROWS,
            "mechanics_bundle_byte_opens": 1,
            "causal_rows_opened": 0,
            "generation_or_native_benchmark_rows_opened": 0,
            "model_or_adapter_parameters_updated": False,
            "full_bandwidth_feedback_installed": False,
            "native_gain_claimed": False,
            "sota_claimed": False,
        }
        result["receipt"] = {
            "algorithm": "sha256",
            "payload_scope": "canonical_result_without_receipt",
            "payload_sha256": canonical_sha256(result),
        }
        distributed.require_consensus(
            context,
            result["receipt"]["payload_sha256"],
            description="cumulative mechanics result",
        )
        if context.is_primary:
            signed_json(output_dir / "result.json", result)
        dist.barrier(group=context.control_group)
        return result
    finally:
        distributed.destroy_distributed_training(context)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-model", type=Path, default=DEFAULT_BASE_MODEL)
    parser.add_argument(
        "--materialization-root", type=Path, default=DEFAULT_MATERIALIZATION
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--preflight-only", action="store_true")
    return parser.parse_args(argv)


if __name__ == "__main__":
    arguments = parse_args()
    outcome = run(
        base_model=arguments.base_model,
        materialization_root=arguments.materialization_root,
        output_dir=arguments.output_dir,
        preflight_only=arguments.preflight_only,
    )
    if int(os.environ.get("RANK", "0")) == 0:
        print(json.dumps(outcome, ensure_ascii=True, sort_keys=True), flush=True)
    raise SystemExit(0 if outcome.get("passed") else 1)
