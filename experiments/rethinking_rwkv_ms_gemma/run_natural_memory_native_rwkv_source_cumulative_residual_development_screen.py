#!/usr/bin/env python3
"""Screen source-cumulative RWKV residual variants on open development data."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any, Callable, Mapping, Sequence

import torch
import torch.distributed as dist
import torch.nn.functional as F


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[1]
for import_root in (SCRIPT_DIR, PROJECT_ROOT):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

from deltamem.core.cumulative_rwkv_residual import (  # noqa: E402
    SourceCumulativeResidualRouter,
)
from deltamem.core.delta import reset_delta_mem_states  # noqa: E402
from experiments.rethinking_rwkv_ms_gemma import (  # noqa: E402
    natural_memory_distributed as distributed,
)
from experiments.rethinking_rwkv_ms_gemma import (  # noqa: E402
    materialize_natural_memory_native_rwkv_source_cumulative_residual as materializer,
)
from experiments.rethinking_rwkv_ms_gemma import (  # noqa: E402
    materialize_natural_memory_native_rwkv_source_cumulative_residual_development as development_materializer,
)
from experiments.rethinking_rwkv_ms_gemma import (  # noqa: E402
    run_natural_memory_native_rwkv_cumulative_virtual_kv_mechanics as parent_runner,
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


SCHEMA = (
    "rwkv_ms_natural_memory_native_rwkv_source_cumulative_residual_"
    "development_screen.v2"
)
SHARD_SCHEMA = f"{SCHEMA}.shard"
PROTOCOL = SCRIPT_DIR / (
    "natural_memory_native_rwkv_source_cumulative_residual_mechanics_protocol_v1.json"
)
LAUNCH_BINDING = SCRIPT_DIR / (
    "natural_memory_native_rwkv_source_cumulative_residual_mechanics_launch_v1.json"
)
DEFAULT_BASE_MODEL = Path("/root/X/.cache/hf/gemma-4-E4B-it-a4c2d58")
DEFAULT_MATERIALIZATION = SCRIPT_DIR / (
    "local_artifacts/natural_memory_native_rwkv_source_cumulative_residual_v1"
)
DEFAULT_OUTPUT = SCRIPT_DIR / (
    "local_artifacts/natural_memory_native_rwkv_source_cumulative_residual_mechanics_v1"
)
DEFAULT_DEVELOPMENT_MATERIALIZATION = SCRIPT_DIR / (
    "local_artifacts/"
    "natural_memory_native_rwkv_source_cumulative_residual_development_v1"
)
DEFAULT_DEVELOPMENT_OUTPUT = SCRIPT_DIR / (
    "local_artifacts/"
    "natural_memory_native_rwkv_source_cumulative_residual_development_screen_v2"
)

RUNTIME_PARENT_COMMIT = "8be756a62a46ea92fd7b62192919bd07a479202d"
RUNTIME_PARENT_RESIDUAL_SHA256 = (
    "db840bda502ec71e6ab930fbdb88f72c684ed8e0b73dc0717db1000072d54560"
)
RUNTIME_PARENT_DELTA_IMPL_SHA256 = (
    "c572971f7157871757d11c91f437b1a73e1a3cde00bf5edb3d1af6b967a92f20"
)
MANIFEST_FILE_SHA256 = (
    "5251cc6f4254718620bd6e1328ac41c6fcb9bf837f836d623f874eedf53e9515"
)
MANIFEST_RECEIPT = (
    "0ef2fb6de7e696dac9881ee223988d3f0e6df531b4957d7830c88599fe60457b"
)
SPLIT_RECEIPT = (
    "6b22c808c6dc0cf722b74cf45981b3fad93a0a3ccdc3a5b023989487b1d637c6"
)

WORLD_SIZE = 4
ROWS = 32
ROWS_PER_RANK = 8
MODULES = 42
ANCHORS = (5, 11, 17, 23)
TERMINAL_ANCHOR = ANCHORS[-1]
ADDRESS_DIM = 64
STATE_DIM = 32
SLOTS = 4
COMPATIBILITY_SCALE = 32.0
RESIDUAL_GAIN = 1.0 / 32.0
DEVELOPMENT_VARIANTS = {
    "renew_at_17_scale_0_5": {
        "anchor_layers": (5, 11, 17),
        "compatibility_scale": 0.5,
    },
    "renew_at_17_scale_1": {
        "anchor_layers": (5, 11, 17),
        "compatibility_scale": 1.0,
    },
    "renew_at_17_scale_2": {
        "anchor_layers": (5, 11, 17),
        "compatibility_scale": 2.0,
    },
    "renew_at_17_scale_4": {
        "anchor_layers": (5, 11, 17),
        "compatibility_scale": 4.0,
    },
    "renew_at_17_scale_8": {
        "anchor_layers": (5, 11, 17),
        "compatibility_scale": 8.0,
    },
    "renew_at_17_scale_16": {
        "anchor_layers": (5, 11, 17),
        "compatibility_scale": 16.0,
    },
    "renew_at_23_scale_4": {
        "anchor_layers": ANCHORS,
        "compatibility_scale": 4.0,
    },
}
HF_ENDPOINT = "https://hf-mirror.com"
TIMEOUT_SECONDS = 1800
BASE_PERMUTATION = torch.tensor((2, 0, 3, 1), dtype=torch.long)
ANCHOR_PERMUTATIONS = {
    layer: tuple(int(value) for value in BASE_PERMUTATION.roll(index).tolist())
    for index, layer in enumerate(ANCHORS)
}
CONTROL_NAMES = (
    "correct_four_way",
    "independent_anchor_slot_permutation",
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
CACHED_NULL_REPLAY_ATOL = 0.0
FULL_CACHED_DIAGNOSTIC_ATOL = 1e-4
BF16_GATE_MASS_ATOL = 1.0 / 512.0


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


def git_blob_sha256(commit: str, relative_path: str) -> str:
    payload = subprocess.check_output(
        ["git", "show", f"{commit}:{relative_path}"], cwd=PROJECT_ROOT
    )
    return hashlib.sha256(payload).hexdigest()


def dependency_paths() -> Mapping[str, Path]:
    return {
        "runner": Path(__file__).resolve(),
        "residual_runtime": PROJECT_ROOT / "deltamem/core/cumulative_rwkv_residual.py",
        "delta_mem_runtime": PROJECT_ROOT / "deltamem/core/delta_impl.py",
        "parent_mechanics_helpers": Path(parent_runner.__file__).resolve(),
        "continuous_mechanics_helpers": Path(mechanics.__file__).resolve(),
        "continuous_retrieval_firewall": Path(retrieval.__file__).resolve(),
        "continuous_write_integration": Path(integration.__file__).resolve(),
        "distributed_runtime": Path(distributed.__file__).resolve(),
        "exact_adapter_loader": Path(exact_v5.__file__).resolve(),
        "fresh_materializer": Path(materializer.__file__).resolve(),
    }


def dependency_bindings() -> Mapping[str, str]:
    return {
        role: sha256_file(path) for role, path in sorted(dependency_paths().items())
    }


def validate_protocol_contract(protocol: Mapping[str, Any]) -> None:
    authorization = protocol.get("authorization_basis", {})
    frozen = protocol.get("frozen_inputs", {})
    materialization = protocol.get("fresh_materialization", {})
    architecture = protocol.get("architecture", {})
    controls = protocol.get("candidate_and_control_bank", {})
    gates = protocol.get("required_gates", {})
    execution = protocol.get("execution", {})
    lifecycle = protocol.get("data_lifecycle", {})
    launch = protocol.get("launch_binding", {})
    if (
        protocol.get("schema")
        != "rwkv_ms_natural_memory_native_rwkv_source_cumulative_residual_mechanics_protocol.v1"
        or authorization.get("runtime_parent_commit") != RUNTIME_PARENT_COMMIT
        or authorization.get("runtime_parent_residual_sha256")
        != RUNTIME_PARENT_RESIDUAL_SHA256
        or authorization.get("runtime_parent_delta_impl_sha256")
        != RUNTIME_PARENT_DELTA_IMPL_SHA256
        or authorization.get("continuous_retrieval_result_file_sha256")
        != mechanics.RETRIEVAL_RESULT_SHA256
        or authorization.get("continuous_retrieval_result_receipt")
        != mechanics.RETRIEVAL_RESULT_RECEIPT
        or authorization.get("fresh_manifest_file_sha256") != MANIFEST_FILE_SHA256
        or authorization.get("fresh_manifest_receipt") != MANIFEST_RECEIPT
        or authorization.get("fresh_split_receipt") != SPLIT_RECEIPT
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
        or frozen.get("frozen_map_file_sha256") != mechanics.MAP_FILE_SHA256
        or frozen.get("frozen_map_digest") != mechanics.MAP_DIGEST
        or frozen.get("conditioning_seed") != mechanics.SEED
        or frozen.get("conditioning_k_gain") != mechanics.K_GAIN
        or frozen.get("conditioning_a_gain") != mechanics.A_GAIN
        or frozen.get("conditioning_b_gain") != mechanics.B_GAIN
        or frozen.get("model_or_map_training") is not False
        or materialization.get("manifest_file_sha256") != MANIFEST_FILE_SHA256
        or materialization.get("manifest_receipt") != MANIFEST_RECEIPT
        or materialization.get("split_receipt") != SPLIT_RECEIPT
        or materialization.get("mechanics_bundle", {}).get("rows") != ROWS
        or materialization.get("mechanics_bundle", {}).get("sha256")
        != "38f6794adc1f07c728df48a9f49a6ed8cd3e9ab0df11f2feb500ea7cb5438397"
        or materialization.get("causal_inventory_from_manifest_only", {}).get(
            "byte_read_authorized"
        )
        is not False
        or architecture.get("name")
        != "source_canonical_score_gated_terminal_rwkv_residual"
        or architecture.get("anchor_layers") != list(ANCHORS)
        or architecture.get("compatibility_scale") != COMPATIBILITY_SCALE
        or architecture.get("residual_gain") != RESIDUAL_GAIN
        or architecture.get("terminal_layer") != TERMINAL_ANCHOR
        or architecture.get("hard_route_forward_soft_route_gradient") is not True
        or architecture.get("explicit_null_arm") is not True
        or architecture.get("attention_kv_mask_or_cache_modified") is not False
        or architecture.get("virtual_positions") != 0
        or architecture.get("full_bandwidth_feedback_installed") is not False
        or controls.get("candidate_order")
        != ["target", "matched_donor", "distractor_1", "distractor_2"]
        or controls.get("control_order") != list(CONTROL_NAMES)
        or controls.get("independent_anchor_permutations")
        != {str(layer): list(ANCHOR_PERMUTATIONS[layer]) for layer in ANCHORS}
        or gates.get("minimum_anchor_layers_passing") != 3
        or gates.get("identity_per_anchor", {}).get("strict_target_top1_fraction")
        != 0.75
        or gates.get("identity_per_anchor", {}).get(
            "mean_target_over_strongest_wrong_margin"
        )
        != 0.05
        or gates.get("identity_per_anchor", {}).get(
            "matched_donor_positive_fraction"
        )
        != 0.75
        or gates.get("identity_per_anchor", {}).get(
            "live_layer_roll_positive_fraction"
        )
        != 0.75
        or gates.get("terminal_target_selected_fraction") != 0.75
        or gates.get("material_predictor_change_fraction") != 0.95
        or gates.get("material_predictor_max_absolute_logit_delta")
        != MATERIAL_LOGIT_DELTA
        or gates.get("cached_null_replay_max_absolute_logit_tolerance")
        != CACHED_NULL_REPLAY_ATOL
        or gates.get("independent_slot_permutation_byte_exact") is not True
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
        or lifecycle.get("causal_path_statted_listed_hashed_or_opened_by_runner")
        is not False
        or lifecycle.get("full_bandwidth_feedback_authorized") is not False
        or launch.get("path") != LAUNCH_BINDING.name
        or launch.get("schema")
        != "rwkv_ms_natural_memory_native_rwkv_source_cumulative_residual_mechanics_launch.v1"
        or launch.get("validated_before_protected_access") is not True
    ):
        raise ValueError("Source-cumulative residual mechanics protocol contract differs")


def validate_protocol(
    base_model: Path, *, validate_large_weights: bool = True
) -> Mapping[str, Any]:
    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    validate_receipt(
        protocol,
        scope="canonical_protocol_without_receipt",
        description="Source-cumulative residual mechanics protocol",
    )
    validate_protocol_contract(protocol)
    source_paths = {
        "deltamem/core/cumulative_rwkv_residual.py": RUNTIME_PARENT_RESIDUAL_SHA256,
        "deltamem/core/delta_impl.py": RUNTIME_PARENT_DELTA_IMPL_SHA256,
    }
    for relative, expected in source_paths.items():
        path = PROJECT_ROOT / relative
        if (
            sha256_file(path) != expected
            or git_blob_sha256(RUNTIME_PARENT_COMMIT, relative) != expected
        ):
            raise ValueError(f"Parent-bound runtime source differs: {relative}")
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
        or sha256_file(mechanics.MAP_FILE) != frozen["frozen_map_file_sha256"]
    ):
        raise ValueError("Frozen adapter or compatibility map differs")
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
        != "rwkv_ms_natural_memory_native_rwkv_source_cumulative_residual_mechanics_launch.v1"
        or protocol.get("launch_binding", {}).get("path") != LAUNCH_BINDING.name
        or protocol.get("launch_binding", {}).get("schema") != launch.get("schema")
        or protocol.get("launch_binding", {}).get("validated_before_protected_access")
        is not True
        or not code_commit
        or head_parent != code_commit
        or code_parent != RUNTIME_PARENT_COMMIT
        or launch.get("runtime_parent_commit") != RUNTIME_PARENT_COMMIT
        or launch.get("protocol_file_sha256") != sha256_file(PROTOCOL)
        or launch.get("protocol_receipt") != protocol["receipt"]["payload_sha256"]
        or launch.get("dependency_bindings") != dependency_bindings()
        or launch.get("dependency_digest") != canonical_sha256(dependency_bindings())
    ):
        raise ValueError("Source-cumulative residual mechanics launch binding differs")


def validate_launch_binding(protocol: Mapping[str, Any]) -> Mapping[str, Any]:
    launch = json.loads(LAUNCH_BINDING.read_text(encoding="utf-8"))
    validate_receipt(
        launch,
        scope="canonical_launch_without_receipt",
        description="Source-cumulative residual mechanics launch",
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
    if git("status", "--porcelain", "--untracked-files=no"):
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
    return parent_runner.tensor_digest(tensor)


def make_router(
    maps: Mapping[str, Any],
    names_by_layer: Mapping[int, str],
    device: torch.device,
    *,
    anchor_layers: Sequence[int] = ANCHORS,
    compatibility_scale: float = COMPATIBILITY_SCALE,
) -> SourceCumulativeResidualRouter:
    anchors = tuple(int(layer) for layer in anchor_layers)
    router = SourceCumulativeResidualRouter(
        maps={layer: maps[names_by_layer[layer]] for layer in anchors},
        anchor_layers=anchors,
        compatibility_scale=compatibility_scale,
        residual_gain=RESIDUAL_GAIN,
        required_receptance_calls=2,
    ).to(device)
    router.eval()
    if any(parameter.requires_grad for parameter in router.parameters()):
        raise RuntimeError("Source-cumulative residual router must remain frozen")
    return router


def _get_parent_module(root: torch.nn.Module, module_name: str) -> torch.nn.Module:
    parent = root
    for part in module_name.split(".")[:-1]:
        parent = getattr(parent, part)
    return parent


def bind_terminal_hook(
    model: torch.nn.Module,
    modules: Sequence[tuple[str, Any]],
    *,
    terminal_layer: int = TERMINAL_ANCHOR,
) -> None:
    for name, module in modules:
        if int(module.layer_idx) != terminal_layer:
            continue
        parent = _get_parent_module(model, name)
        layernorm = getattr(parent, "post_feedforward_layernorm", None)
        if not isinstance(layernorm, torch.nn.Module):
            raise RuntimeError("Terminal anchor has no post-feedforward layernorm")
        module.bind_source_cumulative_residual_layernorm(layernorm)
        return
    raise RuntimeError("Terminal source-cumulative residual module is missing")


def clear_terminal_hooks(modules: Sequence[tuple[str, Any]]) -> None:
    for _, module in modules:
        module.remove_source_cumulative_residual_layernorm_hook()


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
    for layer in ANCHORS:
        name = names_by_layer[layer]
        rolled_name = ordered_names[(layer - 1) % len(ordered_names)]
        source_components = [
            parent_runner.source_state_component(natural_cache, int(source), name)
            for source in sources
        ]
        rolled_components = [
            parent_runner.source_state_component(
                natural_cache, int(source), rolled_name
            )
            for source in sources
        ]
        correct_state = torch.stack(
            [item[0] for item in source_components], dim=1
        ).to(device)
        correct_address = torch.stack(
            [item[1] for item in source_components], dim=0
        ).to(device)
        rolled_state = torch.stack(
            [item[0] for item in rolled_components], dim=1
        ).to(device)
        rolled_address = torch.stack(
            [item[1] for item in rolled_components], dim=0
        ).to(device)
        correct_index = CONTROL_INDEX["correct_four_way"]
        permuted_index = CONTROL_INDEX["independent_anchor_slot_permutation"]
        states[layer][correct_index] = correct_state
        addresses[layer][correct_index] = correct_address
        occupied[layer][correct_index] = True
        permutation = torch.tensor(
            ANCHOR_PERMUTATIONS[layer], dtype=torch.long, device=device
        )
        states[layer][permuted_index] = correct_state.index_select(1, permutation)
        addresses[layer][permuted_index] = correct_address.index_select(
            0, permutation
        )
        occupied[layer][permuted_index] = True
        source_ids[layer][permuted_index] = source_ids[layer][
            permuted_index
        ].index_select(0, permutation)

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


def _canonical_bank(
    state: torch.Tensor,
    addresses: torch.Tensor,
    occupied: torch.Tensor,
    source_ids: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    canonical_sources, order = torch.sort(source_ids, dim=1, stable=True)
    canonical_state = SourceCumulativeResidualRouter._canonical_gather(
        state, order, 2
    )
    canonical_addresses = SourceCumulativeResidualRouter._canonical_gather(
        addresses, order, 1
    )
    canonical_occupied = SourceCumulativeResidualRouter._canonical_gather(
        occupied, order, 1
    )
    return canonical_state, canonical_addresses, canonical_occupied, canonical_sources


def independent_router_equation(
    *,
    compatibility_map: Any,
    receptance: torch.Tensor,
    state: torch.Tensor,
    addresses: torch.Tensor,
    occupied: torch.Tensor,
    source_ids: torch.Tensor,
    running: dict[str, Any],
) -> Mapping[str, torch.Tensor | int]:
    state, addresses, occupied, canonical_sources = _canonical_bank(
        state, addresses, occupied, source_ids
    )

    def rms_normalize(value: torch.Tensor) -> torch.Tensor:
        square_mean = value.float().square().mean(dim=-1, keepdim=True)
        normalized = value.float() / square_mean.clamp_min(1e-12).sqrt()
        return torch.where(
            square_mean.gt(0.0), normalized, torch.zeros_like(normalized)
        )

    flattened_receptance = receptance.float().reshape(
        receptance.size(0), receptance.size(1), -1
    )
    normalized_addresses = rms_normalize(addresses)
    latent = F.linear(
        normalized_addresses,
        compatibility_map.down.to(device=addresses.device),
    )
    mapped_addresses = rms_normalize(
        F.linear(latent, compatibility_map.up.to(device=addresses.device))
    )
    local_scores = torch.einsum(
        "btd,bsd->bts",
        rms_normalize(flattened_receptance),
        mapped_addresses,
    ) / float(flattened_receptance.size(-1))
    local_active = (
        occupied
        & state.float().square().sum(dim=(-1, -2)).sum(dim=1).gt(0.0)
        & addresses.float().square().sum(dim=-1).gt(0.0)
    )
    count = int(running["count"]) + 1
    score_sum = (
        torch.zeros_like(local_scores)
        if running["score_sum"] is None
        else running["score_sum"]
    ) + local_scores
    active = local_active if running["active"] is None else running["active"] & local_active
    accumulated_scores = score_sum / float(count)
    running["count"] = count
    running["score_sum"] = score_sum
    running["active"] = active
    return {
        "count": count,
        "source_ids": canonical_sources,
        "state": state,
        "local_scores": local_scores,
        "accumulated_scores": accumulated_scores,
        "active": active,
    }


def provider_observer(
    router: SourceCumulativeResidualRouter,
    layer: int,
    observations: dict[int, Mapping[str, Any]],
    banks: tuple[Mapping[int, torch.Tensor], ...],
    compatibility_maps: Mapping[int, Any],
    independent_running: dict[str, Any],
):
    base_provider = router.provider_for(layer)

    def provider(**kwargs):
        module = kwargs["module"]
        receptance = module.rwkv_residual_router_receptance
        independent = independent_router_equation(
            compatibility_map=compatibility_maps[layer],
            receptance=receptance,
            state=banks[0][layer],
            addresses=banks[1][layer],
            occupied=banks[2][layer],
            source_ids=banks[3][layer],
            running=independent_running,
        )
        residual = base_provider(**kwargs)
        diagnostic = router.diagnostics[-1]
        checks = {
            "count_exact": int(diagnostic["count"]) == int(independent["count"]),
            "source_ids_exact": torch.equal(
                diagnostic["source_ids"], independent["source_ids"]
            ),
            "local_scores_exact": torch.equal(
                diagnostic["local_scores"], independent["local_scores"]
            ),
            "accumulated_scores_exact": torch.equal(
                diagnostic["accumulated_scores"],
                independent["accumulated_scores"],
            ),
            "active_exact": torch.equal(
                diagnostic["active"], independent["active"]
            ),
        }
        observation: dict[str, Any] = {
            "source_ids": diagnostic["source_ids"].detach().cpu(),
            "local_scores": diagnostic["local_scores"].detach().cpu(),
            "accumulated_scores": diagnostic["accumulated_scores"].detach().cpu(),
            "active": diagnostic["active"].detach().cpu(),
            "receptance": receptance.detach().cpu(),
            "readout_gate": module.rwkv_residual_router_gate.detach().cpu(),
            "receptance_calls": int(module.rwkv_residual_router_receptance_calls),
            "independent_equation_checks": checks,
            "residual_returned": residual is not None,
        }
        if residual is not None:
            observation.update(
                {
                    "residual": residual.detach().cpu(),
                    "diagnostic_residual_exact": torch.equal(
                        residual.detach(), diagnostic["residual"]
                    ),
                    "source_routes": diagnostic["source_routes"].detach().cpu(),
                    "soft_source_routes": diagnostic[
                        "soft_source_routes"
                    ].detach().cpu(),
                    "memory_mass": diagnostic["memory_mass"].detach().cpu(),
                    "selected_slot": diagnostic["selected_slot"].detach().cpu(),
                    "raw_read": diagnostic["raw_read"].detach().cpu(),
                    "native_read": diagnostic["native_read"].detach().cpu(),
                    "hidden_read": diagnostic["hidden_read"].detach().cpu(),
                }
            )
        observations[layer] = observation
        return residual

    return provider


def clear_providers(modules_by_layer: Mapping[int, Any]) -> None:
    for layer in ANCHORS:
        modules_by_layer[layer].clear_source_cumulative_residual_provider()


@torch.no_grad()
def predictor_pass(
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
    first_label, predictor = retrieval.first_prompt_boundary(batch.labels)
    if predictor < 1:
        raise ValueError("Residual mechanics predictor requires a nonempty prefill")
    projected, recurrent = parent_runner.install_target_state(
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
            raise RuntimeError("Residual mechanics prefill cache length differs")
        before_cache = parent_runner.cache_snapshot(cache)
        if router is not None:
            if banks is None or compatibility_maps is None:
                raise ValueError("Routed residual pass requires banks and frozen maps")
            independent_running = {
                "count": 0,
                "score_sum": None,
                "active": None,
            }
            anchor_layers = router.anchor_layers
            router.begin_forward(
                states={layer: banks[0][layer] for layer in anchor_layers},
                address_keys={layer: banks[1][layer] for layer in anchor_layers},
                occupied={layer: banks[2][layer] for layer in anchor_layers},
                source_ids={layer: banks[3][layer] for layer in anchor_layers},
                memory_mass_override=memory_mass_override,
            )
            for layer in anchor_layers:
                modules_by_layer[layer].set_source_cumulative_residual_provider(
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
            if tuple(item["layer"] for item in diagnostics) != router.anchor_layers:
                raise RuntimeError("Residual mechanics router lifecycle differs")
        audit = parent_runner.cache_audit(cache, before_cache, prefix_length=predictor)
    finally:
        clear_providers(modules_by_layer)
        if router is not None and (router.active or router.completed):
            router.abort_forward()
    online_after = mechanics.clone_online_state_cpu(modules)
    audit = {
        **audit,
        "projected_carrier_bytes_unchanged": projected_before
        == mechanics.state_sha256(online_after, mechanics.PROJECTED_ATTRIBUTES),
        "rwkv_state_bytes_unchanged": recurrent_before
        == mechanics.state_sha256(online_after, mechanics.RECURRENT_ATTRIBUTES),
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
    parent_runner.install_target_state(model, modules, target_state, 1)
    clear_providers(modules_by_layer)
    clear_terminal_hooks(modules)
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
        raise ValueError("Source-cumulative development logit shapes differ")
    difference = left.float() - right.float()
    return {
        "byte_exact": bool(torch.equal(left, right)),
        "maximum_absolute_delta": float(difference.abs().max().item()),
        "normalized_l2": float(
            (difference.norm() / right.float().norm().clamp_min(1e-12)).item()
        ),
        "material": bool(difference.abs().max().item() >= MATERIAL_LOGIT_DELTA),
    }


def _target_ce(logits: torch.Tensor, target: int) -> float:
    target_tensor = torch.tensor([target], dtype=torch.long)
    return float(F.cross_entropy(logits.float().unsqueeze(0), target_tensor).item())


def development_row_result(
    *,
    source: int,
    sources: Sequence[int],
    target: int,
    anchor_layers: Sequence[int],
    compatibility_scale: float,
    routed: Mapping[str, Any],
    baseline: Mapping[str, Any],
    baseline_replay: Mapping[str, Any],
    full_null: torch.Tensor,
) -> Mapping[str, Any]:
    anchors = tuple(int(layer) for layer in anchor_layers)
    routed_logits = routed["logits"]
    baseline_logits = baseline["logits"]
    observations = routed["observations"]
    correct_index = CONTROL_INDEX["correct_four_way"]
    permuted_index = CONTROL_INDEX["independent_anchor_slot_permutation"]
    single_index = CONTROL_INDEX["single_target"]
    donor_state_index = CONTROL_INDEX["matched_donor_state_only"]
    donor_address_index = CONTROL_INDEX["matched_donor_address_only"]
    donor_both_index = CONTROL_INDEX["matched_donor_address_and_state"]
    layer_state_index = CONTROL_INDEX["layer_rolled_state_only"]
    layer_address_index = CONTROL_INDEX["layer_rolled_address_only"]
    layer_both_index = CONTROL_INDEX["layer_rolled_address_and_state"]
    zero_indices = (
        CONTROL_INDEX["zero_state"],
        CONTROL_INDEX["zero_address"],
        CONTROL_INDEX["zero_state_and_address"],
    )

    identity = {}
    path_checks = {}
    for layer in anchors:
        observation = observations[layer]
        scores = observation["accumulated_scores"][correct_index, 0]
        canonical_sources = observation["source_ids"][correct_index]
        target_positions = canonical_sources.eq(source).nonzero(as_tuple=False).flatten()
        donor_positions = canonical_sources.eq(int(sources[1])).nonzero(
            as_tuple=False
        ).flatten()
        if target_positions.numel() != 1 or donor_positions.numel() != 1:
            raise RuntimeError("Development canonical source lookup differs")
        target_position = int(target_positions.item())
        donor_position = int(donor_positions.item())
        target_score = scores[target_position]
        strongest_wrong = torch.cat(
            (scores[:target_position], scores[target_position + 1 :])
        ).max()
        rolled_score = observation["accumulated_scores"][
            layer_address_index, 0, target_position
        ]
        identity[str(layer)] = {
            "scores": [float(value) for value in scores],
            "strict_target_top1": bool(target_score > strongest_wrong),
            "target_over_strongest_wrong_margin": float(
                target_score - strongest_wrong
            ),
            "target_over_matched_donor_margin": float(
                target_score - scores[donor_position]
            ),
            "target_over_live_layer_roll_margin": float(target_score - rolled_score),
        }
        path_checks[str(layer)] = {
            "receptance_calls_exactly_two": observation["receptance_calls"] == 2,
            "independent_router_equation_exact": all(
                observation["independent_equation_checks"].values()
            ),
            "residual_only_at_terminal": observation["residual_returned"]
            is (layer == anchors[-1]),
        }

    terminal = observations[anchors[-1]]
    selected_slot = int(terminal["selected_slot"][correct_index, 0].item())
    selected_source = (
        int(terminal["source_ids"][correct_index, selected_slot].item())
        if selected_slot >= 0
        else -1
    )
    selected_score = terminal["accumulated_scores"][correct_index].gather(
        -1,
        terminal["selected_slot"][correct_index].unsqueeze(-1),
    )
    expected_mass = torch.sigmoid(compatibility_scale * selected_score)
    selected_gate_error = float(
        (
            terminal["memory_mass"][correct_index].float()
            - expected_mass.float()
        )
        .abs()
        .max()
        .item()
    )
    residual = terminal["residual"]
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
            routed_logits[donor_both_index], routed_logits[single_index]
        ),
        "layer_state_only_vs_single_target": logit_comparison(
            routed_logits[layer_state_index], routed_logits[single_index]
        ),
        "layer_address_only_vs_single_target": logit_comparison(
            routed_logits[layer_address_index], routed_logits[single_index]
        ),
        "layer_both_vs_single_target": logit_comparison(
            routed_logits[layer_both_index], routed_logits[single_index]
        ),
        "independent_slot_permutation_vs_correct": logit_comparison(
            routed_logits[permuted_index], routed_logits[correct_index]
        ),
        "cached_null_replay_vs_cached_null": logit_comparison(
            baseline_replay["logits"], baseline_logits
        ),
        "full_null_vs_cached_null": logit_comparison(
            full_null[0], baseline_logits[0]
        ),
    }
    ce = {
        "provider_off": _target_ce(baseline_logits[correct_index], target),
        "correct_four_way": _target_ce(routed_logits[correct_index], target),
        "single_target": _target_ce(routed_logits[single_index], target),
        "matched_donor_state_only": _target_ce(
            routed_logits[donor_state_index], target
        ),
        "matched_donor_address_only": _target_ce(
            routed_logits[donor_address_index], target
        ),
        "matched_donor_address_and_state": _target_ce(
            routed_logits[donor_both_index], target
        ),
        "layer_rolled_state_only": _target_ce(routed_logits[layer_state_index], target),
        "layer_rolled_address_only": _target_ce(
            routed_logits[layer_address_index], target
        ),
        "layer_rolled_address_and_state": _target_ce(
            routed_logits[layer_both_index], target
        ),
    }
    zero_exact = {
        CONTROL_NAMES[index]: bool(
            torch.equal(routed_logits[index], baseline_logits[index])
        )
        for index in zero_indices
    }
    audits = (routed["audit"], baseline["audit"], baseline_replay["audit"])
    invariants = {
        "all_anchor_path_checks": all(
            value is True
            for checks in path_checks.values()
            for value in checks.values()
        ),
        "selected_source_is_target": selected_source == source,
        "selected_gate_equation_bf16_close": (
            selected_gate_error <= BF16_GATE_MASS_ATOL
        ),
        "residual_finite_and_bounded": bool(
            torch.isfinite(residual).all()
            and residual.abs().max().item() <= RESIDUAL_GAIN
        ),
        "independent_slot_permutation_diagnostics_exact": all(
            torch.equal(
                terminal[name][correct_index], terminal[name][permuted_index]
            )
            for name in (
                "source_ids",
                "accumulated_scores",
                "active",
                "selected_slot",
                "raw_read",
                "native_read",
                "hidden_read",
                "memory_mass",
                "residual",
            )
        ),
        "independent_slot_permutation_logits_exact": comparisons[
            "independent_slot_permutation_vs_correct"
        ]["byte_exact"],
        "zero_controls_exact_provider_off": all(zero_exact.values()),
        "cached_null_replay_exact": comparisons[
            "cached_null_replay_vs_cached_null"
        ]["byte_exact"],
        "all_logits_finite": bool(
            torch.isfinite(routed_logits).all()
            and torch.isfinite(baseline_logits).all()
            and torch.isfinite(baseline_replay["logits"]).all()
            and torch.isfinite(full_null).all()
        ),
        "all_state_and_cache_invariants": all(
            audit["one_real_position_appended_no_virtual_slots"]
            and all(audit["prefix_cache_bytes_unchanged"].values())
            and audit["projected_carrier_bytes_unchanged"]
            and audit["rwkv_state_bytes_unchanged"]
            for audit in audits
        ),
    }
    return {
        "source_index": source,
        "donor_source_index": int(sources[1]),
        "candidate_sources": list(sources),
        "target_token_id": target,
        "anchor_layers": list(anchors),
        "compatibility_scale": compatibility_scale,
        "identity": identity,
        "terminal_selected_source": selected_source,
        "comparisons": comparisons,
        "target_ce": ce,
        "target_ce_margins": {
            "gain_vs_provider_off": ce["provider_off"] - ce["correct_four_way"],
            "donor_state_minus_target": ce["matched_donor_state_only"]
            - ce["single_target"],
            "donor_address_minus_target": ce["matched_donor_address_only"]
            - ce["single_target"],
            "donor_both_minus_target": ce["matched_donor_address_and_state"]
            - ce["single_target"],
            "layer_state_minus_target": ce["layer_rolled_state_only"]
            - ce["single_target"],
            "layer_address_minus_target": ce["layer_rolled_address_only"]
            - ce["single_target"],
            "layer_both_minus_target": ce["layer_rolled_address_and_state"]
            - ce["single_target"],
        },
        "path_checks": path_checks,
        "zero_controls_byte_exact_provider_off": zero_exact,
        "invariants": invariants,
        "selected_gate_max_absolute_error": selected_gate_error,
        "routed_logits_sha256": tensor_digest(routed_logits),
        "provider_off_logits_sha256": tensor_digest(baseline_logits),
    }


def aggregate_development_variant(
    rows: Sequence[Mapping[str, Any]],
    anchor_layers: Sequence[int],
) -> Mapping[str, Any]:
    anchors = tuple(int(layer) for layer in anchor_layers)
    per_anchor = {}
    anchor_passes = {}
    for layer in anchors:
        metrics = [row["identity"][str(layer)] for row in rows]
        top1 = sum(value["strict_target_top1"] for value in metrics) / len(metrics)
        strongest = [
            value["target_over_strongest_wrong_margin"] for value in metrics
        ]
        donor = [value["target_over_matched_donor_margin"] for value in metrics]
        rolled = [
            value["target_over_live_layer_roll_margin"] for value in metrics
        ]
        per_anchor[str(layer)] = {
            "strict_target_top1_fraction": top1,
            "mean_target_over_strongest_wrong_margin": sum(strongest)
            / len(strongest),
            "matched_donor_positive_fraction": sum(value > 0 for value in donor)
            / len(donor),
            "live_layer_roll_positive_fraction": sum(value > 0 for value in rolled)
            / len(rolled),
        }
        anchor_passes[str(layer)] = (
            top1 >= 0.75
            and per_anchor[str(layer)]["mean_target_over_strongest_wrong_margin"]
            >= 0.05
            and per_anchor[str(layer)]["matched_donor_positive_fraction"] >= 0.75
            and per_anchor[str(layer)]["live_layer_roll_positive_fraction"] >= 0.75
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
    margin_names = tuple(rows[0]["target_ce_margins"])
    causal = {
        name: {
            "mean": sum(row["target_ce_margins"][name] for row in rows) / len(rows),
            "positive_fraction": sum(
                row["target_ce_margins"][name] > 0 for row in rows
            )
            / len(rows),
        }
        for name in margin_names
    }
    thresholded_metrics = {"selected_source_is_target"}
    invariants = {
        name: all(row["invariants"][name] for row in rows)
        for name in rows[0]["invariants"]
        if name not in thresholded_metrics
    }
    terminal_target_fraction = sum(
        row["terminal_selected_source"] == row["source_index"] for row in rows
    ) / len(rows)
    mechanics_pass = (
        all(anchor_passes.values())
        and terminal_target_fraction >= 0.75
        and all(value >= 0.95 for value in material.values())
        and all(invariants.values())
    )
    donor_causal_pass = (
        causal["donor_both_minus_target"]["mean"] >= 0.0
        and causal["donor_both_minus_target"]["positive_fraction"] >= 0.75
    )
    return {
        "mechanics_pass": mechanics_pass,
        "donor_causal_pass": donor_causal_pass,
        "development_pass": mechanics_pass and donor_causal_pass,
        "per_anchor": per_anchor,
        "anchor_passes": anchor_passes,
        "terminal_target_selected_fraction": terminal_target_fraction,
        "material_predictor_change_fraction": material,
        "target_ce_margins": causal,
        "invariants": invariants,
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


def prepare_development_output(context: Any, output_dir: Path) -> None:
    if context.is_primary:
        if output_dir.exists():
            raise ValueError(
                f"Source-cumulative development output must be fresh: {output_dir}"
            )
        output_dir.mkdir(parents=True, exist_ok=False)


def gather_development_natural_cache(
    context: Any,
    local_cache: Mapping[int, Any],
    *,
    expected_rows: int,
) -> dict[int, Any]:
    gathered: list[Any] = [None] * context.world_size
    dist.all_gather_object(gathered, dict(local_cache), group=context.control_group)
    merged: dict[int, Any] = {}
    for payload in gathered:
        if not isinstance(payload, Mapping):
            raise RuntimeError("Development natural cache gather differs")
        for source, value in payload.items():
            source_index = int(source)
            if source_index in merged:
                raise RuntimeError("Development natural cache duplicates a row")
            merged[source_index] = value
    if len(merged) != expected_rows:
        raise RuntimeError("Development natural cache coverage differs")
    return merged


def _development_device_preflight(
    *,
    context: Any,
    modules_by_layer: Mapping[int, Any],
    routers: Mapping[str, SourceCumulativeResidualRouter],
    require_occupied_sidecars: bool,
) -> None:
    for layer in ANCHORS:
        module = modules_by_layer[layer]
        if next(module.parameters()).device != context.device:
            raise RuntimeError("Development anchor module device differs")
        occupied = getattr(module, "projected_kv_occupied", None)
        if require_occupied_sidecars and (
            occupied is None
            or occupied.device != context.device
            or not bool(occupied.any(dim=1).all().item())
        ):
            raise RuntimeError("Development projected sidecar is missing or misplaced")
    for router in routers.values():
        if any(buffer.device != context.device for buffer in router.buffers()):
            raise RuntimeError("Development router buffer device differs")


def run_development_screen(
    *,
    base_model: Path,
    materialization_root: Path,
    output_dir: Path,
    preflight_only: bool = False,
) -> Mapping[str, Any]:
    context = distributed.initialize_distributed_training(
        "cuda", required_world_size=WORLD_SIZE, timeout_seconds=TIMEOUT_SECONDS
    )
    if context is None:
        raise RuntimeError("Run development screen with torchrun --nproc_per_node=4")
    try:
        if (
            context.world_size != WORLD_SIZE
            or context.backend != "nccl"
            or context.control_backend != "gloo"
            or not mechanics.hardware.four_distinct_a100s(context.rank_devices)
        ):
            raise RuntimeError("Development screen requires four distinct A100s")
        if os.environ.get("HF_ENDPOINT") != HF_ENDPOINT:
            raise RuntimeError(f"HF_ENDPOINT must be exactly {HF_ENDPOINT}")

        manifest = consensual_operation(
            context,
            phase="source-cumulative-open-development-manifest",
            operation=lambda: development_materializer.load_manifest(
                materialization_root / "manifest.json"
            ),
        )
        primary_rows = consensual_operation(
            context,
            phase="source-cumulative-open-development-read",
            operation=lambda: (
                development_materializer.read_open_development(
                    materialization_root, manifest
                )
                if context.is_primary
                else None
            ),
        )
        rows = retrieval._broadcast_primary_object(context, primary_rows)
        if (
            not isinstance(rows, list)
            or len(rows) != 64
            or len({int(row["source_index"]) for row in rows}) != 64
        ):
            raise RuntimeError("Open development row coverage differs")
        row_binding = canonical_sha256(
            [
                {
                    "source_index": row["source_index"],
                    "donor_source_index": row["donor_source_index"],
                    "row_sha256": row["row_sha256"],
                    "donor_row_sha256": row["donor_row_sha256"],
                }
                for row in rows
            ]
        )
        distributed.require_consensus(
            context, row_binding, description="open development rows"
        )
        torch.manual_seed(mechanics.SEED)
        torch.cuda.manual_seed_all(mechanics.SEED)
        model, tokenizer, model_audit = exact_v5.load_exact_v5_model(
            base_model, device=context.device
        )
        model.eval()
        modules = mechanics.causal_train.ordered_modules(model)
        ordered_names = tuple(name for name, _ in modules)
        if len(modules) != MODULES:
            raise RuntimeError("Development module inventory differs")
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
            raise RuntimeError("Development anchor attention type differs")
        routers = {
            variant: make_router(
                maps,
                names_by_layer,
                context.device,
                anchor_layers=config["anchor_layers"],
                compatibility_scale=config["compatibility_scale"],
            )
            for variant, config in DEVELOPMENT_VARIANTS.items()
        }
        compatibility_maps = {
            layer: maps[names_by_layer[layer]] for layer in ANCHORS
        }
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        versions_before = parameter_versions(model)
        _development_device_preflight(
            context=context,
            modules_by_layer=modules_by_layer,
            routers=routers,
            require_occupied_sidecars=False,
        )
        preflight = {
            "schema": f"{SCHEMA}.preflight",
            "passed": True,
            "development_manifest_sha256": (
                development_materializer.SEALED_MANIFEST_SHA256
            ),
            "development_rows_opened": 64,
            "protected_mechanics_rows_opened": 0,
            "protected_causal_rows_opened": 0,
            "hardware": {
                "world_size": context.world_size,
                "devices": list(context.rank_devices),
                "four_distinct_a100s": True,
                "hf_endpoint": os.environ["HF_ENDPOINT"],
            },
            "model_audit": model_audit,
            "variants": {
                name: {
                    "anchor_layers": list(config["anchor_layers"]),
                    "compatibility_scale": config["compatibility_scale"],
                }
                for name, config in DEVELOPMENT_VARIANTS.items()
            },
        }
        if preflight_only:
            return preflight
        consensual_operation(
            context,
            phase="source-cumulative-development-output",
            operation=lambda: prepare_development_output(context, output_dir),
        )
        dist.barrier(group=context.control_group)

        examples = retrieval._encode_rows(tokenizer, rows)
        assigned_rows = rows[context.process_rank :: WORLD_SIZE]
        if len(assigned_rows) != 16:
            raise RuntimeError("Development rank assignment differs")
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
                raise RuntimeError("Development natural write audit differs")
            local_cache[source] = {"state": state, "address": address}
            print(
                f"SOURCE_CUMULATIVE_DEV_WRITE rank={context.process_rank} "
                f"row={source} ordinal={ordinal}/16",
                flush=True,
            )
        natural_cache = gather_development_natural_cache(
            context,
            local_cache,
            expected_rows=64,
        )
        candidates = parent_runner.candidate_sources(rows)
        local_results = []
        for ordinal, row in enumerate(assigned_rows, start=1):
            source = int(row["source_index"])
            batch = mechanics.evolution.collate_native_examples(
                [examples[source]],
                pad_token_id=int(tokenizer.pad_token_id),
                device=context.device,
            )
            first_label, _ = retrieval.first_prompt_boundary(batch.labels)
            target_token = int(batch.labels[0, first_label].item())
            banks = control_banks(
                natural_cache,
                candidates[source],
                names_by_layer,
                ordered_names,
                context.device,
            )
            clear_terminal_hooks(modules)
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
            variants = {}
            for variant, config in DEVELOPMENT_VARIANTS.items():
                anchors = config["anchor_layers"]
                clear_terminal_hooks(modules)
                bind_terminal_hook(
                    model,
                    modules,
                    terminal_layer=anchors[-1],
                )
                router = routers[variant]
                _development_device_preflight(
                    context=context,
                    modules_by_layer=modules_by_layer,
                    routers={variant: router},
                    require_occupied_sidecars=True,
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
                variants[variant] = development_row_result(
                    source=source,
                    sources=candidates[source],
                    target=target_token,
                    anchor_layers=anchors,
                    compatibility_scale=config["compatibility_scale"],
                    routed=routed,
                    baseline=baseline,
                    baseline_replay=baseline_replay,
                    full_null=full_null,
                )
            clear_terminal_hooks(modules)
            local_results.append(
                {
                    "source_index": source,
                    "donor_source_index": int(row["donor_source_index"]),
                    "variants": variants,
                }
            )
            reset_delta_mem_states(model)
            mechanics.evolution.release_native_row_allocator_cache(context.device)
            print(
                f"SOURCE_CUMULATIVE_DEV_READ rank={context.process_rank} "
                f"row={source} ordinal={ordinal}/16",
                flush=True,
            )

        shard = {
            "schema": SHARD_SCHEMA,
            "rank": context.process_rank,
            "world_size": WORLD_SIZE,
            "assignment": "sorted_rows_rank_stride_4",
            "development_manifest_sha256": development_materializer.SEALED_MANIFEST_SHA256,
            "development_row_binding": row_binding,
            "variants": {
                name: {
                    "anchor_layers": list(config["anchor_layers"]),
                    "compatibility_scale": config["compatibility_scale"],
                }
                for name, config in DEVELOPMENT_VARIANTS.items()
            },
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
        all_rows = sorted(
            [item for rank_rows in gathered for item in rank_rows],
            key=lambda item: int(item["source_index"]),
        )
        if len(all_rows) != 64 or len({row["source_index"] for row in all_rows}) != 64:
            raise RuntimeError("Development result coverage differs")
        analysis = {
            variant: aggregate_development_variant(
                [row["variants"][variant] for row in all_rows],
                config["anchor_layers"],
            )
            for variant, config in DEVELOPMENT_VARIANTS.items()
        }
        passing = [
            variant
            for variant in DEVELOPMENT_VARIANTS
            if analysis[variant]["development_pass"]
        ]
        selected = (
            max(
                passing,
                key=lambda variant: (
                    analysis[variant]["target_ce_margins"]["gain_vs_provider_off"][
                        "mean"
                    ],
                    -DEVELOPMENT_VARIANTS[variant]["anchor_layers"][-1],
                ),
            )
            if passing
            else None
        )
        if parameter_versions(model) != versions_before:
            raise RuntimeError("Development model parameter versions changed")
        result = {
            "schema": SCHEMA,
            "status": (
                "development_variant_selected"
                if selected is not None
                else "development_screen_failed_no_variant_promoted"
            ),
            "passed": selected is not None,
            "selected_variant": selected,
            "development_manifest_sha256": development_materializer.SEALED_MANIFEST_SHA256,
            "development_manifest_receipt": manifest["receipt"]["payload_sha256"],
            "development_split_receipt": manifest["split_contract"]["receipt"][
                "payload_sha256"
            ],
            "development_row_binding": row_binding,
            "hardware": {
                "world_size": context.world_size,
                "devices": list(context.rank_devices),
                "four_distinct_a100s": True,
                "hf_endpoint": os.environ["HF_ENDPOINT"],
            },
            "model_audit": model_audit,
            "architecture": {
                "family": "source_canonical_selected_score_rwkv_residual",
                "residual_gain": RESIDUAL_GAIN,
                "variants": {
                    name: {
                        "score_anchor_layers": list(config["anchor_layers"]),
                        "compatibility_scale": config["compatibility_scale"],
                        "residual_injection_layer": config["anchor_layers"][-1],
                        "downstream_transformer_layers": (
                            MODULES - config["anchor_layers"][-1] - 1
                        ),
                    }
                    for name, config in DEVELOPMENT_VARIANTS.items()
                },
                "full_bandwidth_transformer_inspiration": (
                    "compare renewed downstream depth while holding selected state, "
                    "maps, gain, and frozen backbone fixed"
                ),
                "full_bandwidth_feedback_installed": False,
            },
            "analysis": analysis,
            "rows": all_rows,
            "shards": [
                {
                    "rank": rank,
                    "path": f"shard-{rank}.json",
                    "sha256": sha256_file(output_dir / f"shard-{rank}.json"),
                }
                for rank in range(WORLD_SIZE)
            ],
            "development_rows_opened": 64,
            "protected_mechanics_rows_opened": 0,
            "protected_causal_rows_opened": 0,
            "model_or_adapter_parameters_updated": False,
            "native_benchmark_opened": False,
        }
        result["receipt"] = {
            "algorithm": "sha256",
            "payload_scope": "canonical_result_without_receipt",
            "payload_sha256": canonical_sha256(result),
        }
        distributed.require_consensus(
            context,
            result["receipt"]["payload_sha256"],
            description="source-cumulative development result",
        )
        if context.is_primary:
            signed_json(output_dir / "result.json", result)
        dist.barrier(group=context.control_group)
        return result
    finally:
        distributed.destroy_distributed_training(context)


def parse_development_args(
    argv: Sequence[str] | None = None,
) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-model", type=Path, default=DEFAULT_BASE_MODEL)
    parser.add_argument(
        "--materialization-root",
        type=Path,
        default=DEFAULT_DEVELOPMENT_MATERIALIZATION,
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_DEVELOPMENT_OUTPUT)
    parser.add_argument("--preflight-only", action="store_true")
    return parser.parse_args(argv)


if __name__ == "__main__":
    arguments = parse_development_args()
    outcome = run_development_screen(
        base_model=arguments.base_model,
        materialization_root=arguments.materialization_root,
        output_dir=arguments.output_dir,
        preflight_only=arguments.preflight_only,
    )
    if int(os.environ.get("RANK", "0")) == 0:
        print(json.dumps(outcome, ensure_ascii=True, sort_keys=True), flush=True)
