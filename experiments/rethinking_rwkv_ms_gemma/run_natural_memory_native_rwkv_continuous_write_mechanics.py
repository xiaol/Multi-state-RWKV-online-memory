#!/usr/bin/env python3
"""Run the sealed continuous-write mechanics gate on exactly four A100s."""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
from types import MethodType
from typing import Any, Callable, Mapping, Sequence

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

import torch
import torch.distributed as dist


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[1]
for import_root in (SCRIPT_DIR, PROJECT_ROOT):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

SIGNED_SOURCE_ROOT_ENV = "RWKV_V5_EXACT_SOURCE_ROOT"
_signed_source_value = os.environ.get(SIGNED_SOURCE_ROOT_ENV)
SIGNED_SOURCE_ROOT = (
    None
    if not _signed_source_value
    else Path(_signed_source_value).expanduser().resolve()
)
if SIGNED_SOURCE_ROOT is not None:
    if not SIGNED_SOURCE_ROOT.is_dir():
        raise RuntimeError(
            f"{SIGNED_SOURCE_ROOT_ENV} is not a directory: {SIGNED_SOURCE_ROOT}"
        )
    try:
        sys.path.remove(str(SIGNED_SOURCE_ROOT))
    except ValueError:
        pass
    sys.path.insert(0, str(SIGNED_SOURCE_ROOT))

from deltamem.core import delta_impl as core_impl  # noqa: E402
from deltamem.core.delta import reset_delta_mem_states  # noqa: E402
from experiments.rethinking_rwkv_ms_gemma import (  # noqa: E402
    materialize_natural_memory_native_rwkv_continuous_write_open_fit as materializer,
)
from experiments.rethinking_rwkv_ms_gemma import (  # noqa: E402
    run_natural_memory_native_rwkv_continuous_write_retrieval as retrieval,
)
from experiments.rethinking_rwkv_ms_gemma import (  # noqa: E402
    run_natural_memory_native_rwkv_v5_shadow_crossfit as exact_v5,
)
from experiments.rethinking_rwkv_ms_gemma import (  # noqa: E402
    rwkv_continuous_write_alignment as alignment,
)
from experiments.rethinking_rwkv_ms_gemma import (  # noqa: E402
    rwkv_continuous_write_fit_split as fit_split,
)
from experiments.rethinking_rwkv_ms_gemma import (  # noqa: E402
    rwkv_continuous_write_integration as integration,
)


SCHEMA = "rwkv_ms_natural_memory_native_rwkv_continuous_write_mechanics.v1"
SHARD_SCHEMA = (
    "rwkv_ms_natural_memory_native_rwkv_continuous_write_mechanics_shard.v1"
)
PROTOCOL = SCRIPT_DIR / (
    "natural_memory_native_rwkv_continuous_write_mechanics_protocol_v1.json"
)
LAUNCH_BINDING = SCRIPT_DIR / (
    "natural_memory_native_rwkv_continuous_write_mechanics_launch_v1.json"
)
PROTOCOL_PAYLOAD_SHA256 = (
    "c482e6874e4e5f5b52be1567df70d2f51c76dd49a50e4554926d56aa63bb735b"
)
PROTOCOL_FILE_SHA256 = (
    "4e29aef94dfc90cb13034e7c972d364ca20b4428c662981dfd62fbe400400e21"
)
PARENT_COMMIT = "198e38e2ff70101670c13c818c5a032eb5027c4f"
RETRIEVAL_RESULT_SHA256 = (
    "5103a66475b7a596e53a58b8c7cb554e7e400f5825d42ca141bc342dfef8784b"
)
RETRIEVAL_RESULT_RECEIPT = (
    "cf001ac0f06afeb58b96084d656e5a22521a7d2229d68436b09572e231e0a6dd"
)
MAP_FILE_SHA256 = (
    "d41857ee834670db359c0e3ddb644e3ce04f4f01d6320da4b2550370cbf5d3d6"
)
MAP_DIGEST = "c4fd930a9efe609c76e3c5a4e3b34581a5848787e3de9bd0c5ac5fb899c0e559"
MANIFEST_FILE_SHA256 = (
    "c437a7d1f2b850a730fe5b28a08ae32ba02678561bb1265a4eef55bda7f4d468"
)
MANIFEST_RECEIPT = (
    "99a878493c3848c96624e2ad658842c99e69769b4a1721b5854ad25af8d0bee2"
)
MECHANICS_FILE_SHA256 = (
    "7a5421df97d295328a95953590010318bfc1361db0402243809bd13c094badd9"
)
MECHANICS_BYTES = 104357
CAUSAL_FILE_SHA256 = (
    "5920ca8c688f4c26e8b55c5c48eefb7c067016bb931dd3b5c210edd1f4d3e925"
)
CAUSAL_BYTES = 86001
WORLD_SIZE = 4
ROWS = 32
ROWS_PER_RANK = 8
MODULES = 42
ADDRESS_DIM = 64
STATE_DIM = 32
MAP_RANK = 16
RIDGE = 1.0
SEED = 151
K_GAIN = 0.25
A_GAIN = 0.25
B_GAIN = 0.25
TIMEOUT_SECONDS = 1800
HF_ENDPOINT = "https://hf-mirror.com"
MATERIAL_DISTANCE_MINIMUM = 0.05
MATERIAL_POSITIVE_ROW_FRACTION_MINIMUM = 0.95
MATERIAL_COMPARISONS = (
    "correct_vs_raw_unconditioned",
    "correct_vs_matched_donor_address_only",
    "correct_vs_layer_rolled_address_only",
    "correct_vs_target_address_on_donor_content",
)
WRITE_CONDITIONS = (
    "continuous_correct",
    "continuous_matched_donor_address_only",
    "continuous_target_address_on_donor_content",
    "natural_donor_continuous",
    "continuous_layer_rolled_address_only",
    "continuous_row_shuffled_address_only",
    "continuous_norm_random_address_only",
    "continuous_zero_address",
    "inherited_exact_v5",
    "raw_unconditioned",
)
READ_CONDITIONS = (
    *WRITE_CONDITIONS,
    "layer_permuted_recurrent",
    "row_shuffled_recurrent",
    "norm_random_recurrent",
    "zero_recurrent",
    "projected_only",
    "state_only",
    "prompt_only",
)
RECURRENT_ATTRIBUTES = exact_v5.causal_train.RECURRENT_ATTRIBUTES
PROJECTED_ATTRIBUTES = exact_v5.causal_train.PROJECTED_ATTRIBUTES
DEFAULT_BASE_MODEL = Path("/root/X/.cache/hf/gemma-4-E4B-it-a4c2d58")
DEFAULT_MATERIALIZATION = SCRIPT_DIR / (
    "local_artifacts/natural_memory_native_rwkv_continuous_write_open_fit_v1"
)
RETRIEVAL_ROOT = SCRIPT_DIR / (
    "local_artifacts/natural_memory_native_rwkv_continuous_write_retrieval_v1"
)
RETRIEVAL_RESULT = RETRIEVAL_ROOT / "result.json"
MAP_FILE = RETRIEVAL_ROOT / "continuous-write-maps.pt"
DEFAULT_OUTPUT = SCRIPT_DIR / (
    "local_artifacts/natural_memory_native_rwkv_continuous_write_mechanics_v1"
)

distributed = exact_v5.distributed
evolution = exact_v5.evolution
causal_train = exact_v5.causal_train
hardware = exact_v5.hardware


def canonical_sha256(value: Any) -> str:
    return distributed.canonical_sha256(value)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_signed_json(path: Path, value: Mapping[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(
        json.dumps(value, ensure_ascii=True, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _validate_receipt(
    value: Mapping[str, Any], *, payload_scope: str, description: str
) -> None:
    unsigned = dict(value)
    receipt = unsigned.pop("receipt", None)
    expected = {
        "algorithm": "sha256",
        "payload_scope": payload_scope,
        "payload_sha256": canonical_sha256(unsigned),
    }
    if receipt != expected:
        raise ValueError(f"{description} receipt differs")


def _consensual_operation(
    context: Any,
    *,
    phase: str,
    operation: Callable[[], Any],
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


def _dependency_paths() -> Mapping[str, Path]:
    return {
        "retrieval_result_validator_and_runtime_helpers": Path(
            retrieval.__file__
        ).resolve(),
        "bias_free_reduced_rank_ridge_and_metrics": Path(
            alignment.__file__
        ).resolve(),
        "full_address_latch_runtime_conditioner_and_override": Path(
            integration.__file__
        ).resolve(),
        "continuous_write_integration_regressions": PROJECT_ROOT
        / "deltamem/tests/test_rwkv_continuous_write_integration.py",
        "continuous_write_mechanics_regressions": PROJECT_ROOT
        / "deltamem/tests/test_natural_memory_native_rwkv_continuous_write_mechanics.py",
        "dataset_qualified_component_disjoint_split": Path(
            fit_split.__file__
        ).resolve(),
        "five_file_open_fit_materializer_and_firewall": Path(
            materializer.__file__
        ).resolve(),
        "model_and_tokenizer_loader": SCRIPT_DIR / "common.py",
        "signed_exact_v5_delta_api": Path(core_impl.__file__).with_name("delta.py"),
        "signed_exact_v5_delta_implementation": Path(core_impl.__file__).resolve(),
        "strict_exact_v5_loader_and_source_validator": Path(
            exact_v5.__file__
        ).resolve(),
        "signed_distributed_runtime": Path(distributed.__file__).resolve(),
        "signed_native_row_encoder_and_write_read_runtime": Path(
            evolution.__file__
        ).resolve(),
        "signed_ordered_module_and_state_intervention_helpers": Path(
            causal_train.__file__
        ).resolve(),
        "signed_four_a100_hardware_validator": Path(hardware.__file__).resolve(),
        "signed_exact_v5_adapter_topology_validator": Path(
            exact_v5.v5_eval.__file__
        ).resolve(),
    }


def dependency_bindings() -> list[dict[str, str]]:
    return [
        {
            "role": role,
            "basename": path.name,
            "sha256": sha256_file(path),
        }
        for role, path in _dependency_paths().items()
    ]


def _validate_source_dependencies(protocol: Mapping[str, Any]) -> None:
    declared = protocol.get("source_dependencies")
    if not isinstance(declared, list):
        raise ValueError("Continuous mechanics source dependencies are missing")
    by_role = {
        str(item.get("role")): item
        for item in declared
        if isinstance(item, Mapping)
    }
    paths = _dependency_paths()
    if set(by_role) != set(paths):
        raise ValueError("Continuous mechanics dependency closure differs")
    for role, path in paths.items():
        item = by_role[role]
        if item.get("basename") != path.name or item.get("sha256") != sha256_file(
            path
        ):
            raise ValueError(f"Continuous mechanics dependency differs: {role}")


def validate_protocol(base_model: Path) -> Mapping[str, Any]:
    if sha256_file(PROTOCOL) != PROTOCOL_FILE_SHA256:
        raise ValueError("Continuous mechanics protocol file hash differs")
    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    _validate_receipt(
        protocol,
        payload_scope="canonical_protocol_without_receipt",
        description="Continuous mechanics protocol",
    )
    if protocol["receipt"]["payload_sha256"] != PROTOCOL_PAYLOAD_SHA256:
        raise ValueError("Continuous mechanics protocol payload hash differs")
    authorization = protocol.get("authorization_basis", {})
    frozen = protocol.get("frozen_inputs", {})
    execution = protocol.get("execution", {})
    mechanics = protocol.get("mechanics_gate", {})
    gates = mechanics.get("material_state_gates", {})
    hard_gates = mechanics.get("hard_exact_gates", {})
    launch = protocol.get("launch_binding", {})
    if (
        protocol.get("schema")
        != "rwkv_ms_natural_memory_native_rwkv_continuous_write_mechanics_protocol.v1"
        or authorization.get("mechanics_code_parent_commit") != PARENT_COMMIT
        or authorization.get("retrieval_result_file_sha256")
        != RETRIEVAL_RESULT_SHA256
        or authorization.get("retrieval_result_receipt")
        != RETRIEVAL_RESULT_RECEIPT
        or authorization.get("frozen_map_file_sha256") != MAP_FILE_SHA256
        or authorization.get("frozen_map_digest") != MAP_DIGEST
        or authorization.get("continuous_write_manifest_file_sha256")
        != MANIFEST_FILE_SHA256
        or authorization.get("continuous_write_manifest_receipt")
        != MANIFEST_RECEIPT
        or frozen.get("base_config_sha256")
        != sha256_file(base_model / "config.json")
        or frozen.get("adapter_weights_sha256")
        != exact_v5.V5_ADAPTER_WEIGHTS_SHA256
        or frozen.get("adapter_config_sha256")
        != exact_v5.V5_ADAPTER_CONFIG_SHA256
        or frozen.get("signed_v5_source_commit") != exact_v5.SIGNED_V5_COMMIT
        or frozen.get("rank") != MAP_RANK
        or frozen.get("ridge") != RIDGE
        or frozen.get("k_gain") != K_GAIN
        or frozen.get("a_gain") != A_GAIN
        or frozen.get("b_gain") != B_GAIN
        or execution.get("world_size") != WORLD_SIZE
        or execution.get("rows_per_rank") != ROWS_PER_RANK
        or execution.get("backend") != "nccl"
        or execution.get("control_backend") != "gloo"
        or execution.get("hf_endpoint") != HF_ENDPOINT
        or mechanics.get("rows") != ROWS
        or mechanics.get("modules") != MODULES
        or mechanics.get("write_conditions") != list(WRITE_CONDITIONS)
        or mechanics.get("read_conditions") != list(READ_CONDITIONS)
        or gates.get("comparisons") != list(MATERIAL_COMPARISONS)
        or gates.get("normalized_l2_minimum") != MATERIAL_DISTANCE_MINIMUM
        or gates.get("global_mean_minimum") != MATERIAL_DISTANCE_MINIMUM
        or gates.get("positive_row_fraction_minimum")
        != MATERIAL_POSITIVE_ROW_FRACTION_MINIMUM
        or gates.get("metric")
        != "per row mean over all 42 modules of ||candidate_delta_state-correct_delta_state||_2 / max(||correct_delta_state||_2,||candidate_delta_state||_2,1e-12)"
        or hard_gates.get(
            "effective_override_is_exact_immutable_latch_object_and_requested_float32_bytes"
        )
        is not True
        or hard_gates.get(
            "continuous_zero_address_each_path_returns_own_raw_feature_objects_unchanged_and_cross_replay_bytes_exact"
        )
        is not True
        or hard_gates.get(
            "recurrent_reads_make_exactly_two_byte_identical_addressed_and_global_basis_calls"
        )
        is not True
        or hard_gates.get(
            "projected_only_bypass_makes_exactly_zero_underlying_rwkv_read_basis_invocations"
        )
        is not True
        or launch.get("path") != LAUNCH_BINDING.name
        or launch.get("schema")
        != "rwkv_ms_natural_memory_native_rwkv_continuous_write_mechanics_launch.v1"
        or launch.get("launch_binding_is_created_only_after_runner_finalization")
        is not True
        or launch.get("launch_binding_validated_before_mechanics_bytes_open")
        is not True
        or protocol.get("mechanics_bundle_byte_read_authorized_by_this_protocol")
        is not True
        or protocol.get("causal_bundle_byte_read_authorized") is not False
        or protocol.get("model_or_adapter_training_authorized") is not False
        or protocol.get("generation_authorized") is not False
        or protocol.get("native_benchmark_authorized") is not False
    ):
        raise ValueError("Continuous mechanics protocol contract differs")
    _validate_source_dependencies(protocol)
    return protocol


def _git_output(*arguments: str) -> str:
    completed = subprocess.run(
        ["git", "-C", str(PROJECT_ROOT), *arguments],
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def validate_launch_binding(protocol: Mapping[str, Any]) -> Mapping[str, Any]:
    launch = json.loads(LAUNCH_BINDING.read_text(encoding="utf-8"))
    _validate_receipt(
        launch,
        payload_scope="canonical_launch_binding_without_receipt",
        description="Continuous mechanics launch binding",
    )
    code_commit = str(launch.get("authorized_code_commit", ""))
    if (
        launch.get("schema")
        != "rwkv_ms_natural_memory_native_rwkv_continuous_write_mechanics_launch.v1"
        or launch.get("code_parent_commit") != PARENT_COMMIT
        or launch.get("protocol_payload_sha256") != PROTOCOL_PAYLOAD_SHA256
        or launch.get("protocol_file_sha256") != PROTOCOL_FILE_SHA256
        or launch.get("protocol_receipt")
        != protocol.get("receipt", {}).get("payload_sha256")
        or launch.get("runner_sha256") != sha256_file(Path(__file__).resolve())
        or launch.get("dependency_bindings_sha256")
        != canonical_sha256(dependency_bindings())
        or not code_commit
    ):
        raise ValueError("Continuous mechanics launch binding differs")
    head = _git_output("rev-parse", "HEAD")
    head_parent = _git_output("rev-parse", "HEAD^")
    code_parent = _git_output("rev-parse", f"{code_commit}^")
    if head_parent != code_commit or code_parent != PARENT_COMMIT:
        raise ValueError("Continuous mechanics exact two-commit ancestry differs")
    launch_relative = str(LAUNCH_BINDING.resolve().relative_to(PROJECT_ROOT))
    if _git_output("diff", "--name-only", code_commit, "HEAD") != launch_relative:
        raise ValueError("Continuous mechanics launch commit changed non-launch files")
    if _git_output("diff", "--name-only", "HEAD"):
        raise ValueError("Continuous mechanics tracked worktree is dirty")
    committed_runner = subprocess.run(
        [
            "git",
            "-C",
            str(PROJECT_ROOT),
            "show",
            f"{code_commit}:{Path(__file__).resolve().relative_to(PROJECT_ROOT)}",
        ],
        check=True,
        capture_output=True,
    ).stdout
    if hashlib.sha256(committed_runner).hexdigest() != launch.get("runner_sha256"):
        raise ValueError("Continuous mechanics committed runner binding differs")
    committed_protocol = subprocess.run(
        [
            "git",
            "-C",
            str(PROJECT_ROOT),
            "show",
            f"{code_commit}:{PROTOCOL.resolve().relative_to(PROJECT_ROOT)}",
        ],
        check=True,
        capture_output=True,
    ).stdout
    if hashlib.sha256(committed_protocol).hexdigest() != PROTOCOL_FILE_SHA256:
        raise ValueError("Continuous mechanics committed protocol binding differs")
    committed_launch = subprocess.run(
        [
            "git",
            "-C",
            str(PROJECT_ROOT),
            "show",
            f"HEAD:{launch_relative}",
        ],
        check=True,
        capture_output=True,
    ).stdout
    if hashlib.sha256(committed_launch).hexdigest() != sha256_file(LAUNCH_BINDING):
        raise ValueError("Continuous mechanics committed launch binding differs")
    if _git_output("rev-parse", "HEAD") != head:
        raise RuntimeError("Continuous mechanics HEAD changed during launch validation")
    return launch


def validate_retrieval_authorization() -> Mapping[str, Any]:
    if sha256_file(RETRIEVAL_RESULT) != RETRIEVAL_RESULT_SHA256:
        raise ValueError("Continuous mechanics retrieval result file hash differs")
    result = retrieval._validate_result(RETRIEVAL_RESULT)
    if (
        result.get("status")
        != "continuous_write_retrieval_passed_mechanics_protocol_draft_authorized"
        or result.get("passed") is not True
        or result.get("mechanics_protocol_drafting_authorized") is not True
        or result.get("mechanics_bytes_open_authorized") is not False
        or result.get("receipt", {}).get("payload_sha256")
        != RETRIEVAL_RESULT_RECEIPT
        or result.get("map_artifact", {}).get("sha256") != MAP_FILE_SHA256
        or result.get("map_artifact", {}).get("frozen_map_digest") != MAP_DIGEST
    ):
        raise ValueError("Continuous mechanics retrieval authorization differs")
    return result


def load_frozen_maps(
    module_names: Sequence[str],
) -> dict[str, alignment.FrozenMapWeights]:
    if sha256_file(MAP_FILE) != MAP_FILE_SHA256:
        raise ValueError("Continuous mechanics frozen map file hash differs")
    payload = torch.load(MAP_FILE, map_location="cpu", weights_only=True)
    if (
        not isinstance(payload, Mapping)
        or payload.get("schema") != retrieval.MAP_SCHEMA
        or payload.get("module_names") != list(module_names)
        or payload.get("rank") != MAP_RANK
        or payload.get("ridge") != RIDGE
        or payload.get("address_dim") != ADDRESS_DIM
        or payload.get("state_dim") != STATE_DIM
        or payload.get("frozen_map_digest") != MAP_DIGEST
    ):
        raise ValueError("Continuous mechanics frozen map payload differs")
    raw_maps = payload.get("maps")
    if not isinstance(raw_maps, Mapping) or set(raw_maps) != set(module_names):
        raise ValueError("Continuous mechanics frozen map inventory differs")
    maps: dict[str, alignment.FrozenMapWeights] = {}
    for name in module_names:
        item = raw_maps[name]
        if not isinstance(item, Mapping):
            raise ValueError("Continuous mechanics frozen map entry differs")
        down = item.get("down")
        up = item.get("up")
        if (
            not isinstance(down, torch.Tensor)
            or not isinstance(up, torch.Tensor)
            or tuple(down.shape) != (MAP_RANK, ADDRESS_DIM)
            or tuple(up.shape) != (STATE_DIM, MAP_RANK)
            or down.dtype != torch.float32
            or up.dtype != torch.float32
            or not bool(torch.isfinite(down).all().item())
            or not bool(torch.isfinite(up).all().item())
        ):
            raise ValueError("Continuous mechanics frozen map tensor differs")
        maps[name] = alignment.FrozenMapWeights(
            down=down.detach().clone(), up=up.detach().clone()
        )
    if retrieval.map_digest(maps, module_names) != MAP_DIGEST:
        raise ValueError("Continuous mechanics frozen map digest differs")
    return maps


def _load_authorized_mechanics_bundle(
    materialization_root: Path,
    manifest: Mapping[str, Any],
    protocol: Mapping[str, Any],
) -> list[dict[str, Any]]:
    if (
        protocol.get("mechanics_bundle_byte_read_authorized_by_this_protocol")
        is not True
        or protocol.get("causal_bundle_byte_read_authorized") is not False
    ):
        raise PermissionError("Continuous mechanics bundle access is not authorized")
    declared = protocol["open_fit_materialization"]["mechanics_bundle"]
    manifest_binding = manifest["file_inventory"]["bundles"]["mechanics"]
    if any(
        declared.get(key) != manifest_binding.get(key)
        for key in (
            "path",
            "rows",
            "bytes",
            "sha256",
            "payload_sha256",
            "source_indices_sha256",
            "qualified_source_ids_sha256",
            "qualified_mapping_pairs_sha256",
            "row_sha256s_sha256",
        )
    ):
        raise ValueError("Continuous mechanics manifest bundle binding differs")
    if (
        declared.get("rows") != ROWS
        or declared.get("bytes") != MECHANICS_BYTES
        or declared.get("sha256") != MECHANICS_FILE_SHA256
    ):
        raise ValueError("Continuous mechanics protected inventory differs")
    rows = materializer._read_bundle(materialization_root, manifest, "mechanics")
    sources = {int(row["source_index"]) for row in rows}
    if (
        len(rows) != ROWS
        or len(sources) != ROWS
        or any(row.get("split") != "mechanics" for row in rows)
        or any(int(row["donor_source_index"]) not in sources for row in rows)
    ):
        raise ValueError("Continuous mechanics bundle row contract differs")
    return sorted(rows, key=lambda row: int(row["source_index"]))


def _raw_tensor_bytes_equal(left: torch.Tensor, right: torch.Tensor) -> bool:
    return retrieval._raw_tensor_bytes_equal(left, right)


def _tensor_digest(tensor: torch.Tensor) -> str:
    value = tensor.detach().contiguous().cpu()
    digest = hashlib.sha256()
    digest.update(str(value.dtype).encode("ascii"))
    digest.update(json.dumps(list(value.shape)).encode("ascii"))
    digest.update(value.view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def state_sha256(
    state: Mapping[str, Mapping[str, torch.Tensor]],
    attributes: Sequence[str],
) -> str:
    digest = hashlib.sha256()
    for name in sorted(state):
        for attribute in attributes:
            tensor = state[name][attribute].detach().contiguous().cpu()
            digest.update(name.encode("utf-8"))
            digest.update(attribute.encode("utf-8"))
            digest.update(str(tensor.dtype).encode("ascii"))
            digest.update(json.dumps(list(tensor.shape)).encode("ascii"))
            digest.update(tensor.view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def clone_online_state_cpu(
    modules: Sequence[tuple[str, Any]],
) -> dict[str, dict[str, torch.Tensor]]:
    references = causal_train.capture_online_state_references(modules)
    return {
        name: {
            attribute: tensor.detach().cpu().clone()
            for attribute, tensor in values.items()
        }
        for name, values in references.items()
    }


def state_subset_to_device(
    state: Mapping[str, Mapping[str, torch.Tensor]],
    attributes: Sequence[str],
    device: torch.device,
) -> dict[str, dict[str, torch.Tensor]]:
    return {
        name: {
            attribute: values[attribute].to(device=device).clone()
            for attribute in attributes
        }
        for name, values in state.items()
    }


def zero_recurrent(
    state: Mapping[str, Mapping[str, torch.Tensor]],
) -> dict[str, dict[str, torch.Tensor]]:
    return {
        name: {
            attribute: (
                torch.zeros_like(values[attribute])
                if attribute == "delta_state"
                else values[attribute].detach().clone()
            )
            for attribute in RECURRENT_ATTRIBUTES
        }
        for name, values in state.items()
    }


def zero_projected(
    state: Mapping[str, Mapping[str, torch.Tensor]],
) -> dict[str, dict[str, torch.Tensor]]:
    return {
        name: {
            attribute: torch.zeros_like(values[attribute])
            for attribute in PROJECTED_ATTRIBUTES
        }
        for name, values in state.items()
    }


def layer_roll_recurrent(
    state: Mapping[str, Mapping[str, torch.Tensor]],
    module_names: Sequence[str],
) -> dict[str, dict[str, torch.Tensor]]:
    return {
        name: {
            attribute: state[module_names[(index - 1) % len(module_names)]][
                attribute
            ]
            .detach()
            .clone()
            for attribute in RECURRENT_ATTRIBUTES
        }
        for index, name in enumerate(module_names)
    }


def norm_random_address(address: torch.Tensor, *, seed: int) -> torch.Tensor:
    if tuple(address.shape) != (MODULES, ADDRESS_DIM):
        raise ValueError("Continuous mechanics random address shape differs")
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed))
    random = torch.randn(address.shape, generator=generator, dtype=torch.float32)
    target_norm = address.float().norm(dim=-1, keepdim=True)
    random = random * (
        target_norm / random.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    )
    return random


def norm_random_recurrent(
    state: Mapping[str, Mapping[str, torch.Tensor]], *, seed: int
) -> dict[str, dict[str, torch.Tensor]]:
    result: dict[str, dict[str, torch.Tensor]] = {}
    for index, name in enumerate(sorted(state)):
        values = state[name]
        recurrent = values["delta_state"]
        generator = torch.Generator(device="cpu")
        generator.manual_seed(int(seed) + index)
        random = torch.randn(
            recurrent.shape, generator=generator, dtype=torch.float32
        )
        random = random * (
            recurrent.float().norm()
            / random.norm().clamp_min(1e-12)
        )
        result[name] = {
            "delta_state": random.to(dtype=recurrent.dtype),
            "rwkv_ms_positions": values["rwkv_ms_positions"].detach().clone(),
            "rwkv_ms_previous_source": values[
                "rwkv_ms_previous_source"
            ].detach().clone(),
        }
    return result


def normalized_delta_state_l2(
    reference: Mapping[str, Mapping[str, torch.Tensor]],
    candidate: Mapping[str, Mapping[str, torch.Tensor]],
    module_names: Sequence[str],
) -> float:
    values = []
    for name in module_names:
        left = reference[name]["delta_state"].float()
        right = candidate[name]["delta_state"].float()
        denominator = torch.maximum(left.norm(), right.norm()).clamp_min(1e-12)
        values.append(float((right - left).norm() / denominator))
    result = sum(values) / len(values)
    if not math.isfinite(result):
        raise RuntimeError("Continuous mechanics normalized state distance is nonfinite")
    return result


def recurrent_metadata_equal(
    left: Mapping[str, Mapping[str, torch.Tensor]],
    right: Mapping[str, Mapping[str, torch.Tensor]],
    module_names: Sequence[str],
) -> bool:
    return all(
        _raw_tensor_bytes_equal(left[name][attribute], right[name][attribute])
        for name in module_names
        for attribute in ("rwkv_ms_positions", "rwkv_ms_previous_source")
    )


def _independent_conditioned_features(
    module: Any,
    k: torch.Tensor,
    v: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    address_seq: torch.Tensor,
    token_mask: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    mode = module.rwkv_continuous_write_mode
    latch = module.rwkv_continuous_write_latch
    if latch is None:
        raise RuntimeError("Continuous mechanics independent formula has no latch")
    if mode == integration.RAW_UNCONDITIONED_MODE:
        return k, v, a, b
    if mode == integration.INHERITED_EXACT_V5_MODE:
        return module.rwkv_continuous_write_original_conditioner(
            k, v, a, b, latch.folded_address_seq, token_mask
        )
    if mode != integration.CONTINUOUS_MODE:
        raise RuntimeError("Continuous mechanics formula mode differs")
    weights = alignment.FrozenMapWeights(
        down=module.rwkv_continuous_write_conditioner.down.detach(),
        up=module.rwkv_continuous_write_conditioner.up.detach(),
    )
    active = (
        address_seq.detach()
        .to(device=k.device)
        .square()
        .sum(dim=-1, keepdim=True)
        .gt(0.0)
    )
    if token_mask is not None:
        active = active & token_mask.to(
            device=k.device, dtype=torch.bool
        ).unsqueeze(-1)
    if not bool(active.any().item()):
        return k, v, a, b
    direction = alignment.mapped_direction(address_seq, weights).to(device=k.device)
    k_float = k.float()
    k_rms = k_float.square().mean(dim=-1, keepdim=True).clamp_min(1e-12).sqrt()
    expected_k = torch.where(
        active, k_float + K_GAIN * k_rms * direction, k_float
    ).to(dtype=k.dtype)
    expected_a = torch.where(
        active,
        a.float() * (1.0 + A_GAIN * torch.tanh(direction)),
        a.float(),
    ).to(dtype=a.dtype)
    expected_b = torch.where(
        active,
        b.float() * (1.0 + B_GAIN * torch.tanh(direction)),
        b.float(),
    ).to(dtype=b.dtype)
    return expected_k, v, expected_a, expected_b


def _observed_conditioned_features(
    module: Any,
    k: torch.Tensor,
    v: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    address_seq: torch.Tensor,
    token_mask: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    raw = (k, v, a, b)
    outputs = module.rwkv_continuous_mechanics_original_conditioned_features(
        k, v, a, b, address_seq, token_mask
    )
    expected = _independent_conditioned_features(
        module, k, v, a, b, address_seq, token_mask
    )
    reference_mode = module.rwkv_continuous_mechanics_reference_mode
    if reference_mode == "record":
        module.rwkv_continuous_mechanics_raw_reference = tuple(
            tensor.detach().clone() for tensor in raw
        )
        input_reference_exact = True
    elif reference_mode == "compare":
        reference = module.rwkv_continuous_mechanics_raw_reference
        if not isinstance(reference, tuple) or len(reference) != 4:
            raise RuntimeError("Continuous mechanics raw reference is missing")
        input_reference_exact = all(
            _raw_tensor_bytes_equal(left, right)
            for left, right in zip(raw, reference, strict=True)
        )
    elif reference_mode == "none":
        input_reference_exact = True
    else:
        raise RuntimeError("Continuous mechanics reference mode differs")
    latch = module.rwkv_continuous_write_latch
    integration_audit = module.rwkv_continuous_write_audit
    if latch is None or not isinstance(integration_audit, Mapping):
        raise RuntimeError("Continuous mechanics integration audit is missing")
    module.rwkv_continuous_mechanics_feature_calls += 1
    module.rwkv_continuous_mechanics_feature_audit = {
        "mode": module.rwkv_continuous_write_mode,
        "formula_byte_exact": all(
            _raw_tensor_bytes_equal(left, right)
            for left, right in zip(outputs, expected, strict=True)
        ),
        "input_reference_exact": input_reference_exact,
        "outputs_finite": all(
            bool(torch.isfinite(tensor).all().item()) for tensor in outputs
        ),
        "value_same_object": outputs[1] is v,
        "value_byte_exact": _raw_tensor_bytes_equal(outputs[1], v),
        "all_outputs_same_objects": all(
            output is source for output, source in zip(outputs, raw, strict=True)
        ),
        "all_outputs_byte_exact_raw": all(
            _raw_tensor_bytes_equal(output, source)
            for output, source in zip(outputs, raw, strict=True)
        ),
        "conditioner_address_is_latch": address_seq is latch.address_seq,
        "conditioner_address_version_unchanged": (
            latch.address_seq._version == latch.address_version
        ),
        "natural_address_version_unchanged": (
            latch.selected_keys._version == latch.selected_keys_version
        ),
        "effective_address_version_unchanged": (
            latch.effective_selected_keys._version
            == latch.effective_selected_keys_version
        ),
        "address_override_applied": latch.address_override_applied,
        "effective_full64_consumed_by_mode": integration_audit.get(
            "effective_full64_consumed_by_mode"
        ),
    }
    return outputs


def install_feature_observer(
    modules: Sequence[tuple[str, Any]],
) -> Mapping[str, Any]:
    installed = []
    for name, module in modules:
        if hasattr(module, "rwkv_continuous_mechanics_original_conditioned_features"):
            raise ValueError(f"Continuous mechanics feature observer exists: {name}")
        module.rwkv_continuous_mechanics_original_conditioned_features = (
            module._rwkv_ms_address_conditioned_write_features
        )
        module.rwkv_continuous_mechanics_reference_mode = "none"
        module.rwkv_continuous_mechanics_raw_reference = None
        module.rwkv_continuous_mechanics_feature_calls = 0
        module.rwkv_continuous_mechanics_feature_audit = None
        module._rwkv_ms_address_conditioned_write_features = MethodType(
            _observed_conditioned_features, module
        )
        installed.append(name)
    return {
        "modules": len(installed),
        "module_names": installed,
        "observer": "no_output_change_raw_feature_and_formula_audit",
    }


def _observed_mechanics_read_basis(
    module: Any,
    state: torch.Tensor,
    memory_source_seq: torch.Tensor,
    token_mask: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    module.rwkv_continuous_mechanics_read_basis_invocations += 1
    return module.rwkv_continuous_mechanics_original_read_basis(
        state, memory_source_seq, token_mask
    )


def install_read_invocation_observer(
    modules: Sequence[tuple[str, Any]],
) -> Mapping[str, Any]:
    installed = []
    for name, module in modules:
        if hasattr(module, "rwkv_continuous_mechanics_original_read_basis"):
            raise ValueError(f"Continuous mechanics read observer exists: {name}")
        module.rwkv_continuous_mechanics_original_read_basis = (
            module._rwkv_ms_token_state_read_basis
        )
        module.rwkv_continuous_mechanics_read_basis_invocations = 0
        module._rwkv_ms_token_state_read_basis = MethodType(
            _observed_mechanics_read_basis, module
        )
        installed.append(name)
    return {
        "modules": len(installed),
        "module_names": installed,
        "observer": "unconditional_rwkv_read_basis_invocation_counter",
    }


def _prepare_feature_observer(
    modules: Sequence[tuple[str, Any]], reference_mode: str
) -> None:
    if reference_mode not in {"none", "record", "compare"}:
        raise ValueError("Continuous mechanics reference mode differs")
    for _, module in modules:
        if reference_mode == "record":
            module.rwkv_continuous_mechanics_raw_reference = None
        if reference_mode == "compare" and not isinstance(
            module.rwkv_continuous_mechanics_raw_reference, tuple
        ):
            raise RuntimeError("Continuous mechanics comparison reference is missing")
        module.rwkv_continuous_mechanics_reference_mode = reference_mode
        module.rwkv_continuous_mechanics_feature_calls = 0
        module.rwkv_continuous_mechanics_feature_audit = None


def _clear_feature_references(modules: Sequence[tuple[str, Any]]) -> None:
    for _, module in modules:
        module.rwkv_continuous_mechanics_reference_mode = "none"
        module.rwkv_continuous_mechanics_raw_reference = None
        module.rwkv_continuous_mechanics_feature_calls = 0
        module.rwkv_continuous_mechanics_feature_audit = None


def _require_effective_address_match(
    effective_address: torch.Tensor,
    requested_override: torch.Tensor | None,
) -> bool:
    if requested_override is None:
        return True
    requested = requested_override.detach().contiguous().cpu().float()
    if not _raw_tensor_bytes_equal(effective_address, requested):
        raise RuntimeError(
            "Continuous mechanics effective address differs from requested override"
        )
    return True


def _collect_write_audit(
    model: torch.nn.Module,
    modules: Sequence[tuple[str, Any]],
    *,
    expected_mode: str,
    expected_override: bool,
    reference_mode: str,
) -> tuple[Mapping[str, Any], torch.Tensor]:
    addresses = []
    audits = []
    for name, module in modules:
        latch = module.rwkv_continuous_write_latch
        feature = module.rwkv_continuous_mechanics_feature_audit
        if (
            latch is None
            or not isinstance(feature, Mapping)
            or module.rwkv_continuous_mechanics_feature_calls != 1
        ):
            raise RuntimeError(f"Continuous mechanics write audit is missing: {name}")
        if (
            feature.get("mode") != expected_mode
            or feature.get("address_override_applied") is not expected_override
            or feature.get("formula_byte_exact") is not True
            or feature.get("outputs_finite") is not True
            or feature.get("conditioner_address_is_latch") is not True
            or feature.get("conditioner_address_version_unchanged") is not True
            or feature.get("natural_address_version_unchanged") is not True
            or feature.get("effective_address_version_unchanged") is not True
            or (
                reference_mode == "compare"
                and feature.get("input_reference_exact") is not True
            )
        ):
            raise RuntimeError(f"Continuous mechanics write invariant differs: {name}")
        if expected_mode == integration.CONTINUOUS_MODE and (
            feature.get("value_same_object") is not True
            or feature.get("value_byte_exact") is not True
            or feature.get("effective_full64_consumed_by_mode") is not True
        ):
            raise RuntimeError(f"Continuous mechanics value identity differs: {name}")
        if expected_mode != integration.CONTINUOUS_MODE and feature.get(
            "effective_full64_consumed_by_mode"
        ) is not False:
            raise RuntimeError(f"Continuous mechanics control consumed full64: {name}")
        addresses.append(latch.effective_selected_keys[0, 0].detach().cpu().float())
        audits.append(dict(feature))
    if integration.pending_effective_full64_address_override_names(model):
        raise RuntimeError("Continuous mechanics write left queued address overrides")
    return {
        "modules": len(audits),
        "mode": expected_mode,
        "address_override_applied": expected_override,
        "formula_byte_exact_all_modules": all(
            audit["formula_byte_exact"] for audit in audits
        ),
        "raw_inputs_reference_exact_all_modules": all(
            audit["input_reference_exact"] for audit in audits
        ),
        "continuous_value_same_object_and_bytes_all_modules": (
            expected_mode != integration.CONTINUOUS_MODE
            or all(
                audit["value_same_object"] and audit["value_byte_exact"]
                for audit in audits
            )
        ),
        "all_outputs_same_objects_all_modules": all(
            audit["all_outputs_same_objects"] for audit in audits
        ),
        "all_outputs_byte_exact_raw_all_modules": all(
            audit["all_outputs_byte_exact_raw"] for audit in audits
        ),
        "effective_address_object_and_versions_exact_all_modules": all(
            audit["conditioner_address_is_latch"]
            and audit["conditioner_address_version_unchanged"]
            and audit["effective_address_version_unchanged"]
            for audit in audits
        ),
    }, torch.stack(addresses)


@torch.no_grad()
def capture_write_condition(
    model: torch.nn.Module,
    batch: Any,
    modules: Sequence[tuple[str, Any]],
    *,
    mode: str,
    override: torch.Tensor | None,
    reference_mode: str,
) -> tuple[dict[str, dict[str, torch.Tensor]], Mapping[str, Any], torch.Tensor]:
    integration.clear_effective_full64_address_overrides(model)
    integration.set_mode(model, mode)
    _prepare_feature_observer(modules, reference_mode)
    if override is not None:
        if tuple(override.shape) != (MODULES, ADDRESS_DIM):
            raise ValueError("Continuous mechanics write override shape differs")
        integration.queue_effective_full64_address_overrides(
            model,
            {
                name: override[index].reshape(1, 1, ADDRESS_DIM).float()
                for index, (name, _) in enumerate(modules)
            },
        )
    try:
        evolution._native_write(model, batch, dtype=torch.bfloat16)
        state = clone_online_state_cpu(modules)
        write_audit, effective_address = _collect_write_audit(
            model,
            modules,
            expected_mode=mode,
            expected_override=override is not None,
            reference_mode=reference_mode,
        )
        effective_address_matches_requested_override = _require_effective_address_match(
            effective_address, override
        )
        write_audit = {
            **dict(write_audit),
            "effective_address_matches_requested_override": (
                effective_address_matches_requested_override
            ),
            "projected_sha256": state_sha256(state, PROJECTED_ATTRIBUTES),
            "recurrent_sha256": state_sha256(state, RECURRENT_ATTRIBUTES),
            "all_state_tensors_finite": all(
                bool(torch.isfinite(values[attribute]).all().item())
                for values in state.values()
                for attribute in (*PROJECTED_ATTRIBUTES, *RECURRENT_ATTRIBUTES)
                if values[attribute].is_floating_point()
            ),
        }
        if write_audit["all_state_tensors_finite"] is not True:
            raise RuntimeError("Continuous mechanics write produced nonfinite state")
        return state, write_audit, effective_address
    finally:
        integration.clear_effective_full64_address_overrides(model)


def _state_objects_and_versions(
    state: Mapping[str, Mapping[str, torch.Tensor]],
    attributes: Sequence[str],
) -> tuple[tuple[str, str, int, int], ...]:
    return tuple(
        (name, attribute, id(state[name][attribute]), state[name][attribute]._version)
        for name in sorted(state)
        for attribute in attributes
    )


def _module_references_exact(
    modules: Sequence[tuple[str, Any]],
    projected: Mapping[str, Mapping[str, torch.Tensor]],
    recurrent: Mapping[str, Mapping[str, torch.Tensor]],
) -> bool:
    return all(
        getattr(module, attribute) is projected[name][attribute]
        for name, module in modules
        for attribute in PROJECTED_ATTRIBUTES
    ) and all(
        getattr(module, attribute) is recurrent[name][attribute]
        for name, module in modules
        for attribute in RECURRENT_ATTRIBUTES
    )


@contextmanager
def _projected_only_bypass(modules: Sequence[tuple[str, Any]]):
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


@torch.no_grad()
def read_condition(
    model: torch.nn.Module,
    batch: Any,
    modules: Sequence[tuple[str, Any]],
    *,
    projected: Mapping[str, Mapping[str, torch.Tensor]],
    recurrent: Mapping[str, Mapping[str, torch.Tensor]],
    projected_only_bypass: bool = False,
) -> tuple[torch.Tensor, Mapping[str, Any]]:
    reset_delta_mem_states(model)
    fixed = causal_train.install_intervened_state(
        modules,
        projected=projected,
        recurrent=recurrent,
        rotate_recurrent_layers=False,
    )
    if not fixed or not _module_references_exact(modules, projected, recurrent):
        raise RuntimeError("Continuous mechanics read state references differ")
    projected_before = state_sha256(projected, PROJECTED_ATTRIBUTES)
    recurrent_before = state_sha256(recurrent, RECURRENT_ATTRIBUTES)
    projected_objects = _state_objects_and_versions(projected, PROJECTED_ATTRIBUTES)
    recurrent_objects = _state_objects_and_versions(recurrent, RECURRENT_ATTRIBUTES)
    _, predictor_index = retrieval.first_prompt_boundary(batch.labels)
    retrieval._clear_read_observer(modules)
    for _, module in modules:
        module.rwkv_continuous_mechanics_read_basis_invocations = 0
    if not projected_only_bypass:
        retrieval._prepare_read_observer(modules, predictor_index)
    if projected_only_bypass:
        with _projected_only_bypass(modules):
            logits = evolution._native_read(model, batch, dtype=torch.bfloat16)
    else:
        logits = evolution._native_read(model, batch, dtype=torch.bfloat16)
    if not all(getattr(module, "write_enabled", None) is False for _, module in modules):
        raise RuntimeError("Continuous mechanics read left writes enabled")
    if not _module_references_exact(modules, projected, recurrent):
        raise RuntimeError("Continuous mechanics read changed state references")
    read_basis_contract_exact = True
    read_basis_calls = []
    read_basis_invocations = []
    if projected_only_bypass:
        read_basis_contract_exact = all(
            int(module.rwkv_continuous_mechanics_read_basis_invocations) == 0
            and int(module.rwkv_continuous_retrieval_read_basis_calls) == 0
            and module.rwkv_continuous_retrieval_full_bytes_identical is False
            for _, module in modules
        )
    else:
        for name, module in modules:
            calls = int(module.rwkv_continuous_retrieval_read_basis_calls)
            invocations = int(
                module.rwkv_continuous_mechanics_read_basis_invocations
            )
            exact = bool(module.rwkv_continuous_retrieval_full_bytes_identical)
            if invocations != 2 or calls != 2 or not exact:
                raise RuntimeError(
                    f"Continuous mechanics addressed/global read differs: {name}"
                )
            read_basis_calls.append(calls)
            read_basis_invocations.append(invocations)
    projected_after = state_sha256(projected, PROJECTED_ATTRIBUTES)
    recurrent_after = state_sha256(recurrent, RECURRENT_ATTRIBUTES)
    audit = {
        "projected_carrier_references_fixed": True,
        "recurrent_references_fixed": True,
        "projected_carrier_bytes_unchanged": projected_before == projected_after,
        "recurrent_bytes_unchanged": recurrent_before == recurrent_after,
        "projected_objects_and_versions_unchanged": (
            projected_objects
            == _state_objects_and_versions(projected, PROJECTED_ATTRIBUTES)
        ),
        "recurrent_objects_and_versions_unchanged": (
            recurrent_objects
            == _state_objects_and_versions(recurrent, RECURRENT_ATTRIBUTES)
        ),
        "read_basis_contract_exact": read_basis_contract_exact,
        "addressed_global_read_basis_byte_exact": (
            None if projected_only_bypass else True
        ),
        "projected_only_rwkv_read_basis_bypassed": (
            True if projected_only_bypass else None
        ),
        "read_basis_calls_per_module": 0 if projected_only_bypass else 2,
        "read_basis_invocations_per_module": 0 if projected_only_bypass else 2,
        "write_disabled": True,
        "logits_finite": bool(torch.isfinite(logits).all().item()),
    }
    retrieval._clear_read_observer(modules)
    required_checks = (
        "projected_carrier_references_fixed",
        "recurrent_references_fixed",
        "projected_carrier_bytes_unchanged",
        "recurrent_bytes_unchanged",
        "projected_objects_and_versions_unchanged",
        "recurrent_objects_and_versions_unchanged",
        "read_basis_contract_exact",
        "write_disabled",
        "logits_finite",
    )
    if not all(audit[key] is True for key in required_checks):
        raise RuntimeError("Continuous mechanics read integrity differs")
    answer_ce = float(exact_v5.contrast.detached_answer_ce(logits, batch.labels)[0])
    return logits.detach().cpu(), {**audit, "answer_ce": answer_ce}


def _normalized_logits(logits: torch.Tensor) -> torch.Tensor:
    if logits.ndim == 2:
        return logits.unsqueeze(0)
    if logits.ndim == 3:
        return logits
    raise ValueError("Continuous mechanics predictor logits rank differs")


def compare_logits(
    candidate: torch.Tensor, correct: torch.Tensor
) -> Mapping[str, Any]:
    left = _normalized_logits(candidate)
    right = _normalized_logits(correct)
    if tuple(left.shape) != tuple(right.shape) or left.dtype != right.dtype:
        raise ValueError("Continuous mechanics condition logits differ in contract")
    changed = left.ne(right).any(dim=-1)
    difference = left.float() - right.float()
    normalized_l2 = difference.norm(dim=-1) / right.float().norm(
        dim=-1
    ).clamp_min(1e-12)
    return {
        "byte_exact": _raw_tensor_bytes_equal(left, right),
        "predictor_vectors": int(changed.numel()),
        "predictor_logit_changed_fraction": float(changed.float().mean()),
        "mean_normalized_logit_l2": float(normalized_l2.mean()),
        "maximum_absolute_logit_delta": float(difference.abs().max()),
    }


def _parameter_versions(
    model: torch.nn.Module,
) -> tuple[tuple[str, int], ...]:
    return tuple(
        (name, int(parameter._version))
        for name, parameter in model.named_parameters()
    )


def _map_runtime_parity(
    modules: Sequence[tuple[str, Any]],
    maps: Mapping[str, alignment.FrozenMapWeights],
) -> Mapping[str, Any]:
    active_exact = []
    zero_exact = []
    finite_nonzero = []
    for index, (name, module) in enumerate(modules):
        active = torch.arange(1, ADDRESS_DIM + 1, dtype=torch.float32).reshape(
            1, ADDRESS_DIM
        )
        active = active.roll(index, dims=-1).to(
            device=module.rwkv_continuous_write_conditioner.down.device
        )
        zero = torch.zeros_like(active)
        device_weights = alignment.FrozenMapWeights(
            down=maps[name].down.to(device=active.device),
            up=maps[name].up.to(device=active.device),
        )
        runtime_active = module.rwkv_continuous_write_conditioner.direction(active)
        runtime_zero = module.rwkv_continuous_write_conditioner.direction(zero)
        offline_active = alignment.mapped_direction(active, device_weights)
        offline_zero = alignment.mapped_direction(zero, device_weights)
        active_exact.append(_raw_tensor_bytes_equal(runtime_active, offline_active))
        zero_exact.append(
            _raw_tensor_bytes_equal(runtime_zero, offline_zero)
            and torch.equal(runtime_zero, torch.zeros_like(runtime_zero))
        )
        finite_nonzero.append(
            bool(torch.isfinite(runtime_active).all().item())
            and bool(runtime_active.norm().gt(0.0).item())
        )
    return {
        "modules": len(modules),
        "active_offline_runtime_byte_exact": all(active_exact),
        "zero_offline_runtime_byte_exact_and_exact_zero": all(zero_exact),
        "active_directions_finite_nonzero": all(finite_nonzero),
        "passed": all(active_exact) and all(zero_exact) and all(finite_nonzero),
    }


def _gather_natural_cache(
    context: Any,
    local_cache: Mapping[int, Any],
) -> dict[int, Any]:
    gathered: list[Any] = [None] * context.world_size
    dist.all_gather_object(gathered, dict(local_cache), group=context.control_group)
    merged: dict[int, Any] = {}
    for payload in gathered:
        if not isinstance(payload, Mapping):
            raise RuntimeError("Continuous mechanics natural cache gather differs")
        for source, value in payload.items():
            source_index = int(source)
            if source_index in merged:
                raise RuntimeError("Continuous mechanics natural cache duplicates a row")
            merged[source_index] = value
    if len(merged) != ROWS:
        raise RuntimeError("Continuous mechanics natural cache coverage differs")
    return merged


def _condition_state_metrics(
    correct: Mapping[str, Mapping[str, torch.Tensor]],
    states: Mapping[str, Mapping[str, Mapping[str, torch.Tensor]]],
    module_names: Sequence[str],
) -> dict[str, float]:
    comparisons = {
        "correct_vs_raw_unconditioned": "raw_unconditioned",
        "correct_vs_matched_donor_address_only": (
            "continuous_matched_donor_address_only"
        ),
        "correct_vs_layer_rolled_address_only": (
            "continuous_layer_rolled_address_only"
        ),
        "correct_vs_target_address_on_donor_content": (
            "continuous_target_address_on_donor_content"
        ),
        "correct_vs_row_shuffled_address_only": (
            "continuous_row_shuffled_address_only"
        ),
        "correct_vs_norm_random_address_only": (
            "continuous_norm_random_address_only"
        ),
        "correct_vs_inherited_exact_v5": "inherited_exact_v5",
        "correct_vs_natural_donor_recurrent": "natural_donor_continuous",
        "correct_vs_layer_permuted_recurrent": "layer_permuted_recurrent",
        "correct_vs_row_shuffled_recurrent": "row_shuffled_recurrent",
        "correct_vs_norm_random_recurrent": "norm_random_recurrent",
        "correct_vs_zero_recurrent": "zero_recurrent",
    }
    return {
        name: normalized_delta_state_l2(
            correct, states[condition], module_names
        )
        for name, condition in comparisons.items()
    }


def _row_shuffle_predecessor(
    source_index: int, ordered_sources: Sequence[int]
) -> int:
    position = ordered_sources.index(source_index)
    return ordered_sources[(position - 1) % len(ordered_sources)]


@torch.no_grad()
def evaluate_row(
    model: torch.nn.Module,
    target_example: Any,
    donor_example: Any,
    *,
    source_index: int,
    donor_source_index: int,
    natural_cache: Mapping[int, Any],
    ordered_sources: Sequence[int],
    module_names: Sequence[str],
    pad_token_id: int,
    device: torch.device,
) -> Mapping[str, Any]:
    modules = causal_train.ordered_modules(model)
    target = evolution.collate_native_examples(
        [target_example], pad_token_id=pad_token_id, device=device
    )
    donor = evolution.collate_native_examples(
        [donor_example], pad_token_id=pad_token_id, device=device
    )
    predecessor = _row_shuffle_predecessor(source_index, ordered_sources)
    target_address = natural_cache[source_index]["address"]
    donor_address = natural_cache[donor_source_index]["address"]
    predecessor_address = natural_cache[predecessor]["address"]
    states: dict[str, dict[str, dict[str, torch.Tensor]]] = {}
    write_audits: dict[str, Mapping[str, Any]] = {}
    try:
        correct, audit, observed_target_address = capture_write_condition(
            model,
            target,
            modules,
            mode=integration.CONTINUOUS_MODE,
            override=None,
            reference_mode="record",
        )
        states["continuous_correct"] = correct
        write_audits["continuous_correct"] = audit
        if (
            state_sha256(correct, (*PROJECTED_ATTRIBUTES, *RECURRENT_ATTRIBUTES))
            != state_sha256(
                natural_cache[source_index]["state"],
                (*PROJECTED_ATTRIBUTES, *RECURRENT_ATTRIBUTES),
            )
            or not _raw_tensor_bytes_equal(observed_target_address, target_address)
        ):
            raise RuntimeError("Continuous mechanics natural target replay differs")

        target_branches = (
            (
                "continuous_matched_donor_address_only",
                integration.CONTINUOUS_MODE,
                donor_address,
            ),
            (
                "continuous_layer_rolled_address_only",
                integration.CONTINUOUS_MODE,
                target_address.roll(1, dims=0),
            ),
            (
                "continuous_row_shuffled_address_only",
                integration.CONTINUOUS_MODE,
                predecessor_address,
            ),
            (
                "continuous_norm_random_address_only",
                integration.CONTINUOUS_MODE,
                norm_random_address(target_address, seed=SEED + source_index),
            ),
            (
                "continuous_zero_address",
                integration.CONTINUOUS_MODE,
                torch.zeros_like(target_address),
            ),
            (
                "inherited_exact_v5",
                integration.INHERITED_EXACT_V5_MODE,
                None,
            ),
            (
                "raw_unconditioned",
                integration.RAW_UNCONDITIONED_MODE,
                None,
            ),
        )
        for name, mode, override in target_branches:
            state, branch_audit, _ = capture_write_condition(
                model,
                target,
                modules,
                mode=mode,
                override=override,
                reference_mode="compare",
            )
            states[name] = state
            write_audits[name] = branch_audit

        natural_donor, donor_audit, observed_donor_address = capture_write_condition(
            model,
            donor,
            modules,
            mode=integration.CONTINUOUS_MODE,
            override=None,
            reference_mode="record",
        )
        states["natural_donor_continuous"] = natural_donor
        write_audits["natural_donor_continuous"] = donor_audit
        if (
            state_sha256(
                natural_donor, (*PROJECTED_ATTRIBUTES, *RECURRENT_ATTRIBUTES)
            )
            != state_sha256(
                natural_cache[donor_source_index]["state"],
                (*PROJECTED_ATTRIBUTES, *RECURRENT_ATTRIBUTES),
            )
            or not _raw_tensor_bytes_equal(observed_donor_address, donor_address)
        ):
            raise RuntimeError("Continuous mechanics natural donor replay differs")
        donor_target, donor_target_audit, _ = capture_write_condition(
            model,
            donor,
            modules,
            mode=integration.CONTINUOUS_MODE,
            override=target_address,
            reference_mode="compare",
        )
        states["continuous_target_address_on_donor_content"] = donor_target
        write_audits[
            "continuous_target_address_on_donor_content"
        ] = donor_target_audit
        _clear_feature_references(modules)

        states["layer_permuted_recurrent"] = layer_roll_recurrent(
            correct, module_names
        )
        states["row_shuffled_recurrent"] = {
            name: {
                attribute: natural_cache[predecessor]["state"][name][attribute]
                .detach()
                .clone()
                for attribute in RECURRENT_ATTRIBUTES
            }
            for name in module_names
        }
        states["norm_random_recurrent"] = norm_random_recurrent(
            correct, seed=SEED * 100000 + source_index
        )
        states["zero_recurrent"] = zero_recurrent(correct)

        target_projected_sha = write_audits["continuous_correct"][
            "projected_sha256"
        ]
        target_content_conditions = (
            "continuous_matched_donor_address_only",
            "continuous_layer_rolled_address_only",
            "continuous_row_shuffled_address_only",
            "continuous_norm_random_address_only",
            "continuous_zero_address",
            "inherited_exact_v5",
            "raw_unconditioned",
        )
        projected_target_fixed = all(
            write_audits[name]["projected_sha256"] == target_projected_sha
            for name in target_content_conditions
        )
        donor_projected_fixed = (
            write_audits["natural_donor_continuous"]["projected_sha256"]
            == write_audits["continuous_target_address_on_donor_content"][
                "projected_sha256"
            ]
        )
        target_metadata_fixed = all(
            recurrent_metadata_equal(correct, states[name], module_names)
            for name in target_content_conditions
        )
        zero_raw_state_exact = state_sha256(
            states["continuous_zero_address"],
            (*PROJECTED_ATTRIBUTES, *RECURRENT_ATTRIBUTES),
        ) == state_sha256(
            states["raw_unconditioned"],
            (*PROJECTED_ATTRIBUTES, *RECURRENT_ATTRIBUTES),
        )
        zero_raw_feature_paths_exact = (
            write_audits["continuous_zero_address"][
                "all_outputs_same_objects_all_modules"
            ]
            and write_audits["continuous_zero_address"][
                "all_outputs_byte_exact_raw_all_modules"
            ]
            and write_audits["raw_unconditioned"][
                "all_outputs_same_objects_all_modules"
            ]
            and write_audits["raw_unconditioned"][
                "all_outputs_byte_exact_raw_all_modules"
            ]
            and write_audits["continuous_zero_address"][
                "raw_inputs_reference_exact_all_modules"
            ]
            and write_audits["raw_unconditioned"][
                "raw_inputs_reference_exact_all_modules"
            ]
        )

        target_projected = state_subset_to_device(
            correct, PROJECTED_ATTRIBUTES, device
        )
        empty_projected = state_subset_to_device(
            zero_projected(correct), PROJECTED_ATTRIBUTES, device
        )
        read_sources: dict[str, Mapping[str, Mapping[str, torch.Tensor]]] = {
            name: states[name]
            for name in (
                "continuous_correct",
                "continuous_matched_donor_address_only",
                "continuous_target_address_on_donor_content",
                "natural_donor_continuous",
                "continuous_layer_rolled_address_only",
                "continuous_row_shuffled_address_only",
                "continuous_norm_random_address_only",
                "continuous_zero_address",
                "inherited_exact_v5",
                "raw_unconditioned",
                "layer_permuted_recurrent",
                "row_shuffled_recurrent",
                "norm_random_recurrent",
                "zero_recurrent",
            )
        }
        read_logits: dict[str, torch.Tensor] = {}
        read_audits: dict[str, Mapping[str, Any]] = {}
        for name, recurrent_source in read_sources.items():
            recurrent_device = state_subset_to_device(
                recurrent_source, RECURRENT_ATTRIBUTES, device
            )
            logits, read_audit = read_condition(
                model,
                target,
                modules,
                projected=target_projected,
                recurrent=recurrent_device,
            )
            read_logits[name] = logits
            read_audits[name] = read_audit

        zero_recurrent_device = state_subset_to_device(
            states["zero_recurrent"], RECURRENT_ATTRIBUTES, device
        )
        projected_only_logits, projected_only_audit = read_condition(
            model,
            target,
            modules,
            projected=target_projected,
            recurrent=zero_recurrent_device,
            projected_only_bypass=True,
        )
        read_logits["projected_only"] = projected_only_logits
        read_audits["projected_only"] = projected_only_audit
        correct_recurrent_device = state_subset_to_device(
            correct, RECURRENT_ATTRIBUTES, device
        )
        state_only_logits, state_only_audit = read_condition(
            model,
            target,
            modules,
            projected=empty_projected,
            recurrent=correct_recurrent_device,
        )
        read_logits["state_only"] = state_only_logits
        read_audits["state_only"] = state_only_audit
        prompt_only_logits, prompt_only_audit = read_condition(
            model,
            target,
            modules,
            projected=empty_projected,
            recurrent=zero_recurrent_device,
        )
        read_logits["prompt_only"] = prompt_only_logits
        read_audits["prompt_only"] = prompt_only_audit
        if set(write_audits) != set(WRITE_CONDITIONS) or set(read_audits) != set(
            READ_CONDITIONS
        ):
            raise RuntimeError("Continuous mechanics condition inventory differs")

        correct_logits = read_logits["continuous_correct"]
        logit_metrics = {
            name: compare_logits(logits, correct_logits)
            for name, logits in read_logits.items()
            if name != "continuous_correct"
        }
        state_metrics = _condition_state_metrics(correct, states, module_names)
        zero_raw_logits_exact = _raw_tensor_bytes_equal(
            read_logits["continuous_zero_address"],
            read_logits["raw_unconditioned"],
        )
        zero_projected_logits_exact = _raw_tensor_bytes_equal(
            read_logits["zero_recurrent"], read_logits["projected_only"]
        )
        integrity = {
            "natural_target_replay_exact": True,
            "natural_donor_replay_exact": True,
            "all_write_formulas_byte_exact": all(
                audit["formula_byte_exact_all_modules"]
                for audit in write_audits.values()
            ),
            "all_target_raw_features_fixed": all(
                write_audits[name]["raw_inputs_reference_exact_all_modules"]
                for name in target_content_conditions
            ),
            "donor_raw_features_fixed_for_target_address": write_audits[
                "continuous_target_address_on_donor_content"
            ]["raw_inputs_reference_exact_all_modules"],
            "continuous_value_same_object_and_bytes": all(
                audit["continuous_value_same_object_and_bytes_all_modules"]
                for name, audit in write_audits.items()
                if name not in {"inherited_exact_v5", "raw_unconditioned"}
            ),
            "effective_address_objects_and_versions_exact": all(
                audit["effective_address_object_and_versions_exact_all_modules"]
                for audit in write_audits.values()
            ),
            "effective_override_values_byte_exact": all(
                audit["effective_address_matches_requested_override"]
                for audit in write_audits.values()
            ),
            "target_projected_carrier_fixed_across_address_controls": projected_target_fixed,
            "donor_projected_carrier_fixed_for_target_address": donor_projected_fixed,
            "target_recurrent_metadata_fixed_across_address_controls": target_metadata_fixed,
            "zero_address_each_path_returns_own_raw_feature_objects_unchanged_and_cross_replay_bytes_exact": zero_raw_feature_paths_exact,
            "zero_address_online_state_exact_raw": zero_raw_state_exact,
            "zero_address_logits_exact_raw": zero_raw_logits_exact,
            "zero_recurrent_logits_exact_projected_only": zero_projected_logits_exact,
            "all_reads_integrity_exact": all(
                audit["projected_carrier_references_fixed"]
                and audit["recurrent_references_fixed"]
                and audit["projected_carrier_bytes_unchanged"]
                and audit["recurrent_bytes_unchanged"]
                and audit["projected_objects_and_versions_unchanged"]
                and audit["recurrent_objects_and_versions_unchanged"]
                and audit["read_basis_contract_exact"]
                and audit["write_disabled"]
                and audit["logits_finite"]
                for audit in read_audits.values()
            ),
        }
        if not all(integrity.values()):
            raise RuntimeError("Continuous mechanics row integrity gate failed")
        return {
            "source_index": source_index,
            "donor_source_index": donor_source_index,
            "row_shuffle_predecessor_source_index": predecessor,
            "state_normalized_l2": state_metrics,
            "material_comparison_positive": {
                name: state_metrics[name] >= MATERIAL_DISTANCE_MINIMUM
                for name in MATERIAL_COMPARISONS
            },
            "read_diagnostics": {
                name: {
                    **dict(logit_metrics[name]),
                    "answer_ce": read_audits[name]["answer_ce"],
                    "answer_ce_minus_correct": (
                        read_audits[name]["answer_ce"]
                        - read_audits["continuous_correct"]["answer_ce"]
                    ),
                }
                for name in logit_metrics
            },
            "correct_answer_ce": read_audits["continuous_correct"]["answer_ce"],
            "conditions": {
                "write": list(WRITE_CONDITIONS),
                "read": list(READ_CONDITIONS),
            },
            "integrity": integrity,
        }
    finally:
        _clear_feature_references(modules)
        integration.clear_effective_full64_address_overrides(model)
        retrieval._clear_read_observer(modules)
        reset_delta_mem_states(model)
        evolution.release_native_row_allocator_cache(device)


def mechanics_analysis(rows: Sequence[Mapping[str, Any]]) -> Mapping[str, Any]:
    if len(rows) != ROWS or len({int(row["source_index"]) for row in rows}) != ROWS:
        raise ValueError("Continuous mechanics analysis row coverage differs")
    aggregate: dict[str, Any] = {}
    checks: dict[str, bool] = {}
    for name in MATERIAL_COMPARISONS:
        values = [float(row["state_normalized_l2"][name]) for row in rows]
        mean_value = sum(values) / len(values)
        positive_fraction = sum(
            value >= MATERIAL_DISTANCE_MINIMUM for value in values
        ) / len(values)
        aggregate[name] = {
            "mean_normalized_l2": mean_value,
            "positive_row_fraction": positive_fraction,
            "minimum_row_normalized_l2": min(values),
            "maximum_row_normalized_l2": max(values),
        }
        checks[f"{name}_mean"] = mean_value >= MATERIAL_DISTANCE_MINIMUM
        checks[f"{name}_positive_row_fraction"] = (
            positive_fraction >= MATERIAL_POSITIVE_ROW_FRACTION_MINIMUM
        )
    integrity_keys = tuple(rows[0]["integrity"])
    integrity = {
        key: all(row["integrity"].get(key) is True for row in rows)
        for key in integrity_keys
    }
    checks.update({f"integrity_{key}": value for key, value in integrity.items()})
    diagnostics: dict[str, Any] = {}
    diagnostic_names = tuple(rows[0]["read_diagnostics"])
    for name in diagnostic_names:
        changed = [
            float(row["read_diagnostics"][name]["predictor_logit_changed_fraction"])
            for row in rows
        ]
        ce_delta = [
            float(row["read_diagnostics"][name]["answer_ce_minus_correct"])
            for row in rows
        ]
        diagnostics[name] = {
            "mean_predictor_logit_changed_fraction": sum(changed) / len(changed),
            "rows_at_least_095_predictor_vectors_changed": sum(
                value >= 0.95 for value in changed
            )
            / len(changed),
            "mean_answer_ce_minus_correct": sum(ce_delta) / len(ce_delta),
            "ce_is_diagnostic_not_causal_preference": True,
        }
    return {
        "evaluation_calls": 1,
        "rows": ROWS,
        "aggregate": aggregate,
        "integrity": integrity,
        "read_diagnostics": diagnostics,
        "checks": checks,
        "passed": all(checks.values()),
    }


def _write_shard(
    output_dir: Path,
    *,
    rank: int,
    rows: Sequence[Mapping[str, Any]],
    binding: Mapping[str, Any],
    assigned_source_rows: Sequence[Mapping[str, Any]],
) -> Mapping[str, Any]:
    assigned_source_indices = [int(row["source_index"]) for row in assigned_source_rows]
    assigned_source_binding = canonical_sha256(
        [
            {
                "source_index": row["source_index"],
                "row_sha256": row["row_sha256"],
                "donor_source_index": row["donor_source_index"],
                "donor_row_sha256": row["donor_row_sha256"],
            }
            for row in assigned_source_rows
        ]
    )
    shard: dict[str, Any] = {
        "schema": SHARD_SCHEMA,
        "rank": rank,
        "world_size": WORLD_SIZE,
        "assignment": "sorted_mechanics_ordinal_round_robin",
        "assigned_source_indices": assigned_source_indices,
        "assigned_source_binding": assigned_source_binding,
        "binding": dict(binding),
        "rows": list(rows),
        "mechanics_rows_opened": ROWS,
        "causal_path_statted_listed_hashed_or_opened_by_experiment_runner": False,
    }
    shard["receipt"] = {
        "algorithm": "sha256",
        "payload_scope": "canonical_mechanics_shard_without_receipt",
        "payload_sha256": canonical_sha256(shard),
    }
    path = output_dir / f"mechanics-shard-{rank}.json"
    _atomic_signed_json(path, shard)
    return shard


def _load_shards(
    output_dir: Path,
    source_rows: Sequence[Mapping[str, Any]],
    *,
    binding: Mapping[str, Any],
) -> tuple[list[Mapping[str, Any]], list[Mapping[str, Any]]]:
    expected_sources = {int(row["source_index"]) for row in source_rows}
    rows: list[Mapping[str, Any]] = []
    provenance = []
    for rank in range(WORLD_SIZE):
        path = output_dir / f"mechanics-shard-{rank}.json"
        shard = json.loads(path.read_text(encoding="utf-8"))
        _validate_receipt(
            shard,
            payload_scope="canonical_mechanics_shard_without_receipt",
            description=f"Continuous mechanics shard {rank}",
        )
        assigned_source_rows = source_rows[rank::WORLD_SIZE]
        expected_rank_sources = [
            int(row["source_index"]) for row in assigned_source_rows
        ]
        expected_source_binding = canonical_sha256(
            [
                {
                    "source_index": row["source_index"],
                    "row_sha256": row["row_sha256"],
                    "donor_source_index": row["donor_source_index"],
                    "donor_row_sha256": row["donor_row_sha256"],
                }
                for row in assigned_source_rows
            ]
        )
        shard_sources = [int(row["source_index"]) for row in shard.get("rows", ())]
        if (
            shard.get("schema") != SHARD_SCHEMA
            or shard.get("rank") != rank
            or shard.get("world_size") != WORLD_SIZE
            or shard.get("assignment")
            != "sorted_mechanics_ordinal_round_robin"
            or len(shard.get("rows", ())) != ROWS_PER_RANK
            or shard.get("assigned_source_indices") != expected_rank_sources
            or shard_sources != expected_rank_sources
            or shard.get("assigned_source_binding") != expected_source_binding
            or shard.get("binding") != dict(binding)
            or shard.get("mechanics_rows_opened") != ROWS
            or shard.get(
                "causal_path_statted_listed_hashed_or_opened_by_experiment_runner"
            )
            is not False
        ):
            raise ValueError("Continuous mechanics shard contract differs")
        rows.extend(shard["rows"])
        provenance.append(
            {
                "path": str(path),
                "rank": rank,
                "rows": len(shard["rows"]),
                "source_indices": shard_sources,
                "assigned_source_binding": expected_source_binding,
                "sha256": sha256_file(path),
                "receipt": shard["receipt"]["payload_sha256"],
            }
        )
    sources = [int(row["source_index"]) for row in rows]
    if set(sources) != expected_sources or len(set(sources)) != ROWS:
        raise ValueError("Continuous mechanics shard source coverage differs")
    return sorted(rows, key=lambda row: int(row["source_index"])), provenance


def _validate_result(path: Path) -> Mapping[str, Any]:
    result = json.loads(path.read_text(encoding="utf-8"))
    _validate_receipt(
        result,
        payload_scope="canonical_result_without_receipt",
        description="Continuous mechanics result",
    )
    if not isinstance(result.get("passed"), bool):
        raise ValueError("Continuous mechanics result passed flag differs")
    passed = result["passed"]
    expected_status = (
        "continuous_write_mechanics_passed_causal_protocol_draft_authorized"
        if passed
        else "continuous_write_mechanics_failed_fixed_map_gain_family_retired"
    )
    analysis = result.get("analysis", {})
    hardware_result = result.get("hardware", {})
    execution = result.get("execution", {})
    provenance = result.get("authorization_provenance", {})
    firewall = result.get("firewall", {})
    code_bindings = result.get("code_bindings", {})
    feature_provenance = result.get("feature_provenance", {})
    launch = json.loads(LAUNCH_BINDING.read_text(encoding="utf-8"))
    shard_provenance = feature_provenance.get("shards", ())
    if (
        result.get("schema") != SCHEMA
        or result.get("status") != expected_status
        or result.get("protocol_payload_sha256") != PROTOCOL_PAYLOAD_SHA256
        or result.get("protocol_file_sha256") != PROTOCOL_FILE_SHA256
        or result.get("mechanics_evaluation_calls") != 1
        or analysis.get("evaluation_calls") != 1
        or analysis.get("passed") is not passed
        or result.get("rows") != ROWS
        or result.get("modules") != MODULES
        or result.get("causal_protocol_drafting_authorized") is not passed
        or result.get("causal_bytes_open_authorized") is not False
        or result.get("causal_authorized") is not False
        or result.get("model_or_adapter_training_authorized") is not False
        or result.get("generation_authorized") is not False
        or result.get("native_benchmark_authorized") is not False
        or result.get("materialization_bundles_opened") != ["mechanics"]
        or hardware_result.get("world_size") != WORLD_SIZE
        or hardware_result.get("backend") != "nccl"
        or hardware_result.get("control_backend") != "gloo"
        or not hardware.four_distinct_a100s(hardware_result.get("rank_devices", ()))
        or execution.get("assignment")
        != "sorted_mechanics_ordinal_round_robin"
        or execution.get("rows_per_rank") != ROWS_PER_RANK
        or execution.get("model_parameters_updated") is not False
        or execution.get("adapter_parameters_updated") is not False
        or execution.get("generation") is not False
        or execution.get("parameter_versions_unchanged") is not True
        or not isinstance(execution.get("parameter_versions_sha256"), str)
        or provenance.get("retrieval_result_sha256") != RETRIEVAL_RESULT_SHA256
        or provenance.get("retrieval_result_receipt") != RETRIEVAL_RESULT_RECEIPT
        or provenance.get("frozen_map_file_sha256") != MAP_FILE_SHA256
        or provenance.get("frozen_map_digest") != MAP_DIGEST
        or provenance.get("manifest_file_sha256") != MANIFEST_FILE_SHA256
        or provenance.get("manifest_receipt") != MANIFEST_RECEIPT
        or provenance.get("mechanics_file_sha256") != MECHANICS_FILE_SHA256
        or provenance.get("launch_receipt")
        != launch.get("receipt", {}).get("payload_sha256")
        or provenance.get("authorized_code_commit")
        != launch.get("authorized_code_commit")
        or code_bindings.get("runner_sha256")
        != sha256_file(Path(__file__).resolve())
        or code_bindings.get("dependencies") != dependency_bindings()
        or len(shard_provenance) != WORLD_SIZE
        or [item.get("rank") for item in shard_provenance]
        != list(range(WORLD_SIZE))
        or firewall.get("mechanics_rows_decoded_tokenized_forwarded_and_scored")
        != ROWS
        or firewall.get("causal_rows_decoded_tokenized_forwarded_or_scored") != 0
        or firewall.get(
            "causal_path_statted_listed_hashed_or_opened_by_experiment_runner"
        )
        is not False
    ):
        raise ValueError("Continuous mechanics result contract differs")
    return result


def run(
    *,
    base_model: Path,
    materialization_root: Path,
    output_dir: Path,
) -> Mapping[str, Any]:
    context = distributed.initialize_distributed_training(
        "cuda", required_world_size=WORLD_SIZE, timeout_seconds=TIMEOUT_SECONDS
    )
    if context is None:
        raise RuntimeError("Run with torchrun --nproc_per_node=4")
    try:
        def validate_preflight() -> tuple[
            Mapping[str, Any],
            Mapping[str, Any],
            Mapping[str, Any],
            Mapping[str, Any],
            Mapping[str, Any],
        ]:
            if (
                context.world_size != WORLD_SIZE
                or context.backend != "nccl"
                or context.control_backend != "gloo"
                or not hardware.four_distinct_a100s(context.rank_devices)
            ):
                raise RuntimeError("Continuous mechanics requires four distinct A100s")
            if os.environ.get("HF_ENDPOINT") != HF_ENDPOINT:
                raise RuntimeError(f"HF_ENDPOINT must be exactly {HF_ENDPOINT}")
            protocol = validate_protocol(base_model)
            launch = validate_launch_binding(protocol)
            retrieval_result = validate_retrieval_authorization()
            source_audit = exact_v5.validate_execution_source()
            manifest = retrieval._load_manifest_only(
                materialization_root, protocol
            )
            return protocol, launch, retrieval_result, source_audit, manifest

        protocol, launch, retrieval_result, source_audit, manifest = _consensual_operation(
            context,
            phase="continuous-mechanics-preflight-without-protected-access",
            operation=validate_preflight,
        )
        preflight_binding = canonical_sha256(
            {
                "protocol": protocol["receipt"]["payload_sha256"],
                "launch": launch["receipt"]["payload_sha256"],
                "retrieval": retrieval_result["receipt"]["payload_sha256"],
                "source": source_audit,
                "manifest": manifest["receipt"]["payload_sha256"],
            }
        )
        distributed.require_consensus(
            context, preflight_binding, description="continuous mechanics preflight"
        )

        def load_runtime() -> tuple[
            torch.nn.Module,
            Any,
            Mapping[str, Any],
            Mapping[str, Any],
            Mapping[str, Any],
            Mapping[str, Any],
            tuple[tuple[str, int], ...],
        ]:
            torch.manual_seed(SEED)
            torch.cuda.manual_seed_all(SEED)
            model, tokenizer, model_audit = exact_v5.load_exact_v5_model(
                base_model, device=context.device
            )
            model.eval()
            modules = causal_train.ordered_modules(model)
            module_names = tuple(name for name, _ in modules)
            maps = load_frozen_maps(module_names)
            install_audit = integration.install(
                model,
                rank=MAP_RANK,
                seed=SEED,
                k_gain=K_GAIN,
                a_gain=A_GAIN,
                b_gain=B_GAIN,
                trainable_map=False,
            )
            for name, module in modules:
                module.rwkv_continuous_write_conditioner.load_frozen_map(
                    maps[name].down, maps[name].up
                )
            integration.set_mode(model, integration.CONTINUOUS_MODE)
            integration.set_capture(model, True)
            feature_observer = install_feature_observer(modules)
            read_observer = retrieval.install_read_observer(model)
            read_invocation_observer = install_read_invocation_observer(modules)
            if (
                len(modules) != MODULES
                or list(module_names)
                != protocol["all_module_inventory"]["ordered_module_names"]
                or install_audit["module_names"] != module_names
                or install_audit.get("effective_full64_override")
                != "one_shot_selected_key_only"
                or feature_observer["module_names"] != list(module_names)
                or read_observer["module_names"] != list(module_names)
                or read_invocation_observer["module_names"] != list(module_names)
                or integration.pending_effective_full64_address_override_names(model)
            ):
                raise RuntimeError("Continuous mechanics module installation differs")
            parity = _map_runtime_parity(modules, maps)
            if parity["passed"] is not True:
                raise RuntimeError("Continuous mechanics map runtime parity failed")
            for parameter in model.parameters():
                parameter.requires_grad_(False)
            if any(parameter.requires_grad for parameter in model.parameters()):
                raise RuntimeError("Continuous mechanics left trainable parameters")
            return (
                model,
                tokenizer,
                model_audit,
                install_audit,
                feature_observer,
                {
                    "read_observer": read_observer,
                    "read_invocation_observer": read_invocation_observer,
                    "map_runtime_parity": parity,
                },
                _parameter_versions(model),
            )

        (
            model,
            tokenizer,
            model_audit,
            install_audit,
            feature_observer,
            observer_audit,
            parameter_versions_before,
        ) = _consensual_operation(
            context, phase="continuous-mechanics-model-load", operation=load_runtime
        )
        dependency_bindings_before = dependency_bindings()
        dependency_bindings_digest = canonical_sha256(dependency_bindings_before)
        runner_sha256_before = sha256_file(Path(__file__).resolve())
        parameter_versions_before_digest = canonical_sha256(parameter_versions_before)
        distributed.require_consensus(
            context,
            parameter_versions_before_digest,
            description="continuous mechanics initial parameter versions",
        )

        def create_output() -> None:
            if not context.is_primary:
                return
            if output_dir.exists():
                raise ValueError(f"Continuous mechanics output must be fresh: {output_dir}")
            output_dir.mkdir(parents=True, exist_ok=False)

        _consensual_operation(
            context, phase="continuous-mechanics-output-create", operation=create_output
        )

        mechanics_rows = _consensual_operation(
            context,
            phase="continuous-mechanics-authorized-bundle-open",
            operation=lambda: _load_authorized_mechanics_bundle(
                materialization_root, manifest, protocol
            ),
        )
        mechanics_binding = canonical_sha256(
            [
                {
                    "source_index": row["source_index"],
                    "row_sha256": row["row_sha256"],
                    "donor_source_index": row["donor_source_index"],
                    "donor_row_sha256": row["donor_row_sha256"],
                }
                for row in mechanics_rows
            ]
        )
        distributed.require_consensus(
            context,
            mechanics_binding,
            description="continuous mechanics protected bundle",
        )
        examples = _consensual_operation(
            context,
            phase="continuous-mechanics-row-encoding",
            operation=lambda: retrieval._encode_rows(tokenizer, mechanics_rows),
        )
        modules = causal_train.ordered_modules(model)
        module_names = tuple(name for name, _ in modules)
        assigned_rows = mechanics_rows[context.process_rank :: WORLD_SIZE]
        if len(assigned_rows) != ROWS_PER_RANK:
            raise RuntimeError("Continuous mechanics rank assignment differs")

        def capture_natural_local() -> Mapping[int, Any]:
            cache: dict[int, Any] = {}
            for ordinal, row in enumerate(assigned_rows, start=1):
                source = int(row["source_index"])
                batch = evolution.collate_native_examples(
                    [examples[source]],
                    pad_token_id=int(tokenizer.pad_token_id),
                    device=context.device,
                )
                state, audit, address = capture_write_condition(
                    model,
                    batch,
                    modules,
                    mode=integration.CONTINUOUS_MODE,
                    override=None,
                    reference_mode="none",
                )
                _clear_feature_references(modules)
                cache[source] = {
                    "state": state,
                    "address": address,
                    "audit": audit,
                }
                print(
                    f"CONTINUOUS_MECHANICS_NATURAL rank={context.process_rank} "
                    f"row={source} ordinal={ordinal}/{ROWS_PER_RANK}",
                    flush=True,
                )
            return cache

        local_natural = _consensual_operation(
            context,
            phase="continuous-mechanics-natural-capture",
            operation=capture_natural_local,
        )
        natural_cache = _consensual_operation(
            context,
            phase="continuous-mechanics-natural-cache-gather",
            operation=lambda: _gather_natural_cache(context, local_natural),
        )
        natural_digest = canonical_sha256(
            [
                {
                    "source_index": source,
                    "address_sha256": _tensor_digest(natural_cache[source]["address"]),
                    "projected_sha256": natural_cache[source]["audit"][
                        "projected_sha256"
                    ],
                    "recurrent_sha256": natural_cache[source]["audit"][
                        "recurrent_sha256"
                    ],
                }
                for source in sorted(natural_cache)
            ]
        )
        distributed.require_consensus(
            context,
            natural_digest,
            description="continuous mechanics natural cache",
        )
        shard_binding = {
            "protocol_payload_sha256": PROTOCOL_PAYLOAD_SHA256,
            "protocol_file_sha256": PROTOCOL_FILE_SHA256,
            "launch_receipt": launch["receipt"]["payload_sha256"],
            "preflight_binding": preflight_binding,
            "mechanics_binding": mechanics_binding,
            "map_digest": MAP_DIGEST,
            "manifest_receipt": MANIFEST_RECEIPT,
            "mechanics_file_sha256": MECHANICS_FILE_SHA256,
            "natural_cache_digest": natural_digest,
            "runner_sha256": runner_sha256_before,
            "dependency_bindings_sha256": dependency_bindings_digest,
        }

        def evaluate_local_rows() -> Mapping[str, Any]:
            evaluated = []
            ordered_sources = [int(row["source_index"]) for row in mechanics_rows]
            rows_by_source = {
                int(row["source_index"]): row for row in mechanics_rows
            }
            for ordinal, row in enumerate(assigned_rows, start=1):
                source = int(row["source_index"])
                donor = int(row["donor_source_index"])
                metrics = evaluate_row(
                    model,
                    examples[source],
                    examples[donor],
                    source_index=source,
                    donor_source_index=donor,
                    natural_cache=natural_cache,
                    ordered_sources=ordered_sources,
                    module_names=module_names,
                    pad_token_id=int(tokenizer.pad_token_id),
                    device=context.device,
                )
                if (
                    row["donor_row_sha256"]
                    != rows_by_source[donor]["row_sha256"]
                ):
                    raise RuntimeError("Continuous mechanics donor binding differs")
                evaluated.append(metrics)
                print(
                    f"CONTINUOUS_MECHANICS_ROW rank={context.process_rank} "
                    f"row={source} ordinal={ordinal}/{ROWS_PER_RANK}",
                    flush=True,
                )
            return _write_shard(
                output_dir,
                rank=context.process_rank,
                rows=evaluated,
                binding=shard_binding,
                assigned_source_rows=assigned_rows,
            )

        _consensual_operation(
            context,
            phase="continuous-mechanics-counterfactual-evaluation",
            operation=evaluate_local_rows,
        )
        evaluated_rows, shard_provenance = _consensual_operation(
            context,
            phase="continuous-mechanics-shard-validation",
            operation=lambda: _load_shards(
                output_dir, mechanics_rows, binding=shard_binding
            ),
        )
        evaluated_digest = canonical_sha256(evaluated_rows)
        distributed.require_consensus(
            context,
            evaluated_digest,
            description="continuous mechanics evaluated rows",
        )
        def validate_parameter_immutability() -> str:
            parameter_versions_after = _parameter_versions(model)
            if parameter_versions_after != parameter_versions_before:
                raise RuntimeError("Continuous mechanics mutated model parameters")
            return canonical_sha256(parameter_versions_after)

        parameter_versions_after_digest = _consensual_operation(
            context,
            phase="continuous-mechanics-parameter-immutability",
            operation=validate_parameter_immutability,
        )
        distributed.require_consensus(
            context,
            parameter_versions_after_digest,
            description="continuous mechanics final parameter versions",
        )
        if parameter_versions_after_digest != parameter_versions_before_digest:
            raise RuntimeError("Continuous mechanics parameter version digest differs")
        del model
        torch.cuda.empty_cache()

        result_error: BaseException | None = None
        if context.is_primary:
            try:
                analysis = mechanics_analysis(evaluated_rows)
                passed = bool(analysis["passed"])
                dependencies_end = dependency_bindings()
                runner_sha256_end = sha256_file(Path(__file__).resolve())
                if (
                    dependencies_end != dependency_bindings_before
                    or runner_sha256_end != runner_sha256_before
                    or sha256_file(PROTOCOL) != PROTOCOL_FILE_SHA256
                ):
                    raise RuntimeError("Continuous mechanics code binding changed")
                result: dict[str, Any] = {
                    "schema": SCHEMA,
                    "status": (
                        "continuous_write_mechanics_passed_causal_protocol_draft_authorized"
                        if passed
                        else "continuous_write_mechanics_failed_fixed_map_gain_family_retired"
                    ),
                    "passed": passed,
                    "causal_protocol_drafting_authorized": passed,
                    "causal_bytes_open_authorized": False,
                    "causal_authorized": False,
                    "model_or_adapter_training_authorized": False,
                    "generation_authorized": False,
                    "native_benchmark_authorized": False,
                    "protocol_payload_sha256": PROTOCOL_PAYLOAD_SHA256,
                    "protocol_file_sha256": PROTOCOL_FILE_SHA256,
                    "protocol_objective": protocol["objective"],
                    "mechanics_evaluation_calls": analysis["evaluation_calls"],
                    "analysis": analysis,
                    "rows": ROWS,
                    "modules": MODULES,
                    "feature_provenance": {
                        "natural_cache_digest": natural_digest,
                        "evaluated_rows_digest": evaluated_digest,
                        "shards": shard_provenance,
                        "shard_binding": shard_binding,
                    },
                    "hardware": {
                        "world_size": context.world_size,
                        "rank_devices": list(context.rank_devices),
                        "backend": context.backend,
                        "control_backend": context.control_backend,
                        "timeout_seconds": TIMEOUT_SECONDS,
                    },
                    "execution": {
                        "hf_endpoint": os.environ.get("HF_ENDPOINT"),
                        "assignment": "sorted_mechanics_ordinal_round_robin",
                        "rows_per_rank": ROWS_PER_RANK,
                        "model_parameters_updated": False,
                        "adapter_parameters_updated": False,
                        "generation": False,
                        "maps_loaded_before_mechanics_bytes_opened": True,
                        "parameter_versions_unchanged": True,
                        "parameter_versions_sha256": parameter_versions_after_digest,
                    },
                    "source_audit": source_audit,
                    "model_audit": {
                        **dict(model_audit),
                        "continuous_write_install": install_audit,
                        "feature_observer": feature_observer,
                        **dict(observer_audit),
                        "parameters_trainable": False,
                    },
                    "authorization_provenance": {
                        "retrieval_result_sha256": RETRIEVAL_RESULT_SHA256,
                        "retrieval_result_receipt": RETRIEVAL_RESULT_RECEIPT,
                        "frozen_map_file_sha256": MAP_FILE_SHA256,
                        "frozen_map_digest": MAP_DIGEST,
                        "manifest_file_sha256": MANIFEST_FILE_SHA256,
                        "manifest_receipt": MANIFEST_RECEIPT,
                        "mechanics_file_sha256": MECHANICS_FILE_SHA256,
                        "launch_receipt": launch["receipt"]["payload_sha256"],
                        "authorized_code_commit": launch["authorized_code_commit"],
                    },
                    "materialization_bundles_opened": ["mechanics"],
                    "firewall": {
                        "manifest_opened_before_protected_access": True,
                        "mechanics_bytes_opened_after_signed_preflight": True,
                        "mechanics_rows_decoded_tokenized_forwarded_and_scored": ROWS,
                        "causal_path_statted_listed_hashed_or_opened_by_experiment_runner": False,
                        "causal_rows_decoded_tokenized_forwarded_or_scored": 0,
                        "publisher_validation_test_hard32_strength_holdout_opened": False,
                    },
                    "claim_boundary": {
                        "mechanics_only": True,
                        "causal_preference_claimed": False,
                        "native_benchmark_gain_claimed": False,
                        "sota_claimed": False,
                        "read_feedback_or_repeated_write_scan_used": False,
                    },
                    "code_bindings": {
                        "runner_sha256": runner_sha256_end,
                        "dependencies": dependencies_end,
                    },
                }
                result["receipt"] = {
                    "algorithm": "sha256",
                    "payload_scope": "canonical_result_without_receipt",
                    "payload_sha256": canonical_sha256(result),
                }
                _atomic_signed_json(output_dir / "result.json", result)
            except BaseException as caught:
                result_error = caught
        distributed.phase_consensus(
            context,
            phase="continuous-mechanics-result-analysis-and-save",
            error=result_error,
        )
        result = _consensual_operation(
            context,
            phase="continuous-mechanics-all-rank-result-validation",
            operation=lambda: _validate_result(output_dir / "result.json"),
        )
        distributed.require_consensus(
            context,
            result["receipt"]["payload_sha256"],
            description="continuous mechanics result receipt",
        )
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
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    result = run(
        base_model=args.base_model.expanduser().resolve(strict=True),
        materialization_root=args.materialization_root.expanduser().resolve(
            strict=True
        ),
        output_dir=args.output_dir.expanduser().resolve(),
    )
    return 0 if result.get("passed") is True else 1


if __name__ == "__main__":
    raise SystemExit(main())
