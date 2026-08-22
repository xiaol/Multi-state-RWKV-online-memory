#!/usr/bin/env python3
"""Train a narrow read path on open FIT rows before sealed causal evaluation."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import random
import subprocess
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

from experiments.rethinking_rwkv_ms_gemma import (  # noqa: E402
    run_natural_memory_native_rwkv_continuous_write_mechanics as mechanics,
)
from experiments.rethinking_rwkv_ms_gemma import (  # noqa: E402
    natural_memory_distributed as distributed,
)
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
    rwkv_continuous_write_integration as integration,
)
from deltamem.core.delta import reset_delta_mem_states  # noqa: E402


SCHEMA = "rwkv_ms_natural_memory_native_rwkv_continuous_write_causal_train.v1"
STEP_SCHEMA = "rwkv_ms_natural_memory_native_rwkv_continuous_write_fit_train_step.v1"
PROTOCOL = SCRIPT_DIR / "natural_memory_native_rwkv_continuous_write_causal_train_protocol_v1.json"
PROTOCOL_PAYLOAD_SHA256 = "562469900b0d4a70214918a4f58ac9b272352fe7b05c162a4c9a0bb0b94742b8"
PROTOCOL_FILE_SHA256 = "c98ba4c8417701a0c7bf2d65a8fb4ca7c80ecfbc9556010fe3366c53906de83d"
LAUNCH = SCRIPT_DIR / "natural_memory_native_rwkv_continuous_write_causal_train_launch_v1.json"
WORLD_SIZE = 4
FIT_ROWS = 64
FIT_PAIRS = 32
UPDATES = 8
LOCAL_PAIRS = 1
LOCAL_ROWS = 2
GLOBAL_BATCH_SIZE = WORLD_SIZE * LOCAL_ROWS
SEED = 157
LEARNING_RATE = 2e-5
MAX_GRAD_NORM = 0.02
SMOOTH_HINGE_TEMPERATURE = 0.05
HF_ENDPOINT = "https://hf-mirror.com"
TRAINABLE_SUFFIXES = (
    ".hrm_rwkv7_core.output.weight",
    ".delta_o_proj",
)
EXPECTED_TENSORS_PER_SUFFIX = mechanics.MODULES
EXPECTED_TRAINABLE_TENSORS = len(TRAINABLE_SUFFIXES) * EXPECTED_TENSORS_PER_SUFFIX
BRANCH_SPECS = (
    ("correct", 0.0, 1.0),
    ("paired_donor_recurrent", 0.02, 0.25),
    ("layer_rolled_recurrent", 0.05, 0.25),
    ("zero_recurrent", 0.05, 0.25),
)
MECHANICS_RESULT = mechanics.DEFAULT_OUTPUT / "result.json"
MECHANICS_RESULT_FILE_SHA256 = "a7215ff987f06a369e19ea5b62e54ae2e99b018b9dbed15616f964806e811456"
MECHANICS_RESULT_RECEIPT = "2621b0d7773f7931fda80676774697fcc4c059abf49f8ebbad683f19f34c1a95"
MANIFEST_FILE_SHA256 = "c437a7d1f2b850a730fe5b28a08ae32ba02678561bb1265a4eef55bda7f4d468"
MANIFEST_RECEIPT = "99a878493c3848c96624e2ad658842c99e69769b4a1721b5854ad25af8d0bee2"
FIT_FILE_SHA256 = "4984e7de044f7befc2c3fdba8a0d8c08f627dcc4b168abbd8090393cca49c2fc"
FIT_PAYLOAD_SHA256 = "41d9e117997e60b808895f4ae8ea63a6fa643ba97d44c6878be5b442eeb76318"
FIT_SOURCE_INDICES_SHA256 = "0712052436346b194e2219147f511ec22e10ae6ab688553f773dbe53c9a94636"
FIT_QUALIFIED_MAPPING_PAIRS_SHA256 = "82109bd7fd84185154d0026cf0b20da52be94faf192b48565866bd6c716488db"
CAUSAL_ROWS = 32
CAUSAL_FILE_SHA256 = "5920ca8c688f4c26e8b55c5c48eefb7c067016bb931dd3b5c210edd1f4d3e925"
CAUSAL_PAYLOAD_SHA256 = "f2c4259366d9376bf52287d1f770880fdfaf44552e88ffb77481069c674cb067"
CAUSAL_SOURCE_INDICES_SHA256 = "fbc9e6314c3248c023458393293884277f8afff2a0bc738f1086fb6648e50024"
CAUSAL_QUALIFIED_MAPPING_PAIRS_SHA256 = "3a039153383f03178de44a4a07808ea31acfc25c38f473df5603f60c58884ab4"
DEFAULT_BASE_MODEL = mechanics.DEFAULT_BASE_MODEL
DEFAULT_MATERIALIZATION = mechanics.DEFAULT_MATERIALIZATION
DEFAULT_OUTPUT = SCRIPT_DIR / (
    "local_artifacts/natural_memory_native_rwkv_continuous_write_causal_train_v1"
)

evolution = exact_v5.evolution
causal_train = exact_v5.causal_train
hardware = exact_v5.hardware


@dataclass(frozen=True)
class PairScheduleStep:
    step: int
    rank_pairs: tuple[tuple[int, int], ...]
    payload_sha256: str


def canonical_sha256(value: Any) -> str:
    return distributed.canonical_sha256(value)


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


def _dependency_paths() -> Mapping[str, Path]:
    return {
        "mechanics_result_validator_and_snapshot_helpers": Path(mechanics.__file__).resolve(),
        "retrieval_result_validator_and_runtime_helpers": Path(retrieval.__file__).resolve(),
        "bias_free_reduced_rank_ridge_and_metrics": Path(mechanics.alignment.__file__).resolve(),
        "full_address_latch_runtime_conditioner_and_override": Path(integration.__file__).resolve(),
        "dataset_qualified_component_disjoint_split": Path(mechanics.fit_split.__file__).resolve(),
        "five_file_open_fit_materializer_and_firewall": Path(materializer.__file__).resolve(),
        "strict_exact_v5_loader_and_source_validator": Path(exact_v5.__file__).resolve(),
        "signed_distributed_runtime": Path(distributed.__file__).resolve(),
        "signed_native_row_encoder_and_write_read_runtime": Path(evolution.__file__).resolve(),
        "ordered_module_state_intervention_and_serialized_graph_helpers": Path(causal_train.__file__).resolve(),
        "four_a100_hardware_validator": Path(hardware.__file__).resolve(),
        "exact_v5_adapter_topology_validator": Path(exact_v5.v5_eval.__file__).resolve(),
    }


def dependency_bindings() -> list[dict[str, str]]:
    return [
        {"role": role, "basename": path.name, "sha256": mechanics.sha256_file(path)}
        for role, path in _dependency_paths().items()
    ]


def _validate_source_dependencies(protocol: Mapping[str, Any]) -> None:
    declared = protocol.get("source_dependencies")
    if not isinstance(declared, list):
        raise ValueError("Continuous-write causal source dependencies are missing")
    by_role = {
        str(item.get("role")): item for item in declared if isinstance(item, Mapping)
    }
    paths = _dependency_paths()
    if set(by_role) != set(paths):
        raise ValueError("Continuous-write causal dependency closure differs")
    for role, path in paths.items():
        item = by_role[role]
        if item.get("basename") != path.name or item.get("sha256") != mechanics.sha256_file(path):
            raise ValueError(f"Continuous-write causal dependency differs: {role}")


def validate_protocol(base_model: Path = DEFAULT_BASE_MODEL) -> Mapping[str, Any]:
    if PROTOCOL_PAYLOAD_SHA256.startswith("TO_BE_FILLED") or PROTOCOL_FILE_SHA256.startswith(
        "TO_BE_FILLED"
    ):
        raise RuntimeError("Continuous-write FIT training protocol is not signed")
    if mechanics.sha256_file(PROTOCOL) != PROTOCOL_FILE_SHA256:
        raise ValueError("Continuous-write FIT training protocol file differs")
    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    _validate_receipt(
        protocol,
        payload_scope="canonical_protocol_without_receipt",
        description="Continuous-write causal protocol",
    )
    digest = protocol["receipt"]["payload_sha256"]
    schedule = protocol.get("fit_training_schedule", {})
    optimizer = protocol.get("optimizer_and_objective", {})
    whitelist = protocol.get("trainable_whitelist", {})
    endpoint = protocol.get("causal_endpoint", {})
    hardware_contract = protocol.get("hardware_and_runtime", {})
    launch_contract = protocol.get("launch_binding", {})
    authorization = protocol.get("authorization_basis", {})
    if (
        protocol.get("schema")
        != "rwkv_ms_natural_memory_native_rwkv_continuous_write_causal_train_protocol.v1"
        or digest != PROTOCOL_PAYLOAD_SHA256
        or authorization.get("mechanics_result_file_sha256")
        != MECHANICS_RESULT_FILE_SHA256
        or authorization.get("mechanics_result_receipt")
        != MECHANICS_RESULT_RECEIPT
        or schedule.get("fit_rows") != FIT_ROWS
        or schedule.get("undirected_donor_pairs") != FIT_PAIRS
        or schedule.get("optimizer_updates") != UPDATES
        or schedule.get("global_batch_rows") != GLOBAL_BATCH_SIZE
        or schedule.get("pairs_per_rank_per_update") != LOCAL_PAIRS
        or optimizer.get("learning_rate") != LEARNING_RATE
        or optimizer.get("maximum_gradient_norm") != MAX_GRAD_NORM
        or optimizer.get("smooth_hinge_temperature") != SMOOTH_HINGE_TEMPERATURE
        or whitelist.get("suffixes") != [suffix.removeprefix(".") for suffix in TRAINABLE_SUFFIXES]
        or whitelist.get("parameter_tensors") != EXPECTED_TRAINABLE_TENSORS
        or endpoint.get("rows") != CAUSAL_ROWS
        or endpoint.get("read_conditions")
        != [
            "continuous_correct",
            "zero_recurrent",
            "projected_only",
            "matched_donor_recurrent",
            "layer_rolled_recurrent",
        ]
        or hardware_contract.get("world_size") != WORLD_SIZE
        or hardware_contract.get("backend") != "nccl"
        or hardware_contract.get("control_backend") != "gloo"
        or hardware_contract.get("hf_endpoint") != HF_ENDPOINT
        or launch_contract.get("path") != LAUNCH.name
        or launch_contract.get("schema")
        != "rwkv_ms_natural_memory_native_rwkv_continuous_write_causal_train_launch.v1"
        or protocol.get("authorization_outputs", {}).get("causal_evaluation_authorized") is not True
    ):
        raise ValueError("Continuous-write FIT training protocol differs")
    if base_model.exists() and mechanics.sha256_file(base_model / "config.json") != protocol.get(
        "frozen_inputs", {}
    ).get("base_config_sha256"):
        raise ValueError("Continuous-write causal base configuration differs")
    _validate_source_dependencies(protocol)
    return protocol


def _git_output(*arguments: str) -> str:
    result = subprocess.run(
        ["git", *arguments],
        cwd=PROJECT_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def validate_launch_binding(protocol: Mapping[str, Any]) -> Mapping[str, Any]:
    if not LAUNCH.exists():
        raise RuntimeError("Continuous-write causal launch is not signed")
    launch = json.loads(LAUNCH.read_text(encoding="utf-8"))
    _validate_receipt(
        launch,
        payload_scope="canonical_launch_binding_without_receipt",
        description="Continuous-write causal launch binding",
    )
    code_commit = str(launch.get("authorized_code_commit", ""))
    if (
        launch.get("schema")
        != "rwkv_ms_natural_memory_native_rwkv_continuous_write_causal_train_launch.v1"
        or launch.get("code_parent_commit")
        != protocol.get("authorization_basis", {}).get("causal_code_parent_commit")
        or launch.get("protocol_payload_sha256") != PROTOCOL_PAYLOAD_SHA256
        or launch.get("protocol_file_sha256") != PROTOCOL_FILE_SHA256
        or launch.get("protocol_receipt") != protocol.get("receipt", {}).get("payload_sha256")
        or launch.get("runner_sha256") != mechanics.sha256_file(Path(__file__).resolve())
        or launch.get("dependency_bindings_sha256") != canonical_sha256(dependency_bindings())
        or launch.get("mechanics_result_file_sha256") != MECHANICS_RESULT_FILE_SHA256
        or launch.get("mechanics_result_receipt") != MECHANICS_RESULT_RECEIPT
        or launch.get("frozen_map_file_sha256") != mechanics.MAP_FILE_SHA256
        or launch.get("frozen_map_digest") != mechanics.MAP_DIGEST
        or launch.get("manifest_file_sha256") != MANIFEST_FILE_SHA256
        or launch.get("manifest_receipt") != MANIFEST_RECEIPT
        or launch.get("fit_file_sha256") != FIT_FILE_SHA256
        or launch.get("fit_payload_sha256") != FIT_PAYLOAD_SHA256
        or launch.get("fit_schedule_sha256")
        != protocol.get("fit_training_schedule", {}).get("canonical_schedule_payload_sha256")
        or launch.get("trainable_parameter_names_sha256")
        != protocol.get("trainable_whitelist", {}).get("sorted_parameter_names_sha256")
        or launch.get("causal_file_sha256") != CAUSAL_FILE_SHA256
        or launch.get("causal_payload_sha256") != CAUSAL_PAYLOAD_SHA256
        or launch.get("causal_source_indices_sha256") != CAUSAL_SOURCE_INDICES_SHA256
        or launch.get("causal_qualified_mapping_pairs_sha256")
        != CAUSAL_QUALIFIED_MAPPING_PAIRS_SHA256
        or launch.get("causal_assignment_policy")
        != "lexicographically_sorted_symmetric_pairs_rank_strided_four_pairs_per_rank"
        or launch.get("world_size") != WORLD_SIZE
        or launch.get("hf_endpoint") != HF_ENDPOINT
        or launch.get("causal_bytes_opened_before_launch") is not False
        or not code_commit
    ):
        raise ValueError("Continuous-write causal launch binding differs")
    head = _git_output("rev-parse", "HEAD")
    head_parent = _git_output("rev-parse", "HEAD^")
    code_parent = _git_output("rev-parse", f"{code_commit}^")
    runner_relative = Path(__file__).resolve().relative_to(PROJECT_ROOT).as_posix()
    protocol_relative = PROTOCOL.resolve().relative_to(PROJECT_ROOT).as_posix()
    launch_relative = LAUNCH.resolve().relative_to(PROJECT_ROOT).as_posix()
    committed_runner = subprocess.run(
        ["git", "show", f"{code_commit}:{runner_relative}"],
        cwd=PROJECT_ROOT,
        check=True,
        capture_output=True,
    ).stdout
    committed_protocol = subprocess.run(
        ["git", "show", f"{code_commit}:{protocol_relative}"],
        cwd=PROJECT_ROOT,
        check=True,
        capture_output=True,
    ).stdout
    committed_launch = subprocess.run(
        ["git", "show", f"{head}:{launch_relative}"],
        cwd=PROJECT_ROOT,
        check=True,
        capture_output=True,
    ).stdout
    if (
        head_parent != code_commit
        or code_parent != protocol.get("authorization_basis", {}).get("causal_code_parent_commit")
        or launch.get("code_parent_commit") != code_parent
        or hashlib.sha256(committed_runner).hexdigest() != launch.get("runner_sha256")
        or hashlib.sha256(committed_protocol).hexdigest() != PROTOCOL_FILE_SHA256
        or committed_launch != LAUNCH.read_bytes()
        or _git_output("diff", "--name-only", code_commit, "HEAD") != launch_relative
        or _git_output("diff", "--name-only", "HEAD")
        or _git_output("rev-parse", "origin/main") != head
    ):
        raise ValueError("Continuous-write causal two-commit launch differs")
    return launch


def validate_mechanics_authorization() -> Mapping[str, Any]:
    if mechanics.sha256_file(MECHANICS_RESULT) != MECHANICS_RESULT_FILE_SHA256:
        raise ValueError("Continuous-write mechanics result file differs")
    result = mechanics._validate_result(MECHANICS_RESULT)
    if (
        result.get("receipt", {}).get("payload_sha256")
        != MECHANICS_RESULT_RECEIPT
        or result.get("passed") is not True
        or result.get("status")
        != "continuous_write_mechanics_passed_causal_protocol_draft_authorized"
        or result.get("causal_protocol_drafting_authorized") is not True
        or result.get("causal_bytes_open_authorized") is not False
    ):
        raise ValueError("Continuous-write mechanics authorization differs")
    return result


def _load_manifest_only(materialization_root: Path) -> Mapping[str, Any]:
    path = materialization_root / "manifest.json"
    if mechanics.sha256_file(path) != MANIFEST_FILE_SHA256:
        raise ValueError("Continuous-write manifest file differs")
    manifest = json.loads(path.read_text(encoding="utf-8"))
    unsigned = dict(manifest)
    receipt = unsigned.pop("receipt", {})
    bundles = manifest.get("file_inventory", {}).get("bundles", {})
    fit = bundles.get("fit", {})
    causal = bundles.get("causal", {})
    if (
        receipt.get("payload_sha256") != MANIFEST_RECEIPT
        or canonical_sha256(unsigned) != MANIFEST_RECEIPT
        or fit.get("path") != "fit.jsonl"
        or fit.get("rows") != FIT_ROWS
        or fit.get("sha256") != FIT_FILE_SHA256
        or fit.get("payload_sha256") != FIT_PAYLOAD_SHA256
        or fit.get("source_indices_sha256") != FIT_SOURCE_INDICES_SHA256
        or fit.get("qualified_mapping_pairs_sha256")
        != FIT_QUALIFIED_MAPPING_PAIRS_SHA256
        or causal.get("path") != "causal.jsonl"
        or causal.get("rows") != CAUSAL_ROWS
        or causal.get("sha256") != CAUSAL_FILE_SHA256
        or causal.get("payload_sha256") != CAUSAL_PAYLOAD_SHA256
        or causal.get("source_indices_sha256") != CAUSAL_SOURCE_INDICES_SHA256
        or causal.get("qualified_mapping_pairs_sha256")
        != CAUSAL_QUALIFIED_MAPPING_PAIRS_SHA256
        or manifest.get("protected_splits_opened") != []
    ):
        raise ValueError("Continuous-write FIT manifest binding differs")
    return manifest


def _load_fit_rows(
    materialization_root: Path,
    manifest: Mapping[str, Any],
) -> list[dict[str, Any]]:
    rows = materializer._read_bundle(materialization_root, manifest, "fit")
    if len(rows) != FIT_ROWS:
        raise ValueError("Continuous-write FIT row count differs")
    return rows


def build_pair_schedule(rows: Sequence[Mapping[str, Any]]) -> tuple[PairScheduleStep, ...]:
    if len(rows) != FIT_ROWS:
        raise ValueError("Continuous-write FIT schedule requires 64 rows")
    mapping = {
        int(row["source_index"]): int(row["donor_source_index"])
        for row in rows
    }
    if len(mapping) != FIT_ROWS or any(
        source == donor or mapping.get(donor) != source
        for source, donor in mapping.items()
    ):
        raise ValueError("Continuous-write FIT donor mapping is not symmetric")
    pairs = sorted({tuple(sorted((source, donor))) for source, donor in mapping.items()})
    if len(pairs) != FIT_PAIRS:
        raise ValueError("Continuous-write FIT pair count differs")
    random.Random(SEED).shuffle(pairs)
    steps: list[PairScheduleStep] = []
    for step_index in range(UPDATES):
        rank_pairs = tuple(
            pairs[step_index * WORLD_SIZE : (step_index + 1) * WORLD_SIZE]
        )
        payload = {
            "step": step_index + 1,
            "rank_pairs": [list(pair) for pair in rank_pairs],
            "policy": "one_complete_symmetric_pair_per_rank",
        }
        steps.append(
            PairScheduleStep(
                step=step_index + 1,
                rank_pairs=rank_pairs,
                payload_sha256=canonical_sha256(payload),
            )
        )
    if len(steps) != UPDATES or {
        source for step in steps for pair in step.rank_pairs for source in pair
    } != set(mapping):
        raise RuntimeError("Continuous-write FIT schedule coverage differs")
    return tuple(steps)


def validate_fit_schedule_binding(
    schedule: Sequence[PairScheduleStep], protocol: Mapping[str, Any]
) -> str:
    payload = [
        {
            "step": step.step,
            "rank_pairs": [list(pair) for pair in step.rank_pairs],
            "policy": "one_complete_symmetric_pair_per_rank",
        }
        for step in schedule
    ]
    contract = protocol.get("fit_training_schedule", {})
    digest = canonical_sha256(payload)
    if (
        payload != contract.get("canonical_schedule_payload")
        or digest != contract.get("canonical_schedule_payload_sha256")
    ):
        raise ValueError("Continuous-write FIT signed schedule differs")
    return digest


def configure_trainable_parameters(
    model: torch.nn.Module,
) -> tuple[tuple[tuple[str, torch.nn.Parameter], ...], Mapping[str, Any]]:
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    family_counts = {suffix: 0 for suffix in TRAINABLE_SUFFIXES}
    selected: list[tuple[str, torch.nn.Parameter]] = []
    for name, parameter in model.named_parameters():
        for suffix in TRAINABLE_SUFFIXES:
            if name.endswith(suffix):
                parameter.requires_grad_(True)
                if parameter.dtype != torch.float32:
                    parameter.data = parameter.data.float()
                family_counts[suffix] += 1
                selected.append((name, parameter))
                break
    named_trainable = distributed.stable_named_parameters(selected)
    names = [name for name, _ in named_trainable]
    passed = (
        len(named_trainable) == EXPECTED_TRAINABLE_TENSORS
        and all(
            family_counts[suffix] == EXPECTED_TENSORS_PER_SUFFIX
            for suffix in TRAINABLE_SUFFIXES
        )
        and all(parameter.dtype == torch.float32 for _, parameter in named_trainable)
        and all(
            not parameter.requires_grad
            for name, parameter in model.named_parameters()
            if name not in set(names)
        )
    )
    audit = {
        "parameter_tensors": len(named_trainable),
        "parameter_names_sha256": canonical_sha256(names),
        "family_counts": family_counts,
        "writer_and_receptance_frozen": True,
        "frozen_map_and_gains_frozen": True,
        "passed": passed,
    }
    if not passed:
        raise RuntimeError(f"Continuous-write trainable isolation failed: {audit!r}")
    return named_trainable, audit


def trainable_sha256(
    named_trainable: Sequence[tuple[str, torch.nn.Parameter]],
) -> str:
    return distributed.tensor_mapping_sha256(
        {name: parameter.detach() for name, parameter in named_trainable}
    )


def frozen_parameter_versions(
    model: torch.nn.Module, trainable_names: Sequence[str]
) -> tuple[tuple[str, int], ...]:
    excluded = set(trainable_names)
    return tuple(
        (name, int(parameter._version))
        for name, parameter in sorted(model.named_parameters())
        if name not in excluded
    )


def smooth_hinge_wrong_coefficient(
    *,
    correct_ce: float,
    wrong_ce: float,
    margin: float,
    weight: float,
    temperature: float = SMOOTH_HINGE_TEMPERATURE,
) -> float:
    values = (correct_ce, wrong_ce, margin, weight, temperature)
    if not all(math.isfinite(value) for value in values) or temperature <= 0.0:
        raise ValueError("Smooth-hinge inputs must be finite with positive temperature")
    logit = (margin - (wrong_ce - correct_ce)) / temperature
    sigmoid = 1.0 if logit >= 40.0 else 0.0 if logit <= -40.0 else 1.0 / (1.0 + math.exp(-logit))
    return -weight * sigmoid


def serialized_branch_coefficients(
    branch_ce: Mapping[str, float],
) -> Mapping[str, float]:
    if set(branch_ce) != {name for name, _, _ in BRANCH_SPECS}:
        raise ValueError("Serialized branch CE coverage differs")
    correct = float(branch_ce["correct"])
    coefficients: dict[str, float] = {}
    correct_coefficient = 1.0
    for name, margin, weight in BRANCH_SPECS[1:]:
        coefficient = smooth_hinge_wrong_coefficient(
            correct_ce=correct,
            wrong_ce=float(branch_ce[name]),
            margin=margin,
            weight=weight,
        )
        coefficients[name] = coefficient
        correct_coefficient += -coefficient
    coefficients["correct"] = correct_coefficient
    return coefficients


@torch.no_grad()
def _capture_natural_snapshot(
    model: torch.nn.Module,
    batch: Any,
    modules: Sequence[tuple[str, Any]],
) -> dict[str, dict[str, torch.Tensor]]:
    integration.clear_effective_full64_address_overrides(model)
    integration.set_mode(model, integration.CONTINUOUS_MODE)
    reset_delta_mem_states(model)
    evolution._native_write(model, batch, dtype=torch.bfloat16)
    snapshot = mechanics.clone_online_state_cpu(modules)
    if integration.pending_effective_full64_address_override_names(model):
        raise RuntimeError("Natural FIT snapshot left an address override")
    reset_delta_mem_states(model)
    return snapshot


def _read_snapshot(
    model: torch.nn.Module,
    batch: Any,
    modules: Sequence[tuple[str, Any]],
    *,
    projected: Mapping[str, Mapping[str, torch.Tensor]],
    recurrent: Mapping[str, Mapping[str, torch.Tensor]],
) -> torch.Tensor:
    projected_device = mechanics.state_subset_to_device(
        projected, mechanics.PROJECTED_ATTRIBUTES, batch.read_input_ids.device
    )
    recurrent_device = mechanics.state_subset_to_device(
        recurrent, mechanics.RECURRENT_ATTRIBUTES, batch.read_input_ids.device
    )
    projected_before = mechanics.state_sha256(
        projected_device, mechanics.PROJECTED_ATTRIBUTES
    )
    recurrent_before = mechanics.state_sha256(
        recurrent_device, mechanics.RECURRENT_ATTRIBUTES
    )
    projected_objects = mechanics._state_objects_and_versions(
        projected_device, mechanics.PROJECTED_ATTRIBUTES
    )
    recurrent_objects = mechanics._state_objects_and_versions(
        recurrent_device, mechanics.RECURRENT_ATTRIBUTES
    )
    reset_delta_mem_states(model)
    fixed = causal_train.install_intervened_state(
        modules,
        projected=projected_device,
        recurrent=recurrent_device,
        rotate_recurrent_layers=False,
    )
    if not fixed or not mechanics._module_references_exact(
        modules, projected_device, recurrent_device
    ):
        raise RuntimeError("Continuous-write FIT projected carrier changed")
    logits = evolution._native_read(model, batch, dtype=torch.bfloat16)
    if (
        not mechanics._module_references_exact(modules, projected_device, recurrent_device)
        or mechanics.state_sha256(projected_device, mechanics.PROJECTED_ATTRIBUTES)
        != projected_before
        or mechanics.state_sha256(recurrent_device, mechanics.RECURRENT_ATTRIBUTES)
        != recurrent_before
        or mechanics._state_objects_and_versions(
            projected_device, mechanics.PROJECTED_ATTRIBUTES
        )
        != projected_objects
        or mechanics._state_objects_and_versions(
            recurrent_device, mechanics.RECURRENT_ATTRIBUTES
        )
        != recurrent_objects
    ):
        raise RuntimeError("Continuous-write FIT read mutated its snapshot")
    return logits


def _mean_ce(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    loss_sum, tokens = distributed.answer_loss_sum_and_count(logits, labels)
    return loss_sum / tokens


def _accumulate_reduced_branch(
    named_trainable: Sequence[tuple[str, torch.nn.Parameter]],
    accumulator: dict[str, torch.Tensor],
) -> None:
    with torch.no_grad():
        for name, parameter in named_trainable:
            if parameter.grad is None:
                raise RuntimeError(f"Serialized branch omitted gradient: {name}")
            if name not in accumulator:
                accumulator[name] = parameter.grad.detach().clone()
            else:
                accumulator[name].add_(parameter.grad)


def _install_accumulated_gradients(
    named_trainable: Sequence[tuple[str, torch.nn.Parameter]],
    accumulator: Mapping[str, torch.Tensor],
) -> None:
    if set(accumulator) != {name for name, _ in named_trainable}:
        raise RuntimeError("Serialized branch gradient coverage differs")
    for name, parameter in named_trainable:
        parameter.grad = accumulator[name]


def train(
    model: torch.nn.Module,
    examples: Mapping[int, Any],
    rows: Sequence[Mapping[str, Any]],
    *,
    context: distributed.DistributedTrainingContext,
    pad_token_id: int,
    named_trainable: Sequence[tuple[str, torch.nn.Parameter]],
    output_dir: Path,
) -> Mapping[str, Any]:
    schedule = build_pair_schedule(rows)
    modules = causal_train.ordered_modules(model)
    optimizer = torch.optim.AdamW(
        [parameter for _, parameter in named_trainable],
        lr=LEARNING_RATE,
        weight_decay=0.0,
        foreach=False,
        fused=False,
    )
    model.eval()
    distributed.broadcast_named_parameters(context, named_trainable, source_rank=0)
    initial_sha256 = trainable_sha256(named_trainable)
    distributed.require_consensus(
        context, initial_sha256, description="continuous-write initial read path"
    )
    progress_path = output_dir / "training_progress.jsonl"
    step_hashes: list[str] = []
    trainable_names = [name for name, _ in named_trainable]
    frozen_versions_before = frozen_parameter_versions(model, trainable_names)
    for schedule_step in schedule:
        pair = schedule_step.rank_pairs[context.process_rank]
        batches = {
            source: evolution.collate_native_examples(
                [examples[source]], pad_token_id=pad_token_id, device=context.device
            )
            for source in pair
        }
        snapshots = {
            source: _capture_natural_snapshot(model, batches[source], modules)
            for source in pair
        }
        snapshot_hashes = {
            source: mechanics.state_sha256(
                snapshots[source],
                (*mechanics.PROJECTED_ATTRIBUTES, *mechanics.RECURRENT_ATTRIBUTES),
            )
            for source in pair
        }
        recurrent_controls: dict[int, dict[str, Mapping[str, Mapping[str, torch.Tensor]]]] = {}
        module_names = tuple(name for name, _ in modules)
        for source in pair:
            donor = pair[1] if source == pair[0] else pair[0]
            recurrent_controls[source] = {
                "correct": snapshots[source],
                "paired_donor_recurrent": snapshots[donor],
                "layer_rolled_recurrent": mechanics.layer_roll_recurrent(
                    snapshots[source], module_names
                ),
                "zero_recurrent": mechanics.zero_recurrent(snapshots[source]),
            }
        branch_metrics: dict[str, list[float]] = {name: [] for name, _, _ in BRANCH_SPECS}
        row_branch_ce: dict[int, dict[str, float]] = {source: {} for source in pair}
        with torch.no_grad():
            for branch_name, _, _ in BRANCH_SPECS:
                for source in pair:
                    logits = _read_snapshot(
                        model,
                        batches[source],
                        modules,
                        projected=snapshots[source],
                        recurrent=recurrent_controls[source][branch_name],
                    )
                    value = float(_mean_ce(logits, batches[source].labels).item())
                    if not math.isfinite(value):
                        raise RuntimeError("Continuous-write FIT branch CE is nonfinite")
                    row_branch_ce[source][branch_name] = value
                    branch_metrics[branch_name].append(value)
                    del logits
                    reset_delta_mem_states(model)
                    evolution.release_native_row_allocator_cache(context.device)
        row_coefficients = {
            source: serialized_branch_coefficients(row_branch_ce[source])
            for source in pair
        }
        accumulated: dict[str, torch.Tensor] = {}
        branch_collectives: list[Mapping[str, Any]] = []
        branch_gradient_audits: dict[str, Mapping[str, Any]] = {}
        for branch_name, _, _ in BRANCH_SPECS:
            optimizer.zero_grad(set_to_none=True)
            for source in pair:
                logits = _read_snapshot(
                    model,
                    batches[source],
                    modules,
                    projected=snapshots[source],
                    recurrent=recurrent_controls[source][branch_name],
                )
                mean_ce = _mean_ce(logits, batches[source].labels)
                coefficient = row_coefficients[source][branch_name]
                (mean_ce * (coefficient / GLOBAL_BATCH_SIZE)).backward()
                del logits, mean_ce
                reset_delta_mem_states(model)
                evolution.release_native_row_allocator_cache(context.device)
            local_validation = distributed.validate_local_gradients(named_trainable)
            if (
                local_validation["passed"] is not True
                or local_validation["active_gradient_tensors"]
                != EXPECTED_TRAINABLE_TENSORS
            ):
                raise RuntimeError(
                    f"Continuous-write serialized branch gradients differ: {local_validation!r}"
                )
            branch_gradient_audits[branch_name] = {
                **dict(local_validation),
                "nonzero_gradient_tensors": sum(
                    parameter.grad is not None
                    and bool(torch.count_nonzero(parameter.grad).item())
                    for _, parameter in named_trainable
                ),
            }
            collective = distributed.sum_gradients(context, named_trainable)
            if collective["gradient_tensors"] != EXPECTED_TRAINABLE_TENSORS:
                raise RuntimeError("Continuous-write branch all-reduce coverage differs")
            branch_collectives.append(collective)
            _accumulate_reduced_branch(named_trainable, accumulated)
        optimizer.zero_grad(set_to_none=True)
        _install_accumulated_gradients(named_trainable, accumulated)
        gradient_norm = torch.nn.utils.clip_grad_norm_(
            [parameter for _, parameter in named_trainable], MAX_GRAD_NORM
        )
        gradient_norm_value = float(gradient_norm.detach().item())
        if not math.isfinite(gradient_norm_value) or gradient_norm_value <= 0.0:
            raise RuntimeError("Continuous-write FIT gradient norm is invalid")
        optimizer.step()
        step_sha256 = trainable_sha256(named_trainable)
        distributed.require_consensus(
            context,
            step_sha256,
            description=f"continuous-write read path after step {schedule_step.step}",
        )
        step_hashes.append(step_sha256)
        record = {
            "schema": STEP_SCHEMA,
            "step": schedule_step.step,
            "schedule_step_sha256": schedule_step.payload_sha256,
            "rank_pairs": [list(value) for value in schedule_step.rank_pairs],
            "serialized_branch_order": [name for name, _, _ in BRANCH_SPECS],
            "branch_all_reduce_calls": len(branch_collectives),
            "branch_gradient_audits_local": branch_gradient_audits,
            "branch_mean_ce_local": {
                name: sum(values) / len(values)
                for name, values in branch_metrics.items()
            },
            "source_state_sha256": {str(source): snapshot_hashes[source] for source in pair},
            "source_projected_carrier_sha256": {
                str(source): mechanics.state_sha256(
                    snapshots[source], mechanics.PROJECTED_ATTRIBUTES
                )
                for source in pair
            },
            "gradient_norm_before_clip": gradient_norm_value,
            "post_update_trainable_sha256": step_sha256,
        }
        if any(
            mechanics.state_sha256(
                snapshots[source],
                (*mechanics.PROJECTED_ATTRIBUTES, *mechanics.RECURRENT_ATTRIBUTES),
            )
            != snapshot_hashes[source]
            for source in pair
        ):
            raise RuntimeError("Continuous-write FIT snapshot changed during reads")
        record["receipt"] = {
            "algorithm": "sha256",
            "payload_scope": "canonical_training_step_without_receipt",
            "payload_sha256": canonical_sha256(record),
        }

        def append_progress() -> None:
            if not context.is_primary:
                return
            with progress_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record, sort_keys=True) + "\n")
                handle.flush()
                os.fsync(handle.fileno())

        mechanics._consensual_operation(
            context,
            phase=f"continuous-write-training-step-{schedule_step.step}-receipt",
            operation=append_progress,
        )
        del batches, snapshots, recurrent_controls, accumulated
        gc.collect()
        torch.cuda.empty_cache()
    final_sha256 = trainable_sha256(named_trainable)
    if frozen_parameter_versions(model, trainable_names) != frozen_versions_before:
        raise RuntimeError("Continuous-write training mutated a frozen parameter")
    del optimizer
    return {
        "updates": UPDATES,
        "rows": FIT_ROWS,
        "symmetric_pairs": FIT_PAIRS,
        "initial_trainable_sha256": initial_sha256,
        "final_trainable_sha256": final_sha256,
        "trainable_changed": initial_sha256 != final_sha256,
        "post_update_trainable_sha256": step_hashes,
        "manual_branch_all_reduce_calls": UPDATES * len(BRANCH_SPECS),
        "frozen_parameter_versions_unchanged": True,
        "signed_training_step_receipts": UPDATES,
        "causal_rows_decoded_tokenized_forwarded_or_scored": 0,
    }


def _write_training_receipt(
    model: torch.nn.Module,
    output_dir: Path,
    named_trainable: Sequence[tuple[str, torch.nn.Parameter]],
    training: Mapping[str, Any],
    *,
    context: distributed.DistributedTrainingContext,
    fit_binding: str,
    launch: Mapping[str, Any],
) -> Mapping[str, Any]:
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    if any(parameter.requires_grad for parameter in model.parameters()):
        raise RuntimeError("Continuous-write read path did not freeze")
    receipt_path = output_dir / "training_receipt.json"

    def save_receipt() -> None:
        if not context.is_primary:
            return
        checkpoint_path = output_dir / "continuous_write_read_path.pt"
        temporary_checkpoint = checkpoint_path.with_name(
            f".{checkpoint_path.name}.tmp-{os.getpid()}"
        )
        torch.save(
            {name: parameter.detach().cpu() for name, parameter in named_trainable},
            temporary_checkpoint,
        )
        os.replace(temporary_checkpoint, checkpoint_path)
        receipt: dict[str, Any] = {
            "schema": "rwkv_ms_natural_memory_native_rwkv_continuous_write_training_receipt.v1",
            "status": "continuous_write_fit_training_frozen_causal_open_authorized",
            "protocol_payload_sha256": PROTOCOL_PAYLOAD_SHA256,
            "launch_receipt": launch["receipt"]["payload_sha256"],
            "fit_binding": fit_binding,
            "training": dict(training),
            "checkpoint_file": checkpoint_path.name,
            "checkpoint_file_sha256": mechanics.sha256_file(checkpoint_path),
            "parameters_frozen_before_causal_open": True,
            "causal_bytes_open_authorized": True,
            "causal_rows_opened": 0,
        }
        receipt["receipt"] = {
            "algorithm": "sha256",
            "payload_scope": "canonical_training_receipt_without_receipt",
            "payload_sha256": canonical_sha256(receipt),
        }
        mechanics._atomic_signed_json(receipt_path, receipt)

    mechanics._consensual_operation(
        context,
        phase="continuous-write-training-receipt-save",
        operation=save_receipt,
    )
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    _validate_receipt(
        receipt,
        payload_scope="canonical_training_receipt_without_receipt",
        description="Continuous-write training receipt",
    )
    if (
        receipt.get("status")
        != "continuous_write_fit_training_frozen_causal_open_authorized"
        or receipt.get("parameters_frozen_before_causal_open") is not True
        or receipt.get("causal_bytes_open_authorized") is not True
        or receipt.get("causal_rows_opened") != 0
    ):
        raise ValueError("Continuous-write training receipt differs")
    checkpoint_path = output_dir / str(receipt["checkpoint_file"])
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if (
        mechanics.sha256_file(checkpoint_path) != receipt.get("checkpoint_file_sha256")
        or distributed.tensor_mapping_sha256(checkpoint)
        != training.get("final_trainable_sha256")
    ):
        raise ValueError("Continuous-write frozen checkpoint differs")
    distributed.require_consensus(
        context,
        mechanics.sha256_file(receipt_path),
        description="continuous-write signed training receipt file",
    )
    return receipt


def _load_causal_rows_after_receipt(
    materialization_root: Path,
    manifest: Mapping[str, Any],
    training_receipt: Mapping[str, Any],
) -> list[dict[str, Any]]:
    _validate_receipt(
        training_receipt,
        payload_scope="canonical_training_receipt_without_receipt",
        description="Continuous-write training receipt",
    )
    if (
        training_receipt.get("status")
        != "continuous_write_fit_training_frozen_causal_open_authorized"
        or training_receipt.get("causal_bytes_open_authorized") is not True
        or training_receipt.get("parameters_frozen_before_causal_open") is not True
    ):
        raise PermissionError("Signed frozen training receipt is required")
    rows = materializer._read_bundle(materialization_root, manifest, "causal")
    if len(rows) != CAUSAL_ROWS:
        raise ValueError("Continuous-write causal row count differs")
    return rows


def build_causal_pair_assignment(
    rows: Sequence[Mapping[str, Any]],
) -> tuple[tuple[tuple[int, int], ...], ...]:
    if len(rows) != CAUSAL_ROWS:
        raise ValueError("Continuous-write causal assignment requires 32 rows")
    rows_by_source = {int(row["source_index"]): row for row in rows}
    mapping = {
        source: int(row["donor_source_index"])
        for source, row in rows_by_source.items()
    }
    if len(mapping) != CAUSAL_ROWS or any(
        source == donor
        or mapping.get(donor) != source
        or rows_by_source[source].get("donor_row_sha256")
        != rows_by_source.get(donor, {}).get("row_sha256")
        for source, donor in mapping.items()
    ):
        raise ValueError("Continuous-write causal donor mapping is not symmetric")
    pairs = sorted({tuple(sorted((source, donor))) for source, donor in mapping.items()})
    if len(pairs) != CAUSAL_ROWS // 2:
        raise ValueError("Continuous-write causal pair count differs")
    assignment = tuple(tuple(pairs[rank::WORLD_SIZE]) for rank in range(WORLD_SIZE))
    if any(len(rank_pairs) != 4 for rank_pairs in assignment):
        raise RuntimeError("Continuous-write causal rank pair assignment differs")
    return assignment


def causal_sources_for_rank(
    assignment: Sequence[Sequence[tuple[int, int]]], rank: int
) -> tuple[int, ...]:
    if len(assignment) != WORLD_SIZE or not 0 <= rank < WORLD_SIZE:
        raise ValueError("Continuous-write causal rank assignment differs")
    sources = tuple(source for pair in assignment[rank] for source in pair)
    if len(sources) != 8 or len(set(sources)) != len(sources):
        raise ValueError("Continuous-write causal rank source coverage differs")
    return sources


def causal_assignment_binding(
    rows: Sequence[Mapping[str, Any]],
    assignment: Sequence[Sequence[tuple[int, int]]],
) -> str:
    rows_by_source = {int(row["source_index"]): row for row in rows}
    return canonical_sha256(
        {
            "assignment": [
                [list(pair) for pair in rank_pairs] for rank_pairs in assignment
            ],
            "rows": [
                {
                    "source_index": source,
                    "row_sha256": rows_by_source[source]["row_sha256"],
                    "donor_source_index": rows_by_source[source]["donor_source_index"],
                    "donor_row_sha256": rows_by_source[source]["donor_row_sha256"],
                }
                for source in sorted(rows_by_source)
            ],
        }
    )


def _capture_assigned_snapshots(
    model: torch.nn.Module,
    examples: Mapping[int, Any],
    source_indices: Sequence[int],
    modules: Sequence[tuple[str, Any]],
    *,
    pad_token_id: int,
    device: torch.device,
) -> dict[int, dict[str, dict[str, torch.Tensor]]]:
    if len(source_indices) != len(set(source_indices)):
        raise ValueError("Continuous-write causal snapshot sources are not unique")
    snapshots: dict[int, dict[str, dict[str, torch.Tensor]]] = {}
    for source in source_indices:
        batch = evolution.collate_native_examples(
            [examples[source]], pad_token_id=pad_token_id, device=device
        )
        snapshots[source] = _capture_natural_snapshot(model, batch, modules)
        del batch
        evolution.release_native_row_allocator_cache(device)
    if set(snapshots) != set(source_indices):
        raise RuntimeError("Continuous-write causal snapshot coverage differs")
    return snapshots


@torch.no_grad()
def evaluate_causal_row(
    model: torch.nn.Module,
    target_example: Any,
    *,
    source_index: int,
    donor_source_index: int,
    snapshots: Mapping[int, Mapping[str, Mapping[str, torch.Tensor]]],
    modules: Sequence[tuple[str, Any]],
    pad_token_id: int,
    device: torch.device,
) -> Mapping[str, Any]:
    module_names = tuple(name for name, _ in modules)
    target = evolution.collate_native_examples(
        [target_example], pad_token_id=pad_token_id, device=device
    )
    target_snapshot = snapshots[source_index]
    donor_snapshot = snapshots[donor_source_index]
    projected = mechanics.state_subset_to_device(
        target_snapshot, mechanics.PROJECTED_ATTRIBUTES, device
    )
    recurrent = {
        "correct": mechanics.state_subset_to_device(
            target_snapshot, mechanics.RECURRENT_ATTRIBUTES, device
        ),
        "paired_donor_recurrent": mechanics.state_subset_to_device(
            donor_snapshot, mechanics.RECURRENT_ATTRIBUTES, device
        ),
        "layer_rolled_recurrent": mechanics.state_subset_to_device(
            mechanics.layer_roll_recurrent(target_snapshot, module_names),
            mechanics.RECURRENT_ATTRIBUTES,
            device,
        ),
        "zero_recurrent": mechanics.state_subset_to_device(
            mechanics.zero_recurrent(target_snapshot),
            mechanics.RECURRENT_ATTRIBUTES,
            device,
        ),
    }
    conditions = (
        ("correct", recurrent["correct"], False),
        ("zero_recurrent", recurrent["zero_recurrent"], False),
        ("projected_only", recurrent["zero_recurrent"], True),
        ("paired_donor_recurrent", recurrent["paired_donor_recurrent"], False),
        ("layer_rolled_recurrent", recurrent["layer_rolled_recurrent"], False),
    )
    condition_ce: dict[str, float] = {}
    condition_loss_sum: dict[str, float] = {}
    condition_token_count: dict[str, int] = {}
    read_audits: dict[str, Mapping[str, Any]] = {}
    condition_logits: dict[str, torch.Tensor] = {}
    for name, recurrent_state, projected_only in conditions:
        logits, audit = mechanics.read_condition(
            model,
            target,
            modules,
            projected=projected,
            recurrent=recurrent_state,
            projected_only_bypass=projected_only,
        )
        loss_sum, token_count = distributed.answer_loss_sum_and_count(
            logits, target.labels.detach().cpu()
        )
        condition_loss_sum[name] = float(loss_sum.item())
        condition_token_count[name] = token_count
        condition_ce[name] = condition_loss_sum[name] / token_count
        read_audits[name] = audit
        condition_logits[name] = logits
        reset_delta_mem_states(model)
        evolution.release_native_row_allocator_cache(device)
    correct_ce = condition_ce["correct"]
    margins = {
        name: condition_ce[name] - correct_ce
        for name in (
            "paired_donor_recurrent",
            "layer_rolled_recurrent",
            "zero_recurrent",
        )
    }
    required_audit_keys = {
        "projected_carrier_references_fixed",
        "recurrent_references_fixed",
        "projected_carrier_bytes_unchanged",
        "recurrent_bytes_unchanged",
        "projected_objects_and_versions_unchanged",
        "recurrent_objects_and_versions_unchanged",
        "read_basis_contract_exact",
        "write_disabled",
        "logits_finite",
    }
    audit_checks = {
        name: all(audit.get(key) is True for key in required_audit_keys)
        for name, audit in read_audits.items()
    }
    return {
        "source_index": source_index,
        "donor_source_index": donor_source_index,
        "condition_mean_ce": condition_ce,
        "condition_loss_sum": condition_loss_sum,
        "condition_token_count": condition_token_count,
        "ce_margins": margins,
        "target_state_sha256": mechanics.state_sha256(
            target_snapshot,
            (*mechanics.PROJECTED_ATTRIBUTES, *mechanics.RECURRENT_ATTRIBUTES),
        ),
        "donor_state_sha256": mechanics.state_sha256(
            donor_snapshot,
            (*mechanics.PROJECTED_ATTRIBUTES, *mechanics.RECURRENT_ATTRIBUTES),
        ),
        "projected_carrier_sha256": mechanics.state_sha256(
            target_snapshot, mechanics.PROJECTED_ATTRIBUTES
        ),
        "read_integrity_by_condition": audit_checks,
        "projected_carrier_fixed_all_conditions": all(audit_checks.values()),
        "zero_recurrent_logits_byte_equal_projected_only": mechanics._raw_tensor_bytes_equal(
            condition_logits["zero_recurrent"], condition_logits["projected_only"]
        ),
        "answer_token_count_identical": len(set(condition_token_count.values())) == 1,
        "all_condition_ce_finite": all(math.isfinite(value) for value in condition_ce.values()),
    }


def analyze_causal_rows(rows: Sequence[Mapping[str, Any]]) -> Mapping[str, Any]:
    if len(rows) != CAUSAL_ROWS or len({int(row["source_index"]) for row in rows}) != CAUSAL_ROWS:
        raise ValueError("Continuous-write causal analysis coverage differs")
    gates = {
        "paired_donor_recurrent": (0.02, 0.75),
        "layer_rolled_recurrent": (0.05, 0.75),
        "zero_recurrent": (0.05, 0.75),
    }
    aggregate: dict[str, Any] = {}
    checks = {
        "all_condition_ce_finite": all(row["all_condition_ce_finite"] for row in rows),
        "projected_carrier_fixed_every_row": all(
            row["projected_carrier_fixed_all_conditions"] for row in rows
        ),
        "zero_recurrent_equals_projected_only_every_row": all(
            row["zero_recurrent_logits_byte_equal_projected_only"] for row in rows
        ),
        "answer_token_count_identical_every_row": all(
            row["answer_token_count_identical"] for row in rows
        ),
    }
    for name, (mean_minimum, fraction_minimum) in gates.items():
        values = [float(row["ce_margins"][name]) for row in rows]
        row_mean_margin = sum(values) / len(values)
        correct_loss_sum = sum(
            float(row["condition_loss_sum"]["correct"]) for row in rows
        )
        wrong_loss_sum = sum(
            float(row["condition_loss_sum"][name]) for row in rows
        )
        correct_tokens = sum(
            int(row["condition_token_count"]["correct"]) for row in rows
        )
        wrong_tokens = sum(
            int(row["condition_token_count"][name]) for row in rows
        )
        if correct_tokens <= 0 or wrong_tokens != correct_tokens:
            raise ValueError("Continuous-write causal token aggregation differs")
        token_weighted_margin = (
            wrong_loss_sum / wrong_tokens - correct_loss_sum / correct_tokens
        )
        positive_fraction = sum(value > 0.0 for value in values) / len(values)
        aggregate[name] = {
            "token_weighted_mean_ce_margin": token_weighted_margin,
            "unweighted_row_mean_ce_margin": row_mean_margin,
            "positive_row_fraction": positive_fraction,
            "mean_minimum": mean_minimum,
            "positive_fraction_minimum": fraction_minimum,
        }
        checks[f"{name}_token_weighted_mean"] = token_weighted_margin >= mean_minimum
        checks[f"{name}_positive_fraction"] = positive_fraction >= fraction_minimum
    return {"rows": len(rows), "aggregate": aggregate, "checks": checks, "passed": all(checks.values())}


def _write_causal_shard(
    output_dir: Path,
    *,
    rank: int,
    rows: Sequence[Mapping[str, Any]],
    binding: Mapping[str, Any],
) -> Mapping[str, Any]:
    shard: dict[str, Any] = {
        "schema": "rwkv_ms_natural_memory_native_rwkv_continuous_write_causal_shard.v1",
        "rank": rank,
        "binding": dict(binding),
        "rows": list(rows),
    }
    shard["receipt"] = {
        "algorithm": "sha256",
        "payload_scope": "canonical_causal_shard_without_receipt",
        "payload_sha256": canonical_sha256(shard),
    }
    path = output_dir / f"causal-shard-{rank}.json"
    mechanics._atomic_signed_json(path, shard)
    return {
        "rank": rank,
        "path": path.name,
        "rows": len(rows),
        "file_sha256": mechanics.sha256_file(path),
        "receipt": shard["receipt"]["payload_sha256"],
    }


def _load_causal_shards(
    output_dir: Path,
    *,
    binding: Mapping[str, Any],
    assignment: Sequence[Sequence[tuple[int, int]]],
) -> tuple[list[Mapping[str, Any]], list[Mapping[str, Any]]]:
    evaluated: list[Mapping[str, Any]] = []
    inventory: list[Mapping[str, Any]] = []
    for rank in range(WORLD_SIZE):
        path = output_dir / f"causal-shard-{rank}.json"
        shard = json.loads(path.read_text(encoding="utf-8"))
        unsigned = dict(shard)
        receipt = unsigned.pop("receipt", {})
        expected_sources = list(causal_sources_for_rank(assignment, rank))
        actual_sources = [int(row["source_index"]) for row in shard.get("rows", [])]
        expected_donors = {
            source: donor
            for left, right in assignment[rank]
            for source, donor in ((left, right), (right, left))
        }
        actual_donors = {
            int(row["source_index"]): int(row.get("donor_source_index", -1))
            for row in shard.get("rows", [])
        }
        if (
            receipt.get("payload_sha256") != canonical_sha256(unsigned)
            or shard.get("rank") != rank
            or shard.get("binding") != binding
            or actual_sources != expected_sources
            or actual_donors != expected_donors
            or len(actual_sources) != 8
        ):
            raise ValueError("Continuous-write causal shard contract differs")
        evaluated.extend(shard["rows"])
        inventory.append(
            {
                "rank": rank,
                "path": path.name,
                "rows": len(shard["rows"]),
                "file_sha256": mechanics.sha256_file(path),
                "receipt": receipt["payload_sha256"],
            }
        )
    evaluated.sort(key=lambda row: int(row["source_index"]))
    expected_all_sources = sorted(
        source for rank_pairs in assignment for pair in rank_pairs for source in pair
    )
    if [int(row["source_index"]) for row in evaluated] != expected_all_sources:
        raise ValueError("Continuous-write causal shard coverage differs")
    return evaluated, inventory


def _scatter_causal_rows(
    context: distributed.DistributedTrainingContext,
    causal_rows: Sequence[Mapping[str, Any]] | None,
) -> tuple[list[dict[str, Any]], tuple[tuple[tuple[int, int], ...], ...], str]:
    metadata: list[Any] = [None, None]
    payloads: list[list[dict[str, Any]]] | None = None
    if context.is_primary:
        if causal_rows is None:
            raise RuntimeError("Continuous-write primary causal rows are missing")
        assignment = build_causal_pair_assignment(causal_rows)
        binding = causal_assignment_binding(causal_rows, assignment)
        rows_by_source = {int(row["source_index"]): dict(row) for row in causal_rows}
        payloads = [
            [rows_by_source[source] for source in causal_sources_for_rank(assignment, rank)]
            for rank in range(WORLD_SIZE)
        ]
        metadata = [assignment, binding]
    dist.broadcast_object_list(metadata, src=0, group=context.control_group)
    raw_assignment, binding = metadata
    assignment = tuple(
        tuple(tuple(int(source) for source in pair) for pair in rank_pairs)
        for rank_pairs in raw_assignment
    )
    received: list[Any] = [None]
    dist.scatter_object_list(
        received,
        scatter_object_input_list=payloads,
        src=0,
        group=context.control_group,
    )
    local_rows = received[0]
    if (
        not isinstance(local_rows, list)
        or [int(row["source_index"]) for row in local_rows]
        != list(causal_sources_for_rank(assignment, context.process_rank))
        or not isinstance(binding, str)
    ):
        raise RuntimeError("Continuous-write causal scatter differs")
    return local_rows, assignment, binding


def _validate_final_result(path: Path) -> Mapping[str, Any]:
    result = json.loads(path.read_text(encoding="utf-8"))
    _validate_receipt(
        result,
        payload_scope="canonical_result_without_receipt",
        description="Continuous-write causal result",
    )
    if (
        result.get("schema") != SCHEMA
        or result.get("causal_bundle_opened") is not True
        or result.get("causal_rows_decoded_tokenized_forwarded_or_scored") != CAUSAL_ROWS
        or result.get("native_benchmark_bytes_opened") is not False
        or result.get("sota_claimed") is not False
    ):
        raise ValueError("Continuous-write causal final result differs")
    return result


def _write_consumed_failure(
    output_dir: Path,
    *,
    error: BaseException,
    protocol_payload_sha256: str,
    training_receipt: str | None,
) -> None:
    failure: dict[str, Any] = {
        "schema": "rwkv_ms_natural_memory_native_rwkv_continuous_write_causal_failure.v1",
        "status": "causal_endpoint_consumed_failure_rerun_forbidden",
        "protocol_payload_sha256": protocol_payload_sha256,
        "training_receipt": training_receipt,
        "causal_bundle_opened": True,
        "causal_endpoint_consumed": True,
        "unchanged_rerun_authorized": False,
        "native_benchmark_authorized": False,
        "raw_fit_or_causal_text_recorded": False,
        "error_type": type(error).__name__,
        "error_message_sha256": hashlib.sha256(str(error).encode("utf-8")).hexdigest(),
    }
    failure["receipt"] = {
        "algorithm": "sha256",
        "payload_scope": "canonical_failure_without_receipt",
        "payload_sha256": canonical_sha256(failure),
    }
    mechanics._atomic_signed_json(output_dir / "failure.json", failure)


def run(
    *,
    base_model: Path,
    materialization_root: Path,
    output_dir: Path,
) -> Mapping[str, Any]:
    context = distributed.initialize_distributed_training(
        "cuda", required_world_size=WORLD_SIZE, timeout_seconds=mechanics.TIMEOUT_SECONDS
    )
    if context is None:
        raise RuntimeError("Run with torchrun --nproc_per_node=4")
    output_created = False
    causal_opened = False
    training_receipt_payload: str | None = None
    try:
        def validate_preflight() -> tuple[
            Mapping[str, Any], Mapping[str, Any], Mapping[str, Any], Mapping[str, Any]
        ]:
            if (
                context.world_size != WORLD_SIZE
                or context.backend != "nccl"
                or context.control_backend != "gloo"
                or not hardware.four_distinct_a100s(context.rank_devices)
                or os.environ.get("HF_ENDPOINT") != HF_ENDPOINT
            ):
                raise RuntimeError(
                    "Continuous-write causal training requires four A100s and HF mirror"
                )
            protocol = validate_protocol(base_model)
            launch = validate_launch_binding(protocol)
            mechanics_result = validate_mechanics_authorization()
            source_audit = exact_v5.validate_execution_source()
            manifest = _load_manifest_only(materialization_root)
            return protocol, launch, mechanics_result, {"source": source_audit, "manifest": manifest}

        protocol, launch, mechanics_result, preflight = mechanics._consensual_operation(
            context,
            phase="continuous-write-causal-preflight-without-protected-access",
            operation=validate_preflight,
        )
        source_audit = preflight["source"]
        manifest = preflight["manifest"]
        preflight_binding = canonical_sha256(
            {
                "protocol": protocol["receipt"]["payload_sha256"],
                "launch": launch["receipt"]["payload_sha256"],
                "mechanics": mechanics_result["receipt"]["payload_sha256"],
                "source": source_audit,
                "manifest": manifest["receipt"]["payload_sha256"],
            }
        )
        distributed.require_consensus(
            context, preflight_binding, description="continuous-write causal preflight"
        )

        def create_output() -> None:
            if not context.is_primary:
                return
            if output_dir.exists():
                raise ValueError(f"Continuous-write causal output must be fresh: {output_dir}")
            output_dir.mkdir(parents=True, exist_ok=False)

        mechanics._consensual_operation(
            context, phase="continuous-write-causal-output-create", operation=create_output
        )
        output_created = True
        fit_rows = mechanics._consensual_operation(
            context,
            phase="continuous-write-open-fit-load",
            operation=lambda: _load_fit_rows(materialization_root, manifest),
        )
        fit_binding = canonical_sha256(
            [
                {
                    "source_index": row["source_index"],
                    "row_sha256": row["row_sha256"],
                    "donor_source_index": row["donor_source_index"],
                    "donor_row_sha256": row["donor_row_sha256"],
                }
                for row in fit_rows
            ]
        )
        distributed.require_consensus(
            context, fit_binding, description="continuous-write FIT rows"
        )

        schedule = build_pair_schedule(fit_rows)
        schedule_digest = validate_fit_schedule_binding(schedule, protocol)

        def load_runtime() -> tuple[
            torch.nn.Module,
            Any,
            Mapping[str, Any],
            Sequence[tuple[str, Any]],
            Mapping[str, Any],
            Mapping[str, Any],
            Sequence[tuple[str, torch.nn.Parameter]],
            str,
        ]:
            torch.manual_seed(SEED)
            torch.cuda.manual_seed_all(SEED)
            model, tokenizer, model_audit = exact_v5.load_exact_v5_model(
                base_model, device=context.device
            )
            modules = causal_train.ordered_modules(model)
            module_names = tuple(name for name, _ in modules)
            maps = mechanics.load_frozen_maps(module_names)
            install_audit = integration.install(
                model,
                rank=mechanics.MAP_RANK,
                seed=SEED,
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
            named_trainable, trainable_audit = configure_trainable_parameters(model)
            map_digest = retrieval.map_digest(maps, module_names)
            if (
                len(modules) != mechanics.MODULES
                or install_audit.get("effective_full64_override")
                != "one_shot_selected_key_only"
                or map_digest != mechanics.MAP_DIGEST
            ):
                raise RuntimeError("Continuous-write causal runtime installation differs")
            return (
                model,
                tokenizer,
                model_audit,
                modules,
                maps,
                {"continuous_install": install_audit, "trainable": trainable_audit},
                named_trainable,
                map_digest,
            )

        (
            model,
            tokenizer,
            model_audit,
            modules,
            maps,
            runtime_audit,
            named_trainable,
            map_digest_before,
        ) = mechanics._consensual_operation(
            context, phase="continuous-write-causal-model-load", operation=load_runtime
        )
        dependency_bindings_before = dependency_bindings()
        runner_sha256_before = mechanics.sha256_file(Path(__file__).resolve())
        fit_examples = mechanics._consensual_operation(
            context,
            phase="continuous-write-fit-tokenization",
            operation=lambda: retrieval._encode_rows(tokenizer, fit_rows),
        )
        training = mechanics._consensual_operation(
            context,
            phase="continuous-write-fit-training",
            operation=lambda: train(
                model,
                fit_examples,
                fit_rows,
                context=context,
                pad_token_id=int(tokenizer.pad_token_id),
                named_trainable=named_trainable,
                output_dir=output_dir,
            ),
        )
        if (
            training.get("updates") != UPDATES
            or training.get("rows") != FIT_ROWS
            or training.get("trainable_changed") is not True
            or training.get("frozen_parameter_versions_unchanged") is not True
            or training.get("signed_training_step_receipts") != UPDATES
        ):
            raise RuntimeError("Continuous-write FIT training gate failed")
        training_receipt = _write_training_receipt(
            model,
            output_dir,
            named_trainable,
            training,
            context=context,
            fit_binding=fit_binding,
            launch=launch,
        )
        distributed.require_consensus(
            context,
            training_receipt["receipt"]["payload_sha256"],
            description="continuous-write training receipt",
        )
        training_receipt_payload = training_receipt["receipt"]["payload_sha256"]

        def install_causal_observers() -> Mapping[str, Any]:
            read_observer = retrieval.install_read_observer(model)
            invocation_observer = mechanics.install_read_invocation_observer(modules)
            module_names = [name for name, _ in modules]
            if (
                read_observer.get("module_names") != module_names
                or invocation_observer.get("module_names") != module_names
            ):
                raise RuntimeError("Continuous-write causal read observer differs")
            return {
                "read_observer": read_observer,
                "read_invocation_observer": invocation_observer,
            }

        observer_audit = mechanics._consensual_operation(
            context,
            phase="continuous-write-causal-observer-install",
            operation=install_causal_observers,
        )
        parameter_versions_before_causal = mechanics._parameter_versions(model)
        trained_digest_before_causal = trainable_sha256(named_trainable)
        distributed.require_consensus(
            context,
            canonical_sha256(parameter_versions_before_causal),
            description="continuous-write frozen causal parameters",
        )

        causal_opened = True
        causal_rows = mechanics._consensual_operation(
            context,
            phase="continuous-write-authorized-single-causal-open",
            operation=lambda: (
                _load_causal_rows_after_receipt(
                    materialization_root, manifest, training_receipt
                )
                if context.is_primary
                else None
            ),
        )
        local_causal_rows, causal_assignment, assignment_digest = mechanics._consensual_operation(
            context,
            phase="continuous-write-causal-complete-pair-scatter",
            operation=lambda: _scatter_causal_rows(context, causal_rows),
        )
        shard_binding = {
            "protocol_payload_sha256": PROTOCOL_PAYLOAD_SHA256,
            "launch_receipt": launch["receipt"]["payload_sha256"],
            "training_receipt": training_receipt["receipt"]["payload_sha256"],
            "trained_parameter_sha256": trained_digest_before_causal,
            "causal_assignment_sha256": assignment_digest,
        }

        def evaluate_local_causal_pairs() -> Mapping[str, Any]:
            local_examples = retrieval._encode_rows(tokenizer, local_causal_rows)
            local_sources = causal_sources_for_rank(
                causal_assignment, context.process_rank
            )
            snapshots = _capture_assigned_snapshots(
                model,
                local_examples,
                local_sources,
                modules,
                pad_token_id=int(tokenizer.pad_token_id),
                device=context.device,
            )
            evaluated = []
            for ordinal, row in enumerate(local_causal_rows, start=1):
                source = int(row["source_index"])
                donor = int(row["donor_source_index"])
                evaluated.append(
                    evaluate_causal_row(
                        model,
                        local_examples[source],
                        source_index=source,
                        donor_source_index=donor,
                        snapshots=snapshots,
                        modules=modules,
                        pad_token_id=int(tokenizer.pad_token_id),
                        device=context.device,
                    )
                )
                print(
                    f"CONTINUOUS_CAUSAL_ROW rank={context.process_rank} "
                    f"row={source} ordinal={ordinal}/8",
                    flush=True,
                )
            del snapshots, local_examples
            gc.collect()
            torch.cuda.empty_cache()
            return _write_causal_shard(
                output_dir,
                rank=context.process_rank,
                rows=evaluated,
                binding=shard_binding,
            )

        mechanics._consensual_operation(
            context,
            phase="continuous-write-causal-evaluation-and-shard-save",
            operation=evaluate_local_causal_pairs,
        )
        evaluated_rows, shard_inventory = mechanics._consensual_operation(
            context,
            phase="continuous-write-causal-shard-validation",
            operation=lambda: _load_causal_shards(
                output_dir,
                binding=shard_binding,
                assignment=causal_assignment,
            ),
        )
        evaluated_digest = canonical_sha256(evaluated_rows)
        distributed.require_consensus(
            context, evaluated_digest, description="continuous-write causal rows"
        )
        if (
            mechanics._parameter_versions(model) != parameter_versions_before_causal
            or trainable_sha256(named_trainable) != trained_digest_before_causal
            or retrieval.map_digest(maps, tuple(name for name, _ in modules))
            != map_digest_before
        ):
            raise RuntimeError("Continuous-write causal evaluation mutated frozen state")

        analysis = analyze_causal_rows(evaluated_rows)
        result_path = output_dir / "result.json"

        def write_result() -> None:
            if not context.is_primary:
                return
            dependencies_end = dependency_bindings()
            runner_sha256_end = mechanics.sha256_file(Path(__file__).resolve())
            if (
                dependencies_end != dependency_bindings_before
                or runner_sha256_end != runner_sha256_before
                or mechanics.sha256_file(PROTOCOL) != PROTOCOL_FILE_SHA256
            ):
                raise RuntimeError("Continuous-write causal code binding changed")
            passed = bool(analysis["passed"])
            result: dict[str, Any] = {
                "schema": SCHEMA,
                "status": (
                    "continuous_write_causal_passed_native_benchmark_protocol_draft_authorized"
                    if passed
                    else "continuous_write_causal_failed_readout_family_retired"
                ),
                "passed": passed,
                "protocol_payload_sha256": PROTOCOL_PAYLOAD_SHA256,
                "protocol_file_sha256": PROTOCOL_FILE_SHA256,
                "launch_receipt": launch["receipt"]["payload_sha256"],
                "preflight_binding": preflight_binding,
                "mechanics_result_receipt": mechanics_result["receipt"]["payload_sha256"],
                "fit_binding": fit_binding,
                "fit_schedule_sha256": schedule_digest,
                "training_receipt": training_receipt["receipt"]["payload_sha256"],
                "training": training,
                "causal_analysis": analysis,
                "causal_assignment_sha256": assignment_digest,
                "causal_evaluated_rows_sha256": evaluated_digest,
                "causal_shards": shard_inventory,
                "model_audit": model_audit,
                "runtime_audit": runtime_audit,
                "observer_audit": observer_audit,
                "source_audit": source_audit,
                "causal_bundle_opened": True,
                "causal_file_read_rank": 0,
                "causal_file_read_calls": 1,
                "causal_rows_decoded_tokenized_forwarded_or_scored": CAUSAL_ROWS,
                "rwkv_snapshot_writes_per_causal_source": 1,
                "second_rwkv_write_scan_used": False,
                "full_bandwidth_transformer_installed": False,
                "native_benchmark_protocol_drafting_authorized": passed,
                "native_benchmark_bytes_opened": False,
                "generation_authorized": False,
                "sota_claimed": False,
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
            mechanics._atomic_signed_json(result_path, result)

        mechanics._consensual_operation(
            context,
            phase="continuous-write-causal-result-save",
            operation=write_result,
        )
        result = mechanics._consensual_operation(
            context,
            phase="continuous-write-causal-all-rank-result-validation",
            operation=lambda: _validate_final_result(result_path),
        )
        distributed.require_consensus(
            context,
            result["receipt"]["payload_sha256"],
            description="continuous-write causal final result",
        )
        return result
    except BaseException as error:
        if causal_opened and output_created:
            failure_error: BaseException | None = None
            if context.is_primary:
                try:
                    _write_consumed_failure(
                        output_dir,
                        error=error,
                        protocol_payload_sha256=PROTOCOL_PAYLOAD_SHA256,
                        training_receipt=training_receipt_payload,
                    )
                except BaseException as caught:
                    failure_error = caught
            try:
                distributed.phase_consensus(
                    context,
                    phase="continuous-write-consumed-endpoint-failure-record",
                    error=failure_error,
                )
            except BaseException:
                pass
        raise
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
        base_model=args.base_model,
        materialization_root=args.materialization_root,
        output_dir=args.output_dir,
    )
    return 0 if result.get("passed") is True else 1


if __name__ == "__main__":
    raise SystemExit(main())
