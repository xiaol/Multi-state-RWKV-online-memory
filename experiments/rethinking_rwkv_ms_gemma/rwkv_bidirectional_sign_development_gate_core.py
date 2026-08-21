"""Fail-closed development gate for bidirectional RWKV sign binding."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
from typing import Any, Mapping, Sequence


HF_ENDPOINT = "https://hf-mirror.com"
SIGNED_SOURCE_ENV = "RWKV_V5_EXACT_SOURCE_ROOT"
if os.environ.get("HF_ENDPOINT") != HF_ENDPOINT:
    raise RuntimeError(f"HF_ENDPOINT must be explicitly set to {HF_ENDPOINT}")
if not os.environ.get(SIGNED_SOURCE_ENV):
    raise RuntimeError(f"{SIGNED_SOURCE_ENV} must be explicitly set")

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[1]
SIGNED_SOURCE_ROOT = Path(os.environ[SIGNED_SOURCE_ENV]).expanduser().resolve(strict=True)
for import_root in (SCRIPT_DIR, PROJECT_ROOT):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))
try:
    sys.path.remove(str(SIGNED_SOURCE_ROOT))
except ValueError:
    pass
sys.path.insert(0, str(SIGNED_SOURCE_ROOT))

import torch
import torch.distributed as dist

from deltamem.core.delta import reset_delta_mem_states
from experiments.rethinking_rwkv_ms_gemma import rwkv_bidirectional_sign_integration as sign
from experiments.rethinking_rwkv_ms_gemma import (
    materialize_natural_memory_native_rwkv_bidirectional_sign_open_fit as open_fit,
)
from experiments.rethinking_rwkv_ms_gemma import (
    run_natural_memory_native_rwkv_plmsc_code_alignment_v2 as plmsc,
)


SCHEMA = "rwkv_ms_bidirectional_sign_development_gate.v2"
RESULT_SCHEMA = "rwkv_ms_bidirectional_sign_development_result.v2"
PASS_STATUS = "bidirectional_sign_development_passed_mechanics_protocol_authorized"
FAIL_STATUS = "bidirectional_sign_development_failed_family_retired"
WORLD_SIZE = 4
ATTEMPT_NUMBER = 2
DEVELOPMENT_ROWS = 64
MECHANICS_ROWS = 17
CAUSAL_ROWS = 17
SEED = 131
FREQUENCY = 64.0
STATE_DIM = 32
ADDRESS_DIM = 64
BASE_MODEL_ID = "google/gemma-4-E4B-it"
BASE_MODEL_REVISION = "a4c2d58be94dda072b918d9db64ee85c8ed34e3f"
PYTHON_VERSION = "3.12.13"
TORCH_VERSION = "2.6.0+cu128"
CUDA_VERSION = "12.8"
EXPECTED_SOURCE_SET_SHA256 = open_fit.EXPECTED_SOURCE_SET_SHA256
EXPECTED_MAPPING_SHA256 = open_fit.EXPECTED_GLOBAL_MAPPING_SHA256
EXPECTED_FIT_SHA256 = open_fit.EXPECTED_SOURCE_SHA256["development"]
EXPECTED_FIT_MAPPING_SHA256 = open_fit.EXPECTED_MAPPING_SHA256["development"]
EXPECTED_MECHANICS_SHA256 = (
    open_fit.EXPECTED_SOURCE_SHA256["mechanics"]
)
EXPECTED_CAUSAL_SHA256 = (
    open_fit.EXPECTED_SOURCE_SHA256["causal"]
)
EXPECTED_COMPONENT_ORDER_SHA256 = (
    open_fit.EXPECTED_ORDERED_COMPONENTS_SHA256
)
OPEN_FIT_INVENTORY = {
    "manifest.json",
    "development.jsonl",
    "mechanics.jsonl",
    "causal.jsonl",
}
OPEN_FIT_MANIFEST_FILE_SHA256 = (
    "fbad372ad50295e9588c10bdeb40807def69e10216fb365acbe38c931aed7773"
)
OPEN_FIT_MANIFEST_RECEIPT = (
    "0b6c3cd1869a5c7c346351a173cd2c16b3fdf27eb5f174c354a35e8a2a34c109"
)
OPEN_FIT_DEVELOPMENT_FILE_SHA256 = (
    "32c43c8cf5b86cca403c89ed7eb39501ec7e043e4b94414b95273b84703640a9"
)
PLMSC_RESULT = (
    SCRIPT_DIR
    / "local_artifacts/natural_memory_native_rwkv_plmsc_code_alignment_v2/result.json"
)
PLMSC_RESULT_FILE_SHA256 = (
    "b7dce00737c928abc13729b19e24ccfe803b9dce6dde62b9d9d944971a295544"
)
PLMSC_RESULT_RECEIPT = (
    "23c7cfdf0cdf0fb747010615cfe271ae7d7c0cddd7bd9a90401179033100fda7"
)
V1_PROTOCOL_FILE_SHA256 = (
    "af4b21a4d523a5282b22b113d9c73761045de2e3e077a14a58844b4e500f250a"
)
V1_PROTOCOL_PAYLOAD_SHA256 = (
    "7c9fb7fb1160ee54851d65d5bcc612a00f1da356a816f30cfd28976fb1ebbdfb"
)
V1_CORE_SHA256 = (
    "08e7695d2c96d1ac7119ff6a4524e321d381eced673d177a6cd7c8b5a72b89d5"
)
V1_EXECUTION_COMMIT = "2cf59d0e2345af893d636bcc28c0659d0d800eea"
V1_OPERATIONAL_FAILURE = (
    SCRIPT_DIR
    / "local_artifacts/"
    "natural_memory_native_rwkv_bidirectional_sign_development_gate_v1/"
    "operational_failure.json"
)
V1_OPERATIONAL_FAILURE_FILE_SHA256 = (
    "ce102f035fe3b4a52a0b1b70670e4478d12780baded62c1888501d41648082ca"
)
V1_OPERATIONAL_FAILURE_RECEIPT = (
    "ce165fe4033476c18b081012387602c2cda47fe0b170b5d65d6de32e79da5023"
)
ROW_CHECK_KEYS = {
    "encoded_state_byte_equal",
    "projected_and_recurrent_sidecars_byte_equal",
    "disabled_and_bound_write_metadata_byte_equal",
    "both_write_codes_match_routed_full_key",
    "write_routes_and_codes_valid",
    "initial_insert_has_no_rebase",
    "native_receptance_byte_equal",
    "decoded_slot_reads_byte_equal",
    "addressed_and_global_routes_byte_equal",
    "left_and_right_read_codes_byte_equal",
    "dual_read_native_basis_byte_equal",
    "all_read_values_finite",
    "final_logits_byte_equal",
    "final_logits_finite",
}
ROW_RESULT_KEYS = {
    "source_index",
    "checks",
    "selected_codes",
    "rebase_events",
    "logit_max_abs",
}
DATA_AUDIT_KEYS = {
    "local_target_sources",
    "local_decoded_sources",
    "development_row_hashes_verified",
    "development_rows_decoded",
    "development_rows_tokenized",
    "development_rows_forwarded",
    "manifest_metadata_opened",
    "development_bundle_opened",
    "mechanics_bundle_opened",
    "causal_bundle_opened",
    "mechanics_rows_decoded",
    "mechanics_rows_tokenized",
    "mechanics_rows_forwarded",
    "causal_rows_decoded",
    "causal_rows_tokenized",
    "causal_rows_forwarded",
}
SHARD_KEYS = {"rank", "sources", "rows", "data_audit"}

distributed = plmsc.distributed
evolution = plmsc.evolution
causal_train = plmsc.causal_train
endpoint = plmsc.endpoint
hardware = plmsc.hardware
shadow = plmsc.shadow


def canonical_sha256(value: Any) -> str:
    return distributed.canonical_sha256(value)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _byte_equal(left: torch.Tensor, right: torch.Tensor) -> bool:
    return bool(
        left.dtype == right.dtype
        and tuple(left.shape) == tuple(right.shape)
        and torch.equal(
            left.detach().contiguous().view(torch.uint8),
            right.detach().contiguous().view(torch.uint8),
        )
    )


def _git_output(root: Path, *arguments: str) -> str:
    return subprocess.run(
        ["git", "-C", str(root), *arguments],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _manifest_path(
    row: Mapping[str, Any],
    *,
    base_model: Path,
) -> Path:
    scope = row.get("scope")
    relative = Path(str(row.get("path")))
    roots = {
        "project": PROJECT_ROOT,
        "signed_source": SIGNED_SOURCE_ROOT,
        "base_model": base_model,
        "v5_adapter": Path(shadow.V5_ADAPTER),
    }
    if scope not in roots or relative.is_absolute() or ".." in relative.parts:
        raise ValueError(f"Invalid manifest path: {row!r}")
    root = roots[str(scope)].resolve(strict=True)
    path = (root / relative).resolve(strict=True)
    if path != root and root not in path.parents:
        raise ValueError(f"Manifest path escapes its scope: {row!r}")
    return path


def _validate_manifests(
    protocol: Mapping[str, Any],
    *,
    base_model: Path,
    open_fit_root: Path,
) -> Mapping[str, Any]:
    manifests = protocol.get("manifests", {})
    rows = manifests.get("files", [])
    if not isinstance(rows, list) or not rows:
        raise ValueError("Development protocol file manifest is empty")
    observed: list[Mapping[str, Any]] = []
    imported_roles = {
        "binding": Path(
            sys.modules[sign.BidirectionalDiagonalSignBinding.__module__].__file__
        ).resolve(),
        "open_fit_materializer": Path(open_fit.__file__).resolve(),
        "chat_template_runtime": Path(
            sys.modules[open_fit.apply_chat_template.__module__].__file__
        ).resolve(),
        "open_fit_manifest": (open_fit_root / "manifest.json").resolve(strict=True),
        "open_fit_development_bundle": (
            open_fit_root / "development.jsonl"
        ).resolve(strict=True),
        "v1_operational_failure": V1_OPERATIONAL_FAILURE.resolve(strict=True),
        "diagonal_sign_dependency": Path(
            sys.modules[sign.deterministic_projection.__module__].__file__
        ).resolve(),
        "integration": Path(sign.__file__).resolve(),
        "gate_core": Path(__file__).resolve(),
        "plmsc_runner": Path(plmsc.__file__).resolve(),
        "distributed_runtime": Path(distributed.__file__).resolve(),
        "native_evolution_runtime": Path(evolution.__file__).resolve(),
        "causal_state_runtime": Path(causal_train.__file__).resolve(),
        "dataset_endpoint_runtime": Path(endpoint.__file__).resolve(),
        "hardware_runtime": Path(hardware.__file__).resolve(),
        "exact_v5_loader": Path(shadow.__file__).resolve(),
        "shadow_model_loader": Path(
            sys.modules[shadow.load_model_and_tokenizer.__module__].__file__
        ).resolve(),
    }
    project_execution_paths: list[str] = []
    seen: set[tuple[str, str]] = set()
    seen_roles: set[str] = set()
    for row in rows:
        if set(row) != {"role", "scope", "path", "sha256"}:
            raise ValueError("Development manifest row schema differs")
        key = (str(row["scope"]), str(row["path"]))
        if key in seen:
            raise ValueError("Development manifest contains duplicate paths")
        seen.add(key)
        role = str(row["role"])
        if role in seen_roles:
            raise ValueError("Development manifest contains duplicate roles")
        seen_roles.add(role)
        path = _manifest_path(row, base_model=base_model)
        if role in imported_roles and path != imported_roles[role]:
            raise ValueError(f"Development import shadowing detected for {role}")
        if path.name in {"mechanics.jsonl", "causal.jsonl"}:
            raise ValueError("Sealed replay bundle must not enter the execution DAG")
        actual = sha256_file(path)
        if actual != row["sha256"]:
            raise ValueError(f"Development manifest file differs: {path}")
        observed.append({**dict(row), "bytes": path.stat().st_size})
        if row["scope"] == "project":
            project_execution_paths.append(str(row["path"]))

    expected_model_files = {
        "chat_template.jinja",
        "config.json",
        "generation_config.json",
        "model.safetensors",
        "tokenizer.json",
        "tokenizer_config.json",
    }
    manifested_model_files = {
        str(row["path"]) for row in rows if row["scope"] == "base_model"
    }
    actual_model_files = {
        path.name for path in base_model.iterdir() if path.is_file()
    }
    if (
        manifested_model_files != expected_model_files
        or actual_model_files != expected_model_files
    ):
        raise ValueError("Development base-model manifest is incomplete")
    expected_adapter_files = {
        "delta_mem_adapter.pt",
        "delta_mem_config.json",
    }
    if {
        str(row["path"]) for row in rows if row["scope"] == "v5_adapter"
    } != expected_adapter_files or {
        path.name for path in Path(shadow.V5_ADAPTER).iterdir() if path.is_file()
    } != expected_adapter_files:
        raise ValueError("Development exact-v5 adapter manifest is incomplete")
    if set(imported_roles) - {str(row["role"]) for row in rows}:
        raise ValueError("Development imported runtime manifest is incomplete")

    if _git_output(SIGNED_SOURCE_ROOT, "rev-parse", "HEAD") != protocol.get(
        "frozen_inputs", {}
    ).get("signed_source_commit"):
        raise ValueError("Development signed source commit differs")
    if _git_output(SIGNED_SOURCE_ROOT, "status", "--short", "--untracked-files=no"):
        raise ValueError("Development signed source has tracked changes")
    for relative in project_execution_paths:
        subprocess.run(
            ["git", "-C", str(PROJECT_ROOT), "ls-files", "--error-unmatch", relative],
            check=True,
            capture_output=True,
        )
    if project_execution_paths:
        for cached in (False, True):
            command = ["git", "-C", str(PROJECT_ROOT), "diff", "--quiet"]
            if cached:
                command.append("--cached")
            command.extend(["--", *project_execution_paths])
            if subprocess.run(command, check=False).returncode != 0:
                raise ValueError("Development execution sources have uncommitted changes")
    return {
        "files": observed,
        "payload_sha256": canonical_sha256(observed),
        "project_execution_paths": sorted(project_execution_paths),
    }


def _validate_open_fit(
    protocol: Mapping[str, Any],
    *,
    open_fit_root: Path,
) -> tuple[
    Mapping[str, tuple[int, ...]],
    Mapping[int, int],
    Sequence[Mapping[str, Any]],
    Mapping[str, Any],
]:
    inventory = {path.name for path in open_fit_root.iterdir()}
    if inventory != OPEN_FIT_INVENTORY:
        raise ValueError("Open-fit materialization inventory differs")
    validated = open_fit.validate_materialization(
        open_fit_root,
        bundles=("development",),
    )
    manifest = validated["manifest"]
    if (
        sha256_file(open_fit_root / "manifest.json")
        != OPEN_FIT_MANIFEST_FILE_SHA256
        or manifest["receipt"]["payload_sha256"] != OPEN_FIT_MANIFEST_RECEIPT
        or manifest["splits"]["development"]["bundle"]["sha256"]
        != OPEN_FIT_DEVELOPMENT_FILE_SHA256
    ):
        raise ValueError("Open-fit signed replay receipt differs")
    splits = manifest["splits"]
    groups = {
        name: tuple(int(value) for value in splits[name]["source_indices"])
        for name in open_fit.BUNDLE_NAMES
    }
    mapping = {
        int(source): int(donor)
        for source, donor in splits["development"]["mapping_pairs"]
    }
    expected_signed_split = {
        "source_namespace": open_fit.SOURCE_NAMESPACE,
        "materialization_schema": open_fit.MANIFEST_SCHEMA,
        "source_set_sha256": EXPECTED_SOURCE_SET_SHA256,
        "mapping_pairs_sha256": EXPECTED_MAPPING_SHA256,
        "ordered_components_sha256": EXPECTED_COMPONENT_ORDER_SHA256,
        "development_sources": list(groups["development"]),
        "development_sha256": EXPECTED_FIT_SHA256,
        "development_mapping_pairs": splits["development"]["mapping_pairs"],
        "development_mapping_sha256": EXPECTED_FIT_MAPPING_SHA256,
        "mechanics_sources": list(groups["mechanics"]),
        "mechanics_sha256": EXPECTED_MECHANICS_SHA256,
        "mechanics_mapping_sha256": open_fit.EXPECTED_MAPPING_SHA256["mechanics"],
        "causal_sources": list(groups["causal"]),
        "causal_sha256": EXPECTED_CAUSAL_SHA256,
        "causal_mapping_sha256": open_fit.EXPECTED_MAPPING_SHA256["causal"],
    }
    if protocol.get("split") != expected_signed_split:
        raise ValueError("Development open-fit 64/17/17 split differs")
    if (
        len(groups["development"]) != DEVELOPMENT_ROWS
        or len(groups["mechanics"]) != MECHANICS_ROWS
        or len(groups["causal"]) != CAUSAL_ROWS
        or canonical_sha256(list(groups["development"])) != EXPECTED_FIT_SHA256
        or canonical_sha256(list(groups["mechanics"])) != EXPECTED_MECHANICS_SHA256
        or canonical_sha256(list(groups["causal"])) != EXPECTED_CAUSAL_SHA256
        or canonical_sha256(
            [[source, mapping[source]] for source in sorted(mapping)]
        )
        != EXPECTED_FIT_MAPPING_SHA256
    ):
        raise ValueError("Development open-fit payload differs")
    for name in open_fit.BUNDLE_NAMES:
        members = set(groups[name])
        split_mapping = {
            int(source): int(donor)
            for source, donor in splits[name]["mapping_pairs"]
        }
        if set(split_mapping) != members or any(
            split_mapping[source] not in members for source in members
        ):
            raise ValueError(f"A donor edge crosses the {name} split")
    return groups, mapping, validated["groups"]["development"], manifest


def validate_protocol(
    protocol_path: Path,
    *,
    base_model: Path,
    open_fit_root: Path,
) -> tuple[
    Mapping[str, Any],
    Mapping[str, tuple[int, ...]],
    Mapping[int, int],
    Sequence[Mapping[str, Any]],
    Mapping[str, Any],
    Mapping[str, Any],
]:
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    unsigned = dict(protocol)
    receipt = unsigned.pop("receipt", {})
    digest = canonical_sha256(unsigned)
    architecture = protocol.get("architecture", {})
    execution = protocol.get("execution", {})
    firewall = protocol.get("firewall", {})
    parent = protocol.get("authorization_basis", {})
    if (
        protocol.get("schema") != SCHEMA
        or set(receipt) != {"algorithm", "payload_scope", "payload_sha256"}
        or receipt.get("algorithm") != "sha256"
        or receipt.get("payload_scope")
        != "canonical_protocol_without_receipt"
        or receipt.get("payload_sha256") != digest
        or architecture.get("state_encoding") != "S_bound=D_value(address)@S@D_key(address)"
        or architecture.get("state_dim") != STATE_DIM
        or architecture.get("address_dim") != ADDRESS_DIM
        or architecture.get("projection_seed") != SEED
        or architecture.get("frequency") != FREQUENCY
        or architecture.get("state_rebase_required") is not True
        or execution.get("world_size") != WORLD_SIZE
        or execution.get("hf_endpoint") != HF_ENDPOINT
        or execution.get("attempts") != 1
        or execution.get("attempt_number") != ATTEMPT_NUMBER
        or execution.get("resume") is not False
        or execution.get("fit_or_training") is not False
        or execution.get("fresh_output_required") is not True
        or execution.get("python") != PYTHON_VERSION
        or execution.get("torch") != TORCH_VERSION
        or execution.get("cuda") != CUDA_VERSION
        or firewall.get("development_rows_opened") != DEVELOPMENT_ROWS
        or firewall.get("development_bundle") != "development.jsonl"
        or firewall.get("manifest_metadata_opened") is not True
        or firewall.get("mechanics_bundle_opened") is not False
        or firewall.get("causal_bundle_opened") is not False
        or firewall.get("mechanics_rows_opened") != 0
        or firewall.get("causal_rows_opened") != 0
        or protocol.get("mechanics_stage_authorized") is not False
        or protocol.get("model_or_adapter_training_authorized") is not False
        or protocol.get("generation_authorized") is not False
        or protocol.get("benchmark_authorized") is not False
        or protocol.get("protected_splits_opened_by_this_protocol") != []
        or parent.get("plmsc_protocol_file_sha256") != plmsc.PROTOCOL_FILE_SHA256
        or parent.get("plmsc_protocol_payload_sha256")
        != plmsc.PROTOCOL_PAYLOAD_SHA256
        or parent.get("plmsc_result_file_sha256") != PLMSC_RESULT_FILE_SHA256
        or parent.get("plmsc_result_receipt") != PLMSC_RESULT_RECEIPT
        or parent.get("v1_protocol_file_sha256") != V1_PROTOCOL_FILE_SHA256
        or parent.get("v1_protocol_payload_sha256")
        != V1_PROTOCOL_PAYLOAD_SHA256
        or parent.get("v1_core_sha256") != V1_CORE_SHA256
        or parent.get("v1_execution_commit") != V1_EXECUTION_COMMIT
        or parent.get("v1_operational_failure_file_sha256")
        != V1_OPERATIONAL_FAILURE_FILE_SHA256
        or parent.get("v1_operational_failure_receipt")
        != V1_OPERATIONAL_FAILURE_RECEIPT
        or parent.get("retry_scope")
        != "one operational retry after pre-mechanics capture-lifecycle failure"
        or protocol.get("frozen_inputs", {}).get("base_model") != BASE_MODEL_ID
        or protocol.get("frozen_inputs", {}).get("base_model_revision")
        != BASE_MODEL_REVISION
        or protocol.get("frozen_inputs", {}).get("open_fit_manifest_sha256")
        != sha256_file(open_fit_root / "manifest.json")
    ):
        raise ValueError("Signed bidirectional development protocol differs")
    if sha256_file(PLMSC_RESULT) != PLMSC_RESULT_FILE_SHA256:
        raise ValueError("Signed PLMSC retirement result file differs")
    plmsc_result = json.loads(PLMSC_RESULT.read_text(encoding="utf-8"))
    unsigned_result = dict(plmsc_result)
    result_receipt = unsigned_result.pop("receipt", {})
    if (
        canonical_sha256(unsigned_result) != PLMSC_RESULT_RECEIPT
        or result_receipt.get("payload_sha256") != PLMSC_RESULT_RECEIPT
        or plmsc_result.get("status") != "plmsc_code_alignment_failed_family_retired"
        or plmsc_result.get("passed") is not False
        or plmsc_result.get("protected_splits_opened") != []
    ):
        raise ValueError("Signed PLMSC retirement result differs")
    if sha256_file(V1_OPERATIONAL_FAILURE) != V1_OPERATIONAL_FAILURE_FILE_SHA256:
        raise ValueError("Bidirectional v1 operational-failure file differs")
    operational_failure = json.loads(
        V1_OPERATIONAL_FAILURE.read_text(encoding="utf-8")
    )
    unsigned_failure = dict(operational_failure)
    failure_receipt = unsigned_failure.pop("receipt", {})
    if (
        operational_failure.get("schema")
        != "rwkv_ms_bidirectional_sign_development_operational_failure.v1"
        or operational_failure.get("status")
        != "bidirectional_sign_development_operational_failure_no_mechanics_result"
        or operational_failure.get("passed") is not False
        or operational_failure.get("attempt") != 1
        or operational_failure.get("execution_commit") != V1_EXECUTION_COMMIT
        or operational_failure.get("protocol", {}).get("file_sha256")
        != V1_PROTOCOL_FILE_SHA256
        or operational_failure.get("protocol", {}).get("payload_sha256")
        != V1_PROTOCOL_PAYLOAD_SHA256
        or operational_failure.get("protocol", {}).get("core_sha256")
        != V1_CORE_SHA256
        or operational_failure.get("runtime", {}).get("development_rows_completed")
        != 0
        or operational_failure.get("firewall", {}).get(
            "mechanics_bundle_opened"
        )
        is not False
        or operational_failure.get("firewall", {}).get("causal_bundle_opened")
        is not False
        or operational_failure.get("firewall", {}).get("protected_splits_opened")
        != []
        or operational_failure.get("retry_authorized_by_this_artifact") is not False
        or set(failure_receipt)
        != {"algorithm", "payload_scope", "payload_sha256"}
        or failure_receipt.get("algorithm") != "sha256"
        or failure_receipt.get("payload_scope")
        != "canonical_operational_failure_without_receipt"
        or failure_receipt.get("payload_sha256")
        != V1_OPERATIONAL_FAILURE_RECEIPT
        or canonical_sha256(unsigned_failure) != V1_OPERATIONAL_FAILURE_RECEIPT
    ):
        raise ValueError("Bidirectional v1 operational-failure receipt differs")
    plmsc.validate_protocol()
    groups, mapping, development_rows, open_fit_manifest = _validate_open_fit(
        protocol,
        open_fit_root=open_fit_root,
    )
    if (
        protocol.get("frozen_inputs", {}).get("open_fit_manifest_receipt")
        != open_fit_manifest["receipt"]["payload_sha256"]
    ):
        raise ValueError("Signed open-fit manifest receipt differs")
    manifests = _validate_manifests(
        protocol,
        base_model=base_model,
        open_fit_root=open_fit_root,
    )
    projection_manifest = protocol.get("projection_sha256", {})
    if len(projection_manifest) != 42 or any(
        set(value) != {"left", "right"} or value["left"] == value["right"]
        for value in projection_manifest.values()
    ):
        raise ValueError("Development projection manifest differs")
    return (
        protocol,
        groups,
        mapping,
        development_rows,
        open_fit_manifest,
        manifests,
    )


def _load_local_examples(
    tokenizer: Any,
    groups: Mapping[str, tuple[int, ...]],
    development_rows: Sequence[Mapping[str, Any]],
    process_rank: int,
) -> tuple[Mapping[int, Any], Mapping[str, Any]]:
    fit = tuple(groups["development"])
    local_targets = fit[process_rank::WORLD_SIZE]
    by_source = {
        int(row["source_index"]): row
        for row in development_rows
    }
    if set(by_source) != set(fit) or len(local_targets) != DEVELOPMENT_ROWS // WORLD_SIZE:
        raise RuntimeError("Development replay coverage differs")
    selected = {source: by_source[source] for source in local_targets}
    examples = {
        source: evolution.encode_native_full_row(
            tokenizer,
            task="scene",
            source_ordinal=source,
            raw_line=str(selected[source]["raw_line"]),
        )
        for source in local_targets
    }
    return examples, {
        "local_target_sources": list(local_targets),
        "local_decoded_sources": list(local_targets),
        "development_row_hashes_verified": len(development_rows),
        "development_rows_decoded": len(local_targets),
        "development_rows_tokenized": len(local_targets),
        "development_rows_forwarded": 0,
        "manifest_metadata_opened": True,
        "development_bundle_opened": True,
        "mechanics_bundle_opened": False,
        "causal_bundle_opened": False,
        "mechanics_rows_decoded": 0,
        "mechanics_rows_tokenized": 0,
        "mechanics_rows_forwarded": 0,
        "causal_rows_decoded": 0,
        "causal_rows_tokenized": 0,
        "causal_rows_forwarded": 0,
    }


def _ordered_modules(model: torch.nn.Module) -> tuple[tuple[str, Any], ...]:
    return causal_train.ordered_modules(model)


def _reset_mode(model: torch.nn.Module, enabled: bool) -> None:
    reset_delta_mem_states(model)
    sign.clear_transient(model)
    sign.set_enabled(model, enabled)
    for _, module in _ordered_modules(model):
        module.rwkv_bidirectional_sign_rebase_events = 0


def _snapshot_state(model: torch.nn.Module) -> Mapping[str, Mapping[str, torch.Tensor]]:
    attributes = (*causal_train.RECURRENT_ATTRIBUTES, *causal_train.PROJECTED_ATTRIBUTES)
    result: dict[str, Mapping[str, torch.Tensor]] = {}
    for name, module in _ordered_modules(model):
        values: dict[str, torch.Tensor] = {}
        for attribute in attributes:
            value = getattr(module, attribute)
            if value is None:
                raise RuntimeError(f"Development state omitted {name}.{attribute}")
            values[attribute] = value.detach().clone()
        result[name] = values
    return result


def _snapshot_write(model: torch.nn.Module) -> Mapping[str, Mapping[str, torch.Tensor]]:
    result: dict[str, Mapping[str, torch.Tensor]] = {}
    for name, module in _ordered_modules(model):
        values = {
            "address": module.rwkv_bidirectional_sign_write_address,
            "left": module.rwkv_bidirectional_sign_write_left_code,
            "right": module.rwkv_bidirectional_sign_write_right_code,
            "routes": module.last_write_routes,
        }
        if any(value is None for value in values.values()):
            raise RuntimeError(f"Development write capture is incomplete for {name}")
        result[name] = {
            key: value.detach().clone() for key, value in values.items()
        }
    return result


def _snapshot_reads(model: torch.nn.Module) -> Mapping[str, Mapping[str, Any]]:
    result: dict[str, Mapping[str, Any]] = {}
    for name, module in _ordered_modules(model):
        captures = module.rwkv_bidirectional_sign_captures
        if set(captures) != {"addressed", "global"}:
            raise RuntimeError(f"Development dual-read capture differs for {name}")
        if module.rwkv_bidirectional_sign_read_sequence:
            raise RuntimeError(f"Development read sequence did not close for {name}")
        result[name] = {
            kind: {
                key: value.detach().clone()
                for key, value in payload.items()
            }
            for kind, payload in captures.items()
        }
    return result


def _write_state_checks(
    model: torch.nn.Module,
    baseline_state: Mapping[str, Mapping[str, torch.Tensor]],
    baseline_write: Mapping[str, Mapping[str, torch.Tensor]],
    bound_state: Mapping[str, Mapping[str, torch.Tensor]],
    bound_write: Mapping[str, Mapping[str, torch.Tensor]],
    write_mask: torch.Tensor,
) -> tuple[Mapping[str, bool], Mapping[str, Any]]:
    state_exact = True
    sidecars_exact = True
    write_exact = True
    code_match = True
    code_valid = True
    selected_codes: dict[str, Mapping[str, int]] = {}
    rebase_events = 0
    for name, module in _ordered_modules(model):
        base = baseline_state[name]
        bound = bound_state[name]
        keys = bound["projected_kv_keys"]
        addresses = keys.unsqueeze(1).expand(-1, bound["delta_state"].shape[1], -1, -1)
        expected = module.rwkv_bidirectional_sign_binding.encode_state(
            addresses,
            base["delta_state"],
        )
        state_exact = state_exact and _byte_equal(bound["delta_state"], expected)
        for attribute in (
            "rwkv_ms_positions",
            "rwkv_ms_previous_source",
            *causal_train.PROJECTED_ATTRIBUTES,
        ):
            sidecars_exact = sidecars_exact and _byte_equal(base[attribute], bound[attribute])
        for key in baseline_write[name]:
            write_exact = write_exact and _byte_equal(
                baseline_write[name][key], bound_write[name][key]
            )
        routes = bound_write[name]["routes"].float()
        valid = write_mask.to(device=routes.device, dtype=torch.bool)
        valid_routes = routes.masked_select(valid.unsqueeze(-1)).reshape(-1, routes.shape[-1])
        if valid_routes.numel() == 0:
            raise RuntimeError("Development write mask selects no tokens")
        chosen = valid_routes.argmax(dim=-1)
        code_valid = code_valid and bool(
            valid_routes.sum(dim=-1).eq(1.0).all().item()
            and chosen.eq(chosen[0]).all().item()
        )
        slot = int(chosen[0].item())
        slot_address = keys[:, slot]
        slot_left, slot_right = module.rwkv_bidirectional_sign_binding.codes(slot_address)
        valid_left = bound_write[name]["left"].masked_select(
            valid.unsqueeze(-1)
        ).reshape(-1, STATE_DIM)
        valid_right = bound_write[name]["right"].masked_select(
            valid.unsqueeze(-1)
        ).reshape(-1, STATE_DIM)
        valid_address = bound_write[name]["address"].masked_select(
            valid.unsqueeze(-1)
        ).reshape(-1, ADDRESS_DIM)
        code_match = code_match and _byte_equal(
            valid_left, slot_left.expand_as(valid_left)
        )
        code_match = code_match and _byte_equal(
            valid_right, slot_right.expand_as(valid_right)
        )
        code_match = code_match and _byte_equal(
            valid_address, slot_address.expand_as(valid_address)
        )
        code_valid = code_valid and bool(
            slot_left.abs().eq(1.0).all().item()
            and slot_right.abs().eq(1.0).all().item()
            and slot_left.ne(slot_right).any().item()
        )
        left_bits = sum(
            int(value > 0) << index
            for index, value in enumerate(slot_left[0].tolist())
        )
        right_bits = sum(
            int(value > 0) << index
            for index, value in enumerate(slot_right[0].tolist())
        )
        selected_codes[name] = {"left": left_bits, "right": right_bits}
        rebase_events += int(module.rwkv_bidirectional_sign_rebase_events)
    return {
        "encoded_state_byte_equal": state_exact,
        "projected_and_recurrent_sidecars_byte_equal": sidecars_exact,
        "disabled_and_bound_write_metadata_byte_equal": write_exact,
        "both_write_codes_match_routed_full_key": code_match,
        "write_routes_and_codes_valid": code_valid,
        "initial_insert_has_no_rebase": rebase_events == 0,
    }, {
        "selected_codes": selected_codes,
        "rebase_events": rebase_events,
    }


def _read_checks(
    baseline: Mapping[str, Mapping[str, Any]],
    bound: Mapping[str, Mapping[str, Any]],
) -> Mapping[str, bool]:
    native_r_exact = True
    decoded_exact = True
    routes_exact = True
    codes_exact = True
    dual_call_exact = True
    finite = True
    for name in baseline:
        for kind in ("addressed", "global"):
            base = baseline[name][kind]
            candidate = bound[name][kind]
            native_r_exact = native_r_exact and _byte_equal(
                base["native_receptance"], candidate["native_receptance"]
            )
            decoded_exact = decoded_exact and _byte_equal(
                base["raw"], candidate["decoded"]
            )
            routes_exact = routes_exact and _byte_equal(base["routes"], candidate["routes"])
            for key in ("slot_addresses", "slot_codes", "right_slot_codes"):
                codes_exact = codes_exact and _byte_equal(base[key], candidate[key])
            finite = finite and all(
                bool(torch.isfinite(value).all().item())
                for value in candidate.values()
                if value.is_floating_point()
            )
        dual_call_exact = dual_call_exact and _byte_equal(
            bound[name]["addressed"]["native_receptance"],
            bound[name]["global"]["native_receptance"],
        )
        dual_call_exact = dual_call_exact and _byte_equal(
            bound[name]["addressed"]["slot_codes"],
            bound[name]["global"]["slot_codes"],
        )
        dual_call_exact = dual_call_exact and _byte_equal(
            bound[name]["addressed"]["right_slot_codes"],
            bound[name]["global"]["right_slot_codes"],
        )
    return {
        "native_receptance_byte_equal": native_r_exact,
        "decoded_slot_reads_byte_equal": decoded_exact,
        "addressed_and_global_routes_byte_equal": routes_exact,
        "left_and_right_read_codes_byte_equal": codes_exact,
        "dual_read_native_basis_byte_equal": dual_call_exact,
        "all_read_values_finite": finite,
    }


@torch.no_grad()
def _run_row(
    model: torch.nn.Module,
    tokenizer: Any,
    example: Any,
    *,
    device: torch.device,
) -> Mapping[str, Any]:
    batch = evolution.collate_native_examples(
        [example],
        pad_token_id=int(tokenizer.pad_token_id),
        device=device,
    )
    write_mask = batch.write_attention_mask.to(device=device, dtype=torch.bool)
    try:
        _reset_mode(model, False)
        evolution._native_write(model, batch, dtype=torch.bfloat16)
        baseline_state = _snapshot_state(model)
        baseline_write = _snapshot_write(model)
        sign.clear_read_capture(model)
        baseline_logits = evolution._native_read(model, batch, dtype=torch.bfloat16)
        baseline_reads = _snapshot_reads(model)

        _reset_mode(model, True)
        evolution._native_write(model, batch, dtype=torch.bfloat16)
        bound_state = _snapshot_state(model)
        bound_write = _snapshot_write(model)
        state_checks, audit = _write_state_checks(
            model,
            baseline_state,
            baseline_write,
            bound_state,
            bound_write,
            write_mask,
        )
        sign.clear_read_capture(model)
        bound_logits = evolution._native_read(model, batch, dtype=torch.bfloat16)
        bound_reads = _snapshot_reads(model)
        read_checks = _read_checks(baseline_reads, bound_reads)
        checks = {
            **state_checks,
            **read_checks,
            "final_logits_byte_equal": _byte_equal(baseline_logits, bound_logits),
            "final_logits_finite": bool(torch.isfinite(bound_logits).all().item()),
        }
        return {
            "source_index": int(example.source_ordinal),
            "checks": checks,
            "selected_codes": audit["selected_codes"],
            "rebase_events": audit["rebase_events"],
            "logit_max_abs": float(
                (baseline_logits.float() - bound_logits.float()).abs().max().item()
            ),
        }
    finally:
        reset_delta_mem_states(model)
        sign.clear_transient(model)
        evolution.release_native_row_allocator_cache(device)


def _code_separation(
    rows: Sequence[Mapping[str, Any]],
    mapping: Mapping[int, int],
) -> Mapping[str, Any]:
    by_source = {int(row["source_index"]): row for row in rows}
    left_distances: list[float] = []
    right_distances: list[float] = []
    independent: list[bool] = []
    modules: set[str] | None = None
    for source in sorted(by_source):
        donor = mapping[source]
        source_codes = by_source[source]["selected_codes"]
        donor_codes = by_source[donor]["selected_codes"]
        modules = set(source_codes) if modules is None else modules
        if set(source_codes) != modules or set(donor_codes) != modules:
            raise RuntimeError("Development selected-code module coverage differs")
        for name in sorted(modules):
            left = int(source_codes[name]["left"])
            right = int(source_codes[name]["right"])
            donor_left = int(donor_codes[name]["left"])
            donor_right = int(donor_codes[name]["right"])
            left_distances.append((left ^ donor_left).bit_count() / STATE_DIM)
            right_distances.append((right ^ donor_right).bit_count() / STATE_DIM)
            independent.append(left != right)
    if modules is None or len(modules) != 42:
        raise RuntimeError("Development code separation has no complete module set")
    return {
        "comparisons": len(left_distances),
        "modules": len(modules),
        "left_positive_fraction": sum(value > 0.0 for value in left_distances)
        / len(left_distances),
        "right_positive_fraction": sum(value > 0.0 for value in right_distances)
        / len(right_distances),
        "left_mean_hamming": sum(left_distances) / len(left_distances),
        "right_mean_hamming": sum(right_distances) / len(right_distances),
        "left_right_independent_fraction": sum(independent) / len(independent),
    }


def _validate_rank_shards(
    shards: Sequence[Mapping[str, Any]],
    development_sources: Sequence[int],
) -> None:
    if (
        len(shards) != WORLD_SIZE
        or len(development_sources) != DEVELOPMENT_ROWS
        or canonical_sha256(list(development_sources)) != EXPECTED_FIT_SHA256
    ):
        raise ValueError("Development rank-shard coverage differs")
    observed: list[int] = []
    for rank, shard in enumerate(shards):
        expected_sources = list(development_sources[rank::WORLD_SIZE])
        if (
            not isinstance(shard, Mapping)
            or set(shard) != SHARD_KEYS
            or shard.get("rank") != rank
            or shard.get("sources") != expected_sources
            or not isinstance(shard.get("rows"), list)
            or len(shard["rows"]) != DEVELOPMENT_ROWS // WORLD_SIZE
            or not isinstance(shard.get("data_audit"), Mapping)
            or set(shard["data_audit"]) != DATA_AUDIT_KEYS
        ):
            raise ValueError("Development rank-shard schema or assignment differs")
        audit = shard["data_audit"]
        if (
            audit.get("local_target_sources") != expected_sources
            or audit.get("local_decoded_sources") != expected_sources
            or audit.get("development_row_hashes_verified") != DEVELOPMENT_ROWS
            or audit.get("development_rows_decoded") != len(expected_sources)
            or audit.get("development_rows_tokenized") != len(expected_sources)
            or audit.get("development_rows_forwarded") != len(expected_sources)
            or audit.get("manifest_metadata_opened") is not True
            or audit.get("development_bundle_opened") is not True
            or audit.get("mechanics_bundle_opened") is not False
            or audit.get("causal_bundle_opened") is not False
            or any(
                audit.get(key) != 0
                for key in (
                    "mechanics_rows_decoded",
                    "mechanics_rows_tokenized",
                    "mechanics_rows_forwarded",
                    "causal_rows_decoded",
                    "causal_rows_tokenized",
                    "causal_rows_forwarded",
                )
            )
        ):
            raise ValueError("Development rank-shard firewall differs")
        row_sources: list[int] = []
        for row in shard["rows"]:
            if (
                not isinstance(row, Mapping)
                or set(row) != ROW_RESULT_KEYS
                or not isinstance(row.get("source_index"), int)
                or not isinstance(row.get("checks"), Mapping)
                or set(row["checks"]) != ROW_CHECK_KEYS
                or not isinstance(row.get("selected_codes"), Mapping)
                or len(row["selected_codes"]) != 42
                or row.get("rebase_events") != 0
                or row.get("logit_max_abs") != 0.0
            ):
                raise ValueError("Development row evidence schema differs")
            for codes in row["selected_codes"].values():
                if (
                    not isinstance(codes, Mapping)
                    or set(codes) != {"left", "right"}
                    or any(
                        not isinstance(codes[axis], int)
                        or isinstance(codes[axis], bool)
                        or codes[axis] < 0
                        or codes[axis] >= 2**STATE_DIM
                        for axis in ("left", "right")
                    )
                ):
                    raise ValueError("Development selected-code evidence differs")
            row_sources.append(int(row["source_index"]))
        if row_sources != expected_sources:
            raise ValueError("Development row order differs from rank assignment")
        observed.extend(row_sources)
    if observed != [
        source
        for rank in range(WORLD_SIZE)
        for source in development_sources[rank::WORLD_SIZE]
    ] or sorted(observed) != list(development_sources):
        raise ValueError("Development gathered source coverage differs")


def _recompute_result_checks(result: Mapping[str, Any]) -> Mapping[str, bool]:
    shards = result.get("rank_shards", [])
    development_sources = result.get("development_sources", [])
    try:
        _validate_rank_shards(shards, development_sources)
        shard_validation = True
    except (KeyError, TypeError, ValueError):
        shard_validation = False
    rows = [
        row
        for shard in shards
        if isinstance(shard, Mapping)
        for row in shard.get("rows", [])
        if isinstance(row, Mapping)
    ]
    row_sources = [int(row["source_index"]) for row in rows if "source_index" in row]
    all_row_checks = all(
        set(row.get("checks", {})) == ROW_CHECK_KEYS
        and all(value is True for value in row["checks"].values())
        for row in rows
    )
    mapping_pairs = result.get("mapping_pairs", [])
    mapping: dict[int, int] = {}
    try:
        mapping = {
            int(source): int(donor)
            for source, donor in mapping_pairs
            if isinstance(source, int)
            and not isinstance(source, bool)
            and isinstance(donor, int)
            and not isinstance(donor, bool)
        }
    except (TypeError, ValueError):
        mapping = {}
    try:
        recomputed_separation = _code_separation(rows, mapping)
    except (KeyError, RuntimeError, TypeError, ValueError):
        recomputed_separation = {}
    separation = result.get("code_separation", {})
    runtime = result.get("runtime", {})
    open_fit_audit = result.get("open_fit_audit", {})
    installation = result.get("installation", {})
    shard_firewall = {
        key: sum(int(shard.get("data_audit", {}).get(key, -1)) for shard in shards)
        for key in (
            "mechanics_rows_decoded",
            "mechanics_rows_tokenized",
            "mechanics_rows_forwarded",
            "causal_rows_decoded",
            "causal_rows_tokenized",
            "causal_rows_forwarded",
        )
    }
    firewall = result.get("firewall", {})
    return {
        "exactly_four_rank_shards": shard_validation,
        "rank_stride_assignments_exact": shard_validation,
        "development64_complete_unique": len(rows) == DEVELOPMENT_ROWS
        and len(set(row_sources)) == DEVELOPMENT_ROWS
        and canonical_sha256(sorted(row_sources)) == EXPECTED_FIT_SHA256
        and list(development_sources) == sorted(row_sources)
        and canonical_sha256(mapping_pairs) == EXPECTED_FIT_MAPPING_SHA256,
        "all_row_exactness_checks_pass": all_row_checks,
        "both_axes_separate_every_donor_pair": separation.get(
            "left_positive_fraction"
        )
        == 1.0
        and separation.get("right_positive_fraction") == 1.0,
        "left_right_codes_independent": separation.get(
            "left_right_independent_fraction"
        )
        == 1.0
        and separation == recomputed_separation,
        "runtime_and_hardware_exact": isinstance(runtime, Mapping)
        and set(runtime)
        == {
            "python",
            "torch",
            "cuda",
            "world_size",
            "backend",
            "control_backend",
            "rank_devices",
            "hf_endpoint",
            "attempt",
        }
        and runtime.get("python") == PYTHON_VERSION
        and runtime.get("torch") == TORCH_VERSION
        and runtime.get("cuda") == CUDA_VERSION
        and runtime.get("world_size") == WORLD_SIZE
        and runtime.get("backend") == "nccl"
        and runtime.get("control_backend") == "gloo"
        and runtime.get("hf_endpoint") == HF_ENDPOINT
        and runtime.get("attempt") == ATTEMPT_NUMBER
        and hardware.four_distinct_a100s(runtime.get("rank_devices", [])),
        "open_fit_receipts_exact": isinstance(open_fit_audit, Mapping)
        and open_fit_audit.get("inventory") == sorted(OPEN_FIT_INVENTORY)
        and open_fit_audit.get("manifest_file_sha256")
        == OPEN_FIT_MANIFEST_FILE_SHA256
        and open_fit_audit.get("manifest_payload_sha256")
        == OPEN_FIT_MANIFEST_RECEIPT
        and open_fit_audit.get("manifest_schema") == open_fit.MANIFEST_SCHEMA
        and open_fit_audit.get("development_bundle_sha256")
        == OPEN_FIT_DEVELOPMENT_FILE_SHA256
        and open_fit_audit.get("development_rows") == DEVELOPMENT_ROWS
        and open_fit_audit.get("mechanics_bundle_opened") is False
        and open_fit_audit.get("causal_bundle_opened") is False,
        "installation_contract_exact": isinstance(installation, Mapping)
        and installation.get("modules") == 42
        and installation.get("state_dim") == STATE_DIM
        and installation.get("head_size") == STATE_DIM
        and installation.get("address_dim") == ADDRESS_DIM
        and installation.get("frequency") == FREQUENCY
        and installation.get("parameter_elements") == 172032
        and installation.get("projections_trainable") is False
        and installation.get("projected_carrier_changed") is False
        and installation.get("state_rebase_on_slot_address_change_implemented")
        is True,
        "mechanics_and_causal_firewall_closed": firewall.get(
            "mechanics_rows_decoded"
        )
        == 0
        and firewall.get("mechanics_rows_tokenized") == 0
        and firewall.get("mechanics_rows_forwarded") == 0
        and firewall.get("causal_rows_decoded") == 0
        and firewall.get("causal_rows_tokenized") == 0
        and firewall.get("causal_rows_forwarded") == 0
        and all(value == 0 for value in shard_firewall.values())
        and firewall.get("manifest_metadata_opened") is True
        and firewall.get("development_bundle_opened") is True
        and firewall.get("mechanics_bundle_opened") is False
        and firewall.get("causal_bundle_opened") is False,
        "no_fit_training_generation_or_weights": result.get("fit_executed") is False
        and result.get("model_updates") == 0
        and result.get("adapter_saved") is False
        and result.get("generation_executed") is False,
    }


def validate_result_payload(result: Mapping[str, Any]) -> Mapping[str, bool]:
    unsigned = dict(result)
    receipt = unsigned.pop("receipt", None)
    expected_keys = {
        "schema",
        "status",
        "passed",
        "protocol_file_sha256",
        "protocol_payload_sha256",
        "launcher_sha256",
        "gate_core_sha256",
        "source_audit",
        "manifest_audit",
        "open_fit_audit",
        "model_audit",
        "installation",
        "runtime",
        "development_sources",
        "rank_shards",
        "mapping_pairs",
        "code_separation",
        "firewall",
        "fit_executed",
        "model_updates",
        "adapter_saved",
        "generation_executed",
        "mechanics_protocol_authorized",
        "generation_authorized",
        "benchmark_authorized",
        "protected_splits_opened",
        "checks",
    }
    if set(unsigned) != expected_keys:
        raise ValueError("Development result top-level schema differs")
    if receipt is not None:
        expected_receipt = {
            "algorithm": "sha256",
            "payload_scope": "canonical_result_without_receipt",
            "payload_sha256": canonical_sha256(unsigned),
        }
        if receipt != expected_receipt:
            raise ValueError("Development result receipt differs")
    checks = _recompute_result_checks(unsigned)
    passed = all(checks.values())
    if (
        unsigned.get("schema") != RESULT_SCHEMA
        or unsigned.get("checks") != checks
        or unsigned.get("passed") is not passed
        or unsigned.get("status") != (PASS_STATUS if passed else FAIL_STATUS)
        or unsigned.get("mechanics_protocol_authorized") is not passed
        or unsigned.get("generation_authorized") is not False
        or unsigned.get("benchmark_authorized") is not False
        or unsigned.get("protected_splits_opened") != []
    ):
        raise ValueError("Development result payload is internally inconsistent")
    return checks


def _validate_output_boundary(
    *,
    protocol_path: Path,
    launcher_path: Path,
    base_model: Path,
    open_fit_root: Path,
    output_dir: Path,
) -> None:
    output_dir = output_dir.resolve()
    if output_dir.exists():
        raise ValueError(f"Development output must be fresh: {output_dir}")
    protected_roots = (
        base_model.resolve(strict=True),
        open_fit_root.resolve(strict=True),
        SIGNED_SOURCE_ROOT,
        Path(shadow.V5_ADAPTER).resolve(strict=True),
    )
    for root in protected_roots:
        if output_dir == root or root in output_dir.parents or output_dir in root.parents:
            raise ValueError("Development output overlaps an immutable input root")
    if output_dir == PROJECT_ROOT or output_dir in PROJECT_ROOT.parents:
        raise ValueError("Development output overlaps the project root")
    for input_path in (protocol_path, launcher_path, Path(__file__).resolve()):
        input_path = input_path.resolve(strict=True)
        if input_path == output_dir or output_dir in input_path.parents:
            raise ValueError("Development output contains an execution input")


def _write_signed_json(path: Path, value: Mapping[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("x", encoding="utf-8") as handle:
        handle.write(
            json.dumps(value, ensure_ascii=True, indent=2, sort_keys=True) + "\n"
        )
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def run(
    *,
    protocol_path: Path,
    launcher_path: Path,
    base_model: Path,
    open_fit_root: Path,
    output_dir: Path,
) -> Mapping[str, Any]:
    context = distributed.initialize_distributed_training(
        "cuda",
        required_world_size=WORLD_SIZE,
        timeout_seconds=1800,
    )
    if context is None:
        raise RuntimeError("Run the development gate with torchrun --nproc_per_node=4")
    try:
        error: BaseException | None = None
        try:
            if (
                context.world_size != WORLD_SIZE
                or context.backend != "nccl"
                or context.control_backend != "gloo"
                or not hardware.four_distinct_a100s(context.rank_devices)
            ):
                raise RuntimeError(
                    "Development gate requires exactly four distinct A100 GPUs"
                )
        except BaseException as exc:
            error = exc
        distributed.phase_consensus(
            context,
            phase="development-hardware",
            error=error,
        )

        validated = None
        error = None
        try:
            validated = validate_protocol(
                protocol_path,
                base_model=base_model,
                open_fit_root=open_fit_root,
            )
            _validate_output_boundary(
                protocol_path=protocol_path,
                launcher_path=launcher_path,
                base_model=base_model,
                open_fit_root=open_fit_root,
                output_dir=output_dir,
            )
        except BaseException as exc:
            error = exc
        distributed.phase_consensus(context, phase="development-preflight", error=error)
        assert validated is not None
        (
            protocol,
            groups,
            mapping,
            development_rows,
            open_fit_manifest,
            manifests,
        ) = validated
        error = None
        if context.is_primary:
            try:
                output_dir.mkdir(parents=True, exist_ok=False)
            except BaseException as exc:
                error = exc
        distributed.phase_consensus(context, phase="development-output-create", error=error)

        torch.manual_seed(SEED)
        torch.cuda.manual_seed_all(SEED)
        model = tokenizer = model_audit = installation = examples = data_audit = None
        source_audit = None
        error = None
        try:
            source_audit = shadow.validate_execution_source()
            model, tokenizer, model_audit = shadow.load_exact_v5_model(
                base_model,
                device=context.device,
            )
            installation = sign.install(
                model,
                state_dim=STATE_DIM,
                head_size=STATE_DIM,
                seed=SEED,
                frequency=FREQUENCY,
                trainable_projection=False,
                expected_projection_sha256=protocol["projection_sha256"],
            )
            if (
                installation["modules"] != 42
                or installation["address_dim"] != ADDRESS_DIM
                or installation["parameter_elements"] != 172032
                or any(
                    parameter.requires_grad
                    for name, parameter in model.named_parameters()
                    if "rwkv_bidirectional_sign_binding" in name
                )
            ):
                raise RuntimeError("Development binding installation differs")
            sign.set_capture(model, True)
            model.eval()
            examples, data_audit = _load_local_examples(
                tokenizer,
                groups,
                development_rows,
                context.process_rank,
            )
        except BaseException as exc:
            error = exc
        distributed.phase_consensus(context, phase="development-model-data", error=error)
        assert model is not None and tokenizer is not None
        assert examples is not None and data_audit is not None
        assert model_audit is not None and installation is not None
        assert source_audit is not None

        local_rows: list[Mapping[str, Any]] = []
        local_targets = tuple(groups["development"])[context.process_rank::WORLD_SIZE]
        for row_index, source in enumerate(local_targets):
            error = None
            row = None
            try:
                row = _run_row(
                    model,
                    tokenizer,
                    examples[source],
                    device=context.device,
                )
            except BaseException as exc:
                error = exc
            distributed.phase_consensus(
                context,
                phase=f"development-row-{row_index}",
                error=error,
            )
            assert row is not None
            local_rows.append(row)
        data_audit["development_rows_forwarded"] = len(local_rows)

        local_shard = {
            "rank": context.process_rank,
            "sources": list(local_targets),
            "rows": local_rows,
            "data_audit": data_audit,
        }
        error = None
        try:
            if (
                set(local_shard) != SHARD_KEYS
                or local_shard["sources"]
                != list(groups["development"])[context.process_rank::WORLD_SIZE]
                or len(local_rows) != DEVELOPMENT_ROWS // WORLD_SIZE
            ):
                raise ValueError("Development local shard differs before gather")
        except BaseException as exc:
            error = exc
        distributed.phase_consensus(
            context,
            phase="development-gather-prepare",
            error=error,
        )
        rank_shards = distributed.gather_objects(context, local_shard)
        separation = None
        error = None
        try:
            _validate_rank_shards(rank_shards, groups["development"])
            all_rows = [row for shard in rank_shards for row in shard["rows"]]
            separation = _code_separation(all_rows, mapping)
        except BaseException as exc:
            error = exc
        distributed.phase_consensus(
            context,
            phase="development-gather-validate",
            error=error,
        )
        assert separation is not None
        firewall = {
            "development_rows_decoded": sum(
                int(shard["data_audit"]["development_rows_decoded"])
                for shard in rank_shards
            ),
            "development_rows_tokenized": sum(
                int(shard["data_audit"]["development_rows_tokenized"])
                for shard in rank_shards
            ),
            "development_rows_forwarded": sum(
                int(shard["data_audit"]["development_rows_forwarded"])
                for shard in rank_shards
            ),
            "manifest_metadata_opened": True,
            "development_bundle_opened": True,
            "mechanics_bundle_opened": False,
            "causal_bundle_opened": False,
            "mechanics_rows_decoded": 0,
            "mechanics_rows_tokenized": 0,
            "mechanics_rows_forwarded": 0,
            "causal_rows_decoded": 0,
            "causal_rows_tokenized": 0,
            "causal_rows_forwarded": 0,
        }
        result: dict[str, Any] | None = None
        error = None
        if context.is_primary:
            try:
                result = {
                    "schema": RESULT_SCHEMA,
                    "status": FAIL_STATUS,
                    "passed": False,
                    "protocol_file_sha256": sha256_file(protocol_path),
                    "protocol_payload_sha256": protocol["receipt"]["payload_sha256"],
                    "launcher_sha256": sha256_file(launcher_path),
                    "gate_core_sha256": sha256_file(Path(__file__).resolve()),
                    "source_audit": source_audit,
                    "manifest_audit": manifests,
                    "open_fit_audit": {
                        "root": open_fit_root.resolve().relative_to(
                            PROJECT_ROOT
                        ).as_posix(),
                        "inventory": sorted(OPEN_FIT_INVENTORY),
                        "manifest_file_sha256": sha256_file(
                            open_fit_root / "manifest.json"
                        ),
                        "manifest_payload_sha256": open_fit_manifest["receipt"][
                            "payload_sha256"
                        ],
                        "manifest_schema": open_fit_manifest["schema"],
                        "development_bundle_sha256": open_fit_manifest["splits"][
                            "development"
                        ]["bundle"]["sha256"],
                        "development_rows": DEVELOPMENT_ROWS,
                        "mechanics_bundle_opened": False,
                        "causal_bundle_opened": False,
                    },
                    "model_audit": model_audit,
                    "installation": installation,
                    "runtime": {
                        "python": platform.python_version(),
                        "torch": torch.__version__,
                        "cuda": torch.version.cuda,
                        "world_size": context.world_size,
                        "backend": context.backend,
                        "control_backend": context.control_backend,
                        "rank_devices": list(context.rank_devices),
                        "hf_endpoint": os.environ.get("HF_ENDPOINT"),
                        "attempt": ATTEMPT_NUMBER,
                    },
                    "development_sources": list(groups["development"]),
                    "rank_shards": list(rank_shards),
                    "mapping_pairs": [
                        [source, mapping[source]] for source in sorted(mapping)
                    ],
                    "code_separation": separation,
                    "firewall": firewall,
                    "fit_executed": False,
                    "model_updates": 0,
                    "adapter_saved": False,
                    "generation_executed": False,
                    "mechanics_protocol_authorized": False,
                    "generation_authorized": False,
                    "benchmark_authorized": False,
                    "protected_splits_opened": [],
                }
                checks = _recompute_result_checks(result)
                passed = all(checks.values())
                result.update(
                    {
                        "checks": checks,
                        "passed": passed,
                        "status": PASS_STATUS if passed else FAIL_STATUS,
                        "mechanics_protocol_authorized": passed,
                    }
                )
                validate_result_payload(result)
            except BaseException as exc:
                error = exc
        distributed.phase_consensus(
            context,
            phase="development-result-finalize",
            error=error,
        )
        result_payload: list[Mapping[str, Any] | None] = [result]
        dist.broadcast_object_list(
            result_payload,
            src=0,
            group=context.control_group,
        )
        if result_payload[0] is None:
            raise RuntimeError("Development result broadcast returned no payload")
        unsigned_result = dict(result_payload[0])
        error = None
        try:
            validate_result_payload(unsigned_result)
        except BaseException as exc:
            error = exc
        distributed.phase_consensus(
            context,
            phase="development-result-broadcast-validate",
            error=error,
        )
        unsigned_digest = canonical_sha256(unsigned_result)
        distributed.require_consensus(
            context,
            unsigned_digest,
            description="development unsigned-result digest",
        )
        signed_result = {
            **unsigned_result,
            "receipt": {
                "algorithm": "sha256",
                "payload_scope": "canonical_result_without_receipt",
                "payload_sha256": unsigned_digest,
            },
        }
        validate_result_payload(signed_result)
        error = None
        if context.is_primary:
            try:
                _write_signed_json(output_dir / "result.json", signed_result)
            except BaseException as exc:
                error = exc
        distributed.phase_consensus(
            context,
            phase="development-result-write",
            error=error,
        )
        persisted_result = None
        persisted_audit = None
        error = None
        try:
            result_path = output_dir / "result.json"
            persisted_result = json.loads(result_path.read_text(encoding="utf-8"))
            validate_result_payload(persisted_result)
            persisted_audit = {
                "file_sha256": sha256_file(result_path),
                "payload_sha256": persisted_result["receipt"]["payload_sha256"],
            }
            if persisted_result != signed_result:
                raise ValueError("Persisted development result differs from broadcast")
        except BaseException as exc:
            error = exc
        distributed.phase_consensus(
            context,
            phase="development-result-reread",
            error=error,
        )
        distributed.require_consensus(
            context,
            persisted_audit,
            description="development persisted-result receipt",
        )
        assert persisted_result is not None
        return persisted_result
    finally:
        distributed.destroy_distributed_training(context)
