#!/usr/bin/env python3
"""Run the sealed continuous-write full64-to-causal32 retrieval gate."""

from __future__ import annotations

import argparse
import hashlib
import json
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


SCHEMA = "rwkv_ms_natural_memory_native_rwkv_continuous_write_retrieval.v1"
FEATURE_SCHEMA = (
    "rwkv_ms_natural_memory_native_rwkv_continuous_write_retrieval_feature.v1"
)
SHARD_SCHEMA = (
    "rwkv_ms_natural_memory_native_rwkv_continuous_write_retrieval_shard.v1"
)
MAP_SCHEMA = "rwkv_ms_natural_memory_native_rwkv_continuous_write_maps.v1"
PROTOCOL = SCRIPT_DIR / (
    "natural_memory_native_rwkv_continuous_write_retrieval_protocol_v1.json"
)
PROTOCOL_PAYLOAD_SHA256 = "8f16350f5c665034d27fe4b22cdfbedc615d00b5afeaa5d453381b5018b22024"
PROTOCOL_FILE_SHA256 = "8b5c3a1920963652466a53615a4ebfd0910a052bdad1762427e88e17159a979f"
WORLD_SIZE = 4
TIMEOUT_SECONDS = 1800
SEED = 149
HF_ENDPOINT = "https://hf-mirror.com"
FIT_ROWS = 64
RETRIEVAL_ROWS = 32
MODULES = 42
ADDRESS_DIM = 64
STATE_DIM = 32
MAP_RANK = 16
RIDGE = 1.0
DONOR_POSITIVE_ROW_FRACTION_MINIMUM = 0.95
DONOR_MEAN_GAP_MINIMUM = 0.05
LAYER_PERMUTED_POSITIVE_ROW_FRACTION_MINIMUM = 0.95
LAYER_PERMUTED_MEAN_GAP_MINIMUM = 0.05
DEFAULT_BASE_MODEL = Path("/root/X/.cache/hf/gemma-4-E4B-it-a4c2d58")
DEFAULT_MATERIALIZATION = SCRIPT_DIR / (
    "local_artifacts/natural_memory_native_rwkv_continuous_write_open_fit_v1"
)
DEFAULT_OUTPUT = SCRIPT_DIR / (
    "local_artifacts/natural_memory_native_rwkv_continuous_write_retrieval_v1"
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


def _broadcast_primary_object(context: Any, value: Any) -> Any:
    payload = [value if context.is_primary else None]
    dist.broadcast_object_list(payload, src=0, group=context.control_group)
    return payload[0]


def _dependency_paths() -> Mapping[str, Path]:
    return {
        "bias_free_reduced_rank_ridge_and_metrics": Path(alignment.__file__).resolve(),
        "full_address_latch_and_runtime_conditioner": Path(
            integration.__file__
        ).resolve(),
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
        "signed_ordered_module_and_read_disable_helpers": Path(
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
            "path": str(path),
            "basename": path.name,
            "sha256": sha256_file(path),
        }
        for role, path in _dependency_paths().items()
    ]


def _validate_source_dependencies(protocol: Mapping[str, Any]) -> None:
    declared = protocol.get("source_dependencies")
    if not isinstance(declared, list):
        raise ValueError("Continuous-write protocol source dependencies are missing")
    by_role = {
        str(item.get("role")): item for item in declared if isinstance(item, Mapping)
    }
    paths = _dependency_paths()
    if set(by_role) != set(paths):
        raise ValueError("Continuous-write protocol dependency closure differs")
    for role, path in paths.items():
        item = by_role[role]
        if item.get("basename") != path.name or item.get("sha256") != sha256_file(path):
            raise ValueError(f"Continuous-write dependency differs: {role}")


def validate_protocol(base_model: Path) -> Mapping[str, Any]:
    if sha256_file(PROTOCOL) != PROTOCOL_FILE_SHA256:
        raise ValueError("Continuous-write retrieval protocol file hash differs")
    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    _validate_receipt(
        protocol,
        payload_scope="canonical_protocol_without_receipt",
        description="Continuous-write retrieval protocol",
    )
    if protocol["receipt"]["payload_sha256"] != PROTOCOL_PAYLOAD_SHA256:
        raise ValueError("Continuous-write retrieval protocol payload hash differs")
    authorization = protocol.get("authorization_basis", {})
    frozen = protocol.get("frozen_inputs", {})
    fit = protocol.get("reduced_rank_fit", {})
    gates = protocol.get("retrieval_gate", {}).get("gates", {})
    execution = protocol.get("execution", {})
    firewall = protocol.get("sealed_firewall", {})
    staged = firewall.get("staged_access", {})
    if (
        protocol.get("schema")
        != "rwkv_ms_natural_memory_native_rwkv_continuous_write_retrieval_protocol.v1"
        or authorization.get("exact_v5_result_sha256") != exact_v5.V5_RESULT_SHA256
        or authorization.get("exact_v5_result_receipt") != exact_v5.V5_RESULT_RECEIPT
        or authorization.get("continuous_write_parent_commit")
        != subprocess.check_output(
            ["git", "-C", str(PROJECT_ROOT), "rev-parse", "HEAD"], text=True
        ).strip()
        or frozen.get("base_config_sha256")
        != sha256_file(base_model / "config.json")
        or frozen.get("adapter_weights_sha256")
        != exact_v5.V5_ADAPTER_WEIGHTS_SHA256
        or frozen.get("adapter_config_sha256")
        != exact_v5.V5_ADAPTER_CONFIG_SHA256
        or frozen.get("required_source_root_environment") != SIGNED_SOURCE_ROOT_ENV
        or frozen.get("signed_v5_source_commit") != exact_v5.SIGNED_V5_COMMIT
        or frozen.get("signed_v5_delta_impl_sha256")
        != exact_v5.SIGNED_V5_DELTA_IMPL_SHA256
        or frozen.get("projected_kv_key_dim") != ADDRESS_DIM
        or frozen.get("state_read_dim") != STATE_DIM
        or frozen.get("rwkv_num_states") != 4
        or frozen.get("hybrid_mode") != "address_keyed_moe_deepembed_ffn"
        or fit.get("fit_rows") != FIT_ROWS
        or fit.get("rank") != MAP_RANK
        or float(fit.get("ridge", -1.0)) != RIDGE
        or fit.get("map_weights_saved_before_retrieval_gate") is not False
        or gates.get("donor_positive_row_fraction_minimum")
        != DONOR_POSITIVE_ROW_FRACTION_MINIMUM
        or gates.get("donor_mean_gap_minimum") != DONOR_MEAN_GAP_MINIMUM
        or gates.get("layer_permuted_positive_row_fraction_minimum")
        != LAYER_PERMUTED_POSITIVE_ROW_FRACTION_MINIMUM
        or gates.get("layer_permuted_mean_gap_minimum")
        != LAYER_PERMUTED_MEAN_GAP_MINIMUM
        or execution.get("world_size") != WORLD_SIZE
        or execution.get("backend") != "nccl"
        or execution.get("control_backend") != "gloo"
        or execution.get("hf_endpoint") != HF_ENDPOINT
        or execution.get("fit_capture_and_map_freeze_complete_before_retrieval_bundle_open")
        is not True
        or staged.get("fit_stage_byte_read_files") != ["manifest.json", "fit.jsonl"]
        or staged.get("retrieval_stage_newly_opened_after_frozen_map_digest")
        != ["retrieval.jsonl"]
        or staged.get("sealed_paths_statted_listed_hashed_or_opened") is not False
        or protocol.get("mechanics_authorized") is not False
        or protocol.get("causal_authorized") is not False
        or protocol.get("model_or_adapter_training_authorized") is not False
        or protocol.get("generation_authorized") is not False
        or protocol.get("native_benchmark_authorized") is not False
        or protocol.get("protected_splits_opened_by_this_protocol") != []
    ):
        raise ValueError("Continuous-write retrieval protocol contract differs")
    _validate_source_dependencies(protocol)
    exact_v5.validate_protocol()
    return protocol


def _load_manifest_only(
    materialization_root: Path,
    protocol: Mapping[str, Any],
) -> Mapping[str, Any]:
    manifest_path = materialization_root / "manifest.json"
    authorization = protocol["authorization_basis"]
    if sha256_file(manifest_path) != authorization["continuous_write_manifest_file_sha256"]:
        raise ValueError("Continuous-write materialization manifest hash differs")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    materializer._validate_manifest(manifest)
    if (
        manifest["receipt"]["payload_sha256"]
        != authorization["continuous_write_manifest_receipt"]
        or manifest["split_contract"]["schema"]
        != authorization["continuous_write_split_schema"]
        or manifest["split_contract"]["receipt"]["payload_sha256"]
        != authorization["continuous_write_split_receipt"]
        or manifest["source"]["namespace"]
        != protocol["open_fit_materialization"]["dataset_namespace"]
        or manifest["source"]["sha256"]
        != protocol["open_fit_materialization"]["dataset_sha256"]
        or manifest["source"]["rows"]
        != protocol["open_fit_materialization"]["dataset_rows"]
        or manifest.get("protected_splits_opened") != []
    ):
        raise ValueError("Continuous-write materialization manifest binding differs")
    return manifest


def _load_open_bundle(
    materialization_root: Path,
    manifest: Mapping[str, Any],
    protocol: Mapping[str, Any],
    split: str,
) -> list[dict[str, Any]]:
    if split not in {"fit", "retrieval"}:
        raise PermissionError("Only FIT and retrieval bundles may be opened")
    binding = protocol["open_fit_materialization"][f"{split}_bundle"]
    manifest_binding = manifest["file_inventory"]["bundles"][split]
    if any(
        binding.get(key) != manifest_binding.get(key)
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
        raise ValueError(f"Continuous-write {split} protocol binding differs")
    return materializer._read_bundle(materialization_root, manifest, split)


def _encode_rows(tokenizer: Any, rows: Sequence[Mapping[str, Any]]) -> dict[int, Any]:
    examples: dict[int, Any] = {}
    for row in rows:
        source = int(row["source_index"])
        example = evolution.encode_native_full_row(
            tokenizer,
            task="scene",
            source_ordinal=source,
            raw_line=str(row["raw_line"]),
        )
        if example.row_sha256 != row["row_sha256"]:
            raise ValueError("Continuous-write encoded row hash differs")
        examples[source] = example
    if len(examples) != len(rows):
        raise ValueError("Continuous-write encoded source coverage differs")
    return examples


def first_prompt_boundary(labels: torch.Tensor) -> tuple[int, int]:
    if labels.ndim != 2 or labels.size(0) != 1 or labels.size(1) < 2:
        raise ValueError("Continuous-write capture requires one causal label row")
    supervised = labels[0].ne(-100).nonzero(as_tuple=False).flatten()
    if supervised.numel() < 1:
        raise ValueError("Continuous-write row has no supervised label")
    first_label = int(supervised[0].item())
    predictor = first_label - 1
    shifted = labels[:, 1:].ne(-100)
    shifted_indices = shifted[0].nonzero(as_tuple=False).flatten()
    if (
        predictor < 0
        or shifted_indices.numel() < 1
        or int(shifted_indices[0].item()) != predictor
        or int(labels[0, predictor].item()) != -100
    ):
        raise RuntimeError("Continuous-write causal predictor boundary differs")
    return first_label, predictor


def _raw_tensor_bytes_equal(left: torch.Tensor, right: torch.Tensor) -> bool:
    return (
        tuple(left.shape) == tuple(right.shape)
        and left.dtype == right.dtype
        and torch.equal(
            left.detach().contiguous().view(torch.uint8),
            right.detach().contiguous().view(torch.uint8),
        )
    )


def _observed_read_basis(
    module: Any,
    state: torch.Tensor,
    memory_source_seq: torch.Tensor,
    token_mask: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    result = module.rwkv_continuous_retrieval_original_read_basis(
        state, memory_source_seq, token_mask
    )
    if not isinstance(result, tuple) or len(result) != 3:
        raise RuntimeError("Continuous-write RWKV read-basis return contract differs")
    capture_index = module.rwkv_continuous_retrieval_predictor_index
    if capture_index is None:
        return result
    call = int(module.rwkv_continuous_retrieval_read_basis_calls)
    if call >= 2:
        raise RuntimeError("Continuous-write read-basis observer ran more than twice")
    receptance = result[0]
    if receptance.ndim != 4 or not 0 <= int(capture_index) < receptance.size(1):
        raise RuntimeError("Continuous-write RWKV receptance capture shape differs")
    selected = receptance[:, int(capture_index)].flatten(start_dim=1)
    if tuple(selected.shape) != (1, STATE_DIM):
        raise RuntimeError("Continuous-write selected RWKV receptance width differs")
    if call == 0:
        module.rwkv_continuous_retrieval_first_result = tuple(
            tensor.detach().clone() for tensor in result
        )
        module.rwkv_continuous_retrieval_receptance = selected.detach().clone()
        module.rwkv_continuous_retrieval_full_bytes_identical = False
    else:
        first = module.rwkv_continuous_retrieval_first_result
        if not isinstance(first, tuple) or len(first) != len(result):
            raise RuntimeError("Continuous-write first read-basis snapshot is missing")
        identical = all(
            _raw_tensor_bytes_equal(left, right)
            for left, right in zip(first, result, strict=True)
        )
        selected_first = module.rwkv_continuous_retrieval_receptance
        if not identical or not _raw_tensor_bytes_equal(selected_first, selected):
            raise RuntimeError("Addressed and global RWKV read-basis bytes differ")
        module.rwkv_continuous_retrieval_full_bytes_identical = True
        module.rwkv_continuous_retrieval_first_result = None
    module.rwkv_continuous_retrieval_result_shapes = [
        list(tensor.shape) for tensor in result
    ]
    module.rwkv_continuous_retrieval_result_dtypes = [
        str(tensor.dtype) for tensor in result
    ]
    module.rwkv_continuous_retrieval_read_basis_calls = call + 1
    return result


def install_read_observer(model: torch.nn.Module) -> Mapping[str, Any]:
    modules = causal_train.ordered_modules(model)
    installed: list[str] = []
    for name, module in modules:
        if hasattr(module, "rwkv_continuous_retrieval_original_read_basis"):
            raise ValueError(f"Continuous-write read observer already installed: {name}")
        module.rwkv_continuous_retrieval_original_read_basis = (
            module._rwkv_ms_token_state_read_basis
        )
        module.rwkv_continuous_retrieval_predictor_index = None
        module.rwkv_continuous_retrieval_read_basis_calls = 0
        module.rwkv_continuous_retrieval_first_result = None
        module.rwkv_continuous_retrieval_receptance = None
        module.rwkv_continuous_retrieval_full_bytes_identical = False
        module.rwkv_continuous_retrieval_result_shapes = None
        module.rwkv_continuous_retrieval_result_dtypes = None
        module._rwkv_ms_token_state_read_basis = MethodType(
            _observed_read_basis, module
        )
        installed.append(name)
    return {
        "modules": len(installed),
        "module_names": installed,
        "observer": "all42_prompt_boundary_exact_v5_r_seq_double_call",
        "forward_output_changed": False,
        "persisted_full_sequence_or_state_features": False,
    }


def _prepare_read_observer(
    modules: Sequence[tuple[str, Any]], predictor_index: int
) -> None:
    for _, module in modules:
        module.rwkv_continuous_retrieval_predictor_index = int(predictor_index)
        module.rwkv_continuous_retrieval_read_basis_calls = 0
        module.rwkv_continuous_retrieval_first_result = None
        module.rwkv_continuous_retrieval_receptance = None
        module.rwkv_continuous_retrieval_full_bytes_identical = False
        module.rwkv_continuous_retrieval_result_shapes = None
        module.rwkv_continuous_retrieval_result_dtypes = None


def _clear_read_observer(modules: Sequence[tuple[str, Any]]) -> None:
    for _, module in modules:
        module.rwkv_continuous_retrieval_predictor_index = None
        module.rwkv_continuous_retrieval_read_basis_calls = 0
        module.rwkv_continuous_retrieval_first_result = None
        module.rwkv_continuous_retrieval_receptance = None
        module.rwkv_continuous_retrieval_full_bytes_identical = False
        module.rwkv_continuous_retrieval_result_shapes = None
        module.rwkv_continuous_retrieval_result_dtypes = None


def _no_identity_binder_or_feedback(model: torch.nn.Module) -> bool:
    modules = causal_train.ordered_modules(model)
    forbidden_model_attributes = (
        "rwkv_identity_binder_bank",
        "rwkv_full_bandwidth_feedback",
        "rwkv_read_feedback_bridge",
    )
    if any(hasattr(model, name) for name in forbidden_model_attributes):
        return False
    return all(
        getattr(module, "rwkv_query_state_identity_fixed_address", None) is None
        and not hasattr(module, "rwkv_identity_binder")
        and not hasattr(module, "rwkv_full_bandwidth_feedback")
        for _, module in modules
    )


def _parameter_versions(model: torch.nn.Module) -> tuple[tuple[str, int], ...]:
    return tuple((name, int(parameter._version)) for name, parameter in model.named_parameters())


@torch.no_grad()
def capture_row(
    model: torch.nn.Module,
    example: Any,
    *,
    pad_token_id: int,
    device: torch.device,
) -> Mapping[str, Any]:
    modules = causal_train.ordered_modules(model)
    batch = evolution.collate_native_examples(
        [example], pad_token_id=pad_token_id, device=device
    )
    try:
        _clear_read_observer(modules)
        integration.set_mode(model, integration.INHERITED_EXACT_V5_MODE)
        integration.set_capture(model, True)
        evolution._native_write(model, batch, dtype=torch.bfloat16)
        if not _no_identity_binder_or_feedback(model):
            raise RuntimeError("Continuous-write capture installed a binder or feedback path")
        addresses: list[torch.Tensor] = []
        latch_versions: list[int] = []
        for name, module in modules:
            latch = module.rwkv_continuous_write_latch
            audit = module.rwkv_continuous_write_audit
            if latch is None or not isinstance(audit, Mapping):
                raise RuntimeError(f"Continuous-write latch or audit is missing: {name}")
            if (
                audit.get("mode") != integration.INHERITED_EXACT_V5_MODE
                or audit.get("conditioner_address") is not latch.address_seq
                or audit.get("conditioner_address_object_id") != id(latch.address_seq)
                or audit.get("latched_address_object_id") != id(latch.address_seq)
                or not torch.equal(audit["latched_selected_keys"], latch.selected_keys)
                or not torch.equal(audit["conditioner_address_value"], latch.address_seq)
                or latch.address_seq._version != latch.address_version
                or latch.folded_address_seq._version != latch.folded_address_version
            ):
                raise RuntimeError(f"Continuous-write immutable latch contract differs: {name}")
            selected = latch.selected_keys
            if tuple(selected.shape) != (1, 1, ADDRESS_DIM):
                raise RuntimeError(f"Continuous-write full address shape differs: {name}")
            addresses.append(selected[0, 0].float().detach().clone())
            latch_versions.append(int(latch.address_seq._version))
        first_label, predictor_index = first_prompt_boundary(batch.labels)
        _prepare_read_observer(modules, predictor_index)
        logits = evolution._native_read(model, batch, dtype=torch.bfloat16)
        del logits
        if not all(getattr(module, "write_enabled", None) is False for _, module in modules):
            raise RuntimeError("Continuous-write causal read left memory writes enabled")
        receptance: list[torch.Tensor] = []
        read_calls: list[int] = []
        full_bytes: list[bool] = []
        result_shapes: list[list[list[int]]] = []
        result_dtypes: list[list[str]] = []
        for index, (name, module) in enumerate(modules):
            selected = module.rwkv_continuous_retrieval_receptance
            calls = int(module.rwkv_continuous_retrieval_read_basis_calls)
            identical = bool(module.rwkv_continuous_retrieval_full_bytes_identical)
            latch = module.rwkv_continuous_write_latch
            if (
                calls != 2
                or selected is None
                or tuple(selected.shape) != (1, STATE_DIM)
                or not identical
                or module.rwkv_continuous_retrieval_first_result is not None
                or latch is None
                or int(latch.address_seq._version) != latch_versions[index]
                or int(latch.address_seq._version) != latch.address_version
            ):
                raise RuntimeError(f"Continuous-write causal read capture differs: {name}")
            receptance.append(selected[0].float().detach().clone())
            read_calls.append(calls)
            full_bytes.append(identical)
            result_shapes.append(module.rwkv_continuous_retrieval_result_shapes)
            result_dtypes.append(module.rwkv_continuous_retrieval_result_dtypes)
        address_tensor = torch.stack(addresses)
        receptance_tensor = torch.stack(receptance)
        if (
            tuple(address_tensor.shape) != (MODULES, ADDRESS_DIM)
            or tuple(receptance_tensor.shape) != (MODULES, STATE_DIM)
            or not bool(torch.isfinite(address_tensor).all().item())
            or not bool(torch.isfinite(receptance_tensor).all().item())
            or bool(address_tensor.norm(dim=-1).le(0.0).any().item())
            or bool(receptance_tensor.norm(dim=-1).le(0.0).any().item())
        ):
            raise RuntimeError("Continuous-write captured features are invalid")
        return {
            "write_address_full64": address_tensor.cpu().tolist(),
            "causal_prompt_boundary_receptance32": receptance_tensor.cpu().tolist(),
            "first_supervised_label_index": first_label,
            "prompt_boundary_predictor_index": predictor_index,
            "predictor_definition": "first_supervised_label_index_minus_one",
            "write_passes": 1,
            "read_passes": 1,
            "read_basis_calls_per_module": read_calls,
            "read_basis_call_roles": ["addressed_recurrent_read", "moe_global_recurrent_read"],
            "addressed_global_full_return_raw_bytes_identical": all(full_bytes),
            "read_basis_result_shapes_per_module": result_shapes,
            "read_basis_result_dtypes_per_module": result_dtypes,
            "latched_address_object_identity_verified": True,
            "latched_address_versions_unchanged": True,
            "read_writes_enabled": False,
            "features_detached_and_cloned": True,
            "model_output_changed_by_observer": False,
            "binder_or_feedback_installed": False,
            "continuous_write_mode": integration.INHERITED_EXACT_V5_MODE,
        }
    finally:
        _clear_read_observer(modules)
        reset_delta_mem_states(model)
        evolution.release_native_row_allocator_cache(device)


def _feature_row(
    row: Mapping[str, Any],
    feature: Mapping[str, Any],
    *,
    process_rank: int,
) -> dict[str, Any]:
    return {
        "schema": FEATURE_SCHEMA,
        "capture_rank": process_rank,
        "split": row["split"],
        "source_index": int(row["source_index"]),
        "qualified_source_id": row["qualified_source_id"],
        "row_sha256": row["row_sha256"],
        "donor_source_index": int(row["donor_source_index"]),
        "qualified_donor_source_id": row["qualified_donor_source_id"],
        "donor_row_sha256": row["donor_row_sha256"],
        **dict(feature),
    }


def _write_stage_shard(
    output_dir: Path,
    *,
    split: str,
    process_rank: int,
    rows: Sequence[Mapping[str, Any]],
) -> Mapping[str, Any]:
    shard: dict[str, Any] = {
        "schema": SHARD_SCHEMA,
        "split": split,
        "rank": process_rank,
        "world_size": WORLD_SIZE,
        "assignment": "source_index_modulo_4",
        "rows": list(rows),
        "mechanics_or_causal_rows_opened": False,
    }
    shard["receipt"] = {
        "algorithm": "sha256",
        "payload_scope": "canonical_feature_shard_without_receipt",
        "payload_sha256": canonical_sha256(shard),
    }
    path = output_dir / f"{split}-shard-{process_rank}.json"
    _atomic_signed_json(path, shard)
    return {
        "path": str(path),
        "rows": len(rows),
        "sha256": sha256_file(path),
        "receipt": shard["receipt"]["payload_sha256"],
    }


def _validate_feature_row(row: Mapping[str, Any], *, split: str, rank: int) -> None:
    address = torch.tensor(row.get("write_address_full64"), dtype=torch.float32)
    receptance = torch.tensor(
        row.get("causal_prompt_boundary_receptance32"), dtype=torch.float32
    )
    if (
        row.get("schema") != FEATURE_SCHEMA
        or row.get("split") != split
        or int(row.get("capture_rank", -1)) != rank
        or int(row.get("source_index", -1)) % WORLD_SIZE != rank
        or tuple(address.shape) != (MODULES, ADDRESS_DIM)
        or tuple(receptance.shape) != (MODULES, STATE_DIM)
        or not bool(torch.isfinite(address).all().item())
        or not bool(torch.isfinite(receptance).all().item())
        or bool(address.norm(dim=-1).le(0.0).any().item())
        or bool(receptance.norm(dim=-1).le(0.0).any().item())
        or row.get("read_basis_calls_per_module") != [2] * MODULES
        or row.get("read_basis_call_roles")
        != ["addressed_recurrent_read", "moe_global_recurrent_read"]
        or row.get("addressed_global_full_return_raw_bytes_identical") is not True
        or row.get("latched_address_object_identity_verified") is not True
        or row.get("latched_address_versions_unchanged") is not True
        or row.get("read_writes_enabled") is not False
        or row.get("model_output_changed_by_observer") is not False
        or row.get("binder_or_feedback_installed") is not False
        or row.get("continuous_write_mode") != integration.INHERITED_EXACT_V5_MODE
    ):
        raise ValueError("Continuous-write feature row contract differs")


def load_stage_shards(
    output_dir: Path,
    *,
    split: str,
    source_rows: Sequence[Mapping[str, Any]],
) -> tuple[list[Mapping[str, Any]], list[Mapping[str, Any]], str]:
    expected = {int(row["source_index"]): row for row in source_rows}
    records: list[Mapping[str, Any]] = []
    provenance: list[Mapping[str, Any]] = []
    for rank in range(WORLD_SIZE):
        path = output_dir / f"{split}-shard-{rank}.json"
        shard = json.loads(path.read_text(encoding="utf-8"))
        _validate_receipt(
            shard,
            payload_scope="canonical_feature_shard_without_receipt",
            description=f"Continuous-write {split} shard {rank}",
        )
        if (
            shard.get("schema") != SHARD_SCHEMA
            or shard.get("split") != split
            or shard.get("rank") != rank
            or shard.get("world_size") != WORLD_SIZE
            or shard.get("assignment") != "source_index_modulo_4"
            or shard.get("mechanics_or_causal_rows_opened") is not False
        ):
            raise ValueError("Continuous-write feature shard contract differs")
        for row in shard["rows"]:
            _validate_feature_row(row, split=split, rank=rank)
            source = int(row["source_index"])
            parent = expected.get(source)
            if parent is None or any(
                row.get(key) != parent.get(key)
                for key in (
                    "qualified_source_id",
                    "row_sha256",
                    "donor_source_index",
                    "qualified_donor_source_id",
                    "donor_row_sha256",
                )
            ):
                raise ValueError("Continuous-write feature source binding differs")
            records.append(row)
        provenance.append(
            {
                "path": str(path),
                "rows": len(shard["rows"]),
                "sha256": sha256_file(path),
                "receipt": shard["receipt"]["payload_sha256"],
            }
        )
    sources = [int(row["source_index"]) for row in records]
    if len(records) != len(expected) or set(sources) != set(expected) or len(set(sources)) != len(sources):
        raise ValueError("Continuous-write feature shard coverage differs")
    ordered = sorted(records, key=lambda row: int(row["source_index"]))
    return ordered, provenance, canonical_sha256(ordered)


def _feature_tensors(
    records: Sequence[Mapping[str, Any]],
) -> tuple[torch.Tensor, torch.Tensor]:
    addresses = torch.tensor(
        [row["write_address_full64"] for row in records], dtype=torch.float32
    )
    receptance = torch.tensor(
        [row["causal_prompt_boundary_receptance32"] for row in records],
        dtype=torch.float32,
    )
    return addresses, receptance


def fit_maps(
    records: Sequence[Mapping[str, Any]], module_names: Sequence[str]
) -> dict[str, alignment.FrozenMapWeights]:
    if len(records) != FIT_ROWS or any(row.get("split") != "fit" for row in records):
        raise ValueError("Continuous-write map fit requires only 64 FIT rows")
    addresses, receptance = _feature_tensors(records)
    return alignment.fit_layer_maps(
        addresses,
        receptance,
        module_names,
        rank=MAP_RANK,
        ridge=RIDGE,
    )


def map_digest(
    maps: Mapping[str, alignment.FrozenMapWeights], module_names: Sequence[str]
) -> str:
    if set(maps) != set(module_names):
        raise ValueError("Continuous-write map digest inventory differs")
    digest = hashlib.sha256()
    for name in module_names:
        digest.update(name.encode("utf-8"))
        digest.update(b"\0")
        for tensor in (maps[name].down, maps[name].up):
            value = tensor.detach().cpu().float().contiguous()
            digest.update(str(tuple(value.shape)).encode("ascii"))
            digest.update(value.numpy().tobytes(order="C"))
    return digest.hexdigest()


def retrieval_analysis(
    records: Sequence[Mapping[str, Any]],
    module_names: Sequence[str],
    maps: Mapping[str, alignment.FrozenMapWeights],
) -> Mapping[str, Any]:
    if len(records) != RETRIEVAL_ROWS or any(
        row.get("split") != "retrieval" for row in records
    ):
        raise ValueError("Continuous-write retrieval requires only 32 retrieval rows")
    addresses, receptance = _feature_tensors(records)
    source_positions = {
        int(row["source_index"]): index for index, row in enumerate(records)
    }
    donor_indices = torch.tensor(
        [source_positions[int(row["donor_source_index"])] for row in records],
        dtype=torch.long,
    )
    query = alignment._rms_normalize(receptance)
    correct_direction = alignment.apply_layer_maps(addresses, module_names, maps)
    donor_direction = correct_direction.index_select(0, donor_indices)
    permuted_direction = alignment.apply_layer_maps(
        addresses.roll(1, dims=1), module_names, maps
    )
    zero_direction = alignment.apply_layer_maps(
        torch.zeros(1, len(module_names), ADDRESS_DIM), module_names, maps
    )
    denominator = float(STATE_DIM)
    correct = (query * correct_direction).sum(dim=-1) / denominator
    donor = (query * donor_direction).sum(dim=-1) / denominator
    permuted = (query * permuted_direction).sum(dim=-1) / denominator
    donor_gap = correct - donor
    permuted_gap = correct - permuted
    finite = all(
        bool(torch.isfinite(value).all().item())
        for value in (
            correct_direction,
            donor_direction,
            permuted_direction,
            correct,
            donor,
            permuted,
            donor_gap,
            permuted_gap,
        )
    )
    active_nonzero = all(
        bool(direction.norm(dim=-1).gt(0.0).all().item())
        for direction in (correct_direction, donor_direction, permuted_direction)
    )
    zero_exact = torch.equal(zero_direction, torch.zeros_like(zero_direction))
    aggregate = {
        "finite": finite,
        "correct_mean_score": float(correct.mean()),
        "matched_donor_mean_score": float(donor.mean()),
        "layer_permuted_mean_score": float(permuted.mean()),
        "donor_positive_module_fraction": float(donor_gap.gt(0.0).float().mean()),
        "donor_positive_row_fraction": float(
            donor_gap.mean(dim=1).gt(0.0).float().mean()
        ),
        "donor_mean_gap": float(donor_gap.mean()),
        "layer_permuted_positive_module_fraction": float(
            permuted_gap.gt(0.0).float().mean()
        ),
        "layer_permuted_positive_row_fraction": float(
            permuted_gap.mean(dim=1).gt(0.0).float().mean()
        ),
        "layer_permuted_mean_gap": float(permuted_gap.mean()),
        "all_active_mapped_directions_nonzero": active_nonzero,
        "exact_zero_address_maps_to_exact_zero": zero_exact,
    }
    checks = {
        "all_scores_and_directions_finite": finite,
        "all_active_mapped_directions_nonzero": active_nonzero,
        "exact_zero_address_maps_to_exact_zero": zero_exact,
        "donor_positive_row_fraction": (
            aggregate["donor_positive_row_fraction"]
            >= DONOR_POSITIVE_ROW_FRACTION_MINIMUM
        ),
        "donor_mean_gap": aggregate["donor_mean_gap"] >= DONOR_MEAN_GAP_MINIMUM,
        "layer_permuted_positive_row_fraction": (
            aggregate["layer_permuted_positive_row_fraction"]
            >= LAYER_PERMUTED_POSITIVE_ROW_FRACTION_MINIMUM
        ),
        "layer_permuted_mean_gap": (
            aggregate["layer_permuted_mean_gap"]
            >= LAYER_PERMUTED_MEAN_GAP_MINIMUM
        ),
    }
    per_row = [
        {
            "source_index": int(row["source_index"]),
            "donor_source_index": int(row["donor_source_index"]),
            "correct_mean_score": float(correct[index].mean()),
            "matched_donor_mean_score": float(donor[index].mean()),
            "layer_permuted_mean_score": float(permuted[index].mean()),
            "correct_minus_donor_mean_gap": float(donor_gap[index].mean()),
            "correct_minus_layer_permuted_mean_gap": float(
                permuted_gap[index].mean()
            ),
            "donor_positive": bool(donor_gap[index].mean().gt(0.0).item()),
            "layer_permuted_positive": bool(
                permuted_gap[index].mean().gt(0.0).item()
            ),
        }
        for index, row in enumerate(records)
    ]
    return {
        "aggregate": aggregate,
        "checks": checks,
        "per_row": per_row,
        "passed": all(checks.values()),
        "evaluation_calls": 1,
    }


def _save_maps(
    output_dir: Path,
    maps: Mapping[str, alignment.FrozenMapWeights],
    module_names: Sequence[str],
    frozen_digest: str,
) -> Mapping[str, Any]:
    path = output_dir / "continuous-write-maps.pt"
    payload = {
        "schema": MAP_SCHEMA,
        "module_names": list(module_names),
        "rank": MAP_RANK,
        "ridge": RIDGE,
        "address_dim": ADDRESS_DIM,
        "state_dim": STATE_DIM,
        "frozen_map_digest": frozen_digest,
        "maps": {
            name: {
                "down": maps[name].down.detach().cpu().float().contiguous(),
                "up": maps[name].up.detach().cpu().float().contiguous(),
            }
            for name in module_names
        },
    }
    torch.save(payload, path)
    return {
        "saved": True,
        "path": str(path),
        "sha256": sha256_file(path),
        "frozen_map_digest": frozen_digest,
    }


def _validate_result(path: Path) -> Mapping[str, Any]:
    result = json.loads(path.read_text(encoding="utf-8"))
    _validate_receipt(
        result,
        payload_scope="canonical_result_without_receipt",
        description="Continuous-write retrieval result",
    )
    passed = result.get("passed") is True
    map_artifact = result.get("map_artifact", {})
    if (
        result.get("schema") != SCHEMA
        or result.get("protocol_payload_sha256") != PROTOCOL_PAYLOAD_SHA256
        or result.get("protocol_file_sha256") != PROTOCOL_FILE_SHA256
        or result.get("retrieval_evaluation_calls") != 1
        or result.get("mechanics_protocol_drafting_authorized") is not passed
        or result.get("mechanics_bytes_open_authorized") is not False
        or result.get("mechanics_authorized") is not False
        or result.get("causal_authorized") is not False
        or result.get("model_or_adapter_training_authorized") is not False
        or result.get("generation_authorized") is not False
        or result.get("native_benchmark_authorized") is not False
        or result.get("protected_splits_opened") != []
        or bool(map_artifact.get("saved")) is not passed
    ):
        raise ValueError("Continuous-write retrieval result contract differs")
    if passed and sha256_file(Path(map_artifact["path"])) != map_artifact["sha256"]:
        raise ValueError("Continuous-write frozen map artifact hash differs")
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
        def validate_preflight() -> tuple[Mapping[str, Any], Mapping[str, Any], Mapping[str, Any]]:
            if (
                context.world_size != WORLD_SIZE
                or context.backend != "nccl"
                or context.control_backend != "gloo"
                or not hardware.four_distinct_a100s(context.rank_devices)
            ):
                raise RuntimeError("Continuous-write retrieval requires four distinct A100s")
            if os.environ.get("HF_ENDPOINT") != HF_ENDPOINT:
                raise RuntimeError(f"HF_ENDPOINT must be exactly {HF_ENDPOINT}")
            protocol = validate_protocol(base_model)
            source_audit = exact_v5.validate_execution_source()
            manifest = _load_manifest_only(materialization_root, protocol)
            return protocol, source_audit, manifest

        protocol, source_audit, manifest = _consensual_operation(
            context,
            phase="continuous-write-retrieval-preflight",
            operation=validate_preflight,
        )
        preflight_binding = canonical_sha256(
            {
                "protocol": protocol["receipt"]["payload_sha256"],
                "source": source_audit,
                "manifest": manifest["receipt"]["payload_sha256"],
            }
        )
        distributed.require_consensus(
            context, preflight_binding, description="continuous-write preflight"
        )

        def create_output() -> None:
            if not context.is_primary:
                return
            if output_dir.exists():
                raise ValueError(f"Continuous-write output must be fresh: {output_dir}")
            output_dir.mkdir(parents=True, exist_ok=False)

        _consensual_operation(
            context, phase="continuous-write-output-create", operation=create_output
        )

        def load_runtime() -> tuple[torch.nn.Module, Any, Mapping[str, Any], Mapping[str, Any], Mapping[str, Any], tuple[tuple[str, int], ...]]:
            torch.manual_seed(SEED)
            torch.cuda.manual_seed_all(SEED)
            model, tokenizer, model_audit = exact_v5.load_exact_v5_model(
                base_model, device=context.device
            )
            model.eval()
            install_audit = integration.install(
                model,
                rank=MAP_RANK,
                seed=SEED,
                trainable_map=False,
            )
            integration.set_mode(model, integration.INHERITED_EXACT_V5_MODE)
            integration.set_capture(model, True)
            observer_audit = install_read_observer(model)
            modules = causal_train.ordered_modules(model)
            names = [name for name, _ in modules]
            expected_names = protocol["all_module_inventory"]["ordered_module_names"]
            if (
                len(modules) != MODULES
                or names != expected_names
                or install_audit["module_names"] != tuple(expected_names)
                or observer_audit["module_names"] != expected_names
            ):
                raise RuntimeError("Continuous-write all-layer inventory differs")
            for parameter in model.parameters():
                parameter.requires_grad_(False)
            if any(parameter.requires_grad for parameter in model.parameters()):
                raise RuntimeError("Continuous-write capture left trainable parameters")
            return (
                model,
                tokenizer,
                model_audit,
                install_audit,
                observer_audit,
                _parameter_versions(model),
            )

        (
            model,
            tokenizer,
            model_audit,
            install_audit,
            observer_audit,
            parameter_versions_before,
        ) = _consensual_operation(
            context, phase="continuous-write-model-load", operation=load_runtime
        )
        dependency_bindings_before = dependency_bindings()
        runner_sha256_before = sha256_file(Path(__file__).resolve())
        module_names = tuple(protocol["all_module_inventory"]["ordered_module_names"])

        fit_rows = _consensual_operation(
            context,
            phase="continuous-write-fit-bundle-open",
            operation=lambda: _load_open_bundle(
                materialization_root, manifest, protocol, "fit"
            ),
        )
        fit_binding = canonical_sha256(
            [
                {
                    "source_index": row["source_index"],
                    "row_sha256": row["row_sha256"],
                    "donor_source_index": row["donor_source_index"],
                }
                for row in fit_rows
            ]
        )
        distributed.require_consensus(
            context, fit_binding, description="continuous-write FIT bundle"
        )

        def capture_fit_shard() -> Mapping[str, Any]:
            examples = _encode_rows(tokenizer, fit_rows)
            by_source = {int(row["source_index"]): row for row in fit_rows}
            sources = [
                source
                for source in sorted(by_source)
                if source % WORLD_SIZE == context.process_rank
            ]
            captured: list[Mapping[str, Any]] = []
            for ordinal, source in enumerate(sources, start=1):
                feature = capture_row(
                    model,
                    examples[source],
                    pad_token_id=int(tokenizer.pad_token_id),
                    device=context.device,
                )
                captured.append(
                    _feature_row(
                        by_source[source], feature, process_rank=context.process_rank
                    )
                )
                print(
                    f"CONTINUOUS_WRITE_FIT rank={context.process_rank} "
                    f"row={source} ordinal={ordinal}/{len(sources)}",
                    flush=True,
                )
            return _write_stage_shard(
                output_dir,
                split="fit",
                process_rank=context.process_rank,
                rows=captured,
            )

        _consensual_operation(
            context,
            phase="continuous-write-fit-capture",
            operation=capture_fit_shard,
        )
        fit_records, fit_provenance, fit_feature_digest = _consensual_operation(
            context,
            phase="continuous-write-fit-shard-validation",
            operation=lambda: load_stage_shards(
                output_dir, split="fit", source_rows=fit_rows
            ),
        )
        distributed.require_consensus(
            context,
            fit_feature_digest,
            description="continuous-write FIT feature digest",
        )

        maps: dict[str, alignment.FrozenMapWeights] = {}
        fit_map_digest: str | None = None
        fit_error: BaseException | None = None
        if context.is_primary:
            try:
                maps = fit_maps(fit_records, module_names)
                fit_map_digest = map_digest(maps, module_names)
            except BaseException as caught:
                fit_error = caught
        distributed.phase_consensus(
            context, phase="continuous-write-rank0-map-fit", error=fit_error
        )
        fit_map_digest = _broadcast_primary_object(context, fit_map_digest)
        if not isinstance(fit_map_digest, str) or len(fit_map_digest) != 64:
            raise RuntimeError("Continuous-write frozen in-memory map digest differs")

        retrieval_rows = _consensual_operation(
            context,
            phase="continuous-write-retrieval-bundle-open-after-map-freeze",
            operation=lambda: _load_open_bundle(
                materialization_root, manifest, protocol, "retrieval"
            ),
        )
        if set(int(row["source_index"]) for row in fit_rows) & set(
            int(row["source_index"]) for row in retrieval_rows
        ):
            raise RuntimeError("Continuous-write FIT and retrieval sources overlap")
        retrieval_binding = canonical_sha256(
            [
                {
                    "source_index": row["source_index"],
                    "row_sha256": row["row_sha256"],
                    "donor_source_index": row["donor_source_index"],
                }
                for row in retrieval_rows
            ]
        )
        distributed.require_consensus(
            context,
            retrieval_binding,
            description="continuous-write retrieval bundle",
        )

        def capture_retrieval_shard() -> Mapping[str, Any]:
            examples = _encode_rows(tokenizer, retrieval_rows)
            by_source = {int(row["source_index"]): row for row in retrieval_rows}
            sources = [
                source
                for source in sorted(by_source)
                if source % WORLD_SIZE == context.process_rank
            ]
            captured: list[Mapping[str, Any]] = []
            for ordinal, source in enumerate(sources, start=1):
                feature = capture_row(
                    model,
                    examples[source],
                    pad_token_id=int(tokenizer.pad_token_id),
                    device=context.device,
                )
                captured.append(
                    _feature_row(
                        by_source[source], feature, process_rank=context.process_rank
                    )
                )
                print(
                    f"CONTINUOUS_WRITE_RETRIEVAL rank={context.process_rank} "
                    f"row={source} ordinal={ordinal}/{len(sources)}",
                    flush=True,
                )
            return _write_stage_shard(
                output_dir,
                split="retrieval",
                process_rank=context.process_rank,
                rows=captured,
            )

        _consensual_operation(
            context,
            phase="continuous-write-retrieval-capture",
            operation=capture_retrieval_shard,
        )
        retrieval_records, retrieval_provenance, retrieval_feature_digest = (
            _consensual_operation(
                context,
                phase="continuous-write-retrieval-shard-validation",
                operation=lambda: load_stage_shards(
                    output_dir,
                    split="retrieval",
                    source_rows=retrieval_rows,
                ),
            )
        )
        distributed.require_consensus(
            context,
            retrieval_feature_digest,
            description="continuous-write retrieval feature digest",
        )
        if _parameter_versions(model) != parameter_versions_before:
            raise RuntimeError("Continuous-write capture mutated model parameters")
        if context.is_primary and map_digest(maps, module_names) != fit_map_digest:
            raise RuntimeError("Continuous-write frozen maps changed before retrieval")
        del model
        torch.cuda.empty_cache()

        result_error: BaseException | None = None
        if context.is_primary:
            try:
                analysis = retrieval_analysis(
                    retrieval_records, module_names, maps
                )
                passed = bool(analysis["passed"])
                map_artifact = (
                    _save_maps(
                        output_dir,
                        maps,
                        module_names,
                        fit_map_digest,
                    )
                    if passed
                    else {
                        "saved": False,
                        "path": None,
                        "sha256": None,
                        "frozen_map_digest": fit_map_digest,
                    }
                )
                dependencies_end = dependency_bindings()
                runner_sha256_end = sha256_file(Path(__file__).resolve())
                if (
                    dependencies_end != dependency_bindings_before
                    or runner_sha256_end != runner_sha256_before
                    or sha256_file(PROTOCOL) != PROTOCOL_FILE_SHA256
                ):
                    raise RuntimeError("Continuous-write code binding changed during run")
                result: dict[str, Any] = {
                    "schema": SCHEMA,
                    "status": (
                        "continuous_write_retrieval_passed_mechanics_protocol_draft_authorized"
                        if passed
                        else "continuous_write_retrieval_failed_family_retired"
                    ),
                    "passed": passed,
                    "mechanics_protocol_drafting_authorized": passed,
                    "mechanics_bytes_open_authorized": False,
                    "mechanics_authorized": False,
                    "causal_authorized": False,
                    "model_or_adapter_training_authorized": False,
                    "generation_authorized": False,
                    "native_benchmark_authorized": False,
                    "protocol_payload_sha256": PROTOCOL_PAYLOAD_SHA256,
                    "protocol_file_sha256": PROTOCOL_FILE_SHA256,
                    "protocol_objective": protocol["objective"],
                    "base_model": str(base_model),
                    "materialization_root": str(materialization_root),
                    "rows": {"fit": len(fit_records), "retrieval": len(retrieval_records)},
                    "module_names": list(module_names),
                    "fit": {
                        "rank": MAP_RANK,
                        "ridge": RIDGE,
                        "fit_only": True,
                        "fit_feature_digest": fit_feature_digest,
                        "frozen_map_digest_before_retrieval_open": fit_map_digest,
                        "map_weights_saved_before_retrieval": False,
                    },
                    "retrieval_evaluation_calls": analysis["evaluation_calls"],
                    "analysis": analysis,
                    "map_artifact": map_artifact,
                    "feature_provenance": {
                        "fit": fit_provenance,
                        "retrieval": retrieval_provenance,
                        "retrieval_feature_digest": retrieval_feature_digest,
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
                        "assignment": "source_index_modulo_4",
                        "fit_capture_and_map_freeze_completed_before_retrieval_bundle_open": True,
                        "rank0_only_map_fit_and_retrieval_analysis": True,
                        "other_ranks_verified_feature_and_result_bindings": True,
                        "model_parameters_updated": False,
                        "adapter_parameters_updated": False,
                        "generation": False,
                    },
                    "source_audit": source_audit,
                    "model_audit": {
                        **dict(model_audit),
                        "continuous_write_install": install_audit,
                        "read_observer": observer_audit,
                        "capture_mode": integration.INHERITED_EXACT_V5_MODE,
                        "parameters_trainable": False,
                        "parameter_versions_unchanged": True,
                    },
                    "firewall": {
                        "byte_read_files_in_order": [
                            "manifest.json",
                            "fit.jsonl",
                            "retrieval.jsonl",
                        ],
                        "fit_map_frozen_before_retrieval_bytes_read": True,
                        "mechanics_path_statted_listed_hashed_or_opened": False,
                        "causal_path_statted_listed_hashed_or_opened": False,
                        "mechanics_rows_decoded_tokenized_forwarded_or_scored": 0,
                        "causal_rows_decoded_tokenized_forwarded_or_scored": 0,
                    },
                    "v5_provenance": {
                        "result_sha256": exact_v5.V5_RESULT_SHA256,
                        "result_receipt": exact_v5.V5_RESULT_RECEIPT,
                        "adapter_weights_sha256": exact_v5.V5_ADAPTER_WEIGHTS_SHA256,
                        "adapter_config_sha256": exact_v5.V5_ADAPTER_CONFIG_SHA256,
                    },
                    "code_bindings": {
                        "runner_sha256": runner_sha256_end,
                        "dependencies": dependencies_end,
                    },
                    "protected_splits_opened": [],
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
            phase="continuous-write-result-analysis-and-save",
            error=result_error,
        )
        result = _consensual_operation(
            context,
            phase="continuous-write-all-rank-result-validation",
            operation=lambda: _validate_result(output_dir / "result.json"),
        )
        distributed.require_consensus(
            context,
            result["receipt"]["payload_sha256"],
            description="continuous-write result receipt",
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
    run(
        base_model=args.base_model.expanduser().resolve(strict=True),
        materialization_root=args.materialization_root.expanduser().resolve(
            strict=True
        ),
        output_dir=args.output_dir.expanduser().resolve(),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
