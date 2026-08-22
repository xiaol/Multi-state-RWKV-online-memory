#!/usr/bin/env python3
"""Screen direct address decoding of continuous-write RWKV state."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
from typing import Any, Mapping, Sequence

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

import torch
import torch.nn.functional as F


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[1]
for import_root in (SCRIPT_DIR, PROJECT_ROOT):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

from experiments.rethinking_rwkv_ms_gemma import (  # noqa: E402
    run_natural_memory_native_rwkv_continuous_write_mechanics as mechanics,
)
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
    run_natural_memory_native_rwkv_continuous_write_causal_train as continuous_causal,
)
from experiments.rethinking_rwkv_ms_gemma import (  # noqa: E402
    run_natural_memory_native_rwkv_continuous_write_retrieval as retrieval,
)
from experiments.rethinking_rwkv_ms_gemma import (  # noqa: E402
    run_natural_memory_native_rwkv_v5_shadow_crossfit as exact_v5,
)
from experiments.rethinking_rwkv_ms_gemma import (  # noqa: E402
    rwkv_address_decoded_token_replacement as ad_rtr,
)
from experiments.rethinking_rwkv_ms_gemma import (  # noqa: E402
    rwkv_continuous_write_integration as integration,
)


SCHEMA = "rwkv_ms_natural_memory_native_rwkv_address_decoded_reconstruction.v1"
SHARD_SCHEMA = (
    "rwkv_ms_natural_memory_native_rwkv_address_decoded_reconstruction_shard.v1"
)
DECODER_SCHEMA = (
    "rwkv_ms_natural_memory_native_rwkv_address_decoded_reconstruction_decoder.v1"
)
PROTOCOL = SCRIPT_DIR / (
    "natural_memory_native_rwkv_address_decoded_reconstruction_protocol_v1.json"
)
LAUNCH = SCRIPT_DIR / (
    "natural_memory_native_rwkv_address_decoded_reconstruction_launch_v1.json"
)
PROTOCOL_PAYLOAD_SHA256 = "e0b614c9fdcd59dfe814b18fe528888cb9f11604376ef6a08c388e0729d7e9d5"
PROTOCOL_FILE_SHA256 = "cdb81d5b0f62ab03e1a5b058b664fd35b82d99c87328257fb735a9f585f45aaf"
CODE_PARENT_COMMIT = "7824a4e45b0029614290da30c9e494412692a585"
CONTINUOUS_RESULT = continuous_causal.DEFAULT_OUTPUT / "result.json"
CONTINUOUS_RESULT_FILE_SHA256 = (
    "71d738ab63ae893c79b42e2cb1a93e25fee5e64daa6bf0d9d3eceb7dff572a09"
)
CONTINUOUS_RESULT_RECEIPT = (
    "5660251fc35005ee6cc054587d83bdd3069f22c52b9d9d7b440912fdbf71c0d0"
)
MANIFEST_FILE_SHA256 = (
    "c437a7d1f2b850a730fe5b28a08ae32ba02678561bb1265a4eef55bda7f4d468"
)
MANIFEST_RECEIPT = (
    "99a878493c3848c96624e2ad658842c99e69769b4a1721b5854ad25af8d0bee2"
)
FIT_FILE_SHA256 = (
    "4984e7de044f7befc2c3fdba8a0d8c08f627dcc4b168abbd8090393cca49c2fc"
)
FIT_PAYLOAD_SHA256 = (
    "41d9e117997e60b808895f4ae8ea63a6fa643ba97d44c6878be5b442eeb76318"
)
RETRIEVAL_FILE_SHA256 = (
    "2083394ac902745617827039c8b61f71ff76dee3f6ddd128d932fd92c400e7c7"
)
RETRIEVAL_PAYLOAD_SHA256 = (
    "02a21f010446131ee994992d662cf16d896d2f4dde5a47f8dadd024b758ae112"
)
WORLD_SIZE = 4
FIT_ROWS = 64
RETRIEVAL_ROWS = 32
MODULES = 42
RIDGE = 1.0
SEED = 163
HF_ENDPOINT = "https://hf-mirror.com"
BASE_MODEL_REVISION = "a4c2d58be94dda072b918d9db64ee85c8ed34e3f"
BASE_CONFIG_SHA256 = "33b10c02df3c2e8536cf323d29d53262aaa2f4d11dbe19bc729373fbe90295d4"
BASE_WEIGHTS_SHA256 = "cfbd3d2f1cd71bd471c37fe2bf8546d5028d41e5736f64e1ca6c6b8893125503"
DEFAULT_BASE_MODEL = mechanics.DEFAULT_BASE_MODEL
DEFAULT_MATERIALIZATION = mechanics.DEFAULT_MATERIALIZATION
DEFAULT_OUTPUT = SCRIPT_DIR / (
    "local_artifacts/natural_memory_native_rwkv_address_decoded_reconstruction_v1"
)

causal_train = exact_v5.causal_train
hardware = exact_v5.hardware


def canonical_sha256(value: Any) -> str:
    return distributed.canonical_sha256(value)


def sha256_file(path: Path) -> str:
    return mechanics.sha256_file(path)


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


def _atomic_signed_json(path: Path, value: Mapping[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(
        json.dumps(value, ensure_ascii=True, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _tensor_digest(tensor: torch.Tensor) -> str:
    value = tensor.detach().contiguous().cpu()
    digest = hashlib.sha256()
    digest.update(str(value.dtype).encode("ascii"))
    digest.update(json.dumps(list(value.shape)).encode("ascii"))
    digest.update(value.view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def _dependency_paths() -> Mapping[str, Path]:
    return {
        "address_decoded_reconstruction_math": Path(ad_rtr.__file__).resolve(),
        "continuous_write_capture_and_map_loader": Path(mechanics.__file__).resolve(),
        "continuous_write_open_row_encoder": Path(retrieval.__file__).resolve(),
        "continuous_write_runtime": Path(integration.__file__).resolve(),
        "open_fit_materializer": Path(materializer.__file__).resolve(),
        "signed_distributed_runtime": Path(distributed.__file__).resolve(),
        "signed_exact_v5_loader": Path(exact_v5.__file__).resolve(),
        "native_write_runtime": Path(evolution.__file__).resolve(),
    }


def dependency_bindings() -> list[dict[str, str]]:
    return [
        {"role": role, "basename": path.name, "sha256": sha256_file(path)}
        for role, path in _dependency_paths().items()
    ]


def _git_output(*args: str) -> str:
    return subprocess.run(
        ["git", *args],
        cwd=PROJECT_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def validate_protocol(base_model: Path = DEFAULT_BASE_MODEL) -> Mapping[str, Any]:
    if PROTOCOL_PAYLOAD_SHA256.startswith("TO_BE") or PROTOCOL_FILE_SHA256.startswith(
        "TO_BE"
    ):
        raise RuntimeError("AD-RTR reconstruction protocol is not signed")
    if sha256_file(PROTOCOL) != PROTOCOL_FILE_SHA256:
        raise ValueError("AD-RTR reconstruction protocol file differs")
    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    _validate_receipt(
        protocol,
        payload_scope="canonical_protocol_without_receipt",
        description="AD-RTR reconstruction protocol",
    )
    authorization = protocol.get("authorization_basis", {})
    frozen = protocol.get("frozen_inputs", {})
    architecture = protocol.get("architecture", {})
    lifecycle = protocol.get("data_lifecycle", {})
    execution = protocol.get("execution", {})
    if (
        protocol.get("schema")
        != "rwkv_ms_natural_memory_native_rwkv_address_decoded_reconstruction_protocol.v1"
        or protocol.get("receipt", {}).get("payload_sha256")
        != PROTOCOL_PAYLOAD_SHA256
        or authorization.get("code_parent_commit") != CODE_PARENT_COMMIT
        or authorization.get("continuous_causal_result_file_sha256")
        != CONTINUOUS_RESULT_FILE_SHA256
        or authorization.get("continuous_causal_result_receipt")
        != CONTINUOUS_RESULT_RECEIPT
        or frozen.get("manifest_file_sha256") != MANIFEST_FILE_SHA256
        or frozen.get("manifest_receipt") != MANIFEST_RECEIPT
        or frozen.get("fit_bundle", {}).get("rows") != FIT_ROWS
        or frozen.get("fit_bundle", {}).get("sha256") != FIT_FILE_SHA256
        or frozen.get("fit_bundle", {}).get("payload_sha256")
        != FIT_PAYLOAD_SHA256
        or frozen.get("retrieval_bundle", {}).get("rows") != RETRIEVAL_ROWS
        or frozen.get("retrieval_bundle", {}).get("sha256")
        != RETRIEVAL_FILE_SHA256
        or frozen.get("retrieval_bundle", {}).get("payload_sha256")
        != RETRIEVAL_PAYLOAD_SHA256
        or frozen.get("frozen_address_map_file_sha256")
        != mechanics.MAP_FILE_SHA256
        or frozen.get("frozen_address_map_digest") != mechanics.MAP_DIGEST
        or frozen.get("base_model_revision") != BASE_MODEL_REVISION
        or frozen.get("base_config_sha256") != BASE_CONFIG_SHA256
        or frozen.get("base_weights_sha256") != BASE_WEIGHTS_SHA256
        or frozen.get("signed_exact_v5_source_head") != exact_v5.SIGNED_V5_COMMIT
        or frozen.get("exact_v5_protocol_receipt")
        != exact_v5.PROTOCOL_PAYLOAD_SHA256
        or frozen.get("exact_v5_result_file_sha256") != exact_v5.V5_RESULT_SHA256
        or frozen.get("exact_v5_result_receipt") != exact_v5.V5_RESULT_RECEIPT
        or frozen.get("exact_v5_adapter_weights_sha256")
        != exact_v5.V5_ADAPTER_WEIGHTS_SHA256
        or frozen.get("exact_v5_adapter_config_sha256")
        != exact_v5.V5_ADAPTER_CONFIG_SHA256
        or architecture.get("modules") != MODULES
        or architecture.get("value_decoder_ridge") != RIDGE
        or architecture.get("model_parameters_updated") is not False
        or architecture.get("full_bandwidth_feedback_installed") is not False
        or lifecycle.get("already_open_bundles_only") != ["fit", "retrieval"]
        or lifecycle.get("mechanics_bundle_opened") is not False
        or lifecycle.get("causal_bundle_opened") is not False
        or lifecycle.get("native_benchmark_bytes_opened") is not False
        or execution.get("world_size") != WORLD_SIZE
        or execution.get("hf_endpoint") != HF_ENDPOINT
        or execution.get("seed") != SEED
        or Path(base_model).resolve() != Path(DEFAULT_BASE_MODEL).resolve()
        or sha256_file(Path(base_model) / "config.json") != BASE_CONFIG_SHA256
        or sha256_file(Path(base_model) / "model.safetensors")
        != BASE_WEIGHTS_SHA256
    ):
        raise ValueError("AD-RTR reconstruction protocol contract differs")
    return protocol


def validate_launch_binding(protocol: Mapping[str, Any]) -> Mapping[str, Any]:
    if not LAUNCH.exists():
        raise RuntimeError("AD-RTR reconstruction launch is not signed")
    launch = json.loads(LAUNCH.read_text(encoding="utf-8"))
    _validate_receipt(
        launch,
        payload_scope="canonical_launch_binding_without_receipt",
        description="AD-RTR reconstruction launch",
    )
    code_commit = str(launch.get("authorized_code_commit", ""))
    launch_relative = LAUNCH.resolve().relative_to(PROJECT_ROOT).as_posix()
    runner_relative = Path(__file__).resolve().relative_to(PROJECT_ROOT).as_posix()
    protocol_relative = PROTOCOL.resolve().relative_to(PROJECT_ROOT).as_posix()
    head = _git_output("rev-parse", "HEAD")
    if (
        launch.get("schema")
        != "rwkv_ms_natural_memory_native_rwkv_address_decoded_reconstruction_launch.v1"
        or launch.get("code_parent_commit") != CODE_PARENT_COMMIT
        or launch.get("protocol_payload_sha256") != PROTOCOL_PAYLOAD_SHA256
        or launch.get("protocol_file_sha256") != PROTOCOL_FILE_SHA256
        or launch.get("protocol_receipt")
        != protocol.get("receipt", {}).get("payload_sha256")
        or launch.get("runner_sha256") != sha256_file(Path(__file__).resolve())
        or launch.get("dependency_bindings_sha256")
        != canonical_sha256(dependency_bindings())
        or launch.get("continuous_causal_result_file_sha256")
        != CONTINUOUS_RESULT_FILE_SHA256
        or launch.get("continuous_causal_result_receipt")
        != CONTINUOUS_RESULT_RECEIPT
        or launch.get("manifest_file_sha256") != MANIFEST_FILE_SHA256
        or launch.get("manifest_receipt") != MANIFEST_RECEIPT
        or launch.get("fit_file_sha256") != FIT_FILE_SHA256
        or launch.get("retrieval_file_sha256") != RETRIEVAL_FILE_SHA256
        or launch.get("frozen_address_map_file_sha256")
        != mechanics.MAP_FILE_SHA256
        or launch.get("frozen_address_map_digest") != mechanics.MAP_DIGEST
        or launch.get("world_size") != WORLD_SIZE
        or launch.get("hf_endpoint") != HF_ENDPOINT
        or launch.get("already_open_bundles_only") != ["fit", "retrieval"]
        or launch.get("mechanics_causal_or_native_bytes_opened_before_launch")
        is not False
        or not code_commit
    ):
        raise ValueError("AD-RTR reconstruction launch binding differs")
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
        _git_output("rev-parse", "HEAD^") != code_commit
        or _git_output("rev-parse", f"{code_commit}^") != CODE_PARENT_COMMIT
        or hashlib.sha256(committed_runner).hexdigest()
        != launch.get("runner_sha256")
        or hashlib.sha256(committed_protocol).hexdigest() != PROTOCOL_FILE_SHA256
        or committed_launch != LAUNCH.read_bytes()
        or _git_output("diff", "--name-only", code_commit, "HEAD")
        != launch_relative
        or _git_output("diff", "--name-only", "HEAD")
        or _git_output("rev-parse", "origin/main") != head
    ):
        raise ValueError("AD-RTR reconstruction two-commit launch differs")
    return launch


def validate_continuous_failure() -> Mapping[str, Any]:
    if sha256_file(CONTINUOUS_RESULT) != CONTINUOUS_RESULT_FILE_SHA256:
        raise ValueError("AD-RTR authorization result file differs")
    result = continuous_causal._validate_final_result(CONTINUOUS_RESULT)
    if (
        result.get("status")
        != "continuous_write_causal_failed_readout_family_retired"
        or result.get("passed") is not False
        or result.get("native_benchmark_bytes_opened") is not False
        or result.get("receipt", {}).get("payload_sha256")
        != CONTINUOUS_RESULT_RECEIPT
    ):
        raise ValueError("AD-RTR authorization result contract differs")
    return result


def _load_manifest_only(materialization_root: Path) -> Mapping[str, Any]:
    path = materialization_root / "manifest.json"
    if sha256_file(path) != MANIFEST_FILE_SHA256:
        raise ValueError("AD-RTR manifest file differs")
    manifest = json.loads(path.read_text(encoding="utf-8"))
    _validate_receipt(
        manifest,
        payload_scope="canonical_manifest_without_receipt",
        description="AD-RTR open-fit manifest",
    )
    bundles = manifest.get("file_inventory", {}).get("bundles", {})
    split_contract = manifest.get("split_contract", {})
    leakage = split_contract.get("leakage_audit", {})
    fit = bundles.get("fit", {})
    heldout = bundles.get("retrieval", {})
    if (
        manifest.get("receipt", {}).get("payload_sha256") != MANIFEST_RECEIPT
        or manifest.get("protected_splits_opened") != []
        or split_contract.get("donor_component_disjoint") is not True
        or split_contract.get("passage_component_disjoint") is not True
        or leakage.get("cross_split_normalized_32_character_shingle_overlap") != 0
        or leakage.get("cross_split_passage_component_count") != 0
        or fit.get("path") != "fit.jsonl"
        or fit.get("rows") != FIT_ROWS
        or fit.get("sha256") != FIT_FILE_SHA256
        or fit.get("payload_sha256") != FIT_PAYLOAD_SHA256
        or heldout.get("path") != "retrieval.jsonl"
        or heldout.get("rows") != RETRIEVAL_ROWS
        or heldout.get("sha256") != RETRIEVAL_FILE_SHA256
        or heldout.get("payload_sha256") != RETRIEVAL_PAYLOAD_SHA256
    ):
        raise ValueError("AD-RTR manifest contract differs")
    return manifest


def _load_rows(
    materialization_root: Path,
    manifest: Mapping[str, Any],
    split: str,
) -> list[dict[str, Any]]:
    if split not in {"fit", "retrieval"}:
        raise PermissionError("AD-RTR may open only FIT and retrieval rows")
    rows = materializer._read_bundle(materialization_root, manifest, split)
    expected = FIT_ROWS if split == "fit" else RETRIEVAL_ROWS
    sources = {int(row["source_index"]) for row in rows}
    donors = {
        int(row["source_index"]): int(row["donor_source_index"]) for row in rows
    }
    if (
        len(rows) != expected
        or len(sources) != expected
        or any(row.get("split") != split for row in rows)
        or any(int(row["donor_source_index"]) not in sources for row in rows)
        or any(donor == source for source, donor in donors.items())
        or any(donors.get(donor) != source for source, donor in donors.items())
    ):
        raise ValueError(f"AD-RTR {split} row contract differs")
    return sorted(rows, key=lambda row: int(row["source_index"]))


def _extract_row_features(
    state: Mapping[str, Mapping[str, torch.Tensor]],
    modules: Sequence[tuple[str, Any]],
    maps: Mapping[str, Any],
    effective_address: torch.Tensor,
    write_audit: Mapping[str, Any],
) -> tuple[Mapping[str, torch.Tensor], Mapping[str, Any]]:
    states = torch.stack(
        [state[name]["delta_state"].float() for name, _ in modules]
    )
    keys = torch.stack(
        [state[name]["projected_kv_keys"].float() for name, _ in modules]
    )
    values = torch.stack(
        [state[name]["projected_kv_values"].float() for name, _ in modules]
    )
    occupied = torch.stack(
        [state[name]["projected_kv_occupied"].bool() for name, _ in modules]
    )
    if (
        tuple(states.shape) != (MODULES, 1, 1, ad_rtr.SLOTS, 32, 32)
        or tuple(keys.shape) != (MODULES, 1, ad_rtr.SLOTS, 64)
        or tuple(values.shape) != (MODULES, 1, ad_rtr.SLOTS, 32)
        or tuple(occupied.shape) != (MODULES, 1, ad_rtr.SLOTS)
    ):
        raise ValueError("AD-RTR captured layered state shape differs")
    states = states[:, 0].contiguous()
    keys = keys[:, 0].contiguous()
    values = values[:, 0].contiguous()
    occupied = occupied[:, 0].contiguous()
    nonzero_slots = states.ne(0).any(dim=(1, 3, 4))
    exactly_one = occupied.sum(dim=-1).eq(1)
    state_slot_match = torch.equal(nonzero_slots, occupied)
    selected_keys = keys[occupied].reshape(MODULES, ad_rtr.ADDRESS_DIM)
    selected_values = values[occupied].reshape(MODULES, ad_rtr.STATE_DIM)
    selected_states = states[:, 0][occupied].reshape(
        MODULES, ad_rtr.STATE_DIM, ad_rtr.STATE_DIM
    )
    effective_address = effective_address.detach().contiguous().cpu().float()
    effective_address_matches = (
        tuple(effective_address.shape) == tuple(selected_keys.shape)
        and torch.equal(effective_address, selected_keys)
    )
    decoded_finite = True
    decoded_nonzero = True
    for index, (name, _) in enumerate(modules):
        decoded = ad_rtr.address_decoded_slots(
            states[index].unsqueeze(0),
            keys[index].unsqueeze(0),
            values[index].unsqueeze(0),
            occupied[index].unsqueeze(0),
            maps[name],
        )
        decoded_finite = decoded_finite and bool(
            torch.isfinite(decoded.contracted).all().item()
        )
        active_decoded = decoded.contracted[occupied[index].unsqueeze(0)]
        decoded_nonzero = decoded_nonzero and bool(
            active_decoded.reshape(-1, ad_rtr.STATE_DIM)
            .ne(0)
            .any(dim=-1)
            .all()
            .item()
        )
    capture_invariants = {
        "write_formula_byte_exact_all_modules": write_audit.get(
            "formula_byte_exact_all_modules"
        )
        is True,
        "write_value_same_object_and_bytes_all_modules": write_audit.get(
            "continuous_value_same_object_and_bytes_all_modules"
        )
        is True,
        "write_effective_address_object_and_versions_exact_all_modules": write_audit.get(
            "effective_address_object_and_versions_exact_all_modules"
        )
        is True,
        "all_state_tensors_finite": write_audit.get("all_state_tensors_finite")
        is True,
        "effective_write_address_matches_occupied_projected_key": (
            effective_address_matches
        ),
        "active_projected_keys_nonzero_all_modules": bool(
            selected_keys.ne(0).any(dim=-1).all().item()
        ),
        "active_projected_values_nonzero_all_modules": bool(
            selected_values.ne(0).any(dim=-1).all().item()
        ),
        "active_rwkv_state_matrices_nonzero_all_modules": bool(
            selected_states.ne(0).any(dim=(-1, -2)).all().item()
        ),
    }
    audit = {
        **capture_invariants,
        "exactly_one_occupied_slot_all_modules": bool(exactly_one.all().item()),
        "occupied_slots_match_nonzero_rwkv_slots": state_slot_match,
        "address_decodes_finite": decoded_finite,
        "active_address_decodes_nonzero": decoded_nonzero,
        "state_sha256": _tensor_digest(states),
        "keys_sha256": _tensor_digest(keys),
        "values_sha256": _tensor_digest(values),
        "occupied_sha256": _tensor_digest(occupied),
        "occupied_slot_indices": occupied.to(dtype=torch.int64).argmax(dim=-1).tolist(),
    }
    required_audits = (
        *capture_invariants,
        "exactly_one_occupied_slot_all_modules",
        "occupied_slots_match_nonzero_rwkv_slots",
        "address_decodes_finite",
        "active_address_decodes_nonzero",
    )
    if not all(audit[name] is True for name in required_audits):
        raise RuntimeError("AD-RTR captured write or feature invariant differs")
    return {
        "state": states,
        "keys": keys,
        "values": values,
        "occupied": occupied,
    }, audit


def _capture_split(
    context: Any,
    *,
    split: str,
    rows: Sequence[Mapping[str, Any]],
    model: torch.nn.Module,
    tokenizer: Any,
    modules: Sequence[tuple[str, Any]],
    maps: Mapping[str, Any],
    output_dir: Path,
) -> Mapping[int, Mapping[str, torch.Tensor]]:
    assigned = list(rows[context.process_rank :: WORLD_SIZE])
    expected = (FIT_ROWS if split == "fit" else RETRIEVAL_ROWS) // WORLD_SIZE
    if len(assigned) != expected:
        raise RuntimeError(f"AD-RTR {split} rank assignment differs")
    examples = retrieval._encode_rows(tokenizer, assigned)
    captured: dict[int, Mapping[str, torch.Tensor]] = {}
    shard_rows = []
    for ordinal, row in enumerate(assigned, start=1):
        error: BaseException | None = None
        try:
            source = int(row["source_index"])
            batch = evolution.collate_native_examples(
                [examples[source]],
                pad_token_id=int(tokenizer.pad_token_id),
                device=context.device,
            )
            state, write_audit, effective_address = mechanics.capture_write_condition(
                model,
                batch,
                modules,
                mode=integration.CONTINUOUS_MODE,
                override=None,
                reference_mode="none",
            )
            mechanics._clear_feature_references(modules)
            features, audit = _extract_row_features(
                state,
                modules,
                maps,
                effective_address,
                write_audit,
            )
            captured[source] = features
            shard_rows.append(
                {
                    "source_index": source,
                    "donor_source_index": int(row["donor_source_index"]),
                    "row_sha256": row["row_sha256"],
                    "donor_row_sha256": row["donor_row_sha256"],
                    "write_audit": {
                        "mode": write_audit["mode"],
                        "formula_byte_exact_all_modules": write_audit[
                            "formula_byte_exact_all_modules"
                        ],
                        "continuous_value_same_object_and_bytes_all_modules": write_audit[
                            "continuous_value_same_object_and_bytes_all_modules"
                        ],
                        "all_state_tensors_finite": write_audit[
                            "all_state_tensors_finite"
                        ],
                        "effective_address_sha256": _tensor_digest(effective_address),
                    },
                    "feature_audit": audit,
                }
            )
        except BaseException as caught:
            error = caught
        distributed.phase_consensus(
            context,
            phase=f"ad-rtr-{split}-capture-row-{ordinal}",
            error=error,
        )
        if error is not None:
            raise error
        print(
            f"AD_RTR_CAPTURE split={split} rank={context.process_rank} "
            f"row={source} ordinal={ordinal}/{expected}",
            flush=True,
        )
    shard = {
        "schema": SHARD_SCHEMA,
        "split": split,
        "rank": context.process_rank,
        "world_size": WORLD_SIZE,
        "assignment": "sorted_source_rank_strided",
        "rows": shard_rows,
    }
    shard["receipt"] = {
        "algorithm": "sha256",
        "payload_scope": "canonical_capture_shard_without_receipt",
        "payload_sha256": canonical_sha256(shard),
    }
    shard_error: BaseException | None = None
    try:
        _atomic_signed_json(
            output_dir / f"{split}-shard-{context.process_rank}.json",
            shard,
        )
    except BaseException as caught:
        shard_error = caught
    distributed.phase_consensus(
        context,
        phase=f"ad-rtr-{split}-capture-shard-write",
        error=shard_error,
    )
    if shard_error is not None:
        raise shard_error
    gathered = distributed.gather_objects(context, captured)
    merged: dict[int, Mapping[str, torch.Tensor]] = {}
    for rank_capture in gathered:
        for source, features in rank_capture.items():
            if int(source) in merged:
                raise ValueError(f"AD-RTR duplicate {split} capture source")
            merged[int(source)] = features
    if len(merged) != len(rows):
        raise ValueError(f"AD-RTR {split} gathered coverage differs")
    return merged


def _stack_capture(
    rows: Sequence[Mapping[str, Any]],
    captured: Mapping[int, Mapping[str, torch.Tensor]],
) -> Mapping[str, torch.Tensor]:
    sources = [int(row["source_index"]) for row in rows]
    return {
        name: torch.stack([captured[source][name] for source in sources])
        for name in ("state", "keys", "values", "occupied")
    }


def _fit_decoders(
    fit: Mapping[str, torch.Tensor],
    module_names: Sequence[str],
    maps: Mapping[str, Any],
) -> Mapping[str, ad_rtr.FullRankRidgeDecoder]:
    decoders = {}
    for index, name in enumerate(module_names):
        decoder, _ = ad_rtr.fit_address_decoded_ridge_decoder(
            fit["state"][:, index],
            fit["keys"][:, index],
            fit["values"][:, index],
            fit["occupied"][:, index],
            maps[name],
            ridge=RIDGE,
        )
        decoders[name] = decoder
    return decoders


def _row_means(score: torch.Tensor, occupied: torch.Tensor) -> torch.Tensor:
    counts = occupied.sum(dim=(1, 2))
    if bool(counts.eq(0).any().item()):
        raise ValueError("AD-RTR held-out row has no occupied module slots")
    return (
        (score * occupied.to(dtype=score.dtype)).sum(dim=(1, 2))
        / counts.to(dtype=score.dtype)
    )


def _all_active_vectors_nonzero(
    value: torch.Tensor, occupied: torch.Tensor
) -> bool:
    if tuple(value.shape[: occupied.ndim]) != tuple(occupied.shape):
        raise ValueError("AD-RTR active-vector mask shape differs")
    active_count = int(occupied.sum().item())
    if active_count < 1:
        return False
    trailing = value.ndim - occupied.ndim
    active = value.masked_select(
        occupied.reshape(*occupied.shape, *([1] * trailing)).expand_as(value)
    )
    return bool(active.reshape(active_count, -1).ne(0).any(dim=-1).all().item())


def _analyze_reconstruction(
    heldout: Mapping[str, torch.Tensor],
    rows: Sequence[Mapping[str, Any]],
    module_names: Sequence[str],
    maps: Mapping[str, Any],
    decoders: Mapping[str, ad_rtr.FullRankRidgeDecoder],
    protocol: Mapping[str, Any],
) -> Mapping[str, Any]:
    source_to_index = {
        int(row["source_index"]): index for index, row in enumerate(rows)
    }
    donor_indices = torch.tensor(
        [source_to_index[int(row["donor_source_index"])] for row in rows],
        dtype=torch.long,
    )
    correct_state = heldout["state"]
    occupied = heldout["occupied"]
    donor_state_raw = correct_state.index_select(0, donor_indices)
    donor_occupied = occupied.index_select(0, donor_indices)
    donor_state = ad_rtr.canonicalize_active_slots(
        donor_state_raw,
        donor_occupied,
        occupied,
        slot_dim=3,
    )
    donor_keys_raw = heldout["keys"].index_select(0, donor_indices)
    donor_keys = ad_rtr.canonicalize_active_slots(
        donor_keys_raw,
        donor_occupied,
        occupied,
        slot_dim=2,
    )
    layer_roll_state_raw = correct_state.roll(1, dims=1)
    layer_roll_occupied = occupied.roll(1, dims=1)
    layer_roll_state = ad_rtr.canonicalize_active_slots(
        layer_roll_state_raw,
        layer_roll_occupied,
        occupied,
        slot_dim=3,
    )
    values = heldout["values"]
    cosine_by_control = {
        name: torch.zeros(
            RETRIEVAL_ROWS, MODULES, ad_rtr.SLOTS, dtype=torch.float32
        )
        for name in ("correct", "matched_donor_state", "wrong_address", "layer_roll")
    }
    per_module = {}
    zero_audit = []
    correct_path_tensors = []
    correct_path_nonzero = []
    for index, name in enumerate(module_names):
        metrics = ad_rtr.reconstruction_control_metrics(
            correct_state=correct_state[:, index],
            matched_donor_state=donor_state[:, index],
            layer_roll_state=layer_roll_state[:, index],
            keys=heldout["keys"][:, index],
            wrong_address_keys=donor_keys[:, index],
            values=values[:, index],
            occupied=occupied[:, index],
            weights=maps[name],
            decoder=decoders[name],
        )
        per_module[name] = metrics
        zero_audit.append(all(metrics["zero_audit"].values()))
        controls = {
            "correct": (correct_state[:, index], heldout["keys"][:, index]),
            "matched_donor_state": (
                donor_state[:, index],
                heldout["keys"][:, index],
            ),
            "wrong_address": (correct_state[:, index], donor_keys[:, index]),
            "layer_roll": (layer_roll_state[:, index], heldout["keys"][:, index]),
        }
        for control, (state, keys) in controls.items():
            slots = ad_rtr.address_decoded_slots(
                state,
                keys,
                values[:, index],
                occupied[:, index],
                maps[name],
            )
            decoded = decoders[name].decode(slots.contracted)
            if control == "correct":
                correct_path_tensors.extend(
                    (slots.directions, slots.contracted, decoded)
                )
                correct_path_nonzero.extend(
                    _all_active_vectors_nonzero(tensor, occupied[:, index])
                    for tensor in (slots.directions, slots.contracted, decoded)
                )
            cosine_by_control[control][:, index] = F.cosine_similarity(
                decoded,
                values[:, index],
                dim=-1,
                eps=1e-6,
            )
    row_cosine = {
        name: _row_means(score, occupied)
        for name, score in cosine_by_control.items()
    }
    row_gaps = {
        name: row_cosine["correct"] - row_cosine[name]
        for name in ("matched_donor_state", "wrong_address", "layer_roll")
    }
    gates = protocol["required_gates"]
    capture_checks = {
        "effective_write_address_matches_occupied_projected_key_all_rows_modules": True,
        "continuous_write_formula_and_value_identity_hold_all_rows_modules": True,
        "all_control_sources_gathered_from_own_occupied_slot_and_scattered_to_target_slot": True,
        "all_rows_have_exactly_one_occupied_projected_slot_per_module": bool(
            occupied.sum(dim=-1).eq(1).all().item()
        ),
        "occupied_projected_slot_matches_only_nonzero_rwkv_slot_per_module": bool(
            correct_state.ne(0).any(dim=(2, 4, 5)).eq(occupied).all().item()
        ),
        "all_addresses_states_targets_decodes_finite": all(
            bool(torch.isfinite(value).all().item())
            for value in (
                *heldout.values(),
                *correct_path_tensors,
                *cosine_by_control.values(),
            )
            if value.is_floating_point()
        ),
        "all_active_addresses_states_targets_decodes_nonzero": all(
            (
                _all_active_vectors_nonzero(heldout["keys"], occupied),
                _all_active_vectors_nonzero(correct_state[:, :, 0], occupied),
                _all_active_vectors_nonzero(values, occupied),
                *correct_path_nonzero,
            )
        ),
        "zero_state_decodes_exact_zero": all(zero_audit),
    }
    correct_mean = float(row_cosine["correct"].mean().item())
    correct_row_fraction = float(
        row_cosine["correct"]
        .ge(float(gates["correct_row_mean_cosine_minimum"]))
        .float()
        .mean()
        .item()
    )
    aggregate = {
        "correct_mean_cosine": correct_mean,
        "correct_minimum_row_cosine": float(row_cosine["correct"].min().item()),
        "correct_row_fraction_at_or_above_minimum": correct_row_fraction,
        "controls": {
            name: {
                "mean_cosine": float(row_cosine[name].mean().item()),
                "correct_minus_control_mean_cosine_gap": float(gap.mean().item()),
                "correct_minus_control_positive_row_fraction": float(
                    gap.gt(0.0).float().mean().item()
                ),
                "minimum_row_gap": float(gap.min().item()),
                "maximum_row_gap": float(gap.max().item()),
            }
            for name, gap in row_gaps.items()
        },
    }
    checks = {
        **capture_checks,
        "correct_mean_cosine": correct_mean
        >= float(gates["correct_mean_cosine_minimum"]),
        "correct_row_fraction_at_or_above_minimum": correct_row_fraction
        >= float(gates["correct_row_fraction_at_or_above_minimum"]),
    }
    for control, prefix in (
        ("matched_donor_state", "correct_minus_matched_donor_state"),
        ("wrong_address", "correct_minus_wrong_address"),
        ("layer_roll", "correct_minus_layer_rolled_state"),
    ):
        control_metrics = aggregate["controls"][control]
        checks[f"{prefix}_mean_cosine_gap"] = control_metrics[
            "correct_minus_control_mean_cosine_gap"
        ] >= float(gates[f"{prefix}_mean_cosine_gap_minimum"])
        checks[f"{prefix}_positive_row_fraction"] = control_metrics[
            "correct_minus_control_positive_row_fraction"
        ] >= float(gates[f"{prefix}_positive_row_fraction_minimum"])
    module_checks = {}
    for name, metrics in per_module.items():
        module_checks[name] = {
            "finite_and_zero_exact": metrics["finite"] is True
            and all(metrics["zero_audit"].values()),
            "correct_mean_cosine": metrics["cosine"]["correct"]
            >= float(gates["module_correct_mean_cosine_minimum"]),
            "matched_donor_state_mean_gap": metrics["mean_gaps"][
                "matched_donor_state"
            ]
            >= float(gates["module_control_mean_cosine_gap_minimum"]),
            "wrong_address_mean_gap": metrics["mean_gaps"]["wrong_address"]
            >= float(gates["module_control_mean_cosine_gap_minimum"]),
            "layer_roll_mean_gap": metrics["mean_gaps"]["layer_roll"]
            >= float(gates["module_control_mean_cosine_gap_minimum"]),
            "matched_donor_state_positive_row_fraction": metrics[
                "positive_row_fractions"
            ]["matched_donor_state"]
            >= float(gates["module_control_positive_row_fraction_minimum"]),
            "wrong_address_positive_row_fraction": metrics[
                "positive_row_fractions"
            ]["wrong_address"]
            >= float(gates["module_control_positive_row_fraction_minimum"]),
            "layer_roll_positive_row_fraction": metrics[
                "positive_row_fractions"
            ]["layer_roll"]
            >= float(gates["module_control_positive_row_fraction_minimum"]),
        }
        module_checks[name]["passed"] = all(module_checks[name].values())
    passed_modules = sum(item["passed"] is True for item in module_checks.values())
    module_pass_fraction = passed_modules / MODULES
    checks["module_identity_pass_fraction"] = module_pass_fraction >= float(
        gates["module_identity_pass_fraction_minimum"]
    )
    return {
        "rows": RETRIEVAL_ROWS,
        "modules": MODULES,
        "aggregate": aggregate,
        "checks": checks,
        "per_module": per_module,
        "module_gate": {
            "passed_modules": passed_modules,
            "module_pass_fraction": module_pass_fraction,
            "required_fraction": float(
                gates["module_identity_pass_fraction_minimum"]
            ),
            "checks": module_checks,
        },
        "passed": all(checks.values()),
    }


def _decoder_digest(
    decoders: Mapping[str, ad_rtr.FullRankRidgeDecoder],
    module_names: Sequence[str],
) -> str:
    digest = hashlib.sha256()
    for name in module_names:
        digest.update(name.encode("utf-8"))
        digest.update(decoders[name].weight.contiguous().view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def _save_decoders(
    path: Path,
    decoders: Mapping[str, ad_rtr.FullRankRidgeDecoder],
    module_names: Sequence[str],
) -> Mapping[str, Any]:
    digest = _decoder_digest(decoders, module_names)
    payload = {
        "schema": DECODER_SCHEMA,
        "module_names": list(module_names),
        "ridge": RIDGE,
        "weights": {name: decoders[name].weight.detach().cpu() for name in module_names},
        "decoder_digest": digest,
    }
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    torch.save(payload, temporary)
    os.replace(temporary, path)
    return {
        "path": path.name,
        "sha256": sha256_file(path),
        "decoder_digest": digest,
        "modules": len(module_names),
        "ridge": RIDGE,
    }


def _validate_decoder_artifact(
    path: Path,
    *,
    module_names: Sequence[str],
    expected_digest: str,
    expected_sha256: str | None = None,
) -> Mapping[str, Any]:
    if expected_sha256 is not None and sha256_file(path) != expected_sha256:
        raise ValueError("AD-RTR decoder artifact file differs")
    payload = torch.load(path, map_location="cpu", weights_only=True)
    names = tuple(module_names)
    weights = payload.get("weights", {}) if isinstance(payload, Mapping) else {}
    if (
        not isinstance(payload, Mapping)
        or payload.get("schema") != DECODER_SCHEMA
        or tuple(payload.get("module_names", ())) != names
        or payload.get("ridge") != RIDGE
        or payload.get("decoder_digest") != expected_digest
        or set(weights) != set(names)
    ):
        raise ValueError("AD-RTR decoder artifact contract differs")
    decoders = {
        name: ad_rtr.FullRankRidgeDecoder(
            weight=weights[name].detach().float(),
            ridge=RIDGE,
        )
        for name in names
    }
    if _decoder_digest(decoders, names) != expected_digest:
        raise ValueError("AD-RTR decoder artifact tensor digest differs")
    return {
        "path": path.name,
        "sha256": sha256_file(path),
        "decoder_digest": expected_digest,
        "modules": len(names),
        "ridge": RIDGE,
    }


def _capture_shard_bindings(output_dir: Path) -> list[Mapping[str, Any]]:
    bindings = []
    sources_by_split: dict[str, set[int]] = {"fit": set(), "retrieval": set()}
    required_feature_audits = (
        "write_formula_byte_exact_all_modules",
        "write_value_same_object_and_bytes_all_modules",
        "write_effective_address_object_and_versions_exact_all_modules",
        "all_state_tensors_finite",
        "effective_write_address_matches_occupied_projected_key",
        "active_projected_keys_nonzero_all_modules",
        "active_projected_values_nonzero_all_modules",
        "active_rwkv_state_matrices_nonzero_all_modules",
        "exactly_one_occupied_slot_all_modules",
        "occupied_slots_match_nonzero_rwkv_slots",
        "address_decodes_finite",
        "active_address_decodes_nonzero",
    )
    for split, expected_rows in (
        ("fit", FIT_ROWS // WORLD_SIZE),
        ("retrieval", RETRIEVAL_ROWS // WORLD_SIZE),
    ):
        for rank in range(WORLD_SIZE):
            path = output_dir / f"{split}-shard-{rank}.json"
            shard = json.loads(path.read_text(encoding="utf-8"))
            _validate_receipt(
                shard,
                payload_scope="canonical_capture_shard_without_receipt",
                description=f"AD-RTR {split} capture shard {rank}",
            )
            rows = shard.get("rows", [])
            row_sources = {int(row["source_index"]) for row in rows}
            if (
                shard.get("schema") != SHARD_SCHEMA
                or shard.get("split") != split
                or shard.get("rank") != rank
                or shard.get("world_size") != WORLD_SIZE
                or shard.get("assignment") != "sorted_source_rank_strided"
                or len(rows) != expected_rows
                or len(row_sources) != expected_rows
                or any(
                    row.get("write_audit", {}).get("mode")
                    != integration.CONTINUOUS_MODE
                    or row.get("write_audit", {}).get(
                        "formula_byte_exact_all_modules"
                    )
                    is not True
                    or row.get("write_audit", {}).get(
                        "continuous_value_same_object_and_bytes_all_modules"
                    )
                    is not True
                    or row.get("write_audit", {}).get("all_state_tensors_finite")
                    is not True
                    or any(
                        row.get("feature_audit", {}).get(name) is not True
                        for name in required_feature_audits
                    )
                    for row in rows
                )
            ):
                raise ValueError("AD-RTR capture shard contract differs")
            if sources_by_split[split].intersection(row_sources):
                raise ValueError("AD-RTR capture shards repeat a source")
            sources_by_split[split].update(row_sources)
            bindings.append(
                {
                    "path": path.name,
                    "sha256": sha256_file(path),
                    "receipt": shard["receipt"]["payload_sha256"],
                    "split": split,
                    "rank": rank,
                    "rows": expected_rows,
                }
            )
    if (
        len(sources_by_split["fit"]) != FIT_ROWS
        or len(sources_by_split["retrieval"]) != RETRIEVAL_ROWS
    ):
        raise ValueError("AD-RTR capture shard coverage differs")
    return bindings


def _validate_result(path: Path) -> Mapping[str, Any]:
    result = json.loads(path.read_text(encoding="utf-8"))
    _validate_receipt(
        result,
        payload_scope="canonical_result_without_receipt",
        description="AD-RTR reconstruction result",
    )
    passed = result.get("passed")
    expected_status = (
        "address_decoded_reconstruction_passed_fresh_split_authorized"
        if passed is True
        else "address_decoded_reconstruction_failed_linear_decoder_family_retired"
    )
    module_names = tuple(result.get("module_names", ()))
    source_audit = result.get("source_audit", {})
    model_audit = result.get("model_audit", {})
    analysis = result.get("analysis", {})
    analysis_checks = analysis.get("checks", {})
    module_gate = analysis.get("module_gate", {})
    module_checks = module_gate.get("checks", {})
    module_passed = sum(
        item.get("passed") is True
        for item in module_checks.values()
        if isinstance(item, Mapping)
    )
    if (
        not isinstance(passed, bool)
        or result.get("schema") != SCHEMA
        or result.get("status") != expected_status
        or result.get("analysis", {}).get("passed") is not passed
        or result.get("fit_rows") != FIT_ROWS
        or result.get("retrieval_rows") != RETRIEVAL_ROWS
        or result.get("modules") != MODULES
        or len(module_names) != MODULES
        or len(set(module_names)) != MODULES
        or result.get("decoder_frozen_and_persisted_before_retrieval_open")
        is not True
        or result.get("fresh_split_protocol_drafting_authorized") is not passed
        or result.get("mechanics_causal_generation_or_native_bytes_opened") is not False
        or result.get("model_parameters_updated") is not False
        or result.get("full_bandwidth_feedback_installed") is not False
        or result.get("native_gain_claimed") is not False
        or result.get("sota_claimed") is not False
        or source_audit.get("signed_v5_source_commit")
        != exact_v5.SIGNED_V5_COMMIT
        or source_audit.get("exact_v5_protocol_receipt")
        != exact_v5.PROTOCOL_PAYLOAD_SHA256
        or source_audit.get("exact_v5_result_receipt")
        != exact_v5.V5_RESULT_RECEIPT
        or source_audit.get("adapter_weights_sha256")
        != exact_v5.V5_ADAPTER_WEIGHTS_SHA256
        or source_audit.get("adapter_config_sha256")
        != exact_v5.V5_ADAPTER_CONFIG_SHA256
        or source_audit.get("base_model_revision") != BASE_MODEL_REVISION
        or source_audit.get("base_config_sha256") != BASE_CONFIG_SHA256
        or source_audit.get("base_weights_sha256") != BASE_WEIGHTS_SHA256
        or model_audit.get("adapter_weights_sha256")
        != exact_v5.V5_ADAPTER_WEIGHTS_SHA256
        or model_audit.get("adapter_config_sha256")
        != exact_v5.V5_ADAPTER_CONFIG_SHA256
        or not analysis_checks
        or any(not isinstance(value, bool) for value in analysis_checks.values())
        or analysis.get("passed") is not all(analysis_checks.values())
        or set(module_checks) != set(module_names)
        or module_gate.get("passed_modules") != module_passed
        or module_gate.get("module_pass_fraction") != module_passed / MODULES
        or module_gate.get("required_fraction") != 0.95
        or analysis_checks.get("module_identity_pass_fraction")
        != (module_passed / MODULES >= 0.95)
    ):
        raise ValueError("AD-RTR reconstruction result contract differs")
    decoder = result.get("decoder_artifact", {})
    if decoder.get("path") != "address-decoded-value-decoders.pt":
        raise ValueError("AD-RTR result decoder path differs")
    actual_decoder = _validate_decoder_artifact(
        path.parent / decoder["path"],
        module_names=module_names,
        expected_digest=str(decoder.get("decoder_digest", "")),
        expected_sha256=str(decoder.get("sha256", "")),
    )
    if actual_decoder != decoder:
        raise ValueError("AD-RTR result decoder binding differs")
    if result.get("capture_shards") != _capture_shard_bindings(path.parent):
        raise ValueError("AD-RTR result capture shard bindings differ")
    return result


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
    try:
        def preflight() -> tuple[Mapping[str, Any], ...]:
            if (
                context.world_size != WORLD_SIZE
                or context.backend != "nccl"
                or context.control_backend != "gloo"
                or not hardware.four_distinct_a100s(context.rank_devices)
                or os.environ.get("HF_ENDPOINT") != HF_ENDPOINT
            ):
                raise RuntimeError("AD-RTR reconstruction requires four A100s and HF mirror")
            protocol = validate_protocol(base_model)
            launch = validate_launch_binding(protocol)
            prior = validate_continuous_failure()
            v5_protocol, v5_result = exact_v5.validate_protocol()
            source_audit = exact_v5.validate_execution_source()
            source_audit = {
                **dict(source_audit),
                "exact_v5_protocol_receipt": v5_protocol["receipt"][
                    "payload_sha256"
                ],
                "exact_v5_result_receipt": v5_result["receipt"]["payload_sha256"],
                "adapter_weights_sha256": exact_v5.V5_ADAPTER_WEIGHTS_SHA256,
                "adapter_config_sha256": exact_v5.V5_ADAPTER_CONFIG_SHA256,
                "base_model_revision": BASE_MODEL_REVISION,
                "base_config_sha256": BASE_CONFIG_SHA256,
                "base_weights_sha256": BASE_WEIGHTS_SHA256,
            }
            manifest = _load_manifest_only(materialization_root)
            return protocol, launch, prior, source_audit, manifest

        protocol, launch, prior, source_audit, manifest = mechanics._consensual_operation(
            context,
            phase="ad-rtr-reconstruction-preflight",
            operation=preflight,
        )

        def load_runtime() -> tuple[Any, ...]:
            torch.manual_seed(SEED)
            torch.cuda.manual_seed_all(SEED)
            model, tokenizer, model_audit = exact_v5.load_exact_v5_model(
                base_model, device=context.device
            )
            model.eval()
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
            feature_audit = mechanics.install_feature_observer(modules)
            for parameter in model.parameters():
                parameter.requires_grad_(False)
            if (
                len(modules) != MODULES
                or install_audit["module_names"] != module_names
                or feature_audit["module_names"] != list(module_names)
                or any(parameter.requires_grad for parameter in model.parameters())
            ):
                raise RuntimeError("AD-RTR reconstruction runtime installation differs")
            return model, tokenizer, modules, module_names, maps, model_audit, install_audit

        (
            model,
            tokenizer,
            modules,
            module_names,
            maps,
            model_audit,
            install_audit,
        ) = mechanics._consensual_operation(
            context,
            phase="ad-rtr-reconstruction-model-load",
            operation=load_runtime,
        )

        def create_output() -> None:
            if context.is_primary:
                if output_dir.exists():
                    raise ValueError(f"AD-RTR reconstruction output must be fresh: {output_dir}")
                output_dir.mkdir(parents=True, exist_ok=False)

        mechanics._consensual_operation(
            context,
            phase="ad-rtr-reconstruction-output-create",
            operation=create_output,
        )
        fit_rows = mechanics._consensual_operation(
            context,
            phase="ad-rtr-open-fit-bundle",
            operation=lambda: _load_rows(materialization_root, manifest, "fit"),
        )
        fit_capture = mechanics._consensual_operation(
            context,
            phase="ad-rtr-fit-capture",
            operation=lambda: _capture_split(
                context,
                split="fit",
                rows=fit_rows,
                model=model,
                tokenizer=tokenizer,
                modules=modules,
                maps=maps,
                output_dir=output_dir,
            ),
        )
        fit = _stack_capture(fit_rows, fit_capture)
        decoders = _fit_decoders(fit, module_names, maps)
        decoder_digest = _decoder_digest(decoders, module_names)
        distributed.require_consensus(
            context, decoder_digest, description="AD-RTR frozen decoder"
        )
        decoder_path = output_dir / "address-decoded-value-decoders.pt"

        def persist_decoder() -> None:
            if context.is_primary:
                _save_decoders(decoder_path, decoders, module_names)

        mechanics._consensual_operation(
            context,
            phase="ad-rtr-fit-decoder-persist",
            operation=persist_decoder,
        )
        decoder_artifact = mechanics._consensual_operation(
            context,
            phase="ad-rtr-fit-decoder-validation",
            operation=lambda: _validate_decoder_artifact(
                decoder_path,
                module_names=module_names,
                expected_digest=decoder_digest,
            ),
        )
        distributed.require_consensus(
            context,
            canonical_sha256(decoder_artifact),
            description="AD-RTR persisted decoder artifact",
        )
        retrieval_rows = mechanics._consensual_operation(
            context,
            phase="ad-rtr-open-retrieval-bundle-after-decoder-freeze",
            operation=lambda: _load_rows(materialization_root, manifest, "retrieval"),
        )
        heldout_capture = mechanics._consensual_operation(
            context,
            phase="ad-rtr-retrieval-capture",
            operation=lambda: _capture_split(
                context,
                split="retrieval",
                rows=retrieval_rows,
                model=model,
                tokenizer=tokenizer,
                modules=modules,
                maps=maps,
                output_dir=output_dir,
            ),
        )
        heldout = _stack_capture(retrieval_rows, heldout_capture)
        capture_shards = mechanics._consensual_operation(
            context,
            phase="ad-rtr-capture-shard-validation",
            operation=lambda: _capture_shard_bindings(output_dir),
        )
        distributed.require_consensus(
            context,
            canonical_sha256(capture_shards),
            description="AD-RTR capture shard bindings",
        )
        analysis = _analyze_reconstruction(
            heldout,
            retrieval_rows,
            module_names,
            maps,
            decoders,
            protocol,
        )
        analysis_digest = canonical_sha256(analysis)
        distributed.require_consensus(
            context, analysis_digest, description="AD-RTR reconstruction analysis"
        )
        result_path = output_dir / "result.json"

        def persist_result() -> None:
            if not context.is_primary:
                return
            passed = analysis["passed"] is True
            result = {
                "schema": SCHEMA,
                "status": (
                    "address_decoded_reconstruction_passed_fresh_split_authorized"
                    if passed
                    else "address_decoded_reconstruction_failed_linear_decoder_family_retired"
                ),
                "passed": passed,
                "protocol_payload_sha256": PROTOCOL_PAYLOAD_SHA256,
                "protocol_file_sha256": PROTOCOL_FILE_SHA256,
                "launch_receipt": launch["receipt"]["payload_sha256"],
                "continuous_causal_result_receipt": prior["receipt"]["payload_sha256"],
                "manifest_receipt": manifest["receipt"]["payload_sha256"],
                "source_audit": source_audit,
                "hardware": {
                    "world_size": context.world_size,
                    "backend": context.backend,
                    "control_backend": context.control_backend,
                    "rank_devices": list(context.rank_devices),
                },
                "model_audit": model_audit,
                "continuous_install": install_audit,
                "code_bindings": {
                    "runner_sha256": sha256_file(Path(__file__).resolve()),
                    "dependencies": dependency_bindings(),
                },
                "fit_rows": FIT_ROWS,
                "retrieval_rows": RETRIEVAL_ROWS,
                "modules": MODULES,
                "module_names": list(module_names),
                "write_scans_per_source": 1,
                "already_open_bundles_read": ["fit", "retrieval"],
                "decoder_artifact": decoder_artifact,
                "capture_shards": capture_shards,
                "decoder_frozen_and_persisted_before_retrieval_open": True,
                "analysis": analysis,
                "fresh_split_protocol_drafting_authorized": passed,
                "mechanics_causal_generation_or_native_bytes_opened": False,
                "model_parameters_updated": False,
                "frozen_address_map_updated": False,
                "full_bandwidth_feedback_installed": False,
                "native_gain_claimed": False,
                "sota_claimed": False,
            }
            result["receipt"] = {
                "algorithm": "sha256",
                "payload_scope": "canonical_result_without_receipt",
                "payload_sha256": canonical_sha256(result),
            }
            _atomic_signed_json(result_path, result)
        mechanics._consensual_operation(
            context,
            phase="ad-rtr-result-persist",
            operation=persist_result,
        )
        result = mechanics._consensual_operation(
            context,
            phase="ad-rtr-result-validation",
            operation=lambda: _validate_result(result_path),
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
        base_model=args.base_model,
        materialization_root=args.materialization_root,
        output_dir=args.output_dir,
    )
    print(json.dumps(result, ensure_ascii=True, sort_keys=True))
    return 0 if result.get("passed") is True else 1


if __name__ == "__main__":
    raise SystemExit(main())
