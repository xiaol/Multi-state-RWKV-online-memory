#!/usr/bin/env python3
"""Run the signed exact-v5 PLMSC v2 write/read code-alignment screen."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import sys
from types import MethodType
from typing import Any, Callable, Mapping, Sequence

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

import torch
import torch.distributed as dist
from torch import nn
from torch.nn import functional as F


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[1]
for import_root in (SCRIPT_DIR, PROJECT_ROOT):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

from experiments.rethinking_rwkv_ms_gemma import (  # noqa: E402
    run_natural_memory_native_rwkv_plat_prompt_latch_crossfit as plat,
)

predictor = plat.parent
shadow = predictor.shadow

from experiments.rethinking_rwkv_ms_gemma import (  # noqa: E402
    rwkv_query_state_identity as write_address_capture,
)

signed_delta_api = sys.modules["deltamem.core.delta"]


SCHEMA = "rwkv_ms_natural_memory_native_plmsc_code_alignment.v2"
ROW_SCHEMA = "rwkv_ms_natural_memory_native_plmsc_code_alignment_feature.v2"
SHARD_SCHEMA = "rwkv_ms_natural_memory_native_plmsc_code_alignment_shard.v2"
FEATURE_ROW_KEYS = {
    "schema",
    "capture_rank",
    "source_index",
    "row_sha256",
    "donor_source_index",
    "donor_row_sha256",
    "split",
    "anchors",
    "write_slot_address",
    "prompt_boundary_rwkv_receptance",
    "first_supervised_label_index",
    "prompt_boundary_predictor_index",
    "predictor_definition",
    "predictor_vectors_per_row",
    "answer_or_later_predictor_features_captured",
    "write_passes",
    "read_passes",
    "read_basis_calls_per_anchor",
    "read_basis_call_roles",
    "read_basis_observations_byte_identical_per_anchor",
    "read_basis_prompt_boundary_sha256_per_anchor_per_call",
    "read_basis_return_shapes_per_anchor_per_call",
    "read_basis_return_dtypes_per_anchor_per_call",
    "read_writes_enabled",
    "features_detached_and_cloned",
    "model_output_changed_by_capture",
    "binder_bridge_or_code_module_installed_during_capture",
    "receipt",
}
PROTOCOL = (
    SCRIPT_DIR / "natural_memory_native_rwkv_plmsc_code_alignment_protocol_v2.json"
)
PROTOCOL_PAYLOAD_SHA256 = "a66d2b855491e0c814d0c524d9d66f70cd03164c9aac14b30da1fb47e769142e"
PROTOCOL_FILE_SHA256 = "15fd83f0cc9eb636f6264d5d2fb80a830e612ac144a123a4b4e7be5d483ed5ed"
PROTOCOL_SCHEMA = "rwkv_ms_natural_memory_native_plmsc_code_alignment_protocol.v2"
WORLD_SIZE = 4
DISTRIBUTED_TIMEOUT_SECONDS = 1800
HF_ENDPOINT = "https://hf-mirror.com"
DEFAULT_OUTPUT_DIR = (
    SCRIPT_DIR / "local_artifacts/natural_memory_native_rwkv_plmsc_code_alignment_v2"
)

SPLIT_SALT = "rwkv-write-slot-code-v1:"
FIT_ROWS = 64
MECHANICS_ROWS = 34
CAUSAL_ROWS = 34
CAPTURE_ROWS = FIT_ROWS + MECHANICS_ROWS
PRIOR_EXCLUDED_ROWS = 44
PLAT_EXCLUDED_ROWS = 44
MECHANICS_COMPONENT_INDICES = tuple(range(0, 12))
CAUSAL_COMPONENT_INDICES = tuple(range(12, 25))
FIT_COMPONENT_INDICES = tuple(range(25, 48))
SPLIT_PAYLOAD_SHA256 = "ab2a225c4a4710316a88b61fa48b5a48bb5ff0772c28d04455db20370fd0f737"
ELIGIBLE_MAPPING_SHA256 = "a15184300c04d707a135fd6f1ffd69c460985ddf07357a543fb3ee063530f6c6"
CAPTURE_SOURCES_SHA256 = "2d84030580d983ad6ab41956ebc7c9d5c92322de79dee346c5ea1e1adf6e6381"
PRIOR_EXCLUDED_SHA256 = "8f1fb3ec2fee2e8d01d7dec0b081d78d9b6b628f243b481dd082305b88a66eea"
PLAT_EXCLUDED_SHA256 = "bd7b7ddec5beab243a56b964140432ad18041c203dfbb0d430399e847f03f938"

ANCHORS = (10, 21, 31, 41)
STATE_WIDTH = 32
READ_BASIS_CALL_ROLES = ("addressed_recurrent", "global_recurrent")
READ_BASIS_CALLS_PER_ANCHOR = len(READ_BASIS_CALL_ROLES)
CODEBOOK_SIZE = 64
SEED = 120
TRAIN_STEPS = 512
LEARNING_RATE = 0.01
WEIGHT_DECAY = 0.001
TEMPERATURE = 0.25
DONOR_MARGIN = 0.2
LOSS_WEIGHTS = {
    "agreement": 1.0,
    "donor_margin": 1.0,
    "balance": 0.1,
    "sharpness": 0.01,
}

CORRECT_ANCHOR_GATE = 0.95
CORRECT_ROW_GATE = 0.95
DONOR_ANCHOR_COLLISION_GATE = 0.03
DONOR_ROW_COLLISION_COUNT_GATE = 1
LAYER_PERMUTED_ANCHOR_COLLISION_GATE = 0.03
LAYER_PERMUTED_ROW_COLLISION_COUNT_GATE = 1
MINIMUM_DISTINCT_CODES = 8
MAXIMUM_CODE_FRACTION = 0.25

PLAT_RESULT = (
    SCRIPT_DIR
    / "local_artifacts/natural_memory_native_rwkv_plat_prompt_latch_crossfit_v1/result.json"
)
PLAT_RESULT_SHA256 = "4f7c4aa1f715157e95cd753842b79d28f94b3356e318ef5c7c09e911456f8aac"
PLAT_RESULT_RECEIPT = "de1a677ce8b77e5c1f16eb9f9601d93c1140feffdf8adc60922e8cf671b01979"
PLAT_RUNNER_SHA256 = "72343bdd12c0bc125c6c0cedc60dcf181b0573b1cd985ca54707652bce98150f"
PLAT_PROTOCOL_FILE_SHA256 = "f33f9a7fe718c87a36adf0d99d99a3582161c247d477d3bb339b1dc7e8702347"
PLAT_SPLIT_PAYLOAD_SHA256 = "c1c1d27335563b47845633f5c3d33ea5409d5dfaf279dea9f084fd0b472c6f25"
V1_OPERATIONAL_FAILURE_RELATIVE = (
    "local_artifacts/natural_memory_native_rwkv_plmsc_code_alignment_v1/operational_failure.json"
)
V1_OPERATIONAL_FAILURE = SCRIPT_DIR / V1_OPERATIONAL_FAILURE_RELATIVE
V1_OPERATIONAL_FAILURE_SHA256 = (
    "f402b6e78dffe69bf6c2233e2beaee3791b3f179b2f927a3c338cc97798d0968"
)
V1_OPERATIONAL_FAILURE_RECEIPT = (
    "283adb5c5d8ed021eb4301156fd40a1ea41db101c49cc97c0b7a4127003ba1d5"
)
V1_PROTOCOL_FILE_SHA256 = (
    "8428742275084b901d677395514228e2410323322d5041d92af6b6578f0cb93d"
)
V1_PROTOCOL_PAYLOAD_SHA256 = (
    "5f849ec1c6fbc590f4bb8df6c976136213c928dc3270acc9e2291ec88ca76400"
)
V1_RUNNER_SHA256 = "2a30f50e921f018e7e3cedb0641cb582c3ffdf308f7261e54765ec07b6c0b6d9"
V2_AUTHORIZATION_PARENT_COMMIT = "a3f63da49ddba4d021d77498dbf80917a35789c7"
V1_INCIDENT_BOUNDARY = {
    "mechanics_rows_tokenized_before_first_forward": 34,
    "first_forward_source_by_rank": {"0": 40, "1": 5, "2": 14, "3": 7},
    "first_forward_split_by_rank": {
        "0": "fit",
        "1": "mechanics",
        "2": "fit",
        "3": "fit",
    },
    "saved_feature_rows": 0,
    "mechanics_metrics_computed": False,
    "identical_failure_calls_per_anchor": 2,
    "failure_rank_split_counts": {"fit": 3, "mechanics": 1},
    "correction_basis": "signed_static_addressed_and_global_call_sites_not_mechanics_values",
}

distributed = shadow.distributed
evolution = shadow.evolution
causal_train = shadow.causal_train
endpoint = shadow.endpoint
hardware = shadow.hardware

PREDICTOR_RESULT = plat.PARENT_RESULT
PREDICTOR_RESULT_SHA256 = plat.PARENT_RESULT_SHA256
PREDICTOR_RESULT_RECEIPT = plat.PARENT_RECEIPT
PREDICTOR_RUNNER_SHA256 = plat.PARENT_RUNNER_SHA256
SHADOW_RUNNER_SHA256 = plat.PARENT_SHADOW_RUNNER_SHA256
V5_RESULT_SHA256 = shadow.V5_RESULT_SHA256
V5_RESULT_RECEIPT = shadow.V5_RESULT_RECEIPT
V5_ADAPTER_WEIGHTS_SHA256 = shadow.V5_ADAPTER_WEIGHTS_SHA256
V5_ADAPTER_CONFIG_SHA256 = shadow.V5_ADAPTER_CONFIG_SHA256
SIGNED_SOURCE_ROOT_ENV = shadow.SIGNED_SOURCE_ROOT_ENV
SIGNED_V5_COMMIT = shadow.SIGNED_V5_COMMIT
SIGNED_V5_DELTA_IMPL_SHA256 = shadow.SIGNED_V5_DELTA_IMPL_SHA256

V5_TOPOLOGY_SHA256 = "27995bb507d0cf8b474def0042e28f998cf1f43a844425a0e99129549c615c51"
MODEL_COMMON_SHA256 = "8c2863eeea3701f557edf180391586fbc838ce357f3f4e53d8a2292fe4f853e0"
WRITE_ADDRESS_HELPER_SHA256 = "4ea2c1402831064762d30fe002f359363fe9685b63b3e047d99c7731cec91129"
DISTRIBUTED_RUNTIME_SHA256 = "09fd08b4750469c1364c28a935b443895f988f17ccc8102ede3ecce6bed6f44d"
SIGNED_V5_DELTA_API_SHA256 = "144023cf0ef24970f93d2ffaf80a7265dfb137a32c1326df84e43aee80898f0c"
EVOLUTION_RUNTIME_SHA256 = "6abbe06f249ec8fba942f0f865ff9d92485a13fe95d8c9eab0f308e0c0e258e7"
CAUSAL_RUNTIME_SHA256 = "3036e7c75c1dedd31ab7f3d8aa79126c849a5c64e5e37395bbe3e2c43822fbc7"
ENDPOINT_RUNTIME_SHA256 = "ddb2da83266967ab877c928d7d21662f9bcfbd225c84fdbdd619fcbc2c159756"
HARDWARE_RUNTIME_SHA256 = "b42cb3455679b1799a72b262958df5fd5a85211bc5b37730c24c70309309d654"
BASE_MODEL = "google/gemma-4-E4B-it"
BASE_MODEL_REVISION = "a4c2d58be94dda072b918d9db64ee85c8ed34e3f"
BASE_CONFIG_SHA256 = "33b10c02df3c2e8536cf323d29d53262aaa2f4d11dbe19bc729373fbe90295d4"
V5_TOPOLOGY_PATH = (
    SCRIPT_DIR
    / "run_natural_memory_native_rwkv_address_keyed_moe_deepembed_ffn_causal_train.py"
)
MODEL_COMMON_PATH = SCRIPT_DIR / "common.py"


def canonical_sha256(value: Any) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def tensor_raw_bytes_sha256(tensor: torch.Tensor) -> str:
    contiguous = tensor.detach().contiguous().cpu()
    header = json.dumps(
        {"dtype": str(contiguous.dtype), "shape": list(contiguous.shape)},
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("ascii")
    byte_view = contiguous.view(torch.uint8)
    return hashlib.sha256(header + b"\x00" + byte_view.numpy().tobytes()).hexdigest()


def tensor_raw_bytes_equal(left: torch.Tensor, right: torch.Tensor) -> bool:
    if left.shape != right.shape or left.dtype != right.dtype:
        return False
    left_bytes = left.detach().contiguous().view(torch.uint8)
    right_bytes = right.detach().contiguous().view(torch.uint8)
    return bool(torch.equal(left_bytes, right_bytes))


def _signed_json(path: Path, file_sha256: str, receipt_sha256: str) -> Mapping[str, Any]:
    if sha256_file(path) != file_sha256:
        raise ValueError(f"Signed JSON file differs: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise ValueError(f"Signed JSON must contain an object: {path}")
    unsigned = dict(value)
    receipt = unsigned.pop("receipt", {})
    if (
        not isinstance(receipt, Mapping)
        or receipt.get("payload_sha256") != receipt_sha256
        or canonical_sha256(unsigned) != receipt_sha256
    ):
        raise ValueError(f"Signed JSON receipt differs: {path}")
    return value


def _dependency_payload() -> list[Mapping[str, str]]:
    return [
        {
            "role": "plat_split_and_parent_contract",
            "basename": Path(plat.__file__).name,
            "sha256": PLAT_RUNNER_SHA256,
        },
        {
            "role": "predictor_metadata_and_causal_boundary_contract",
            "basename": Path(predictor.__file__).name,
            "sha256": PREDICTOR_RUNNER_SHA256,
        },
        {
            "role": "exact_v5_loader_and_capture_contract",
            "basename": Path(shadow.__file__).name,
            "sha256": SHADOW_RUNNER_SHA256,
        },
        {
            "role": "v5_endpoint_topology_contract",
            "basename": V5_TOPOLOGY_PATH.name,
            "sha256": V5_TOPOLOGY_SHA256,
        },
        {
            "role": "model_and_tokenizer_loader",
            "basename": MODEL_COMMON_PATH.name,
            "sha256": MODEL_COMMON_SHA256,
        },
        {
            "role": "native_capture_runtime",
            "basename": Path(evolution.__file__).name,
            "sha256": EVOLUTION_RUNTIME_SHA256,
        },
        {
            "role": "anchor_module_runtime",
            "basename": Path(causal_train.__file__).name,
            "sha256": CAUSAL_RUNTIME_SHA256,
        },
        {
            "role": "dataset_endpoint_contract",
            "basename": Path(endpoint.__file__).name,
            "sha256": ENDPOINT_RUNTIME_SHA256,
        },
        {
            "role": "a100_hardware_contract",
            "basename": Path(hardware.__file__).name,
            "sha256": HARDWARE_RUNTIME_SHA256,
        },
        {
            "role": "write_address_capture_helper",
            "basename": Path(write_address_capture.__file__).name,
            "sha256": WRITE_ADDRESS_HELPER_SHA256,
        },
        {
            "role": "distributed_runtime",
            "basename": Path(distributed.__file__).name,
            "sha256": DISTRIBUTED_RUNTIME_SHA256,
        },
        {
            "role": "signed_exact_v5_delta_api",
            "basename": Path(signed_delta_api.__file__).name,
            "sha256": SIGNED_V5_DELTA_API_SHA256,
        },
        {
            "role": "signed_exact_v5_delta_implementation",
            "basename": Path(shadow.core_impl.__file__).name,
            "sha256": SIGNED_V5_DELTA_IMPL_SHA256,
        },
    ]


def _validate_dependencies() -> None:
    bindings = (
        (Path(plat.__file__), PLAT_RUNNER_SHA256),
        (Path(predictor.__file__), PREDICTOR_RUNNER_SHA256),
        (Path(shadow.__file__), SHADOW_RUNNER_SHA256),
        (V5_TOPOLOGY_PATH, V5_TOPOLOGY_SHA256),
        (MODEL_COMMON_PATH, MODEL_COMMON_SHA256),
        (Path(evolution.__file__), EVOLUTION_RUNTIME_SHA256),
        (Path(causal_train.__file__), CAUSAL_RUNTIME_SHA256),
        (Path(endpoint.__file__), ENDPOINT_RUNTIME_SHA256),
        (Path(hardware.__file__), HARDWARE_RUNTIME_SHA256),
        (Path(write_address_capture.__file__), WRITE_ADDRESS_HELPER_SHA256),
        (Path(distributed.__file__), DISTRIBUTED_RUNTIME_SHA256),
        (Path(signed_delta_api.__file__), SIGNED_V5_DELTA_API_SHA256),
        (Path(shadow.core_impl.__file__), SIGNED_V5_DELTA_IMPL_SHA256),
    )
    for path, expected in bindings:
        if sha256_file(path.resolve()) != expected:
            raise ValueError(f"PLMSC code dependency differs: {path}")
    signed_root = shadow.SIGNED_SOURCE_ROOT
    imported_delta_api = Path(signed_delta_api.__file__).resolve()
    imported_core = Path(shadow.core_impl.__file__).resolve()
    if (
        signed_root is None
        or not Path(evolution.__file__).resolve().is_relative_to(signed_root)
        or not Path(causal_train.__file__).resolve().is_relative_to(signed_root)
        or not Path(endpoint.__file__).resolve().is_relative_to(signed_root)
        or not Path(hardware.__file__).resolve().is_relative_to(signed_root)
        or not imported_delta_api.is_relative_to(signed_root)
        or not imported_core.is_relative_to(signed_root)
    ):
        raise RuntimeError(
            "PLMSC imported Delta-Mem core is outside the signed exact-v5 source root"
        )


def _ordered_components(
    mapping: Mapping[int, int], eligible: set[int]
) -> tuple[tuple[int, ...], ...]:
    components = plat.donor_components(mapping, eligible)
    return tuple(
        sorted(
            components,
            key=lambda component: (
                hashlib.sha256(
                    (SPLIT_SALT + canonical_sha256(list(component))).encode("ascii")
                ).hexdigest(),
                component,
            ),
        )
    )


def derive_three_way_split(
    predictor_result: Mapping[str, Any],
    plat_result: Mapping[str, Any],
) -> tuple[dict[int, str], Mapping[str, Any], Mapping[int, Mapping[str, Any]]]:
    nested = plat_result.get("nested_split", {})
    eligible = {int(value) for value in nested.get("train_sources", [])}
    excluded_prior = {
        int(value) for value in nested.get("excluded_prior_heldout_sources", [])
    }
    excluded_plat = {int(value) for value in nested.get("heldout_sources", [])}
    parent_rows = predictor_result.get("crossfit_split", {}).get("rows", [])
    source_donor_metadata = {
        int(row["source_index"]): {
            "source_index": int(row["source_index"]),
            "donor_source_index": int(row["donor_source_index"]),
        }
        for row in parent_rows
    }
    if (
        len(eligible) != FIT_ROWS + MECHANICS_ROWS + CAUSAL_ROWS
        or len(excluded_prior) != PRIOR_EXCLUDED_ROWS
        or len(excluded_plat) != PLAT_EXCLUDED_ROWS
        or eligible & excluded_prior
        or eligible & excluded_plat
        or excluded_prior & excluded_plat
        or eligible | excluded_prior | excluded_plat != set(source_donor_metadata)
    ):
        raise ValueError("PLMSC source scope or failed-heldout exclusions differ")
    mapping = {
        source: int(source_donor_metadata[source]["donor_source_index"])
        for source in sorted(eligible)
    }
    if set(mapping.values()) - eligible:
        raise ValueError("PLMSC eligible donor mapping leaves the 132-row source set")
    mapping_pairs = [[source, mapping[source]] for source in sorted(mapping)]
    if canonical_sha256(mapping_pairs) != ELIGIBLE_MAPPING_SHA256:
        raise ValueError("PLMSC eligible donor mapping differs")
    components = _ordered_components(mapping, eligible)
    if len(components) != len(FIT_COMPONENT_INDICES) + len(
        MECHANICS_COMPONENT_INDICES
    ) + len(CAUSAL_COMPONENT_INDICES):
        raise RuntimeError("PLMSC donor component count differs")
    mechanics = {
        source
        for index in MECHANICS_COMPONENT_INDICES
        for source in components[index]
    }
    causal = {
        source for index in CAUSAL_COMPONENT_INDICES for source in components[index]
    }
    fit = {source for index in FIT_COMPONENT_INDICES for source in components[index]}
    if (
        len(fit) != FIT_ROWS
        or len(mechanics) != MECHANICS_ROWS
        or len(causal) != CAUSAL_ROWS
        or fit | mechanics | causal != eligible
        or fit & mechanics
        or fit & causal
        or mechanics & causal
    ):
        raise RuntimeError("PLMSC three-way component assignment differs")
    split = {
        source: (
            "fit" if source in fit else "mechanics" if source in mechanics else "causal"
        )
        for source in sorted(eligible)
    }
    if any(split[source] != split[donor] for source, donor in mapping.items()):
        raise RuntimeError("A donor edge crosses the PLMSC three-way split")
    payload = {
        "selection_salt": SPLIT_SALT,
        "component_count": len(components),
        "component_sizes": [len(component) for component in components],
        "mechanics_component_indices": list(MECHANICS_COMPONENT_INDICES),
        "causal_component_indices": list(CAUSAL_COMPONENT_INDICES),
        "fit_sources": sorted(fit),
        "mechanics_sources": sorted(mechanics),
        "causal_sources": sorted(causal),
        "excluded_prior_heldout_sources": sorted(excluded_prior),
        "excluded_plat_heldout_sources": sorted(excluded_plat),
    }
    capture_sources = sorted(fit | mechanics)
    if (
        canonical_sha256(payload) != SPLIT_PAYLOAD_SHA256
        or canonical_sha256(capture_sources) != CAPTURE_SOURCES_SHA256
        or canonical_sha256(sorted(excluded_prior)) != PRIOR_EXCLUDED_SHA256
        or canonical_sha256(sorted(excluded_plat)) != PLAT_EXCLUDED_SHA256
    ):
        raise ValueError("PLMSC precommitted split payload differs")
    capture_sources = fit | mechanics
    eligible_row_contracts: dict[int, Mapping[str, Any]] = {}
    for row in parent_rows:
        source = int(row["source_index"])
        if source not in eligible:
            continue
        contract: dict[str, Any] = {
            "source_index": source,
            "donor_source_index": int(row["donor_source_index"]),
        }
        if source in capture_sources:
            contract.update(
                {
                    "row_sha256": str(row["row_sha256"]),
                    "donor_row_sha256": str(row["donor_row_sha256"]),
                }
            )
        eligible_row_contracts[source] = contract
    if (
        len(eligible_row_contracts) != FIT_ROWS + MECHANICS_ROWS + CAUSAL_ROWS
        or sum("row_sha256" in row for row in eligible_row_contracts.values())
        != CAPTURE_ROWS
        or any(
            set(eligible_row_contracts[source])
            != {"source_index", "donor_source_index"}
            for source in causal
        )
    ):
        raise RuntimeError("PLMSC eligible row firewall contracts differ")
    return split, payload, eligible_row_contracts


def validate_protocol() -> tuple[
    Mapping[str, Any],
    Mapping[str, Any],
    Mapping[str, Any],
    Mapping[str, Any],
    Mapping[int, Mapping[str, Any]],
]:
    if sha256_file(PROTOCOL) != PROTOCOL_FILE_SHA256:
        raise ValueError("PLMSC protocol file hash differs")
    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    unsigned = dict(protocol)
    receipt = unsigned.pop("receipt", {})
    protocol_digest = canonical_sha256(unsigned)
    authorization = protocol.get("authorization_basis", {})
    frozen = protocol.get("frozen_inputs", {})
    signed_split = protocol.get("precommitted_three_way_split", {})
    capture = protocol.get("exact_v5_capture", {})
    fit = protocol.get("code_alignment_fit", {})
    gates = protocol.get("locked_mechanics_gates", {})
    execution = protocol.get("execution", {})
    firewall = protocol.get("causal_firewall", {})
    if (
        protocol.get("schema") != PROTOCOL_SCHEMA
        or protocol_digest != PROTOCOL_PAYLOAD_SHA256
        or receipt.get("payload_sha256") != protocol_digest
        or authorization.get("plat_parent_result_sha256") != PLAT_RESULT_SHA256
        or authorization.get("plat_parent_result_receipt") != PLAT_RESULT_RECEIPT
        or authorization.get("plat_parent_protocol_sha256") != PLAT_PROTOCOL_FILE_SHA256
        or authorization.get("plat_parent_protocol_payload_sha256")
        != plat.PROTOCOL_PAYLOAD_SHA256
        or authorization.get("plat_parent_split_payload_sha256")
        != PLAT_SPLIT_PAYLOAD_SHA256
        or authorization.get("predictor_parent_result_sha256")
        != PREDICTOR_RESULT_SHA256
        or authorization.get("predictor_parent_result_receipt")
        != PREDICTOR_RESULT_RECEIPT
        or authorization.get("v5_result_sha256") != V5_RESULT_SHA256
        or authorization.get("v5_result_receipt") != V5_RESULT_RECEIPT
        or authorization.get("v5_adapter_weights_sha256")
        != V5_ADAPTER_WEIGHTS_SHA256
        or authorization.get("v5_adapter_config_sha256")
        != V5_ADAPTER_CONFIG_SHA256
        or authorization.get("v2_authorization_parent_commit")
        != V2_AUTHORIZATION_PARENT_COMMIT
        or authorization.get("v1_operational_failure")
        != V1_OPERATIONAL_FAILURE_RELATIVE
        or authorization.get("v1_operational_failure_sha256")
        != V1_OPERATIONAL_FAILURE_SHA256
        or authorization.get("v1_operational_failure_receipt")
        != V1_OPERATIONAL_FAILURE_RECEIPT
        or authorization.get("v1_protocol_file_sha256")
        != V1_PROTOCOL_FILE_SHA256
        or authorization.get("v1_protocol_payload_sha256")
        != V1_PROTOCOL_PAYLOAD_SHA256
        or authorization.get("v1_runner_sha256") != V1_RUNNER_SHA256
        or authorization.get("v1_retry_authorized") is not False
        or authorization.get("v1_incident_boundary") != V1_INCIDENT_BOUNDARY
        or authorization.get("code_dependencies") != _dependency_payload()
        or frozen.get("base_model") != BASE_MODEL
        or frozen.get("base_model_revision") != BASE_MODEL_REVISION
        or frozen.get("base_config_sha256") != BASE_CONFIG_SHA256
        or frozen.get("dataset") != endpoint.DATASET_RELATIVE_PATH
        or frozen.get("dataset_sha256") != endpoint.DATASET_SHA256
        or frozen.get("authorized_open_rows") != endpoint.EVALUATION_ROWS
        or frozen.get("authorized_rows_payload_sha256")
        != endpoint.AUTHORIZED_ROWS_PAYLOAD_SHA256
        or frozen.get("required_source_root_environment") != SIGNED_SOURCE_ROOT_ENV
        or frozen.get("signed_v5_source_commit") != SIGNED_V5_COMMIT
        or frozen.get("signed_v5_delta_impl_sha256")
        != SIGNED_V5_DELTA_IMPL_SHA256
        or frozen.get("learned_write_installed") is not False
        or frozen.get("outer_ffn_layers") != list(ANCHORS)
        or frozen.get("config_overrides") != []
        or frozen.get("initialize_missing_parameters") is not False
        or frozen.get("hf_endpoint") != HF_ENDPOINT
        or signed_split.get("selection_salt") != SPLIT_SALT
        or signed_split.get("fit_rows") != FIT_ROWS
        or signed_split.get("mechanics_rows") != MECHANICS_ROWS
        or signed_split.get("causal_rows") != CAUSAL_ROWS
        or signed_split.get("payload_sha256") != SPLIT_PAYLOAD_SHA256
        or signed_split.get("eligible_donor_mapping_pairs_sha256")
        != ELIGIBLE_MAPPING_SHA256
        or signed_split.get("capture_sources_sha256") != CAPTURE_SOURCES_SHA256
        or signed_split.get("donor_component_disjoint") is not True
        or signed_split.get("crossing_donor_edges") != 0
        or capture.get("capture_rows") != CAPTURE_ROWS
        or capture.get("capture_splits") != ["fit", "mechanics"]
        or capture.get("causal_rows_captured") != 0
        or capture.get("anchors") != list(ANCHORS)
        or capture.get("prompt_boundary_predictor_count_per_row") != 1
        or capture.get("read_basis_calls_per_anchor")
        != READ_BASIS_CALLS_PER_ANCHOR
        or capture.get("read_basis_call_roles") != list(READ_BASIS_CALL_ROLES)
        or capture.get("canonical_read_basis_call_role")
        != READ_BASIS_CALL_ROLES[0]
        or capture.get("duplicate_prompt_boundary_raw_byte_identity_required")
        is not True
        or capture.get("per_call_prompt_boundary_sha256_recorded") is not True
        or capture.get("prompt_boundary_sha256_scope")
        != "dtype_shape_and_exact_raw_bytes_of_only_the_selected_prompt_boundary_vector"
        or capture.get("full_return_shapes_and_dtypes_recorded_per_call") is not True
        or capture.get("full_state_or_sequence_hashed") is not False
        or capture.get("capture_seed") != SEED
        or capture.get("model_gradients") is not False
        or capture.get("model_output_changed") is not False
        or capture.get("write_or_read_dynamics_changed") is not False
        or capture.get("binder_bridge_or_code_module_installed_during_capture")
        is not False
        or capture.get("generation") is not False
        or capture.get("adapter_saved") is not False
        or fit.get("fit_rows") != FIT_ROWS
        or fit.get("anchors") != list(ANCHORS)
        or fit.get("state_width") != STATE_WIDTH
        or fit.get("codebook_size") != CODEBOOK_SIZE
        or fit.get("parameters_per_anchor") != 2 * STATE_WIDTH * CODEBOOK_SIZE
        or fit.get("total_parameters")
        != len(ANCHORS) * 2 * STATE_WIDTH * CODEBOOK_SIZE
        or fit.get("head_seed") != SEED
        or fit.get("temperature") != TEMPERATURE
        or fit.get("loss_weights") != LOSS_WEIGHTS
        or fit.get("donor_margin") != DONOR_MARGIN
        or fit.get("optimizer") != "AdamW"
        or fit.get("steps") != TRAIN_STEPS
        or fit.get("learning_rate") != LEARNING_RATE
        or fit.get("weight_decay") != WEIGHT_DECAY
        or fit.get("gradient_clipping") is not None
        or fit.get("thresholds") != "none"
        or fit.get("mechanics_or_causal_rows_used_for_fit_thresholds_architecture_hyperparameters_or_selection")
        is not False
        or fit.get("model_or_adapter_weights_updated") is not False
        or fit.get("code_map_weights_saved") is not False
        or gates.get("evaluation_rows") != MECHANICS_ROWS
        or gates.get("single_locked_evaluation") is not True
        or gates.get("correct_anchor_hard_code_match_fraction_minimum")
        != CORRECT_ANCHOR_GATE
        or gates.get("correct_complete_row_hard_code_match_fraction_minimum")
        != CORRECT_ROW_GATE
        or gates.get("donor_anchor_collision_fraction_maximum")
        != DONOR_ANCHOR_COLLISION_GATE
        or gates.get("donor_complete_row_collision_count_maximum")
        != DONOR_ROW_COLLISION_COUNT_GATE
        or gates.get("layer_permuted_anchor_collision_fraction_maximum")
        != LAYER_PERMUTED_ANCHOR_COLLISION_GATE
        or gates.get("layer_permuted_complete_row_collision_count_maximum")
        != LAYER_PERMUTED_ROW_COLLISION_COUNT_GATE
        or gates.get("minimum_distinct_hard_codes_per_anchor_per_side")
        != MINIMUM_DISTINCT_CODES
        or gates.get("maximum_single_hard_code_fraction_per_anchor_per_side")
        != MAXIMUM_CODE_FRACTION
        or gates.get("mechanics_used_for_fit_thresholds_architecture_hyperparameters_or_selection")
        is not False
        or execution.get("world_size") != WORLD_SIZE
        or execution.get("model_backend") != "nccl"
        or execution.get("control_backend") != "gloo"
        or execution.get("timeout_seconds") != DISTRIBUTED_TIMEOUT_SECONDS
        or execution.get("rank0_only_fit") is not True
        or execution.get("hf_endpoint") != HF_ENDPOINT
        or execution.get("output_directory")
        != str(DEFAULT_OUTPUT_DIR.relative_to(SCRIPT_DIR))
        or execution.get("fresh_output_required") is not True
        or execution.get("four_rank_failures_reach_consensus_before_later_collectives")
        is not True
        or firewall.get("causal_rows") != CAUSAL_ROWS
        or firewall.get("row_text_opened") is not False
        or firewall.get("tokenized") is not False
        or firewall.get("model_forward") is not False
        or firewall.get("features_captured") is not False
        or firewall.get("used_for_fit") is not False
        or firewall.get("used_for_thresholds") is not False
        or firewall.get("used_for_architecture_or_hyperparameters") is not False
        or firewall.get("used_for_model_or_candidate_selection") is not False
        or firewall.get("used_for_stopping_or_retry_decisions") is not False
        or firewall.get("result_metrics_reported") is not False
        or protocol.get("stage2_authorized") is not False
        or protocol.get("model_or_adapter_training_authorized") is not False
        or protocol.get("generation_authorized") is not False
        or protocol.get("adapter_saved") is not False
        or protocol.get("protected_splits_opened_by_this_protocol") != []
    ):
        raise ValueError("Signed PLMSC protocol differs")

    _validate_dependencies()
    failure = _signed_json(
        V1_OPERATIONAL_FAILURE,
        V1_OPERATIONAL_FAILURE_SHA256,
        V1_OPERATIONAL_FAILURE_RECEIPT,
    )
    if (
        failure.get("status")
        != "plmsc_v1_observer_call_count_contract_failed_before_feature_save"
        or failure.get("failure_class") != "instrumentation_topology_mismatch"
        or failure.get("root_cause", {}).get("observed_calls_per_anchor_before_failure")
        != READ_BASIS_CALLS_PER_ANCHOR
        or failure.get("authorization", {}).get("bounded_v2_protocol_draft_authorized")
        is not True
        or failure.get("authorization", {}).get("v1_retry_authorized") is not False
        or failure.get("data_firewall", {}).get("causal_rows_opened") is not False
        or failure.get("data_firewall", {}).get("saved_feature_rows") != 0
        or failure.get("data_firewall", {}).get("first_forward_source_by_rank")
        != V1_INCIDENT_BOUNDARY["first_forward_source_by_rank"]
        or failure.get("data_firewall", {}).get("first_forward_split_by_rank")
        != V1_INCIDENT_BOUNDARY["first_forward_split_by_rank"]
        or failure.get("data_firewall", {}).get("mechanics_metrics_computed") is not False
        or failure.get("mechanics_gate_evaluated") is not False
    ):
        raise ValueError("Signed PLMSC v1 operational failure contract differs")
    _, predictor_result, plat_nested = plat.validate_protocol()
    plat_result = _signed_json(PLAT_RESULT, PLAT_RESULT_SHA256, PLAT_RESULT_RECEIPT)
    signed_nested = dict(plat_result.get("nested_split", {}))
    nested_receipt = signed_nested.pop("payload_sha256", None)
    if (
        plat_result.get("schema") != plat.SCHEMA
        or plat_result.get("status") != "plat_prompt_latch_crossfit_failed_family_retired"
        or plat_result.get("passed") is not False
        or plat_result.get("plat_mechanics_design_authorized") is not False
        or plat_result.get("stage2_authorized") is not False
        or plat_result.get("model_or_adapter_training_authorized") is not False
        or plat_result.get("generation_authorized") is not False
        or plat_result.get("protected_splits_opened") != []
        or plat_result.get("protocol_payload_sha256") != plat.PROTOCOL_PAYLOAD_SHA256
        or plat_result.get("code_bindings", {}).get("runner_sha256")
        != PLAT_RUNNER_SHA256
        or nested_receipt != PLAT_SPLIT_PAYLOAD_SHA256
        or signed_nested != dict(plat_nested)
    ):
        raise ValueError("Signed failed PLAT parent contract differs")
    v5_protocol, v5_result = shadow.validate_protocol()
    del v5_protocol
    if (
        predictor_result.get("status") != "predictor_crossfit_failed_stage2_not_run"
        or predictor_result.get("passed") is not False
        or predictor_result.get("stage2_executed") is not False
        or predictor_result.get("protected_splits_opened") != []
        or v5_result.get("status")
        != "address_keyed_moe_deepembed_ffn_serialized_graphs_heldout_passed_generation_authorized"
        or v5_result.get("receipt", {}).get("payload_sha256") != V5_RESULT_RECEIPT
        or v5_result.get("protected_splits_opened") != []
    ):
        raise ValueError("Signed predictor or exact-v5 authorization differs")
    split, split_payload, eligible_row_contracts = derive_three_way_split(
        predictor_result, plat_result
    )
    signed_split_payload = {
        key: signed_split[key]
        for key in (
            "selection_salt",
            "component_count",
            "component_sizes",
            "mechanics_component_indices",
            "causal_component_indices",
            "fit_sources",
            "mechanics_sources",
            "causal_sources",
            "excluded_prior_heldout_sources",
            "excluded_plat_heldout_sources",
        )
    }
    if signed_split_payload != split_payload:
        raise ValueError("Signed PLMSC split does not reproduce from parent metadata")
    return protocol, predictor_result, plat_result, split_payload, eligible_row_contracts


def _capture_read_basis(
    module: Any,
    state: torch.Tensor,
    memory_source_seq: torch.Tensor,
    token_mask: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    result = module.rwkv_plmsc_original_token_state_read_basis(
        state, memory_source_seq, token_mask
    )
    if (
        not isinstance(result, tuple)
        or len(result) != 3
        or not all(isinstance(value, torch.Tensor) for value in result)
    ):
        raise RuntimeError("PLMSC RWKV read-basis return contract differs")
    receptance = result[0]
    if receptance.ndim != 4:
        raise RuntimeError("PLMSC RWKV receptance must be [batch, token, head, rank]")
    capture_index = module.rwkv_plmsc_prompt_boundary_predictor_index
    if capture_index is not None:
        if not 0 <= int(capture_index) < receptance.size(1):
            raise RuntimeError("PLMSC prompt-boundary capture index is out of range")
        selected = receptance[:, int(capture_index)].flatten(start_dim=1)
        if tuple(selected.shape) != (1, STATE_WIDTH):
            raise RuntimeError("PLMSC selected RWKV receptance shape differs")
        if not bool(torch.isfinite(selected).all().item()):
            raise RuntimeError("PLMSC selected RWKV receptance is nonfinite")
        calls = int(module.rwkv_plmsc_read_basis_calls)
        if calls >= READ_BASIS_CALLS_PER_ANCHOR:
            raise RuntimeError(
                "PLMSC anchor read-basis observer exceeded exactly two calls per read"
            )
        capture = selected.detach().clone()
        shapes = [list(value.shape) for value in result]
        dtypes = [str(value.dtype) for value in result]
        digest = tensor_raw_bytes_sha256(capture)
        module.rwkv_plmsc_prompt_boundary_r_seq_captures.append(capture)
        module.rwkv_plmsc_read_basis_return_shapes.append(shapes)
        module.rwkv_plmsc_read_basis_return_dtypes.append(dtypes)
        module.rwkv_plmsc_prompt_boundary_sha256.append(digest)
        module.rwkv_plmsc_read_basis_calls = calls + 1
        if calls == 1:
            first = module.rwkv_plmsc_prompt_boundary_r_seq_captures[0]
            if (
                module.rwkv_plmsc_read_basis_return_shapes[0] != shapes
                or module.rwkv_plmsc_read_basis_return_dtypes[0] != dtypes
                or not tensor_raw_bytes_equal(first, capture)
                or module.rwkv_plmsc_prompt_boundary_sha256[0] != digest
            ):
                raise RuntimeError(
                    "PLMSC addressed/global prompt-boundary read bases differ"
                )
    return result


def install_anchor_read_capture(model: torch.nn.Module) -> Mapping[str, Any]:
    installed: list[Mapping[str, Any]] = []
    for module_name, module in causal_train.ordered_modules(model):
        layer = int(module.layer_idx)
        if layer not in ANCHORS:
            continue
        if hasattr(module, "rwkv_plmsc_original_token_state_read_basis"):
            raise ValueError(f"PLMSC read-basis capture is already installed: {module_name}")
        if int(module.state_read_dim) != STATE_WIDTH:
            raise ValueError(f"PLMSC anchor state width differs: {module_name}")
        module.rwkv_plmsc_original_token_state_read_basis = (
            module._rwkv_ms_token_state_read_basis
        )
        module.rwkv_plmsc_prompt_boundary_predictor_index = None
        module.rwkv_plmsc_prompt_boundary_r_seq_captures = []
        module.rwkv_plmsc_read_basis_return_shapes = []
        module.rwkv_plmsc_read_basis_return_dtypes = []
        module.rwkv_plmsc_prompt_boundary_sha256 = []
        module.rwkv_plmsc_read_basis_calls = 0
        module._rwkv_ms_token_state_read_basis = MethodType(
            _capture_read_basis, module
        )
        installed.append({"layer": layer, "module_name": module_name})
    if [item["layer"] for item in installed] != list(ANCHORS):
        raise RuntimeError("PLMSC read-basis anchors differ")
    return {
        "observer": "anchor_hrm_rwkv7_r_seq",
        "anchors": list(ANCHORS),
        "modules": installed,
        "captured_width": STATE_WIDTH,
        "read_basis_calls_per_anchor": READ_BASIS_CALLS_PER_ANCHOR,
        "read_basis_call_roles": list(READ_BASIS_CALL_ROLES),
        "canonical_read_basis_call_role": READ_BASIS_CALL_ROLES[0],
        "addressed_global_prompt_boundary_raw_byte_identity_required": True,
        "full_state_or_sequence_hashed": False,
        "forward_output_changed": False,
        "write_or_read_dynamics_changed": False,
        "trainable_parameters_added": 0,
        "binder_bridge_or_code_module_installed": False,
    }


def first_prompt_boundary(labels: torch.Tensor) -> tuple[int, int]:
    if labels.ndim != 2 or labels.size(0) != 1 or labels.size(1) < 2:
        raise ValueError("PLMSC capture requires one batched causal label row")
    supervised = labels[0].ne(-100).nonzero(as_tuple=False).flatten()
    if supervised.numel() < 1:
        raise ValueError("PLMSC row has no supervised label")
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
        raise RuntimeError("PLMSC first-label-minus-one causal boundary differs")
    return first_label, predictor


def _anchor_modules(model: torch.nn.Module) -> Mapping[int, tuple[str, Any]]:
    selected = {
        int(module.layer_idx): (module_name, module)
        for module_name, module in causal_train.ordered_modules(model)
        if int(module.layer_idx) in ANCHORS
    }
    if tuple(sorted(selected)) != ANCHORS:
        raise RuntimeError("PLMSC exact-v5 anchor modules differ")
    return selected


@torch.no_grad()
def capture_row(
    model: torch.nn.Module,
    example: Any,
    *,
    pad_token_id: int,
    device: torch.device,
) -> Mapping[str, Any]:
    modules = causal_train.ordered_modules(model)
    anchors = _anchor_modules(model)
    batch = evolution.collate_native_examples(
        [example], pad_token_id=pad_token_id, device=device
    )
    try:
        reset = shadow.reset_delta_mem_states
        reset(model)
        evolution._native_write(model, batch, dtype=torch.bfloat16)
        all_addresses = write_address_capture.capture_write_addresses(model)
        write_addresses: list[torch.Tensor] = []
        for layer in ANCHORS:
            module_name, _ = anchors[layer]
            address = all_addresses[module_name]
            if tuple(address.shape) != (1, 1, STATE_WIDTH):
                raise RuntimeError("PLMSC write-slot address shape differs")
            write_addresses.append(address[0, 0].float().detach().clone())
        first_label, predictor_position = first_prompt_boundary(batch.labels)
        for _, module in anchors.values():
            module.rwkv_plmsc_prompt_boundary_predictor_index = predictor_position
            module.rwkv_plmsc_prompt_boundary_r_seq_captures = []
            module.rwkv_plmsc_read_basis_return_shapes = []
            module.rwkv_plmsc_read_basis_return_dtypes = []
            module.rwkv_plmsc_prompt_boundary_sha256 = []
            module.rwkv_plmsc_read_basis_calls = 0
        logits = evolution._native_read(model, batch, dtype=torch.bfloat16)
        if not predictor.reads_are_write_disabled(modules):
            raise RuntimeError("PLMSC binder-disabled read left memory writes enabled")
        query_vectors: list[torch.Tensor] = []
        read_basis_calls: list[int] = []
        observations_identical: list[bool] = []
        prompt_boundary_sha256: list[list[str]] = []
        return_shapes: list[list[list[list[int]]]] = []
        return_dtypes: list[list[list[str]]] = []
        for layer in ANCHORS:
            _, module = anchors[layer]
            captures = module.rwkv_plmsc_prompt_boundary_r_seq_captures
            shapes = module.rwkv_plmsc_read_basis_return_shapes
            dtypes = module.rwkv_plmsc_read_basis_return_dtypes
            digests = module.rwkv_plmsc_prompt_boundary_sha256
            calls = int(module.rwkv_plmsc_read_basis_calls)
            read_basis_calls.append(calls)
            if (
                calls != READ_BASIS_CALLS_PER_ANCHOR
                or len(captures) != READ_BASIS_CALLS_PER_ANCHOR
                or len(shapes) != READ_BASIS_CALLS_PER_ANCHOR
                or len(dtypes) != READ_BASIS_CALLS_PER_ANCHOR
                or len(digests) != READ_BASIS_CALLS_PER_ANCHOR
            ):
                raise RuntimeError(
                    "PLMSC anchor read-basis observer must run exactly twice per read"
                )
            identical = (
                shapes[0] == shapes[1]
                and dtypes[0] == dtypes[1]
                and digests[0] == digests[1]
                and tensor_raw_bytes_equal(captures[0], captures[1])
            )
            if not identical:
                raise RuntimeError(
                    "PLMSC addressed/global prompt-boundary read bases differ"
                )
            query_vectors.append(captures[0][0].float().detach().clone())
            observations_identical.append(True)
            prompt_boundary_sha256.append(list(digests))
            return_shapes.append(list(shapes))
            return_dtypes.append(list(dtypes))
        del logits
        write_tensor = torch.stack(write_addresses)
        query_tensor = torch.stack(query_vectors)
        if (
            tuple(write_tensor.shape) != (len(ANCHORS), STATE_WIDTH)
            or tuple(query_tensor.shape) != (len(ANCHORS), STATE_WIDTH)
            or not bool(torch.isfinite(write_tensor).all().item())
            or not bool(torch.isfinite(query_tensor).all().item())
            or bool(write_tensor.norm(dim=-1).le(1e-6).any().item())
            or bool(query_tensor.norm(dim=-1).le(1e-6).any().item())
        ):
            raise RuntimeError("PLMSC write/query features are invalid or collapsed")
        if any(
            getattr(module, "rwkv_query_state_identity_fixed_address", None)
            is not None
            for _, module in modules
        ) or hasattr(model, "rwkv_identity_binder_bank"):
            raise RuntimeError("PLMSC capture unexpectedly installed a binder")
        return {
            "anchors": list(ANCHORS),
            "write_slot_address": write_tensor.cpu().tolist(),
            "prompt_boundary_rwkv_receptance": query_tensor.cpu().tolist(),
            "first_supervised_label_index": first_label,
            "prompt_boundary_predictor_index": predictor_position,
            "predictor_definition": "first_supervised_label_index_minus_one",
            "predictor_vectors_per_row": 1,
            "answer_or_later_predictor_features_captured": False,
            "write_passes": 1,
            "read_passes": 1,
            "read_basis_calls_per_anchor": read_basis_calls,
            "read_basis_call_roles": list(READ_BASIS_CALL_ROLES),
            "read_basis_observations_byte_identical_per_anchor": observations_identical,
            "read_basis_prompt_boundary_sha256_per_anchor_per_call": prompt_boundary_sha256,
            "read_basis_return_shapes_per_anchor_per_call": return_shapes,
            "read_basis_return_dtypes_per_anchor_per_call": return_dtypes,
            "read_writes_enabled": False,
            "features_detached_and_cloned": True,
            "model_output_changed_by_capture": False,
            "binder_bridge_or_code_module_installed_during_capture": False,
        }
    finally:
        for _, module in anchors.values():
            module.rwkv_plmsc_prompt_boundary_predictor_index = None
            module.rwkv_plmsc_prompt_boundary_r_seq_captures = []
            module.rwkv_plmsc_read_basis_return_shapes = []
            module.rwkv_plmsc_read_basis_return_dtypes = []
            module.rwkv_plmsc_prompt_boundary_sha256 = []
            module.rwkv_plmsc_read_basis_calls = 0
        shadow.reset_delta_mem_states(model)
        evolution.release_native_row_allocator_cache(device)


def _load_local_examples(
    tokenizer: Any,
    dataset_root: Path,
    eligible_row_contracts: Mapping[int, Mapping[str, Any]],
    split_payload: Mapping[str, Any],
    process_rank: int,
) -> tuple[Mapping[int, Any], Mapping[str, Any]]:
    capture_sources = set(split_payload["fit_sources"]) | set(
        split_payload["mechanics_sources"]
    )
    causal_sources = set(split_payload["causal_sources"])
    excluded_prior = set(split_payload["excluded_prior_heldout_sources"])
    excluded_plat = set(split_payload["excluded_plat_heldout_sources"])
    local_sources = {
        source for source in capture_sources if source % WORLD_SIZE == process_rank
    }
    dataset_path = dataset_root / endpoint.DATASET_RELATIVE_PATH
    if sha256_file(dataset_path) != endpoint.DATASET_SHA256:
        raise ValueError("PLMSC native development dataset differs")
    retained: dict[int, bytes] = {}
    verified_capture: set[int] = set()
    causal_opaque_skipped = 0
    excluded_prior_opaque_skipped = 0
    excluded_plat_opaque_skipped = 0
    line_count = 0
    with dataset_path.open("rb") as handle:
        for source_index, raw_with_newline in enumerate(handle):
            line_count += 1
            if source_index in causal_sources:
                causal_opaque_skipped += 1
                continue
            if source_index in excluded_prior:
                excluded_prior_opaque_skipped += 1
                continue
            if source_index in excluded_plat:
                excluded_plat_opaque_skipped += 1
                continue
            if source_index not in capture_sources:
                continue
            raw_line = raw_with_newline.rstrip(b"\r\n")
            expected = eligible_row_contracts[source_index]
            verified_capture.add(source_index)
            if hashlib.sha256(raw_line).hexdigest() != expected["row_sha256"]:
                raise ValueError("PLMSC signed capture-source row hash differs")
            if source_index in local_sources:
                retained[source_index] = raw_line
    if (
        line_count != 361
        or verified_capture != capture_sources
        or set(retained) != local_sources
        or set(retained) & (causal_sources | excluded_prior | excluded_plat)
        or causal_opaque_skipped != CAUSAL_ROWS
        or excluded_prior_opaque_skipped != PRIOR_EXCLUDED_ROWS
        or excluded_plat_opaque_skipped != PLAT_EXCLUDED_ROWS
    ):
        raise RuntimeError("PLMSC dataset metadata or local capture coverage differs")
    retained_forbidden = set(retained) & (
        causal_sources | excluded_prior | excluded_plat
    )
    examples = {
        source: evolution.encode_native_full_row(
            tokenizer,
            task="scene",
            source_ordinal=source,
            raw_line=retained[source].decode("utf-8"),
        )
        for source in sorted(local_sources)
    }
    if any(
        examples[source].row_sha256
        != eligible_row_contracts[source]["row_sha256"]
        for source in examples
    ):
        raise RuntimeError("PLMSC encoded capture row hash differs")
    return examples, {
        "dataset_sha256": endpoint.DATASET_SHA256,
        "capture_row_hashes_verified": len(verified_capture),
        "local_capture_rows_decoded_and_tokenized": len(examples),
        "causal_rows_opaque_skipped": causal_opaque_skipped,
        "excluded_prior_rows_opaque_skipped": excluded_prior_opaque_skipped,
        "excluded_plat_rows_opaque_skipped": excluded_plat_opaque_skipped,
        "retained_causal_or_excluded_payloads": len(retained_forbidden),
        "causal_rows_decoded": 0,
        "causal_rows_tokenized": 0,
        "causal_rows_model_forwarded": 0,
        "causal_features_captured": 0,
    }


def _write_signed_json(path: Path, value: Mapping[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=True, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _signed_feature_row(value: Mapping[str, Any]) -> Mapping[str, Any]:
    row = dict(value)
    row["receipt"] = {
        "algorithm": "sha256",
        "payload_scope": "canonical_feature_without_receipt",
        "payload_sha256": canonical_sha256(row),
    }
    return row


def _valid_read_basis_evidence(
    row: Mapping[str, Any], query: torch.Tensor
) -> bool:
    identities = row.get("read_basis_observations_byte_identical_per_anchor")
    digests = row.get("read_basis_prompt_boundary_sha256_per_anchor_per_call")
    shapes = row.get("read_basis_return_shapes_per_anchor_per_call")
    dtypes = row.get("read_basis_return_dtypes_per_anchor_per_call")
    if (
        row.get("read_basis_call_roles") != list(READ_BASIS_CALL_ROLES)
        or identities != [True] * len(ANCHORS)
        or not all(
            isinstance(value, list) and len(value) == len(ANCHORS)
            for value in (digests, shapes, dtypes)
        )
    ):
        return False
    for anchor_index in range(len(ANCHORS)):
        anchor_digests = digests[anchor_index]
        anchor_shapes = shapes[anchor_index]
        anchor_dtypes = dtypes[anchor_index]
        if (
            not isinstance(anchor_digests, list)
            or len(anchor_digests) != READ_BASIS_CALLS_PER_ANCHOR
            or anchor_digests[0] != anchor_digests[1]
            or any(
                not isinstance(digest, str)
                or len(digest) != 64
                or any(character not in "0123456789abcdef" for character in digest)
                for digest in anchor_digests
            )
            or tensor_raw_bytes_sha256(query[anchor_index : anchor_index + 1])
            != anchor_digests[0]
            or not isinstance(anchor_shapes, list)
            or len(anchor_shapes) != READ_BASIS_CALLS_PER_ANCHOR
            or anchor_shapes[0] != anchor_shapes[1]
            or any(
                not isinstance(call_shapes, list)
                or len(call_shapes) != 3
                or any(
                    not isinstance(shape, list)
                    or not all(isinstance(dimension, int) for dimension in shape)
                    for shape in call_shapes
                )
                for call_shapes in anchor_shapes
            )
            or len(anchor_shapes[0][0]) != 4
            or anchor_shapes[0][0][0] != 1
            or anchor_shapes[0][0][2] * anchor_shapes[0][0][3] != STATE_WIDTH
            or not isinstance(anchor_dtypes, list)
            or len(anchor_dtypes) != READ_BASIS_CALLS_PER_ANCHOR
            or anchor_dtypes[0] != anchor_dtypes[1]
            or any(
                not isinstance(call_dtypes, list)
                or len(call_dtypes) != 3
                or not all(isinstance(dtype, str) for dtype in call_dtypes)
                for call_dtypes in anchor_dtypes
            )
        ):
            return False
    return True


def _validate_feature_row(
    row: Mapping[str, Any],
    split: Mapping[int, str],
    signed_rows: Mapping[int, Mapping[str, Any]],
) -> None:
    unsigned = dict(row)
    receipt = unsigned.pop("receipt", {})
    source = int(row["source_index"])
    expected = signed_rows[source]
    write = torch.tensor(row.get("write_slot_address"), dtype=torch.float32)
    query = torch.tensor(
        row.get("prompt_boundary_rwkv_receptance"), dtype=torch.float32
    )
    if (
        set(row) != FEATURE_ROW_KEYS
        or receipt.get("payload_sha256") != canonical_sha256(unsigned)
        or row.get("schema") != ROW_SCHEMA
        or source not in split
        or split[source] not in {"fit", "mechanics"}
        or row.get("split") != split[source]
        or source % WORLD_SIZE != int(row["capture_rank"])
        or row.get("row_sha256") != expected["row_sha256"]
        or int(row.get("donor_source_index"))
        != int(expected["donor_source_index"])
        or row.get("donor_row_sha256") != expected["donor_row_sha256"]
        or row.get("anchors") != list(ANCHORS)
        or tuple(write.shape) != (len(ANCHORS), STATE_WIDTH)
        or tuple(query.shape) != (len(ANCHORS), STATE_WIDTH)
        or not bool(torch.isfinite(write).all().item())
        or not bool(torch.isfinite(query).all().item())
        or bool(write.norm(dim=-1).le(1e-6).any().item())
        or bool(query.norm(dim=-1).le(1e-6).any().item())
        or row.get("predictor_definition")
        != "first_supervised_label_index_minus_one"
        or int(row.get("prompt_boundary_predictor_index"))
        != int(row.get("first_supervised_label_index")) - 1
        or row.get("predictor_vectors_per_row") != 1
        or row.get("answer_or_later_predictor_features_captured") is not False
        or row.get("write_passes") != 1
        or row.get("read_passes") != 1
        or row.get("read_basis_calls_per_anchor")
        != [READ_BASIS_CALLS_PER_ANCHOR] * len(ANCHORS)
        or not _valid_read_basis_evidence(row, query)
        or row.get("read_writes_enabled") is not False
        or row.get("features_detached_and_cloned") is not True
        or row.get("model_output_changed_by_capture") is not False
        or row.get("binder_bridge_or_code_module_installed_during_capture")
        is not False
    ):
        raise ValueError("PLMSC signed feature row differs")


def load_feature_shards(
    output_dir: Path,
    split_payload: Mapping[str, Any],
    signed_rows: Mapping[int, Mapping[str, Any]],
) -> tuple[list[Mapping[str, Any]], list[Mapping[str, Any]]]:
    split = {
        **{source: "fit" for source in split_payload["fit_sources"]},
        **{source: "mechanics" for source in split_payload["mechanics_sources"]},
        **{source: "causal" for source in split_payload["causal_sources"]},
    }
    records: list[Mapping[str, Any]] = []
    provenance: list[Mapping[str, Any]] = []
    for rank in range(WORLD_SIZE):
        path = output_dir / f"shard-{rank}.json"
        shard = json.loads(path.read_text(encoding="utf-8"))
        unsigned = dict(shard)
        receipt = unsigned.pop("receipt", {})
        rows = shard.get("rows", [])
        if (
            shard.get("schema") != SHARD_SCHEMA
            or shard.get("rank") != rank
            or receipt.get("payload_sha256") != canonical_sha256(unsigned)
            or not isinstance(rows, list)
        ):
            raise ValueError("PLMSC signed feature shard differs")
        for row in rows:
            _validate_feature_row(row, split, signed_rows)
            if int(row["capture_rank"]) != rank:
                raise ValueError("PLMSC feature is stored in the wrong shard")
            records.append(row)
        provenance.append(
            {
                "basename": path.name,
                "rows": len(rows),
                "sha256": sha256_file(path),
                "receipt": receipt["payload_sha256"],
            }
        )
    sources = [int(row["source_index"]) for row in records]
    expected_capture = sorted(
        set(split_payload["fit_sources"]) | set(split_payload["mechanics_sources"])
    )
    if (
        len(records) != CAPTURE_ROWS
        or len(set(sources)) != CAPTURE_ROWS
        or sorted(sources) != expected_capture
        or set(sources) & set(split_payload["causal_sources"])
        or set(sources) & set(split_payload["excluded_prior_heldout_sources"])
        or set(sources) & set(split_payload["excluded_plat_heldout_sources"])
    ):
        raise RuntimeError("PLMSC shard capture firewall or coverage differs")
    return records, provenance


class PairedAnchorCodeMaps(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.write_maps = nn.ModuleList(
            nn.Linear(STATE_WIDTH, CODEBOOK_SIZE, bias=False) for _ in ANCHORS
        )
        self.query_maps = nn.ModuleList(
            nn.Linear(STATE_WIDTH, CODEBOOK_SIZE, bias=False) for _ in ANCHORS
        )
        for projection in (*self.write_maps, *self.query_maps):
            nn.init.xavier_uniform_(projection.weight)

    @staticmethod
    def _validate(value: torch.Tensor) -> torch.Tensor:
        if value.ndim != 3 or tuple(value.shape[1:]) != (
            len(ANCHORS),
            STATE_WIDTH,
        ):
            raise ValueError("PLMSC code-map input must be [row, anchor, state]")
        if not bool(torch.isfinite(value).all().item()):
            raise ValueError("PLMSC code-map input is non-finite")
        return F.normalize(value, dim=-1, eps=1e-6)

    def write_logits(self, value: torch.Tensor) -> torch.Tensor:
        normalized = self._validate(value)
        return torch.stack(
            [projection(normalized[:, index]) for index, projection in enumerate(self.write_maps)],
            dim=1,
        )

    def query_logits(self, value: torch.Tensor) -> torch.Tensor:
        normalized = self._validate(value)
        return torch.stack(
            [projection(normalized[:, index]) for index, projection in enumerate(self.query_maps)],
            dim=1,
        )


def _code_map_sha256(code_maps: PairedAnchorCodeMaps) -> str:
    digest = hashlib.sha256()
    for name, tensor in sorted(code_maps.state_dict().items()):
        value = tensor.detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(tuple(value.shape)).encode("ascii"))
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def _record_tensors(
    records: Sequence[Mapping[str, Any]], split: str
) -> tuple[list[Mapping[str, Any]], torch.Tensor, torch.Tensor]:
    selected = sorted(
        (row for row in records if row["split"] == split),
        key=lambda row: int(row["source_index"]),
    )
    expected = FIT_ROWS if split == "fit" else MECHANICS_ROWS
    if len(selected) != expected:
        raise RuntimeError(f"PLMSC {split} record count differs")
    write = torch.tensor(
        [row["write_slot_address"] for row in selected], dtype=torch.float32
    )
    query = torch.tensor(
        [row["prompt_boundary_rwkv_receptance"] for row in selected],
        dtype=torch.float32,
    )
    if (
        tuple(write.shape) != (expected, len(ANCHORS), STATE_WIDTH)
        or tuple(query.shape) != tuple(write.shape)
        or not bool(torch.isfinite(torch.cat((write.flatten(), query.flatten()))).all())
        or bool(write.norm(dim=-1).le(1e-6).any().item())
        or bool(query.norm(dim=-1).le(1e-6).any().item())
    ):
        raise RuntimeError(f"PLMSC {split} feature tensors differ")
    return selected, write, query


def _donor_indices(records: Sequence[Mapping[str, Any]]) -> torch.Tensor:
    index = {int(row["source_index"]): position for position, row in enumerate(records)}
    donors: list[int] = []
    for row in records:
        donor = int(row["donor_source_index"])
        if donor not in index:
            raise RuntimeError("PLMSC donor leaves the selected component split")
        donors.append(index[donor])
    return torch.tensor(donors, dtype=torch.long)


def _distributions(logits: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    log_probabilities = F.log_softmax(logits / TEMPERATURE, dim=-1)
    return log_probabilities, log_probabilities.exp()


def _loss_parts(
    write_logits: torch.Tensor,
    query_logits: torch.Tensor,
    donor_indices: torch.Tensor,
) -> Mapping[str, torch.Tensor]:
    write_log, write_probability = _distributions(write_logits)
    query_log, query_probability = _distributions(query_logits)
    agreement = -0.5 * (
        (write_probability.detach() * query_log).sum(dim=-1)
        + (query_probability.detach() * write_log).sum(dim=-1)
    ).mean()
    correct_affinity = (query_probability * write_probability).sum(dim=-1)
    donor_affinity = (
        query_probability * write_probability.index_select(0, donor_indices)
    ).sum(dim=-1)
    donor_margin = F.relu(DONOR_MARGIN - correct_affinity + donor_affinity).mean()
    marginal = 0.5 * (
        write_probability.mean(dim=0) + query_probability.mean(dim=0)
    )
    balance = (
        marginal
        * (marginal.clamp_min(1e-12).log() + math.log(float(CODEBOOK_SIZE)))
    ).sum(dim=-1).mean()
    sharpness = -0.5 * (
        (write_probability * write_log).sum(dim=-1)
        + (query_probability * query_log).sum(dim=-1)
    ).mean() / math.log(float(CODEBOOK_SIZE))
    total = (
        LOSS_WEIGHTS["agreement"] * agreement
        + LOSS_WEIGHTS["donor_margin"] * donor_margin
        + LOSS_WEIGHTS["balance"] * balance
        + LOSS_WEIGHTS["sharpness"] * sharpness
    )
    return {
        "total": total,
        "agreement": agreement,
        "donor_margin": donor_margin,
        "balance": balance,
        "sharpness": sharpness,
    }


def train_fit_only(
    records: Sequence[Mapping[str, Any]],
) -> tuple[PairedAnchorCodeMaps, Mapping[str, Any]]:
    fit_records, write, query = _record_tensors(records, "fit")
    donors = _donor_indices(fit_records)
    torch.manual_seed(SEED)
    code_maps = PairedAnchorCodeMaps()
    if sum(parameter.numel() for parameter in code_maps.parameters()) != 16384:
        raise RuntimeError("PLMSC code-map parameter count differs")
    initial_binding = _code_map_sha256(code_maps)
    optimizer = torch.optim.AdamW(
        code_maps.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )
    history: list[Mapping[str, float]] = []
    for _ in range(TRAIN_STEPS):
        optimizer.zero_grad(set_to_none=True)
        parts = _loss_parts(
            code_maps.write_logits(write), code_maps.query_logits(query), donors
        )
        if not all(bool(torch.isfinite(value).item()) for value in parts.values()):
            raise RuntimeError("PLMSC fit objective is non-finite")
        parts["total"].backward()
        optimizer.step()
        if any(
            not bool(torch.isfinite(parameter).all().item())
            for parameter in code_maps.parameters()
        ):
            raise RuntimeError("PLMSC code-map parameter is non-finite")
        history.append({name: float(value.detach().item()) for name, value in parts.items()})
    code_maps.eval()
    for parameter in code_maps.parameters():
        parameter.requires_grad_(False)
    final_binding = _code_map_sha256(code_maps)
    return code_maps, {
        "architecture": {
            "anchors": list(ANCHORS),
            "maps_per_anchor": ["Ww", "Wq"],
            "input_width": STATE_WIDTH,
            "codebook_size": CODEBOOK_SIZE,
            "bias": False,
            "input_normalization": "L2",
            "temperature": TEMPERATURE,
            "parameters": 16384,
            "monomial_transform": None,
        },
        "optimizer": {
            "name": "AdamW",
            "seed": SEED,
            "steps": TRAIN_STEPS,
            "learning_rate": LEARNING_RATE,
            "weight_decay": WEIGHT_DECAY,
            "gradient_clipping": None,
        },
        "objective": {
            "loss_weights": dict(LOSS_WEIGHTS),
            "donor_margin": DONOR_MARGIN,
            "agreement": "symmetric_stop_gradient_soft_cross_entropy",
            "donor": "probability_dot_margin_hinge",
            "balance": "per_anchor_joint_write_query_marginal_kl_to_uniform",
            "sharpness": "mean_distribution_entropy_divided_by_log64",
            "labels_used_only_to_locate_first_supervised_label_minus_one": True,
            "answer_or_later_predictor_features_used": False,
        },
        "loss": {
            "initial": dict(history[0]),
            "final": dict(history[-1]),
            "all_steps_finite": True,
        },
        "initial_code_map_sha256": initial_binding,
        "frozen_code_map_sha256": final_binding,
        "code_map_weights_saved": False,
        "mechanics_or_causal_rows_used": False,
        "thresholds": None,
    }


def _branch_metrics(
    query_probability: torch.Tensor,
    write_probability: torch.Tensor,
    query_code: torch.Tensor,
    write_code: torch.Tensor,
) -> Mapping[str, Any]:
    matches = query_code.eq(write_code)
    affinities = (query_probability * write_probability).sum(dim=-1)
    return {
        "anchor_match_or_collision_count": int(matches.sum().item()),
        "anchor_match_or_collision_fraction": float(matches.float().mean().item()),
        "complete_row_match_or_collision_count": int(matches.all(dim=1).sum().item()),
        "complete_row_match_or_collision_fraction": float(
            matches.all(dim=1).float().mean().item()
        ),
        "mean_probability_dot_affinity": float(affinities.mean().item()),
        "per_anchor_match_or_collision_fraction": [
            float(value) for value in matches.float().mean(dim=0).tolist()
        ],
        "finite": bool(torch.isfinite(affinities).all().item()),
    }


def _usage_metrics(codes: torch.Tensor) -> list[Mapping[str, Any]]:
    rows = int(codes.size(0))
    result: list[Mapping[str, Any]] = []
    for anchor_index, layer in enumerate(ANCHORS):
        counts = torch.bincount(codes[:, anchor_index], minlength=CODEBOOK_SIZE)
        result.append(
            {
                "layer": layer,
                "distinct_codes": int(counts.gt(0).sum().item()),
                "maximum_single_code_count": int(counts.max().item()),
                "maximum_single_code_fraction": float(counts.max().item() / rows),
            }
        )
    return result


def _finite_tree(value: Any) -> bool:
    if isinstance(value, bool) or value is None or isinstance(value, str):
        return True
    if isinstance(value, int):
        return True
    if isinstance(value, float):
        return math.isfinite(value)
    if isinstance(value, Mapping):
        return all(_finite_tree(item) for item in value.values())
    if isinstance(value, Sequence):
        return all(_finite_tree(item) for item in value)
    return False


@torch.no_grad()
def evaluate_mechanics_once(
    code_maps: PairedAnchorCodeMaps,
    records: Sequence[Mapping[str, Any]],
) -> Mapping[str, Any]:
    mechanics_records, write, query = _record_tensors(records, "mechanics")
    donor_indices = _donor_indices(mechanics_records)
    correct_write_logits = code_maps.write_logits(write)
    query_logits = code_maps.query_logits(query)
    donor_write_logits = code_maps.write_logits(write.index_select(0, donor_indices))
    cyclic_write = torch.roll(write, shifts=1, dims=1)
    permuted_write_logits = code_maps.write_logits(cyclic_write)
    _, correct_write_probability = _distributions(correct_write_logits)
    _, query_probability = _distributions(query_logits)
    _, donor_write_probability = _distributions(donor_write_logits)
    _, permuted_write_probability = _distributions(permuted_write_logits)
    query_code = query_logits.argmax(dim=-1)
    correct_write_code = correct_write_logits.argmax(dim=-1)
    donor_write_code = donor_write_logits.argmax(dim=-1)
    permuted_write_code = permuted_write_logits.argmax(dim=-1)
    correct = _branch_metrics(
        query_probability,
        correct_write_probability,
        query_code,
        correct_write_code,
    )
    donor = _branch_metrics(
        query_probability,
        donor_write_probability,
        query_code,
        donor_write_code,
    )
    layer_permuted = _branch_metrics(
        query_probability,
        permuted_write_probability,
        query_code,
        permuted_write_code,
    )
    usage = {
        "correct_write": _usage_metrics(correct_write_code),
        "correct_query": _usage_metrics(query_code),
    }
    finite = _finite_tree(
        {"correct": correct, "donor": donor, "layer_permuted": layer_permuted, "usage": usage}
    ) and all(
        bool(torch.isfinite(value).all().item())
        for value in (
            correct_write_logits,
            query_logits,
            donor_write_logits,
            permuted_write_logits,
            correct_write_probability,
            query_probability,
            donor_write_probability,
            permuted_write_probability,
        )
    )
    noncollapsed = all(
        item["distinct_codes"] >= MINIMUM_DISTINCT_CODES
        and item["maximum_single_code_fraction"] <= MAXIMUM_CODE_FRACTION
        for side in usage.values()
        for item in side
    )
    donor_row_fraction_gate = DONOR_ROW_COLLISION_COUNT_GATE / MECHANICS_ROWS
    permuted_row_fraction_gate = (
        LAYER_PERMUTED_ROW_COLLISION_COUNT_GATE / MECHANICS_ROWS
    )
    checks = {
        "correct_anchor_match_fraction": (
            correct["anchor_match_or_collision_fraction"] >= CORRECT_ANCHOR_GATE
        ),
        "correct_complete_row_match_fraction": (
            correct["complete_row_match_or_collision_fraction"] >= CORRECT_ROW_GATE
        ),
        "donor_anchor_collision_fraction": (
            donor["anchor_match_or_collision_fraction"]
            <= DONOR_ANCHOR_COLLISION_GATE
        ),
        "donor_complete_row_collision_count": (
            donor["complete_row_match_or_collision_count"]
            <= DONOR_ROW_COLLISION_COUNT_GATE
        ),
        "donor_complete_row_collision_fraction": (
            donor["complete_row_match_or_collision_fraction"]
            <= donor_row_fraction_gate
        ),
        "layer_permuted_anchor_collision_fraction": (
            layer_permuted["anchor_match_or_collision_fraction"]
            <= LAYER_PERMUTED_ANCHOR_COLLISION_GATE
        ),
        "layer_permuted_complete_row_collision_count": (
            layer_permuted["complete_row_match_or_collision_count"]
            <= LAYER_PERMUTED_ROW_COLLISION_COUNT_GATE
        ),
        "layer_permuted_complete_row_collision_fraction": (
            layer_permuted["complete_row_match_or_collision_fraction"]
            <= permuted_row_fraction_gate
        ),
        "all_logits_probabilities_losses_and_metrics_finite": finite,
        "correct_write_and_query_codes_noncollapsed": noncollapsed,
    }
    return {
        "evaluation_rows": MECHANICS_ROWS,
        "evaluation_calls": 1,
        "code_metrics": {
            "correct": correct,
            "matched_donor": donor,
            "cyclic_layer_permuted": layer_permuted,
        },
        "noncollapse": usage,
        "checks": checks,
        "passed": all(checks.values()),
        "thresholds_fit_or_selected_on_mechanics": False,
        "architecture_hyperparameters_or_candidate_selected_on_mechanics": False,
    }


def analyze(records: Sequence[Mapping[str, Any]]) -> Mapping[str, Any]:
    code_maps, fit_audit = train_fit_only(records)
    mechanics = evaluate_mechanics_once(code_maps, records)
    finite = bool(fit_audit["loss"]["all_steps_finite"]) and bool(
        mechanics["checks"]["all_logits_probabilities_losses_and_metrics_finite"]
    )
    passed = bool(mechanics["passed"] and finite)
    return {
        "fit": fit_audit,
        "mechanics": mechanics,
        "passed": passed,
        "code_map_frozen_before_mechanics": True,
        "mechanics_evaluated_once": True,
        "causal_rows_opened": False,
        "causal_rows_used_for_any_decision": False,
        "monomial_transform_fit_or_applied": False,
        "model_or_adapter_parameters_updated": False,
        "code_map_weights_saved": False,
    }


def _consensual_operation(
    context: Any,
    *,
    phase: str,
    operation: Callable[[], Any],
) -> Any:
    result: Any = None
    local_error: BaseException | None = None
    try:
        result = operation()
    except BaseException as error:
        local_error = error
    distributed.phase_consensus(context, phase=phase, error=local_error)
    if local_error is not None:
        raise local_error
    return result


def _validated_source_state() -> tuple[
    Mapping[str, Any],
    Mapping[str, Any],
    Mapping[str, Any],
    Mapping[int, Mapping[str, Any]],
    Mapping[str, Any],
    str,
]:
    protocol, predictor_result, plat_result, split_payload, signed_rows = (
        validate_protocol()
    )
    source_execution = shadow.validate_execution_source()
    source_audit = {
        "protocol_payload_sha256": PROTOCOL_PAYLOAD_SHA256,
        "protocol_file_sha256": PROTOCOL_FILE_SHA256,
        "plat_result_sha256": PLAT_RESULT_SHA256,
        "plat_result_receipt": PLAT_RESULT_RECEIPT,
        "predictor_result_sha256": PREDICTOR_RESULT_SHA256,
        "predictor_result_receipt": PREDICTOR_RESULT_RECEIPT,
        "v5_result_sha256": V5_RESULT_SHA256,
        "v5_result_receipt": V5_RESULT_RECEIPT,
        "v5_adapter_weights_sha256": V5_ADAPTER_WEIGHTS_SHA256,
        "v5_adapter_config_sha256": V5_ADAPTER_CONFIG_SHA256,
        "v2_authorization_parent_commit": V2_AUTHORIZATION_PARENT_COMMIT,
        "v1_operational_failure_sha256": V1_OPERATIONAL_FAILURE_SHA256,
        "v1_operational_failure_receipt": V1_OPERATIONAL_FAILURE_RECEIPT,
        "v1_protocol_file_sha256": V1_PROTOCOL_FILE_SHA256,
        "v1_protocol_payload_sha256": V1_PROTOCOL_PAYLOAD_SHA256,
        "v1_runner_sha256": V1_RUNNER_SHA256,
        "v1_incident_boundary": dict(V1_INCIDENT_BOUNDARY),
        "code_dependencies": _dependency_payload(),
        "exact_source": dict(source_execution),
        "split_payload_sha256": SPLIT_PAYLOAD_SHA256,
    }
    return (
        protocol,
        predictor_result,
        plat_result,
        signed_rows,
        source_audit,
        canonical_sha256(source_audit),
    )


def _validate_base_and_dataset(base_model: Path, dataset_root: Path) -> None:
    if os.environ.get("HF_ENDPOINT") != HF_ENDPOINT:
        raise RuntimeError(f"HF_ENDPOINT must be exactly {HF_ENDPOINT}")
    if sha256_file(base_model / "config.json") != BASE_CONFIG_SHA256:
        raise ValueError("PLMSC base-model config differs")
    if sha256_file(dataset_root / endpoint.DATASET_RELATIVE_PATH) != endpoint.DATASET_SHA256:
        raise ValueError("PLMSC native development dataset differs")


def _validate_result(path: Path) -> Mapping[str, Any]:
    result = json.loads(path.read_text(encoding="utf-8"))
    unsigned = dict(result)
    receipt = unsigned.pop("receipt", {})
    if (
        result.get("schema") != SCHEMA
        or receipt.get("payload_sha256") != canonical_sha256(unsigned)
        or result.get("protocol_payload_sha256") != PROTOCOL_PAYLOAD_SHA256
        or result.get("code_bindings", {}).get("runner_sha256")
        != sha256_file(Path(__file__).resolve())
        or result.get("protected_splits_opened") != []
        or result.get("causal_firewall", {}).get("causal_rows_opened") is not False
        or result.get("stage2_authorized") is not False
        or result.get("model_or_adapter_training_authorized") is not False
        or result.get("generation_authorized") is not False
        or result.get("adapter_saved") is not False
    ):
        raise ValueError("PLMSC result receipt or authorization contract differs")
    return result


def run(*, base_model: Path, dataset_root: Path, output_dir: Path) -> Mapping[str, Any]:
    if output_dir.resolve() != DEFAULT_OUTPUT_DIR.resolve():
        raise ValueError(f"PLMSC v2 output must be exactly {DEFAULT_OUTPUT_DIR}")
    context = distributed.initialize_distributed_training(
        "cuda",
        required_world_size=WORLD_SIZE,
        timeout_seconds=DISTRIBUTED_TIMEOUT_SECONDS,
    )
    if context is None:
        raise RuntimeError("Run PLMSC with torchrun --nproc_per_node=4")
    try:
        (
            protocol,
            predictor_result,
            plat_result,
            signed_rows,
            source_audit,
            source_digest,
        ) = _consensual_operation(
            context,
            phase="plmsc-signed-source-and-protocol-validation",
            operation=_validated_source_state,
        )
        del predictor_result, plat_result
        distributed.require_consensus(
            context, source_digest, description="PLMSC signed source binding"
        )
        split_payload = protocol["precommitted_three_way_split"]
        split_payload = {
            key: split_payload[key]
            for key in (
                "selection_salt",
                "component_count",
                "component_sizes",
                "mechanics_component_indices",
                "causal_component_indices",
                "fit_sources",
                "mechanics_sources",
                "causal_sources",
                "excluded_prior_heldout_sources",
                "excluded_plat_heldout_sources",
            )
        }

        def validate_runtime() -> None:
            if context.world_size != WORLD_SIZE or not hardware.four_distinct_a100s(
                context.rank_devices
            ):
                raise RuntimeError("PLMSC requires exactly four distinct A100 GPUs")
            if context.control_backend != "gloo":
                raise RuntimeError("PLMSC control consensus requires gloo")
            if context.backend != "nccl":
                raise RuntimeError("PLMSC model/data capture requires nccl")
            _validate_base_and_dataset(base_model, dataset_root)

        _consensual_operation(
            context,
            phase="plmsc-runtime-input-and-hardware-validation",
            operation=validate_runtime,
        )

        def create_output() -> None:
            if not context.is_primary:
                return
            if output_dir.exists():
                raise ValueError(f"PLMSC output must be fresh: {output_dir}")
            output_dir.mkdir(parents=True, exist_ok=False)

        _consensual_operation(
            context, phase="plmsc-fresh-output-create", operation=create_output
        )

        def load_runtime() -> tuple[torch.nn.Module, Any, Mapping[str, Any], Mapping[str, Any]]:
            torch.manual_seed(SEED)
            torch.cuda.manual_seed_all(SEED)
            loaded_model, tokenizer, model_audit = shadow.load_exact_v5_model(
                base_model, device=context.device
            )
            observer_audit = install_anchor_read_capture(loaded_model)
            for parameter in loaded_model.parameters():
                parameter.requires_grad_(False)
            if any(parameter.requires_grad for parameter in loaded_model.parameters()):
                raise RuntimeError("PLMSC exact-v5 capture left model gradients enabled")
            return loaded_model, tokenizer, model_audit, observer_audit

        model, tokenizer, model_audit, observer_audit = _consensual_operation(
            context, phase="plmsc-exact-v5-model-load", operation=load_runtime
        )

        def capture_local_shard() -> Mapping[str, Any]:
            examples, dataset_audit = _load_local_examples(
                tokenizer,
                dataset_root,
                signed_rows,
                split_payload,
                context.process_rank,
            )
            split = {
                **{source: "fit" for source in split_payload["fit_sources"]},
                **{
                    source: "mechanics"
                    for source in split_payload["mechanics_sources"]
                },
                **{source: "causal" for source in split_payload["causal_sources"]},
            }
            rows: list[Mapping[str, Any]] = []
            for ordinal, source in enumerate(sorted(examples), start=1):
                donor = int(signed_rows[source]["donor_source_index"])
                if split[source] not in {"fit", "mechanics"} or split[donor] != split[source]:
                    raise RuntimeError("PLMSC local capture crossed its donor component")
                feature = capture_row(
                    model,
                    examples[source],
                    pad_token_id=int(tokenizer.pad_token_id),
                    device=context.device,
                )
                rows.append(
                    _signed_feature_row(
                        {
                            "schema": ROW_SCHEMA,
                            "capture_rank": context.process_rank,
                            "source_index": source,
                            "row_sha256": signed_rows[source]["row_sha256"],
                            "donor_source_index": donor,
                            "donor_row_sha256": signed_rows[source]["donor_row_sha256"],
                            "split": split[source],
                            **feature,
                        }
                    )
                )
                print(
                    f"PLMSC_CAPTURE rank={context.process_rank} row={source} "
                    f"ordinal={ordinal}/{len(examples)}",
                    flush=True,
                )
            shard: dict[str, Any] = {
                "schema": SHARD_SCHEMA,
                "rank": context.process_rank,
                "world_size": WORLD_SIZE,
                "assignment": "source_index_modulo_4",
                "rows": rows,
                "dataset_audit": dataset_audit,
                "causal_rows_opened": False,
            }
            shard["receipt"] = {
                "algorithm": "sha256",
                "payload_scope": "canonical_shard_without_receipt",
                "payload_sha256": canonical_sha256(shard),
            }
            _write_signed_json(
                output_dir / f"shard-{context.process_rank}.json", shard
            )
            return {
                "rows": len(rows),
                "receipt": shard["receipt"]["payload_sha256"],
            }

        local_shard_binding = _consensual_operation(
            context, phase="plmsc-fit-and-mechanics-feature-capture", operation=capture_local_shard
        )
        del local_shard_binding
        del model
        torch.cuda.empty_cache()

        def analyze_and_save() -> None:
            if not context.is_primary:
                return
            records, provenance = load_feature_shards(
                output_dir, split_payload, signed_rows
            )
            analysis = analyze(records)
            passed = bool(analysis["passed"])
            result: dict[str, Any] = {
                "schema": SCHEMA,
                "status": (
                    "plmsc_code_alignment_passed_causal_protocol_draft_only"
                    if passed
                    else "plmsc_code_alignment_failed_family_retired"
                ),
                "passed": passed,
                "causal_protocol_drafting_authorized": passed,
                "stage2_authorized": False,
                "model_or_adapter_training_authorized": False,
                "generation_authorized": False,
                "adapter_saved": False,
                "native_benchmark_or_sota_claim_authorized": False,
                "protocol_payload_sha256": PROTOCOL_PAYLOAD_SHA256,
                "protocol_file_sha256": PROTOCOL_FILE_SHA256,
                "protocol_objective": protocol["objective"],
                "base_model": str(base_model),
                "base_model_revision": BASE_MODEL_REVISION,
                "base_config_sha256": BASE_CONFIG_SHA256,
                "dataset_file": str(dataset_root / endpoint.DATASET_RELATIVE_PATH),
                "dataset_sha256": endpoint.DATASET_SHA256,
                "source_audit": source_audit,
                "three_way_split": {
                    **dict(split_payload),
                    "payload_sha256": SPLIT_PAYLOAD_SHA256,
                    "eligible_donor_mapping_pairs_sha256": ELIGIBLE_MAPPING_SHA256,
                    "capture_sources_sha256": CAPTURE_SOURCES_SHA256,
                    "donor_component_disjoint": True,
                    "crossing_donor_edges": 0,
                },
                "feature_provenance": provenance,
                "analysis": analysis,
                "causal_firewall": {
                    "causal_sources_committed": split_payload["causal_sources"],
                    "causal_rows_opened": False,
                    "causal_rows_decoded": 0,
                    "causal_rows_tokenized": 0,
                    "causal_rows_model_forwarded": 0,
                    "causal_features_captured": 0,
                    "causal_metrics_reported": False,
                    "causal_rows_used_for_fit_thresholds_architecture_hyperparameters_selection_stopping_or_retry": False,
                },
                "hardware": {
                    "world_size": WORLD_SIZE,
                    "rank_devices": list(context.rank_devices),
                    "model_backend": context.backend,
                    "control_backend": context.control_backend,
                    "timeout_seconds": DISTRIBUTED_TIMEOUT_SECONDS,
                },
                "execution": {
                    "rank0_only_code_map_fit": True,
                    "other_ranks_verified_source_and_result_bindings": True,
                    "hf_endpoint": os.environ.get("HF_ENDPOINT"),
                    "fresh_output": True,
                },
                "model_audit": {
                    **dict(model_audit),
                    "read_basis_observer": observer_audit,
                    "model_parameters_updated": False,
                    "model_output_changed_by_capture": False,
                    "binder_bridge_or_code_module_installed_during_capture": False,
                },
                "no_adapter_or_code_map_weights_saved": True,
                "protected_splits_opened": [],
                "code_bindings": {
                    "runner_sha256": sha256_file(Path(__file__).resolve()),
                    "dependencies": _dependency_payload(),
                },
            }
            result["receipt"] = {
                "algorithm": "sha256",
                "payload_scope": "canonical_result_without_receipt",
                "payload_sha256": canonical_sha256(result),
            }
            _write_signed_json(output_dir / "result.json", result)

        _consensual_operation(
            context, phase="plmsc-rank0-fit-mechanics-and-result-save", operation=analyze_and_save
        )
        result = _consensual_operation(
            context,
            phase="plmsc-all-rank-result-verification",
            operation=lambda: _validate_result(output_dir / "result.json"),
        )
        distributed.require_consensus(
            context,
            result["receipt"]["payload_sha256"],
            description="PLMSC result receipt",
        )
        return result
    finally:
        distributed.destroy_distributed_training(context)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-model", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args(argv)
    run(
        base_model=args.base_model.expanduser().resolve(strict=True),
        dataset_root=args.dataset_root.expanduser().resolve(strict=True),
        output_dir=args.output_dir.expanduser().resolve(),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
