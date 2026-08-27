#!/usr/bin/env python3
"""Train a DeepEmbed-shaped multiplicative RWKV adapter in Gemma PLE."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import replace
import json
import os
from pathlib import Path
import sys
from typing import Any, Iterator, Mapping, Sequence

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

import torch

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[1]
for import_root in (SCRIPT_DIR, PROJECT_ROOT):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

from experiments.rethinking_rwkv_ms_gemma import (  # noqa: E402
    run_natural_memory_native_rwkv_ple_causal_train as additive,
)


shared = additive.shared
stable = additive.stable
binder = additive.binder
value_identity = additive.value_identity
causal_train = additive.causal_train
evolution = additive.evolution
deepembed = additive.deepembed
ORIGINAL_ADDITIVE_LOAD_MODEL = additive.load_model

PROTOCOL = SCRIPT_DIR / "natural_memory_native_rwkv_ple_scale_causal_train_protocol_v1.json"
PROTOCOL_PAYLOAD_SHA256 = "bd4b0e68fee74147638d203e45ec2b708f6740ba8139933680c56e412120aaf4"
SCHEMA = "rwkv_ms_natural_memory_native_rwkv_ple_scale_causal_train.v1"
STEP_SCHEMA = "rwkv_ms_natural_memory_native_rwkv_ple_scale_causal_train_step.v1"
INPUT_SCHEMA = "rwkv_ms_natural_memory_native_rwkv_ple_scale_causal_train_input.v1"
SEED = 20260901
UPDATES = 8
IDENTITY_MARGIN = 0.2
IDENTITY_WEIGHT = 1.0
HELDOUT_ORDINALS = additive.HELDOUT_ORDINALS
HELDOUT_PAYLOAD_SHA256 = additive.HELDOUT_PAYLOAD_SHA256
PASS_STATUS = "rwkv_ple_scale_heldout_passed_generation_authorized"
FAIL_STATUS = "rwkv_ple_scale_heldout_failed_generation_blocked"
PLE_RANK = additive.PLE_RANK
PLE_GAIN = additive.PLE_GAIN
PLE_FUSION = "multiplicative"
DISTRIBUTED_TIMEOUT_SECONDS = additive.DISTRIBUTED_TIMEOUT_SECONDS
EXPECTED_PLE_TENSORS = additive.EXPECTED_PLE_TENSORS
EXPECTED_BINDER_TENSORS = additive.EXPECTED_BINDER_TENSORS
EXPECTED_TRAINABLE_TENSORS = additive.EXPECTED_TRAINABLE_TENSORS
WARMSTART_ADAPTER = additive.WARMSTART_ADAPTER
WARMSTART_RESULT = additive.WARMSTART_RESULT
WARMSTART_RESULT_SHA256 = additive.WARMSTART_RESULT_SHA256
WARMSTART_RESULT_RECEIPT = additive.WARMSTART_RESULT_RECEIPT
WARMSTART_CONFIG_SHA256 = additive.WARMSTART_CONFIG_SHA256


def _canonical(value: Any) -> str:
    return shared.canonical_sha256(value)


def _ple_config() -> Any:
    return replace(
        deepembed.screen.build_config(deepembed.SELECTED_CANDIDATE),
        delta_heads=(),
        rwkv_ms_hybrid_mode="address_keyed_moe_ple",
        rwkv_ms_outer_ffn_gain=0.0,
        rwkv_ms_outer_ffn_layers=(),
        rwkv_ms_ple_rank=PLE_RANK,
        rwkv_ms_ple_gain=PLE_GAIN,
        rwkv_ms_ple_fusion=PLE_FUSION,
        rwkv_ms_write_address_value_adapter=False,
    )


def load_model(
    base_model: Path,
    *,
    device: torch.device,
    candidate: Mapping[str, Any] | None = None,
) -> tuple[torch.nn.Module, Any, Mapping[str, Any]]:
    previous_config = additive._ple_config
    additive._ple_config = _ple_config
    try:
        model, tokenizer, audit = ORIGINAL_ADDITIVE_LOAD_MODEL(
            base_model,
            device=device,
            candidate=candidate,
        )
    finally:
        additive._ple_config = previous_config
    for _, module in additive.iter_delta_mem_modules(model):
        if module.rwkv_ms_ple_fusion != PLE_FUSION:
            raise RuntimeError("Multiplicative PLE fusion was not attached")
    return model, tokenizer, {
        **dict(audit),
        "rwkv_ple_fusion": PLE_FUSION,
        "deepembed_analogy": "multiplicative_pre_per_layer_projection_scale",
    }


def validate_protocol() -> Mapping[str, Any]:
    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    unsigned = dict(protocol)
    receipt = unsigned.pop("receipt", None)
    architecture = protocol.get("architecture", {})
    training = protocol.get("training", {})
    endpoint = protocol.get("heldout_causal_endpoint", {})
    digest = _canonical(unsigned)
    if (
        digest != PROTOCOL_PAYLOAD_SHA256
        or not isinstance(receipt, Mapping)
        or receipt.get("payload_sha256") != digest
        or protocol.get("schema") != SCHEMA
        or architecture.get("hybrid_mode") != "address_keyed_moe_ple"
        or architecture.get("ple_fusion") != PLE_FUSION
        or architecture.get("native_target") != "per_layer_projection_input"
        or architecture.get("deepembed_hooks") != 0
        or architecture.get("ple_hooks") != 42
        or architecture.get("zero_effect_initialization") is not True
        or architecture.get("projected_carrier_fixed") is not True
        or training.get("optimizer_updates") != UPDATES
        or training.get("distributed_timeout_seconds") != DISTRIBUTED_TIMEOUT_SECONDS
        or training.get("identity_margin") != IDENTITY_MARGIN
        or training.get("joint_trainable_parameter_tensors") != EXPECTED_TRAINABLE_TENSORS
        or endpoint.get("source_ordinals") != list(HELDOUT_ORDINALS)
        or endpoint.get("source_donor_payload_sha256") != HELDOUT_PAYLOAD_SHA256
        or protocol.get("protected_splits_opened_by_this_protocol") != []
    ):
        raise ValueError("RWKV multiplicative PLE protocol differs")
    additive._validate_warmstart()
    return protocol


@contextmanager
def training_bindings() -> Iterator[None]:
    previous = {
        "load_model": additive.load_model,
        "_ple_config": additive._ple_config,
        "validate_protocol": additive.validate_protocol,
        "PROTOCOL": additive.PROTOCOL,
        "PROTOCOL_PAYLOAD_SHA256": additive.PROTOCOL_PAYLOAD_SHA256,
        "SCHEMA": additive.SCHEMA,
        "STEP_SCHEMA": additive.STEP_SCHEMA,
        "INPUT_SCHEMA": additive.INPUT_SCHEMA,
        "SEED": additive.SEED,
        "UPDATES": additive.UPDATES,
        "IDENTITY_MARGIN": additive.IDENTITY_MARGIN,
        "IDENTITY_WEIGHT": additive.IDENTITY_WEIGHT,
        "HELDOUT_ORDINALS": additive.HELDOUT_ORDINALS,
        "HELDOUT_PAYLOAD_SHA256": additive.HELDOUT_PAYLOAD_SHA256,
        "PASS_STATUS": additive.PASS_STATUS,
        "FAIL_STATUS": additive.FAIL_STATUS,
    }
    additive.load_model = load_model
    additive._ple_config = _ple_config
    additive.validate_protocol = validate_protocol
    for name, value in {
        "PROTOCOL": PROTOCOL,
        "PROTOCOL_PAYLOAD_SHA256": PROTOCOL_PAYLOAD_SHA256,
        "SCHEMA": SCHEMA,
        "STEP_SCHEMA": STEP_SCHEMA,
        "INPUT_SCHEMA": INPUT_SCHEMA,
        "SEED": SEED,
        "UPDATES": UPDATES,
        "IDENTITY_MARGIN": IDENTITY_MARGIN,
        "IDENTITY_WEIGHT": IDENTITY_WEIGHT,
        "HELDOUT_ORDINALS": HELDOUT_ORDINALS,
        "HELDOUT_PAYLOAD_SHA256": HELDOUT_PAYLOAD_SHA256,
        "PASS_STATUS": PASS_STATUS,
        "FAIL_STATUS": FAIL_STATUS,
    }.items():
        setattr(additive, name, value)
    try:
        with additive.training_bindings():
            yield
    finally:
        for name, value in previous.items():
            setattr(additive, name, value)


def run(**kwargs: Any) -> Mapping[str, Any]:
    with training_bindings():
        return shared.run(**kwargs)


def main(argv: Sequence[str] | None = None) -> int:
    with training_bindings():
        return shared.main(argv)


if __name__ == "__main__":
    raise SystemExit(main())
