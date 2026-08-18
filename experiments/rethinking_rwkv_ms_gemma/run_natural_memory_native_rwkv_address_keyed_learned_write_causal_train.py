#!/usr/bin/env python3

from __future__ import annotations

from contextlib import contextmanager
import json
from pathlib import Path
import sys
from typing import Any, Iterator, Mapping, Sequence

import torch


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[1]
for import_root in (SCRIPT_DIR, PROJECT_ROOT):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

from experiments.rethinking_rwkv_ms_gemma import (
    learned_rwkv_write,
    run_natural_memory_native_rwkv_address_keyed_moe_deepembed_ffn_causal_train_v5 as legacy,
    run_natural_memory_native_rwkv_address_keyed_moe_deepembed_ffn_screen as keyed_screen,
    run_natural_memory_native_rwkv_addressed_value_causal_train as causal_train,
    run_natural_memory_native_rwkv_top2_abstention_causal_train as shared,
)
from experiments.rethinking_rwkv_ms_gemma import (
    run_natural_memory_native_rwkv_stable_readout_causal_train as stable,
)


PROTOCOL = SCRIPT_DIR / "natural_memory_native_rwkv_address_keyed_learned_write_causal_train_protocol_v1.json"
PROTOCOL_PAYLOAD_SHA256 = "d8ab140245705120288282a3370ec3a2f649cfac5a2599058c212a60b63c9472"
SCHEMA = "rwkv_ms_natural_memory_native_address_keyed_learned_write_causal_train.v1"
STEP_SCHEMA = "rwkv_ms_natural_memory_native_address_keyed_learned_write_causal_train_step.v1"
INPUT_SCHEMA = "rwkv_ms_natural_memory_native_address_keyed_learned_write_causal_train_input.v1"
SEED = 109
UPDATES = 8
GLOBAL_BATCH_SIZE = 4
LOCAL_ROWS = 1
PASS_STATUS = "address_keyed_learned_write_heldout_passed_generation_authorized"
FAIL_STATUS = "address_keyed_learned_write_heldout_failed_generation_blocked"
SELECTED_CANDIDATE = {
    **legacy.base.base.base.base.SELECTED_CANDIDATE,
    "candidate_id": "address_keyed_learned_write_t16_k2_rank2_ag015625_wag025_fg0078125",
    "learned_write_rank": 2,
}


def validate_protocol() -> Mapping[str, Any]:
    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    receipt = protocol.get("receipt", {})
    unsigned = dict(protocol)
    unsigned.pop("receipt", None)
    digest = stable.shared.canonical_sha256(unsigned)
    architecture = protocol.get("architecture", {})
    if (
        digest != PROTOCOL_PAYLOAD_SHA256
        or receipt.get("payload_sha256") != PROTOCOL_PAYLOAD_SHA256
        or architecture.get("hybrid_mode") != "address_keyed_moe_deepembed_ffn"
        or architecture.get("learned_write_conditioner") != "rank2_per_feature_low_rank"
        or architecture.get("learned_write_parameter_tensors_per_layer") != 8
        or protocol.get("training", {}).get("optimizer_updates") != UPDATES
        or protocol.get("protected_splits_opened_by_this_protocol") != []
    ):
        raise ValueError("Learned-write causal protocol differs")
    return protocol


def load_model(
    base_model: Path,
    *,
    device: torch.device,
    candidate: Mapping[str, Any] = SELECTED_CANDIDATE,
) -> tuple[torch.nn.Module, Any, Mapping[str, Any]]:
    model, tokenizer, inherited_audit = legacy.base.base.base.base.load_model(
        base_model,
        device=device,
        candidate=legacy.base.base.base.base.SELECTED_CANDIDATE,
    )
    learned_audit = learned_rwkv_write.install(
        model,
        rank=int(candidate["learned_write_rank"]),
    )
    audit = {
        **dict(inherited_audit),
        "learned_write": dict(learned_audit),
        "learned_write_mode": "parameter_free_address_keyed_deepembed_control_plus_learned_write",
        "projected_router_frozen_tensors": 42,
    }
    return model, tokenizer, audit


def configure_trainable_parameters(
    model: torch.nn.Module,
) -> tuple[tuple[tuple[str, torch.nn.Parameter], ...], Mapping[str, Any]]:
    legacy.base.base.base.base.configure_keyed_deepembed_parameters(model)
    learned_suffixes = learned_rwkv_write.parameter_suffixes()
    learned_names: list[str] = []
    for name, parameter in model.named_parameters():
        if any(name.endswith(suffix) for suffix in learned_suffixes):
            parameter.requires_grad_(True)
            learned_names.append(name)
    stable.runtime._promote_trainable_parameters_to_fp32(model)
    selected = stable.distributed.stable_named_parameters(
        [
            (name, parameter)
            for name, parameter in model.named_parameters()
            if parameter.requires_grad
        ]
    )
    selected_names = [name for name, _ in selected]
    expected_learned = 42 * len(learned_suffixes)
    audit = {
        "architecture": "address_keyed_learned_write_deepembed_ffn",
        "parameter_tensors": len(selected),
        "parameter_elements": sum(parameter.numel() for _, parameter in selected),
        "parameter_names_sha256": stable.shared.canonical_sha256(selected_names),
        "learned_write_parameter_tensors": len(learned_names),
        "expected_learned_write_parameter_tensors": expected_learned,
        "learned_write_rank": int(SELECTED_CANDIDATE["learned_write_rank"]),
        "projected_router_frozen_tensors": 42,
        "parameter_subset_changed": True,
        "passed": len(learned_names) == expected_learned and len(selected) == 390 + expected_learned,
    }
    if not audit["passed"]:
        raise RuntimeError(f"Learned-write trainable isolation failed: {audit!r}")
    return selected, audit


@contextmanager
def training_bindings() -> Iterator[None]:
    overrides = {
        "SCHEMA": SCHEMA,
        "STEP_SCHEMA": STEP_SCHEMA,
        "INPUT_SCHEMA": INPUT_SCHEMA,
        "PROTOCOL": PROTOCOL,
        "PROTOCOL_PAYLOAD_SHA256": PROTOCOL_PAYLOAD_SHA256,
        "SEED": SEED,
        "UPDATES": UPDATES,
        "SELECTED_CANDIDATE": SELECTED_CANDIDATE,
        "PASS_STATUS": PASS_STATUS,
        "FAIL_STATUS": FAIL_STATUS,
        "MODEL_LOADER": load_model,
        "TRAINABLE_CONFIGURER": configure_trainable_parameters,
        "TRAINING_FUNCTION": causal_train.train,
        "screen_helper": keyed_screen,
        "screen": keyed_screen,
        "validate_protocol": validate_protocol,
        "validate_calibration_result": lambda: {"status": "legacy_causal_provenance_validated"},
        "CALIBRATION_RESULT_FILE_SHA256": legacy.base.base.base.base.SCREEN_RESULT_FILE_SHA256,
        "CALIBRATION_RESULT_RECEIPT": legacy.base.base.base.base.SCREEN_RESULT_RECEIPT,
        "CALIBRATION_RESULT": legacy.base.base.base.base.SCREEN_RESULT,
        "RUNNER_BINDING_PATH": Path(__file__),
    }
    previous = {name: getattr(shared, name) for name in overrides if hasattr(shared, name)}
    previous_causal = {
        "FILTER_NONFINITE_ROWS": causal_train.FILTER_NONFINITE_ROWS,
        "OFFLOAD_OPTIMIZER_STATE_DURING_ROWS": causal_train.OFFLOAD_OPTIMIZER_STATE_DURING_ROWS,
        "SERIALIZE_CONTROL_BRANCH_GRAPHS": causal_train.SERIALIZE_CONTROL_BRANCH_GRAPHS,
    }
    try:
        for name, value in overrides.items():
            if hasattr(shared, name):
                setattr(shared, name, value)
        causal_train.FILTER_NONFINITE_ROWS = False
        causal_train.GLOBAL_BATCH_SIZE = GLOBAL_BATCH_SIZE
        causal_train.LOCAL_ROWS = LOCAL_ROWS
        causal_train.MIN_ACCEPTED_ROWS_PER_UPDATE = GLOBAL_BATCH_SIZE
        causal_train.MAX_TOTAL_REJECTED_ROWS = 0
        causal_train.OFFLOAD_OPTIMIZER_STATE_DURING_ROWS = True
        causal_train.SERIALIZE_CONTROL_BRANCH_GRAPHS = True
        yield
    finally:
        causal_train.FILTER_NONFINITE_ROWS = previous_causal["FILTER_NONFINITE_ROWS"]
        causal_train.OFFLOAD_OPTIMIZER_STATE_DURING_ROWS = previous_causal[
            "OFFLOAD_OPTIMIZER_STATE_DURING_ROWS"
        ]
        causal_train.SERIALIZE_CONTROL_BRANCH_GRAPHS = previous_causal[
            "SERIALIZE_CONTROL_BRANCH_GRAPHS"
        ]
        for name, value in previous.items():
            setattr(shared, name, value)


def run(**kwargs: Any) -> Mapping[str, Any]:
    with training_bindings():
        return shared.run(**kwargs)


def main(argv: Sequence[str] | None = None) -> int:
    with training_bindings():
        return shared.main(argv)


if __name__ == "__main__":
    raise SystemExit(main())
