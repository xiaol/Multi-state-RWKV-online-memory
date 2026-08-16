#!/usr/bin/env python3
"""Train a projected-carrier vector-gate hybrid with causal controls."""

from __future__ import annotations

from contextlib import contextmanager
import json
import os
from pathlib import Path
import sys
from typing import Any, Iterator, Mapping, Sequence

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = Path(__file__).resolve().parents[2]
for import_root in (SCRIPT_DIR, PROJECT_ROOT):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

import torch  # noqa: E402

from deltamem.core.delta import iter_delta_mem_modules  # noqa: E402
from experiments.rethinking_rwkv_ms_gemma import (  # noqa: E402
    run_natural_memory_native_projected_rwkv_hybrid_screen as hybrid_screen,
)
from experiments.rethinking_rwkv_ms_gemma import (  # noqa: E402
    run_natural_memory_native_rwkv_addressed_value_causal_train as causal_train,
)
from experiments.rethinking_rwkv_ms_gemma import (  # noqa: E402
    run_natural_memory_native_rwkv_positive_only_causal_train as positive,
)
from experiments.rethinking_rwkv_ms_gemma import (  # noqa: E402
    run_natural_memory_native_rwkv_sharp_router_screen as screen_helper,
)
from experiments.rethinking_rwkv_ms_gemma import (  # noqa: E402
    run_natural_memory_native_rwkv_top2_abstention_causal_train as shared,
)
from experiments.rethinking_rwkv_ms_gemma import (  # noqa: E402
    run_natural_memory_native_rwkv_top2_abstention_screen as top2_screen,
)


SCHEMA = "rwkv_ms_natural_memory_native_vector_gate_causal_train.v1"
STEP_SCHEMA = "rwkv_ms_natural_memory_native_vector_gate_causal_train_step.v1"
INPUT_SCHEMA = "rwkv_ms_natural_memory_native_vector_gate_causal_train_input.v1"
PROTOCOL = (
    SCRIPT_DIR / "natural_memory_native_rwkv_vector_gate_causal_train_protocol_v1.json"
)
PROTOCOL_PAYLOAD_SHA256 = (
    "4c2d9724c153f085797652b91e343324b8b12356830672871c7fe809a5bbb447"
)
SEED = 74
MIN_ACCEPTED_ROWS_PER_UPDATE = 6
MAX_TOTAL_REJECTED_ROWS = 8
SELECTED_CANDIDATE = {
    "candidate_id": "vector_gate_t16_k2_gate025_g0125",
    "hybrid_mode": "vector_gate",
    "hybrid_gain": 0.125,
    "read_temperature": 16.0,
    "read_top_k": 2,
    "fusion_gate_probability": 0.25,
    "detach_read_scores": True,
}
HELDOUT_ORDINALS = (
    444, 942, 719, 528, 92, 400, 1372, 129,
    1399, 52, 870, 399, 272, 921, 753, 506,
    962, 672, 876, 1065, 65, 810, 291, 1408,
    912, 1076, 1324, 235, 1402, 340, 1259, 830,
)
HELDOUT_PAYLOAD_SHA256 = (
    "8fb5c15f465b668b49313761e23e806bfb7ed8c965ff007ecf7249e998ce0822"
)
PRIOR_NATIVE_RESULT = (
    SCRIPT_DIR
    / "local_artifacts/"
    "natural_memory_native_rwkv_scalar_agreement_eval_v2/result.json"
)
PRIOR_NATIVE_RESULT_FILE_SHA256 = (
    "b3ccbaaa0e3c31116bdf84c6df76667c83d2e33bc06b8757f469efa18ee7795f"
)
PRIOR_NATIVE_RESULT_RECEIPT = (
    "9590428d136660e378b0b92ce79fcde5d46b7518314f9a629a04d8a9966c2e9e"
)


def load_model(
    base_model: Path,
    *,
    device: torch.device,
    candidate: Mapping[str, Any] = SELECTED_CANDIDATE,
) -> tuple[torch.nn.Module, Any, Mapping[str, Any]]:
    model, tokenizer, inherited_audit = hybrid_screen.load_model(
        base_model,
        device=device,
        delta_config=top2_screen.build_config(candidate),
    )
    screen_helper.configure_candidate(model, candidate)
    modules = tuple(iter_delta_mem_modules(model))
    configured = all(
        module.rwkv_ms_hybrid_mode == "vector_gate"
        and module.rwkv_ms_hybrid_gain == 0.125
        and module.rwkv_ms_read_temperature == 16.0
        and module.rwkv_ms_read_top_k == 2
        and module.rwkv_ms_detach_read_scores is True
        and module.memory_fusion_mode == "content_gated_add"
        for _, module in modules
    )
    audit = {
        **dict(inherited_audit),
        "all_wrappers_vector_gate_content_gated": configured,
    }
    if not configured:
        raise RuntimeError(f"Vector-gate attachment failed: {audit!r}")
    return model, tokenizer, audit


def validate_protocol() -> Mapping[str, Any]:
    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    receipt = protocol.get("receipt")
    if not isinstance(receipt, Mapping):
        raise ValueError("Vector-gate protocol receipt is missing")
    unsigned = dict(protocol)
    unsigned.pop("receipt")
    digest = shared.canonical_sha256(unsigned)
    if digest != PROTOCOL_PAYLOAD_SHA256 or receipt.get("payload_sha256") != digest:
        raise ValueError("Vector-gate protocol payload differs")
    if shared.sha256_file(PRIOR_NATIVE_RESULT) != (
        PRIOR_NATIVE_RESULT_FILE_SHA256
    ):
        raise ValueError("Prior scalar native result binding differs")
    prior = json.loads(PRIOR_NATIVE_RESULT.read_text(encoding="utf-8"))
    unsigned_prior = dict(prior)
    prior_receipt = unsigned_prior.pop("receipt", {})
    if (
        shared.canonical_sha256(unsigned_prior) != PRIOR_NATIVE_RESULT_RECEIPT
        or prior_receipt.get("payload_sha256") != PRIOR_NATIVE_RESULT_RECEIPT
        or prior.get("status") != "scalar_agreement_native_gain_not_established"
        or prior.get("passed") is not False
    ):
        raise ValueError("Prior scalar native failure does not authorize training")
    endpoint = protocol.get("heldout_causal_endpoint", {})
    training = protocol.get("training", {})
    if (
        protocol.get("architecture", {}).get("selected_candidate")
        != SELECTED_CANDIDATE
        or endpoint.get("source_ordinals") != list(HELDOUT_ORDINALS)
        or endpoint.get("source_donor_payload_sha256") != HELDOUT_PAYLOAD_SHA256
        or training.get("minimum_accepted_rows_per_update")
        != MIN_ACCEPTED_ROWS_PER_UPDATE
        or training.get("maximum_total_rejected_rows")
        != MAX_TOTAL_REJECTED_ROWS
        or training.get("optimizer_updates") != shared.UPDATES
    ):
        raise ValueError("Vector-gate training contract differs")
    if protocol.get("protected_splits_opened_by_this_protocol") != []:
        raise ValueError("Vector-gate training may not open protected data")
    return protocol


@contextmanager
def training_bindings() -> Iterator[None]:
    with positive.training_bindings():
        causal_bindings = {
            "SCHEMA": SCHEMA,
            "STEP_SCHEMA": STEP_SCHEMA,
            "PROTOCOL_PAYLOAD_SHA256": PROTOCOL_PAYLOAD_SHA256,
            "FILTER_NONFINITE_ROWS": True,
            "MIN_ACCEPTED_ROWS_PER_UPDATE": MIN_ACCEPTED_ROWS_PER_UPDATE,
            "MAX_TOTAL_REJECTED_ROWS": MAX_TOTAL_REJECTED_ROWS,
        }
        shared_bindings = {
            "SCHEMA": SCHEMA,
            "STEP_SCHEMA": STEP_SCHEMA,
            "INPUT_SCHEMA": INPUT_SCHEMA,
            "PROTOCOL": PROTOCOL,
            "PROTOCOL_PAYLOAD_SHA256": PROTOCOL_PAYLOAD_SHA256,
            "SEED": SEED,
            "SELECTED_CANDIDATE": SELECTED_CANDIDATE,
            "HELDOUT_ORDINALS": HELDOUT_ORDINALS,
            "HELDOUT_PAYLOAD_SHA256": HELDOUT_PAYLOAD_SHA256,
            "RUNNER_BINDING_PATH": Path(__file__),
            "PASS_STATUS": "vector_gate_heldout_passed_generation_authorized",
            "FAIL_STATUS": "vector_gate_heldout_failed_generation_blocked",
            "REQUIRE_RECURRENT_SUBSET_CHANGED": True,
            "TRAINING_FUNCTION": causal_train.train,
            "MODEL_LOADER": load_model,
            "validate_protocol": validate_protocol,
        }
        previous_causal = {name: getattr(causal_train, name) for name in causal_bindings}
        previous_shared = {name: getattr(shared, name) for name in shared_bindings}
        try:
            for name, value in causal_bindings.items():
                setattr(causal_train, name, value)
            for name, value in shared_bindings.items():
                setattr(shared, name, value)
            yield
        finally:
            for name, value in previous_shared.items():
                setattr(shared, name, value)
            for name, value in previous_causal.items():
                setattr(causal_train, name, value)


def validate_calibration_result() -> Mapping[str, Any]:
    return shared.validate_calibration_result()


def run(**kwargs: Any) -> Mapping[str, Any]:
    with training_bindings():
        return shared.run(**kwargs)


def main(argv: Sequence[str] | None = None) -> int:
    with training_bindings():
        return shared.main(argv)


if __name__ == "__main__":
    raise SystemExit(main())
