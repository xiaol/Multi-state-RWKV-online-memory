#!/usr/bin/env python3
"""Train the projected-addressed RWKV vector gate with causal controls."""

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
    run_natural_memory_native_rwkv_addressed_vector_gate_screen as screen,
)
from experiments.rethinking_rwkv_ms_gemma import (  # noqa: E402
    run_natural_memory_native_rwkv_aligned_vector_gate_specificity_train as specificity,
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


SCHEMA = "rwkv_ms_natural_memory_native_addressed_vector_gate_causal_train.v1"
STEP_SCHEMA = (
    "rwkv_ms_natural_memory_native_addressed_vector_gate_causal_train_step.v1"
)
INPUT_SCHEMA = (
    "rwkv_ms_natural_memory_native_addressed_vector_gate_causal_train_input.v1"
)
PROTOCOL = (
    SCRIPT_DIR
    / "natural_memory_native_rwkv_addressed_vector_gate_causal_train_protocol_v1.json"
)
PROTOCOL_PAYLOAD_SHA256 = (
    "7ecbd8f9572a0d2f6be7eceee47cc6c62a82cd0ba8b58f942844238309769f37"
)
SCREEN_RESULT = (
    SCRIPT_DIR
    / "local_artifacts/natural_memory_native_rwkv_addressed_vector_gate_screen_v1/"
    "result.json"
)
SCREEN_RESULT_FILE_SHA256 = (
    "fad0b6d4ec69de89532ad97f51630172dde6d1f84900a762c6aab650e0619859"
)
SCREEN_RESULT_RECEIPT = (
    "db95af434dca190c1934a4acb90a51fc3b5238b17b22fdbedd3505bc8e5d5102"
)
SEED = 81
UPDATES = 16
CONTRAST_WEIGHT = 1.0
MIN_ACCEPTED_ROWS_PER_UPDATE = 6
MAX_TOTAL_REJECTED_ROWS = 8
ENDPOINT_CANDIDATE_ROWS = 315
TRAINING_PREFIX_SHA256 = (
    "121c2eea4904545b2f8c44ef2e4b16e905ca76809af9c34de5b7f848825872fd"
)
HELDOUT_ORDINALS = (
    1234, 1112, 851, 1346, 1284, 965, 1183, 46,
    408, 380, 1102, 1427, 5, 791, 725, 908,
    539, 762, 348, 1423, 193, 526, 1247, 750,
    1376, 1347, 442, 837, 453, 556, 1094, 569,
)
HELDOUT_PAYLOAD_SHA256 = (
    "c26ec6f4704211d4f5f21991d17d9a7d887e161dd38d0c46cdebda773de7322f"
)
SELECTED_CANDIDATE = screen.SELECTED_CANDIDATE
PASS_STATUS = "addressed_vector_gate_heldout_passed_generation_authorized"
FAIL_STATUS = "addressed_vector_gate_heldout_failed_generation_blocked"


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
        module.rwkv_ms_hybrid_mode == "addressed_vector_gate"
        and module.rwkv_ms_hybrid_gain == 0.125
        and module.rwkv_ms_read_temperature == 16.0
        and module.rwkv_ms_read_top_k == 2
        and module.rwkv_ms_detach_read_scores is True
        and module.memory_fusion_mode == "content_gated_add"
        for _, module in modules
    )
    audit = {
        **dict(inherited_audit),
        "all_wrappers_addressed_vector_gate_content_gated": configured,
    }
    if not configured:
        raise RuntimeError(f"Addressed-vector-gate attachment failed: {audit!r}")
    return model, tokenizer, audit


def validate_screen_result() -> Mapping[str, Any]:
    if shared.sha256_file(SCREEN_RESULT) != SCREEN_RESULT_FILE_SHA256:
        raise ValueError("Addressed-vector screen result file differs")
    result = json.loads(SCREEN_RESULT.read_text(encoding="utf-8"))
    receipt = result.get("receipt")
    if not isinstance(receipt, Mapping):
        raise ValueError("Addressed-vector screen receipt is missing")
    unsigned = dict(result)
    unsigned.pop("receipt")
    if (
        shared.canonical_sha256(unsigned) != SCREEN_RESULT_RECEIPT
        or receipt.get("payload_sha256") != SCREEN_RESULT_RECEIPT
        or result.get("status") != screen.PASS_STATUS
        or result.get("passed") is not True
        or result.get("training_authorized") is not True
        or result.get("selected_candidate") != SELECTED_CANDIDATE
        or result.get("protected_splits_opened") != []
    ):
        raise ValueError("Addressed-vector screen did not authorize training")
    return result


def validate_protocol() -> Mapping[str, Any]:
    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    receipt = protocol.get("receipt")
    if not isinstance(receipt, Mapping):
        raise ValueError("Addressed-vector training protocol receipt is missing")
    unsigned = dict(protocol)
    unsigned.pop("receipt")
    digest = shared.canonical_sha256(unsigned)
    authorization = protocol.get("authorization_basis", {})
    training = protocol.get("training", {})
    endpoint = protocol.get("heldout_causal_endpoint", {})
    if (
        digest != PROTOCOL_PAYLOAD_SHA256
        or receipt.get("payload_sha256") != digest
        or authorization.get("screen_protocol_payload_sha256")
        != screen.PROTOCOL_PAYLOAD_SHA256
        or authorization.get("screen_result_file_sha256")
        != SCREEN_RESULT_FILE_SHA256
        or authorization.get("screen_result_receipt") != SCREEN_RESULT_RECEIPT
        or authorization.get("selected_candidate") != SELECTED_CANDIDATE
        or training.get("optimizer_updates") != UPDATES
        or training.get("contrast_weight_per_active_control") != CONTRAST_WEIGHT
        or training.get("minimum_accepted_rows_per_update")
        != MIN_ACCEPTED_ROWS_PER_UPDATE
        or training.get("maximum_total_rejected_rows")
        != MAX_TOTAL_REJECTED_ROWS
        or endpoint.get("candidate_rows_after_exclusions")
        != ENDPOINT_CANDIDATE_ROWS
        or endpoint.get("source_ordinals") != list(HELDOUT_ORDINALS)
        or endpoint.get("source_donor_payload_sha256")
        != HELDOUT_PAYLOAD_SHA256
        or protocol.get("protected_splits_opened_by_this_protocol") != []
    ):
        raise ValueError("Addressed-vector training protocol differs")
    validate_screen_result()
    return protocol


@contextmanager
def training_bindings() -> Iterator[None]:
    with specificity.training_bindings():
        causal_bindings = {
            "SCHEMA": SCHEMA,
            "STEP_SCHEMA": STEP_SCHEMA,
            "PROTOCOL_PAYLOAD_SHA256": PROTOCOL_PAYLOAD_SHA256,
            "TRAIN_UPDATES": UPDATES,
            "CONTRAST_WEIGHT": CONTRAST_WEIGHT,
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
            "CALIBRATION_RESULT": SCREEN_RESULT,
            "CALIBRATION_RESULT_FILE_SHA256": SCREEN_RESULT_FILE_SHA256,
            "CALIBRATION_RESULT_RECEIPT": SCREEN_RESULT_RECEIPT,
            "SEED": SEED,
            "UPDATES": UPDATES,
            "TRAINING_PREFIX_SHA256": TRAINING_PREFIX_SHA256,
            "SELECTED_CANDIDATE": SELECTED_CANDIDATE,
            "HELDOUT_ORDINALS": HELDOUT_ORDINALS,
            "HELDOUT_PAYLOAD_SHA256": HELDOUT_PAYLOAD_SHA256,
            "RUNNER_BINDING_PATH": Path(__file__),
            "PASS_STATUS": PASS_STATUS,
            "FAIL_STATUS": FAIL_STATUS,
            "REQUIRE_RECURRENT_SUBSET_CHANGED": True,
            "TRAINING_FUNCTION": causal_train.train,
            "MODEL_LOADER": load_model,
            "validate_protocol": validate_protocol,
            "validate_calibration_result": validate_screen_result,
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


def run(**kwargs: Any) -> Mapping[str, Any]:
    with training_bindings():
        return shared.run(**kwargs)


def main(argv: Sequence[str] | None = None) -> int:
    with training_bindings():
        return shared.main(argv)


if __name__ == "__main__":
    raise SystemExit(main())
