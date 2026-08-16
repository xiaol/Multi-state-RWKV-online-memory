#!/usr/bin/env python3
"""Train the aligned vector gate with stronger state-specific contrast."""

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

from experiments.rethinking_rwkv_ms_gemma import (  # noqa: E402
    run_natural_memory_native_rwkv_addressed_value_causal_train as causal_train,
)
from experiments.rethinking_rwkv_ms_gemma import (  # noqa: E402
    run_natural_memory_native_rwkv_aligned_vector_gate_causal_train as aligned,
)
from experiments.rethinking_rwkv_ms_gemma import (  # noqa: E402
    run_natural_memory_native_rwkv_top2_abstention_causal_train as shared,
)


SCHEMA = "rwkv_ms_natural_memory_native_aligned_vector_gate_specificity_train.v1"
STEP_SCHEMA = (
    "rwkv_ms_natural_memory_native_aligned_vector_gate_specificity_train_step.v1"
)
INPUT_SCHEMA = (
    "rwkv_ms_natural_memory_native_aligned_vector_gate_specificity_train_input.v1"
)
PROTOCOL = (
    SCRIPT_DIR
    / "natural_memory_native_rwkv_aligned_vector_gate_specificity_train_protocol_v1.json"
)
PROTOCOL_PAYLOAD_SHA256 = (
    "e72258c5339b258a07387f87cd28126165926ae05b5960aa4eafecd96c808fbe"
)
PRIOR_RESULT = (
    SCRIPT_DIR
    / "local_artifacts/natural_memory_native_rwkv_aligned_vector_gate_causal_train_v1/"
    "result.json"
)
PRIOR_RESULT_FILE_SHA256 = (
    "27aab0f513424ad0b137817e793cfc291f529767fe271af422330312b67e1670"
)
PRIOR_RESULT_RECEIPT = (
    "5f4a0f233e8b74a7d5ca17376954144e235e65db9bc2330ef11c8afc7581fff4"
)
SEED = 79
UPDATES = 16
CONTRAST_WEIGHT = 1.0
MIN_ACCEPTED_ROWS_PER_UPDATE = 6
MAX_TOTAL_REJECTED_ROWS = 8
ENDPOINT_CANDIDATE_ROWS = 412
TRAINING_PREFIX_SHA256 = (
    "121c2eea4904545b2f8c44ef2e4b16e905ca76809af9c34de5b7f848825872fd"
)
HELDOUT_ORDINALS = (
    270, 795, 600, 1097, 304, 429, 435, 1311,
    1262, 925, 287, 127, 606, 67, 356, 199,
    1313, 917, 1302, 553, 316, 744, 885, 1254,
    632, 1127, 180, 676, 197, 454, 1394, 890,
)
HELDOUT_PAYLOAD_SHA256 = (
    "d48074234ab7ce894e267c79ae92c96db0cb05620499a7dea2d930b94d846cf2"
)


def validate_prior_result() -> Mapping[str, Any]:
    if shared.sha256_file(PRIOR_RESULT) != PRIOR_RESULT_FILE_SHA256:
        raise ValueError("Aligned-vector-gate prior result file differs")
    result = json.loads(PRIOR_RESULT.read_text(encoding="utf-8"))
    receipt = result.get("receipt")
    if not isinstance(receipt, Mapping):
        raise ValueError("Aligned-vector-gate prior receipt is missing")
    unsigned = dict(result)
    unsigned.pop("receipt")
    endpoint = result.get("heldout_causal_endpoint", {})
    margins = endpoint.get("mean_ce_margins", {})
    if (
        shared.canonical_sha256(unsigned) != PRIOR_RESULT_RECEIPT
        or receipt.get("payload_sha256") != PRIOR_RESULT_RECEIPT
        or result.get("status")
        != "aligned_vector_gate_heldout_failed_generation_blocked"
        or result.get("training_passed") is not True
        or result.get("passed") is not False
        or margins.get("zero_minus_correct", 0.0) <= 0.0
        or margins.get("donor_minus_correct", 0.0) >= 0.0
        or margins.get("layer_permuted_minus_correct", 0.0) >= 0.0
        or result.get("protected_splits_opened") != []
    ):
        raise ValueError("Aligned-vector-gate failure does not authorize specificity training")
    return result


def validate_protocol() -> Mapping[str, Any]:
    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    receipt = protocol.get("receipt")
    if not isinstance(receipt, Mapping):
        raise ValueError("Specificity-training protocol receipt is missing")
    unsigned = dict(protocol)
    unsigned.pop("receipt")
    digest = shared.canonical_sha256(unsigned)
    authorization = protocol.get("authorization_basis", {})
    training = protocol.get("training", {})
    endpoint = protocol.get("heldout_causal_endpoint", {})
    if (
        digest != PROTOCOL_PAYLOAD_SHA256
        or receipt.get("payload_sha256") != digest
        or authorization.get("prior_result_file_sha256")
        != PRIOR_RESULT_FILE_SHA256
        or authorization.get("prior_result_receipt") != PRIOR_RESULT_RECEIPT
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
        raise ValueError("Specificity-training protocol differs")
    validate_prior_result()
    aligned.validate_screen_result()
    return protocol


@contextmanager
def training_bindings() -> Iterator[None]:
    with aligned.training_bindings():
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
            "SEED": SEED,
            "UPDATES": UPDATES,
            "TRAINING_PREFIX_SHA256": TRAINING_PREFIX_SHA256,
            "SELECTED_CANDIDATE": aligned.SELECTED_CANDIDATE,
            "HELDOUT_ORDINALS": HELDOUT_ORDINALS,
            "HELDOUT_PAYLOAD_SHA256": HELDOUT_PAYLOAD_SHA256,
            "RUNNER_BINDING_PATH": Path(__file__),
            "PASS_STATUS": (
                "aligned_vector_gate_specificity_heldout_passed_"
                "generation_authorized"
            ),
            "FAIL_STATUS": (
                "aligned_vector_gate_specificity_heldout_failed_"
                "generation_blocked"
            ),
            "REQUIRE_RECURRENT_SUBSET_CHANGED": True,
            "TRAINING_FUNCTION": causal_train.train,
            "MODEL_LOADER": aligned.load_model,
            "validate_protocol": validate_protocol,
            "validate_calibration_result": aligned.validate_screen_result,
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
