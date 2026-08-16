#!/usr/bin/env python3
"""Train an alignment-gated RWKV residual with causal controls."""

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
    run_natural_memory_native_rwkv_alignment_residual_screen as screen,
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


SCHEMA = "rwkv_ms_natural_memory_native_alignment_residual_causal_train.v1"
STEP_SCHEMA = "rwkv_ms_natural_memory_native_alignment_residual_causal_train_step.v1"
INPUT_SCHEMA = "rwkv_ms_natural_memory_native_alignment_residual_causal_train_input.v1"
PROTOCOL = (
    SCRIPT_DIR
    / "natural_memory_native_rwkv_alignment_residual_causal_train_protocol_v1.json"
)
PROTOCOL_PAYLOAD_SHA256 = (
    "023a970f0c3d65c48e6bf3563bd6c54a7dc9e86f618ca8d891208b1fbf74f9b4"
)
SEED = 76
MIN_ACCEPTED_ROWS_PER_UPDATE = 6
MAX_TOTAL_REJECTED_ROWS = 8
SELECTED_CANDIDATE = screen.SELECTED_CANDIDATE
HELDOUT_ORDINALS = (
    764, 967, 816, 423, 604, 120, 880, 491,
    571, 613, 924, 1027, 480, 179, 688, 614,
    474, 684, 1113, 131, 10, 1345, 867, 1218,
    1244, 800, 489, 683, 1130, 958, 54, 139,
)
HELDOUT_PAYLOAD_SHA256 = (
    "9902c8dad34d135e0ca2bc2e191971b652500997f64c1dc56943f420c6463e1d"
)
SCREEN_RESULT = (
    SCRIPT_DIR
    / "local_artifacts/natural_memory_native_rwkv_alignment_residual_screen_v1/"
    "result.json"
)
SCREEN_RESULT_FILE_SHA256 = (
    "200a04981e13087678c2c304659b44f31670b5721225536df0a8920e39f04f3a"
)
SCREEN_RESULT_RECEIPT = (
    "43d07e6afa8ced47992932993414c695335abb6173275dc9fe551de29b12036a"
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
        module.rwkv_ms_hybrid_mode == "alignment_residual"
        and module.rwkv_ms_hybrid_gain == 0.125
        and module.rwkv_ms_read_temperature == 16.0
        and module.rwkv_ms_read_top_k == 2
        and module.rwkv_ms_detach_read_scores is True
        and module.memory_fusion_mode == "content_gated_add"
        for _, module in modules
    )
    audit = {
        **dict(inherited_audit),
        "all_wrappers_alignment_residual_content_gated": configured,
    }
    if not configured:
        raise RuntimeError(f"Alignment-residual attachment failed: {audit!r}")
    return model, tokenizer, audit


def validate_protocol() -> Mapping[str, Any]:
    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    receipt = protocol.get("receipt")
    if not isinstance(receipt, Mapping):
        raise ValueError("Alignment-residual training protocol receipt is missing")
    unsigned = dict(protocol)
    unsigned.pop("receipt")
    digest = shared.canonical_sha256(unsigned)
    endpoint = protocol.get("heldout_causal_endpoint", {})
    training = protocol.get("training", {})
    authorization = protocol.get("authorization_basis", {})
    if (
        digest != PROTOCOL_PAYLOAD_SHA256
        or receipt.get("payload_sha256") != digest
        or authorization.get("selected_candidate") != SELECTED_CANDIDATE
        or authorization.get("screen_result_file_sha256")
        != SCREEN_RESULT_FILE_SHA256
        or authorization.get("screen_result_receipt") != SCREEN_RESULT_RECEIPT
        or endpoint.get("source_ordinals") != list(HELDOUT_ORDINALS)
        or endpoint.get("source_donor_payload_sha256")
        != HELDOUT_PAYLOAD_SHA256
        or training.get("minimum_accepted_rows_per_update")
        != MIN_ACCEPTED_ROWS_PER_UPDATE
        or training.get("maximum_total_rejected_rows")
        != MAX_TOTAL_REJECTED_ROWS
        or training.get("optimizer_updates") != shared.UPDATES
        or protocol.get("protected_splits_opened_by_this_protocol") != []
    ):
        raise ValueError("Alignment-residual training protocol differs")
    return protocol


def validate_screen_result() -> Mapping[str, Any]:
    if shared.sha256_file(SCREEN_RESULT) != SCREEN_RESULT_FILE_SHA256:
        raise ValueError("Alignment-residual screen result file differs")
    result = json.loads(SCREEN_RESULT.read_text(encoding="utf-8"))
    receipt = result.get("receipt")
    if not isinstance(receipt, Mapping):
        raise ValueError("Alignment-residual screen result receipt is missing")
    unsigned = dict(result)
    unsigned.pop("receipt")
    if (
        shared.canonical_sha256(unsigned) != SCREEN_RESULT_RECEIPT
        or receipt.get("payload_sha256") != SCREEN_RESULT_RECEIPT
        or result.get("status")
        != "alignment_residual_screen_passed_training_authorized"
        or result.get("passed") is not True
        or result.get("training_authorized") is not True
        or result.get("selected_candidate") != SELECTED_CANDIDATE
        or result.get("protected_splits_opened") != []
    ):
        raise ValueError("Alignment-residual screen did not authorize training")
    return result


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
            "CALIBRATION_RESULT": SCREEN_RESULT,
            "CALIBRATION_RESULT_FILE_SHA256": SCREEN_RESULT_FILE_SHA256,
            "CALIBRATION_RESULT_RECEIPT": SCREEN_RESULT_RECEIPT,
            "RUNNER_BINDING_PATH": Path(__file__),
            "PASS_STATUS": (
                "alignment_residual_heldout_passed_generation_authorized"
            ),
            "FAIL_STATUS": (
                "alignment_residual_heldout_failed_generation_blocked"
            ),
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


def validate_calibration_result() -> Mapping[str, Any]:
    return validate_screen_result()


def run(**kwargs: Any) -> Mapping[str, Any]:
    with training_bindings():
        return shared.run(**kwargs)


def main(argv: Sequence[str] | None = None) -> int:
    with training_bindings():
        return shared.main(argv)


if __name__ == "__main__":
    raise SystemExit(main())
