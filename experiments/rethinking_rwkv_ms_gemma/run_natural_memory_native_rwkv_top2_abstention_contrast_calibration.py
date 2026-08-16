#!/usr/bin/env python3
"""Calibrate top-2 RWKV abstention under causal contrast supervision."""

from __future__ import annotations

import argparse
import gc
import json
import os
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

import torch
from torch.distributed.elastic.multiprocessing.errors import record
from transformers import set_seed


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = Path(__file__).resolve().parents[2]
for import_root in (SCRIPT_DIR, PROJECT_ROOT):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

from deltamem.core.delta import iter_delta_mem_modules, save_delta_mem_adapter  # noqa: E402
from experiments.rethinking_rwkv_ms_gemma import (  # noqa: E402
    natural_memory_distributed as distributed,
)
from experiments.rethinking_rwkv_ms_gemma import (  # noqa: E402
    run_natural_memory_native_recurrent_rwkv_preflight as preflight,
)
from experiments.rethinking_rwkv_ms_gemma import (  # noqa: E402
    run_natural_memory_native_rwkv_addressed_value_causal_train as causal_train,
)
from experiments.rethinking_rwkv_ms_gemma import (  # noqa: E402
    run_natural_memory_native_rwkv_addressed_value_screen as addressed_screen,
)
from experiments.rethinking_rwkv_ms_gemma import (  # noqa: E402
    run_natural_memory_native_rwkv_recurrent_value_causal_train as recurrent_train,
)
from experiments.rethinking_rwkv_ms_gemma import (  # noqa: E402
    run_natural_memory_native_rwkv_top2_abstention_screen as screen,
)
from experiments.rethinking_rwkv_ms_gemma import (  # noqa: E402
    run_natural_memory_native_rwkv_sharp_router_screen as screen_helper,
)
from experiments.rethinking_rwkv_ms_gemma import (  # noqa: E402
    run_natural_memory_native_scene_contrast_dropout as contrast,
)


SCHEMA = "rwkv_ms_natural_memory_native_top2_abstention_contrast_calibration.v1"
STEP_SCHEMA = (
    "rwkv_ms_natural_memory_native_top2_abstention_contrast_calibration_step.v1"
)
INPUT_SCHEMA = (
    "rwkv_ms_natural_memory_native_top2_abstention_contrast_calibration_input.v1"
)
PROTOCOL = (
    SCRIPT_DIR
    / "natural_memory_native_rwkv_top2_abstention_contrast_calibration_protocol_v1.json"
)
PROTOCOL_PAYLOAD_SHA256 = (
    "0b328adb331346d8b008045497e593dab6ee554df9484dcd5297e44e7a6ce032"
)
SCREEN_RESULT = (
    SCRIPT_DIR
    / "local_artifacts/natural_memory_native_rwkv_top2_abstention_screen_v1/"
    "result.json"
)
SCREEN_RESULT_FILE_SHA256 = (
    "bd6ead9b8205baba3e5bd0bdd31c4b2a9928aeb23f7e49a0cc0db6aa51caf672"
)
SCREEN_RESULT_RECEIPT = (
    "e5872d606a7e981af79b190a7e2f403d8568d3eff990651eea59ce6e02d61ba7"
)
SELECTED_CANDIDATE = dict(screen.CANDIDATES[0])
HF_MIRROR_ENDPOINT = "https://hf-mirror.com"
BASE_MODEL = screen.shared.BASE_MODEL
DATASET_ROOT = screen.shared.DATASET_ROOT
WORLD_SIZE = 4
SEED = 67
UPDATES = 1


def canonical_sha256(value: Any) -> str:
    return distributed.canonical_sha256(value)


def sha256_file(path: Path) -> str:
    return preflight.sha256_file(path)


def write_json(path: Path, value: Mapping[str, Any]) -> None:
    if path.exists():
        raise ValueError(f"Top-2 contrast calibration output must be fresh: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=True, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def validate_protocol() -> Mapping[str, Any]:
    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    receipt = protocol.get("receipt")
    if not isinstance(receipt, Mapping):
        raise ValueError("Top-2 contrast calibration protocol receipt is missing")
    unsigned = dict(protocol)
    unsigned.pop("receipt")
    digest = canonical_sha256(unsigned)
    if digest != PROTOCOL_PAYLOAD_SHA256 or receipt.get("payload_sha256") != digest:
        raise ValueError("Top-2 contrast calibration protocol payload differs")
    authorization = protocol.get("authorization_basis", {})
    required = {
        "screen_protocol_payload_sha256": screen.PROTOCOL_PAYLOAD_SHA256,
        "screen_result_file": (
            "local_artifacts/"
            "natural_memory_native_rwkv_top2_abstention_screen_v1/result.json"
        ),
        "screen_result_file_sha256": SCREEN_RESULT_FILE_SHA256,
        "screen_result_receipt": SCREEN_RESULT_RECEIPT,
        "screen_status": "screen_failed_sharp_router_training_blocked",
        "observed_outcome": (
            "Every candidate preserved causal materiality, state invariants, and "
            "finite logits, but the best differentiable top-2 route reached only "
            "0.5024469495 weakest-layer mean peak probability against the locked "
            "0.55 gate."
        ),
        "protocol_response": (
            "Do not lower the concentration gate; calibrate the first "
            "preregistered top-2 abstention candidate under direct causal "
            "contrast supervision."
        ),
        "selected_candidate": SELECTED_CANDIDATE,
    }
    if authorization != required:
        raise ValueError("Top-2 contrast calibration authorization differs")
    if protocol.get("protected_splits_opened_by_this_protocol") != []:
        raise ValueError("Top-2 contrast calibration may not open protected data")
    return protocol


def validate_screen_result() -> Mapping[str, Any]:
    if sha256_file(SCREEN_RESULT) != SCREEN_RESULT_FILE_SHA256:
        raise ValueError("Top-2 abstention screen result file hash differs")
    result = json.loads(SCREEN_RESULT.read_text(encoding="utf-8"))
    receipt = result.get("receipt")
    if not isinstance(receipt, Mapping):
        raise ValueError("Top-2 abstention screen result receipt is missing")
    unsigned = dict(result)
    unsigned.pop("receipt")
    digest = canonical_sha256(unsigned)
    first = result.get("candidate_results", [None])[0]
    first_identity = (
        None
        if not isinstance(first, Mapping)
        else {name: first.get(name) for name in SELECTED_CANDIDATE}
    )
    if (
        digest != SCREEN_RESULT_RECEIPT
        or receipt.get("payload_sha256") != digest
        or result.get("status") != "screen_failed_sharp_router_training_blocked"
        or result.get("passed") is not False
        or result.get("selected_candidate") is not None
        or first_identity != SELECTED_CANDIDATE
        or first.get("checks", {}).get(
            "minimum_layer_mean_router_peak_on_all_ranks"
        )
        is not False
        or any(
            value is not True
            for name, value in first.get("checks", {}).items()
            if name != "minimum_layer_mean_router_peak_on_all_ranks"
        )
        or result.get("protected_splits_opened") != []
    ):
        raise ValueError("Top-2 abstention failure does not authorize calibration")
    return result


def run(
    *,
    context: distributed.DistributedTrainingContext,
    output_dir: Path,
    base_model: Path = BASE_MODEL,
    dataset_root: Path = DATASET_ROOT,
) -> Mapping[str, Any]:
    if context.world_size != WORLD_SIZE:
        raise ValueError("Top-2 contrast calibration requires exactly four ranks")
    if os.environ.get("HF_ENDPOINT") != HF_MIRROR_ENDPOINT:
        raise ValueError(f"HF_ENDPOINT must be exactly {HF_MIRROR_ENDPOINT}")
    protocol = validate_protocol()
    screen_result = validate_screen_result()
    base_model = base_model.expanduser().resolve(strict=True)
    dataset_root = dataset_root.expanduser().resolve(strict=True)
    if sha256_file(base_model / "config.json") != preflight.EXPECTED_BASE_CONFIG_SHA256:
        raise ValueError("Top-2 contrast calibration pinned base config differs")
    resolved_output = output_dir.expanduser().resolve()
    freshness_error: BaseException | None = None
    if context.is_primary and resolved_output.exists():
        freshness_error = ValueError(
            f"Top-2 contrast calibration output must be fresh: {resolved_output}"
        )
    distributed.phase_consensus(
        context,
        phase="top2-contrast-calibration-output-freshness",
        error=freshness_error,
    )
    creation_error: BaseException | None = None
    if context.is_primary:
        try:
            resolved_output.mkdir(parents=True, exist_ok=False)
        except BaseException as error:
            creation_error = error
    distributed.phase_consensus(
        context,
        phase="top2-contrast-calibration-output-creation",
        error=creation_error,
    )

    set_seed(SEED)
    model, tokenizer, model_audit = screen.load_model(
        base_model,
        device=context.device,
    )
    screen_helper.configure_candidate(model, SELECTED_CANDIDATE)
    named_trainable, trainable_audit = recurrent_train.configure_trainable_parameters(
        model
    )
    modules = tuple(iter_delta_mem_modules(model))
    candidate_configured = all(
        module.rwkv_ms_read_temperature == 16.0
        and module.rwkv_ms_read_top_k == 2
        and module.memory_fusion_mode == "content_gated_add"
        for _, module in modules
    )
    if not candidate_configured:
        raise RuntimeError("Top-2 contrast candidate configuration failed")
    rows = contrast.load_scene_rows(tokenizer, dataset_root)
    donor_mapping, donor_deltas, donor_payload = contrast.build_donor_mapping(rows)
    schedule, schedule_payload = contrast.build_schedule(
        rows,
        donor_mapping,
        donor_deltas,
    )
    if (
        canonical_sha256(schedule_payload) != contrast.FULL_SCHEDULE_SHA256
        or canonical_sha256(donor_payload) != contrast.DONOR_MAPPING_SHA256
    ):
        raise RuntimeError("Top-2 contrast calibration schedule binding differs")
    input_binding = {
        "schema": INPUT_SCHEMA,
        "protocol_payload_sha256": PROTOCOL_PAYLOAD_SHA256,
        "protocol_objective": protocol["objective"],
        "screen_result_file_sha256": SCREEN_RESULT_FILE_SHA256,
        "screen_result_receipt": SCREEN_RESULT_RECEIPT,
        "screen_status": screen_result["status"],
        "selected_candidate": SELECTED_CANDIDATE,
        "seed": SEED,
        "updates": UPDATES,
        "world_size": WORLD_SIZE,
        "global_batch_size": causal_train.GLOBAL_BATCH_SIZE,
        "local_rows": causal_train.LOCAL_ROWS,
        "learning_rate": causal_train.LEARNING_RATE,
        "max_gradient_norm": causal_train.MAX_GRAD_NORM,
        "contrast_weight": causal_train.CONTRAST_WEIGHT,
        "margin": causal_train.MARGIN,
        "base_model": str(base_model),
        "base_config_sha256": preflight.EXPECTED_BASE_CONFIG_SHA256,
        "dataset_root": str(dataset_root),
        "scene_fit_file_sha256": contrast.SCENE_FILE_SHA256,
        "donor_mapping_payload_sha256": canonical_sha256(donor_payload),
        "training_schedule_sha256": canonical_sha256(schedule_payload),
        "schedule_prefix_sha256": canonical_sha256(schedule_payload[:UPDATES]),
        "hf_endpoint": os.environ.get("HF_ENDPOINT"),
        "rank_devices": list(context.rank_devices),
        "model_audit": model_audit,
        "trainable_audit": trainable_audit,
        "runner_sha256": sha256_file(Path(__file__)),
        "protected_splits_opened": [],
    }
    distributed.require_consensus(
        context,
        canonical_sha256(input_binding),
        description="top-2 contrast calibration input binding",
    )
    binding_error: BaseException | None = None
    if context.is_primary:
        try:
            write_json(resolved_output / "input_binding.json", input_binding)
        except BaseException as error:
            binding_error = error
    distributed.phase_consensus(
        context,
        phase="top2-contrast-calibration-input-binding-save",
        error=binding_error,
    )

    previous_step_schema = causal_train.STEP_SCHEMA
    previous_protocol_hash = causal_train.PROTOCOL_PAYLOAD_SHA256
    causal_train.STEP_SCHEMA = STEP_SCHEMA
    causal_train.PROTOCOL_PAYLOAD_SHA256 = PROTOCOL_PAYLOAD_SHA256
    try:
        training = causal_train.train(
            model,
            rows,
            schedule,
            updates=UPDATES,
            context=context,
            pad_token_id=int(tokenizer.pad_token_id),
            output_dir=resolved_output,
            named_trainable=named_trainable,
        )
    finally:
        causal_train.STEP_SCHEMA = previous_step_schema
        causal_train.PROTOCOL_PAYLOAD_SHA256 = previous_protocol_hash
    rank_runtime = distributed.gather_objects(
        context,
        {
            "rank": context.process_rank,
            "peak_cuda_memory_bytes": training["peak_cuda_memory_bytes"],
        },
    )
    passed = (
        addressed_screen.four_distinct_a100s(context.rank_devices)
        and trainable_audit["projected_router_frozen_tensors"]
        == preflight.EXPECTED_LAYERS
        and training["initial_adapter_sha256"] != training["final_adapter_sha256"]
        and training["recurrent_subset_changed"] is True
        and training["maximum_global_inactive_parameter_tensors"] == 0
        and training["projected_carrier_fixed_every_row"] is True
        and training["first_update_recurrent_gradient_audit"]["passed"] is True
    )
    result: dict[str, Any] = {}
    save_error: BaseException | None = None
    if context.is_primary:
        try:
            adapter_dir = resolved_output / "adapter"
            save_delta_mem_adapter(model, adapter_dir, screen.build_config())
            adapter_files = contrast.gate.snapshot_directory_files(adapter_dir)
            result = {
                "schema": SCHEMA,
                "status": (
                    "calibration_passed_contrastive_training_authorized"
                    if passed
                    else "calibration_failed_contrastive_training_blocked"
                ),
                "passed": passed,
                "protocol_payload_sha256": PROTOCOL_PAYLOAD_SHA256,
                "seed": SEED,
                "updates": UPDATES,
                "input_binding": input_binding,
                "training": training,
                "adapter_files": adapter_files,
                "adapter_files_sha256": contrast.gate._sha256_json(adapter_files),
                "rank_runtime": list(rank_runtime),
                "contrastive_training_authorized": passed,
                "native_benchmark_authorized": False,
                "protected_splits_opened": [],
                "code_bindings": {
                    "runner_sha256": sha256_file(Path(__file__)),
                    "training_helper_sha256": sha256_file(Path(causal_train.__file__)),
                    "screen_runner_sha256": sha256_file(Path(screen.__file__)),
                    "protocol_file_sha256": sha256_file(PROTOCOL),
                    "screen_result_file_sha256": sha256_file(SCREEN_RESULT),
                    "contrast_runner_sha256": sha256_file(Path(contrast.__file__)),
                    "distributed_sha256": sha256_file(Path(distributed.__file__)),
                    "delta_impl_sha256": sha256_file(
                        PROJECT_ROOT / "deltamem/core/delta_impl.py"
                    ),
                    "rwkv_core_sha256": sha256_file(
                        PROJECT_ROOT / "deltamem/core/hrm_rwkv7.py"
                    ),
                },
            }
            result["receipt"] = {
                "algorithm": "sha256",
                "payload_scope": "canonical_result_without_receipt",
                "payload_sha256": canonical_sha256(result),
            }
            write_json(resolved_output / "result.json", result)
        except BaseException as error:
            save_error = error
    distributed.phase_consensus(
        context,
        phase="top2-contrast-calibration-result-save",
        error=save_error,
    )
    del model, rows
    gc.collect()
    torch.cuda.empty_cache()
    return result if context.is_primary else {
        "status": "worker_complete",
        "passed": passed,
        "seed": SEED,
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--base-model", type=Path, default=BASE_MODEL)
    parser.add_argument("--dataset-root", type=Path, default=DATASET_ROOT)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args(argv)


@record
def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    context = distributed.initialize_distributed_training(args.device)
    if context is None:
        raise ValueError("Top-2 contrast calibration requires four-rank torchrun")
    try:
        result = run(
            context=context,
            output_dir=args.output_dir,
            base_model=args.base_model,
            dataset_root=args.dataset_root,
        )
    finally:
        distributed.destroy_distributed_training(context)
    print(
        json.dumps(
            {
                "rank": context.process_rank,
                "status": result["status"],
                "passed": result["passed"],
                "result_receipt": (
                    None
                    if not context.is_primary
                    else result["receipt"]["payload_sha256"]
                ),
            },
            ensure_ascii=True,
            sort_keys=True,
        ),
        flush=True,
    )
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
