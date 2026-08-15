#!/usr/bin/env python3
"""Train internally routed RWKV values against causal controls."""

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
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = Path(__file__).resolve().parents[2]
for import_root in (SCRIPT_DIR, PROJECT_ROOT):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

from deltamem.core.delta import (  # noqa: E402
    attach_delta_mem,
    freeze_non_delta_mem_params,
    iter_delta_mem_modules,
    save_delta_mem_adapter,
)
from experiments.rethinking_rwkv_ms_gemma import (  # noqa: E402
    natural_memory_distributed as distributed,
)
from experiments.rethinking_rwkv_ms_gemma import (  # noqa: E402
    run_natural_memory_native_recurrent_rwkv_preflight as preflight,
)
from experiments.rethinking_rwkv_ms_gemma import (  # noqa: E402
    run_natural_memory_native_recurrent_rwkv_bf16_calibration as recurrent_calibration,
)
from experiments.rethinking_rwkv_ms_gemma import (  # noqa: E402
    run_natural_memory_native_rwkv_addressed_value_screen as addressed_screen,
)
from experiments.rethinking_rwkv_ms_gemma import (  # noqa: E402
    run_natural_memory_native_rwkv_chunk_addressed_value_causal_train as causal_train,
)
from experiments.rethinking_rwkv_ms_gemma import (  # noqa: E402
    run_natural_memory_native_rwkv_recurrent_value_calibration as calibration,
)
from experiments.rethinking_rwkv_ms_gemma import (  # noqa: E402
    run_natural_memory_native_rwkv_recurrent_value_screen as screen,
)
from experiments.rethinking_rwkv_ms_gemma import (  # noqa: E402
    run_natural_memory_native_scene_contrast_dropout as contrast,
)
from experiments.rethinking_rwkv_ms_gemma import (  # noqa: E402
    run_synthetic_compositional_associative_retrieval_v3 as runtime,
)


SCHEMA = "rwkv_ms_natural_memory_native_recurrent_value_causal_train.v2"
STEP_SCHEMA = "rwkv_ms_natural_memory_native_recurrent_value_causal_train_step.v2"
INPUT_SCHEMA = "rwkv_ms_natural_memory_native_recurrent_value_causal_train_input.v2"
PROTOCOL = (
    SCRIPT_DIR
    / "natural_memory_native_rwkv_recurrent_value_causal_train_protocol_v2.json"
)
PROTOCOL_PAYLOAD_SHA256 = (
    "1c632f06bd77161a05b26212b0f64fc104bf8a5d2ec7d9dc3e4406cb86a4eb3c"
)
CALIBRATION_RESULT = (
    SCRIPT_DIR
    / "local_artifacts/"
    "natural_memory_native_rwkv_recurrent_value_calibration_v1/result.json"
)
CALIBRATION_RESULT_FILE_SHA256 = (
    "9e04b66f80feb9a6fbbdf1bff0286aa062f9b23151f5f730f33ecc9559318c0d"
)
CALIBRATION_RESULT_RECEIPT = (
    "d3fcc0f91dcdc2ffeb7c2567f36e6b71318fe02648c0b236f4499ff2ee9dfc18"
)
FAILED_PREFLIGHT_PROTOCOL_PAYLOAD_SHA256 = (
    "a964200b1b5b9dd3ea8f9ab65f90be63279c9ff60fe0f4bb93a06fd0e4a5ebd8"
)
FAILED_PREFLIGHT_INPUT_BINDING = (
    SCRIPT_DIR
    / "local_artifacts/"
    "natural_memory_native_rwkv_recurrent_value_causal_preflight_v1/"
    "input_binding.json"
)
FAILED_PREFLIGHT_INPUT_BINDING_FILE_SHA256 = (
    "c21331058ac8dee68799c5f520c07b627f0f0b43d4ca0a35a6d0fe9f96742e5c"
)
FAILED_PREFLIGHT_RUNNER_SHA256 = (
    "78ba3eb626681c07de563117901f874b91397f546f10fb7011fb7a1ea47056d6"
)
SELECTED_CANDIDATE = calibration.SELECTED_CANDIDATE
HF_MIRROR_ENDPOINT = "https://hf-mirror.com"
BASE_MODEL = calibration.BASE_MODEL
DATASET_ROOT = calibration.DATASET_ROOT
WORLD_SIZE = 4
SEED = 60
GLOBAL_BATCH_SIZE = 8
LOCAL_ROWS = 2
PREFLIGHT_UPDATES = 1
TRAIN_UPDATES = 8
LEARNING_RATE = 1e-4
MAX_GRAD_NORM = 0.1
CONTRAST_WEIGHT = 0.25
MARGIN = 0.05
PROJECTED_ROUTER_SUFFIX = ".projected_kv_key_proj"


def canonical_sha256(value: Any) -> str:
    return distributed.canonical_sha256(value)


def sha256_file(path: Path) -> str:
    return preflight.sha256_file(path)


def write_json(path: Path, value: Mapping[str, Any]) -> None:
    if path.exists():
        raise ValueError(f"Recurrent-value causal output must be fresh: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=True, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def validate_protocol() -> Mapping[str, Any]:
    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    receipt = protocol.get("receipt")
    if not isinstance(receipt, Mapping):
        raise ValueError("Recurrent-value causal protocol receipt is missing")
    unsigned = dict(protocol)
    unsigned.pop("receipt")
    digest = canonical_sha256(unsigned)
    if digest != PROTOCOL_PAYLOAD_SHA256 or receipt.get("payload_sha256") != digest:
        raise ValueError("Recurrent-value causal protocol payload differs")
    required_authorization = {
        "calibration_protocol_payload_sha256": calibration.PROTOCOL_PAYLOAD_SHA256,
        "calibration_result_file": (
            "local_artifacts/"
            "natural_memory_native_rwkv_recurrent_value_calibration_v1/"
            "result.json"
        ),
        "calibration_result_file_sha256": CALIBRATION_RESULT_FILE_SHA256,
        "calibration_result_receipt": CALIBRATION_RESULT_RECEIPT,
        "calibration_status": "calibration_passed_causal_training_authorized",
        "selected_candidate": SELECTED_CANDIDATE,
        "failed_preflight_v1": {
            "protocol_payload_sha256": FAILED_PREFLIGHT_PROTOCOL_PAYLOAD_SHA256,
            "input_binding_file": (
                "local_artifacts/"
                "natural_memory_native_rwkv_recurrent_value_causal_preflight_v1/"
                "input_binding.json"
            ),
            "input_binding_file_sha256": (
                FAILED_PREFLIGHT_INPUT_BINDING_FILE_SHA256
            ),
            "runner_sha256": FAILED_PREFLIGHT_RUNNER_SHA256,
            "observed_exception": (
                "RuntimeError: Addressed causal optimizer has inactive parameters"
            ),
            "diagnosis": (
                "The complete projected bundle is causally inert in recurrent_value "
                "mode, but v1 still included each layer's projected_kv_key_proj in "
                "the optimizer."
            ),
        },
    }
    if protocol.get("authorization_basis") != required_authorization:
        raise ValueError("Recurrent-value causal authorization differs")
    required_training = {
        "hf_endpoint": HF_MIRROR_ENDPOINT,
        "seed": SEED,
        "learning_rate": LEARNING_RATE,
        "max_gradient_norm": MAX_GRAD_NORM,
        "global_batch_rows": GLOBAL_BATCH_SIZE,
        "local_rows_per_rank": LOCAL_ROWS,
        "preflight_optimizer_updates": PREFLIGHT_UPDATES,
        "screen_optimizer_updates": TRAIN_UPDATES,
        "contrast_weight_per_active_control": CONTRAST_WEIGHT,
        "contrast_margin": MARGIN,
    }
    training = protocol.get("training", {})
    if any(training.get(key) != value for key, value in required_training.items()):
        raise ValueError("Recurrent-value causal training contract differs")
    if protocol.get("protected_splits_opened_by_this_protocol") != []:
        raise ValueError("Recurrent-value causal training may not open protected data")
    if (
        sha256_file(FAILED_PREFLIGHT_INPUT_BINDING)
        != FAILED_PREFLIGHT_INPUT_BINDING_FILE_SHA256
    ):
        raise ValueError("Failed recurrent-value preflight binding differs")
    return protocol


def validate_calibration_result() -> Mapping[str, Any]:
    if sha256_file(CALIBRATION_RESULT) != CALIBRATION_RESULT_FILE_SHA256:
        raise ValueError("Recurrent-value calibration result file hash differs")
    result = json.loads(CALIBRATION_RESULT.read_text(encoding="utf-8"))
    receipt = result.get("receipt")
    if not isinstance(receipt, Mapping):
        raise ValueError("Recurrent-value calibration result receipt is missing")
    unsigned = dict(result)
    unsigned.pop("receipt")
    digest = canonical_sha256(unsigned)
    required = {
        "schema": calibration.SCHEMA,
        "status": "calibration_passed_causal_training_authorized",
        "passed": True,
        "causal_training_authorized": True,
        "protected_splits_opened": [],
    }
    if (
        digest != CALIBRATION_RESULT_RECEIPT
        or receipt.get("payload_sha256") != digest
        or any(result.get(key) != value for key, value in required.items())
    ):
        raise ValueError("Recurrent-value calibration did not authorize training")
    return result


def load_model(
    base_model: Path,
    *,
    device: torch.device,
) -> tuple[
    torch.nn.Module,
    Any,
    Any,
    tuple[tuple[str, torch.nn.Parameter], ...],
    Mapping[str, Any],
]:
    tokenizer = AutoTokenizer.from_pretrained(
        base_model,
        local_files_only=True,
        trust_remote_code=False,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        dtype=torch.bfloat16,
        attn_implementation="sdpa",
        local_files_only=True,
        low_cpu_mem_usage=True,
        trust_remote_code=False,
    ).to(device)
    runtime._disable_training_cache(model)
    delta_config = screen.build_config()
    replaced = attach_delta_mem(model, delta_config)
    named_trainable, trainable_audit = configure_trainable_parameters(model)
    checkpointed_mlps = runtime.checkpoint_frozen_mlp_activations(model)
    modules = tuple(iter_delta_mem_modules(model))
    wrappers_valid = all(
        module.memory_backend == "rwkv_ms"
        and module.memory_readout_mode == "projected_kv_rwkv_hybrid"
        and module.rwkv_ms_write_mode == "recurrent"
        and module.rwkv_ms_hybrid_mode == "recurrent_value"
        and module.rwkv_ms_hybrid_gain == 0.03125
        for _, module in modules
    )
    audit = {
        "wrapped_layers": len(modules),
        "replaced_layers": len(replaced),
        "all_wrappers_recurrent_value": wrappers_valid,
        "checkpointed_frozen_mlps": len(checkpointed_mlps),
        "backbone_dtype": "bfloat16",
        "trainable_master_dtype": "float32",
        "trainables": trainable_audit,
    }
    if (
        len(replaced) != preflight.EXPECTED_LAYERS
        or len(modules) != preflight.EXPECTED_LAYERS
        or not wrappers_valid
    ):
        raise RuntimeError(f"Recurrent-value causal attachment failed: {audit!r}")
    return model, tokenizer, delta_config, named_trainable, audit


def configure_trainable_parameters(
    model: torch.nn.Module,
) -> tuple[tuple[tuple[str, torch.nn.Parameter], ...], Mapping[str, Any]]:
    freeze_non_delta_mem_params(model)
    projected_router_frozen: list[str] = []
    for name, parameter in model.named_parameters():
        if name.endswith(PROJECTED_ROUTER_SUFFIX):
            parameter.requires_grad_(False)
            projected_router_frozen.append(name)
    runtime._promote_trainable_parameters_to_fp32(model)
    selected = distributed.stable_named_parameters(
        [
            (name, parameter)
            for name, parameter in model.named_parameters()
            if parameter.requires_grad
        ]
    )
    selected_names = [name for name, _ in selected]
    recurrent_output_names = [
        name
        for name in selected_names
        if name.endswith(recurrent_calibration.RECURRENT_READOUT_SUFFIX)
    ]
    passed = (
        bool(selected)
        and len(projected_router_frozen) == preflight.EXPECTED_LAYERS
        and len(recurrent_output_names) == preflight.EXPECTED_LAYERS
        and not any(name.endswith(PROJECTED_ROUTER_SUFFIX) for name in selected_names)
    )
    audit = {
        "architecture": "recurrent_value",
        "parameter_tensors": len(selected),
        "parameter_elements": sum(parameter.numel() for _, parameter in selected),
        "parameter_names_sha256": canonical_sha256(selected_names),
        "projected_router_frozen_tensors": len(projected_router_frozen),
        "projected_router_frozen_names_sha256": canonical_sha256(
            sorted(projected_router_frozen)
        ),
        "recurrent_output_trainable_tensors": len(recurrent_output_names),
        "recurrent_output_trainable_names_sha256": canonical_sha256(
            recurrent_output_names
        ),
        "passed": passed,
    }
    if not passed:
        raise RuntimeError(f"Recurrent-value trainable isolation failed: {audit!r}")
    return selected, audit


def train(
    model: torch.nn.Module,
    rows: Sequence[contrast.SceneContrastRow],
    schedule: Sequence[contrast.ContrastScheduleStep],
    *,
    updates: int,
    context: distributed.DistributedTrainingContext,
    pad_token_id: int,
    output_dir: Path,
    named_trainable: Sequence[tuple[str, torch.nn.Parameter]],
) -> Mapping[str, Any]:
    previous_step_schema = causal_train.STEP_SCHEMA
    previous_protocol_hash = causal_train.PROTOCOL_PAYLOAD_SHA256
    causal_train.STEP_SCHEMA = STEP_SCHEMA
    causal_train.PROTOCOL_PAYLOAD_SHA256 = PROTOCOL_PAYLOAD_SHA256
    try:
        return causal_train.train(
            model,
            rows,
            schedule,
            updates=updates,
            context=context,
            pad_token_id=pad_token_id,
            output_dir=output_dir,
            named_trainable=named_trainable,
        )
    finally:
        causal_train.STEP_SCHEMA = previous_step_schema
        causal_train.PROTOCOL_PAYLOAD_SHA256 = previous_protocol_hash


def run(
    *,
    context: distributed.DistributedTrainingContext,
    output_dir: Path,
    updates: int,
    base_model: Path = BASE_MODEL,
    dataset_root: Path = DATASET_ROOT,
) -> Mapping[str, Any]:
    if context.world_size != WORLD_SIZE:
        raise ValueError("Recurrent-value causal training requires four ranks")
    if updates not in (PREFLIGHT_UPDATES, TRAIN_UPDATES):
        raise ValueError("Recurrent-value causal updates must be 1 or 8")
    if os.environ.get("HF_ENDPOINT") != HF_MIRROR_ENDPOINT:
        raise ValueError(f"HF_ENDPOINT must be exactly {HF_MIRROR_ENDPOINT}")
    protocol = validate_protocol()
    calibration_result = validate_calibration_result()
    base_model = base_model.expanduser().resolve(strict=True)
    dataset_root = dataset_root.expanduser().resolve(strict=True)
    if sha256_file(base_model / "config.json") != preflight.EXPECTED_BASE_CONFIG_SHA256:
        raise ValueError("Recurrent-value causal pinned base config differs")
    resolved_output = output_dir.expanduser().resolve()
    freshness_error: BaseException | None = None
    if context.is_primary and resolved_output.exists():
        freshness_error = ValueError(
            f"Recurrent-value causal output must be fresh: {resolved_output}"
        )
    distributed.phase_consensus(
        context,
        phase="recurrent-value-causal-output-freshness",
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
        phase="recurrent-value-causal-output-creation",
        error=creation_error,
    )

    set_seed(SEED)
    model, tokenizer, delta_config, named_trainable, model_audit = load_model(
        base_model,
        device=context.device,
    )
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
        raise RuntimeError("Recurrent-value causal schedule binding differs")
    input_binding = {
        "schema": INPUT_SCHEMA,
        "protocol_payload_sha256": PROTOCOL_PAYLOAD_SHA256,
        "protocol_objective": protocol["objective"],
        "calibration_result_file_sha256": CALIBRATION_RESULT_FILE_SHA256,
        "calibration_result_receipt": CALIBRATION_RESULT_RECEIPT,
        "calibration_status": calibration_result["status"],
        "failed_preflight_input_binding_file_sha256": (
            FAILED_PREFLIGHT_INPUT_BINDING_FILE_SHA256
        ),
        "selected_candidate": SELECTED_CANDIDATE,
        "seed": SEED,
        "updates": updates,
        "world_size": context.world_size,
        "global_batch_size": GLOBAL_BATCH_SIZE,
        "local_rows": LOCAL_ROWS,
        "learning_rate": LEARNING_RATE,
        "max_gradient_norm": MAX_GRAD_NORM,
        "contrast_weight": CONTRAST_WEIGHT,
        "margin": MARGIN,
        "base_model": str(base_model),
        "base_config_sha256": preflight.EXPECTED_BASE_CONFIG_SHA256,
        "dataset_root": str(dataset_root),
        "scene_fit_file_sha256": contrast.SCENE_FILE_SHA256,
        "donor_mapping_payload_sha256": canonical_sha256(donor_payload),
        "training_schedule_sha256": canonical_sha256(schedule_payload),
        "schedule_prefix_sha256": canonical_sha256(schedule_payload[:updates]),
        "hf_endpoint": os.environ.get("HF_ENDPOINT"),
        "rank_devices": list(context.rank_devices),
        "model_audit": model_audit,
        "runner_sha256": sha256_file(Path(__file__)),
        "protected_splits_opened": [],
    }
    distributed.require_consensus(
        context,
        canonical_sha256(input_binding),
        description="recurrent-value causal input binding",
    )
    binding_error: BaseException | None = None
    if context.is_primary:
        try:
            write_json(resolved_output / "input_binding.json", input_binding)
        except BaseException as error:
            binding_error = error
    distributed.phase_consensus(
        context,
        phase="recurrent-value-causal-input-binding-save",
        error=binding_error,
    )
    training = train(
        model,
        rows,
        schedule,
        updates=updates,
        context=context,
        pad_token_id=int(tokenizer.pad_token_id),
        output_dir=resolved_output,
        named_trainable=named_trainable,
    )
    rank_runtime = distributed.gather_objects(
        context,
        {
            "rank": context.process_rank,
            "peak_cuda_memory_bytes": training["peak_cuda_memory_bytes"],
        },
    )
    passed = (
        addressed_screen.four_distinct_a100s(context.rank_devices)
        and model_audit["trainables"]["projected_router_frozen_tensors"]
        == preflight.EXPECTED_LAYERS
        and training["initial_adapter_sha256"] != training["final_adapter_sha256"]
        and training["recurrent_subset_changed"] is True
        and training["maximum_global_inactive_parameter_tensors"] == 0
        and training["projected_carrier_fixed_every_row"] is True
        and training["first_update_recurrent_gradient_audit"]["passed"] is True
    )
    save_error: BaseException | None = None
    result: dict[str, Any] = {}
    if context.is_primary:
        try:
            adapter_dir = resolved_output / "adapter"
            save_delta_mem_adapter(model, adapter_dir, delta_config)
            adapter_files = contrast.gate.snapshot_directory_files(adapter_dir)
            result = {
                "schema": SCHEMA,
                "status": (
                    "preflight_passed"
                    if updates == PREFLIGHT_UPDATES and passed
                    else "training_complete_open_evaluation_authorized"
                    if passed
                    else "training_failed_evaluation_blocked"
                ),
                "passed": passed,
                "protocol_payload_sha256": PROTOCOL_PAYLOAD_SHA256,
                "seed": SEED,
                "updates": updates,
                "input_binding": input_binding,
                "training": training,
                "adapter_files": adapter_files,
                "adapter_files_sha256": contrast.gate._sha256_json(adapter_files),
                "rank_runtime": list(rank_runtime),
                "open_native_evaluation_authorized": (
                    passed and updates == TRAIN_UPDATES
                ),
                "protected_splits_opened": [],
                "code_bindings": {
                    "runner_sha256": sha256_file(Path(__file__)),
                    "training_helper_sha256": sha256_file(Path(causal_train.__file__)),
                    "protocol_file_sha256": sha256_file(PROTOCOL),
                    "calibration_result_file_sha256": sha256_file(
                        CALIBRATION_RESULT
                    ),
                    "failed_preflight_input_binding_file_sha256": sha256_file(
                        FAILED_PREFLIGHT_INPUT_BINDING
                    ),
                    "contrast_runner_sha256": sha256_file(Path(contrast.__file__)),
                    "distributed_sha256": sha256_file(Path(distributed.__file__)),
                    "delta_impl_sha256": sha256_file(
                        PROJECT_ROOT / "deltamem/core/delta_impl.py"
                    ),
                    "rwkv_core_sha256": sha256_file(
                        PROJECT_ROOT / "deltamem/core/hrm_rwkv7.py"
                    ),
                    "rwkv_write_scan_sha256": sha256_file(
                        PROJECT_ROOT / "deltamem/kernels/rwkv_ms_write_scan.py"
                    ),
                    "rwkv_write_scan_cuda_sha256": sha256_file(
                        PROJECT_ROOT / "deltamem/kernels/rwkv_ms_write_scan_cuda.cu"
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
        phase="recurrent-value-causal-result-save",
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
    parser.add_argument("--updates", type=int, required=True, choices=(1, 8))
    parser.add_argument("--base-model", type=Path, default=BASE_MODEL)
    parser.add_argument("--dataset-root", type=Path, default=DATASET_ROOT)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args(argv)


@record
def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    context = distributed.initialize_distributed_training(args.device)
    if context is None:
        raise ValueError("Recurrent-value causal training requires four-rank torchrun")
    try:
        result = run(
            context=context,
            output_dir=args.output_dir,
            updates=args.updates,
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
