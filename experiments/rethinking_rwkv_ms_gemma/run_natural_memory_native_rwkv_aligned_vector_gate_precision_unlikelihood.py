#!/usr/bin/env python3
"""Train only the aligned-vector content gate against native precision negatives."""

from __future__ import annotations

import argparse
import gc
import json
import os
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

import torch
from torch.distributed.elastic.multiprocessing.errors import record

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = Path(__file__).resolve().parents[2]
for import_root in (SCRIPT_DIR, PROJECT_ROOT):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

from deltamem.core.delta import (  # noqa: E402
    iter_delta_mem_modules,
    load_delta_mem_adapter,
    save_delta_mem_adapter,
)
from experiments.rethinking_rwkv_ms_gemma import (  # noqa: E402
    natural_memory_distributed as distributed,
)
from experiments.rethinking_rwkv_ms_gemma import (  # noqa: E402
    run_natural_memory_native_rwkv_aligned_vector_gate_causal_train as aligned,
)
from experiments.rethinking_rwkv_ms_gemma import (  # noqa: E402
    run_natural_memory_native_rwkv_top2_abstention_screen as top2_screen,
)
from experiments.rethinking_rwkv_ms_gemma import (  # noqa: E402
    run_natural_memory_native_scene_contrast_dropout as contrast,
)
from experiments.rethinking_rwkv_ms_gemma import (  # noqa: E402
    run_natural_memory_native_scene_precision_unlikelihood as engine,
)
from experiments.rethinking_rwkv_ms_gemma import (  # noqa: E402
    run_natural_memory_native_rwkv_sharp_router_screen as screen_helper,
)

# The existing locked generation engine expects this compatibility namespace for
# the base model and preflight constants.
shared = aligned.shared


SCHEMA = "rwkv_ms_natural_memory_native_aligned_vector_gate_precision_unlikelihood_train.v1"
STEP_SCHEMA = (
    "rwkv_ms_natural_memory_native_aligned_vector_gate_precision_unlikelihood_step.v1"
)
PATCH_SCHEMA = (
    "rwkv_ms_natural_memory_native_aligned_vector_gate_precision_unlikelihood_patch.v1"
)
PROTOCOL = SCRIPT_DIR / "natural_memory_native_rwkv_aligned_vector_gate_precision_unlikelihood_protocol_v1.json"
PROTOCOL_PAYLOAD_SHA256 = "4a6a93aeda67be9503e2f036838b66957ca536380966a94ee4a1b18140c7daf8"
WORLD_SIZE = 4
SEED = 131
UPDATES = 16
BASE_MODEL = contrast.BASE_MODEL
DATASET_ROOT = contrast.NATIVE_DATASET_ROOT
STARTING_ADAPTER = (
    SCRIPT_DIR
    / "local_artifacts/natural_memory_native_rwkv_aligned_vector_gate_specificity_train_v1/adapter"
)
STARTING_ADAPTER_FILES_SHA256 = "2503a387cec74a53dcb1dc163733a267b3091e029f77e582cf5908c7921ce0ac"
STARTING_ADAPTER_CONFIG_SHA256 = "28830f91f455c77f6b565cacb3069262d991d2030e3564961e0c957f13c8245f"
STARTING_RESULT_RECEIPT = "96ccb53dbcf9f8927125a2b9ff0d6118007c11f54770d54c5fc4a3cdfee915a7"
STARTING_RESULT_FILE_SHA256 = "44e024dde6097a27b803a3b2fa8a23fdd432de6db7c46d21fc75cd1d04ccca6b"


def canonical_sha256(value: Any) -> str:
    return distributed.canonical_sha256(value)


def sha256_file(path: Path) -> str:
    return engine.sha256_file(path)


def validate_protocol() -> Mapping[str, Any]:
    payload = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    receipt = payload.get("receipt")
    if not isinstance(receipt, Mapping):
        raise ValueError("Aligned precision protocol receipt is missing")
    unsigned = dict(payload)
    unsigned.pop("receipt")
    digest = canonical_sha256(unsigned)
    training = payload.get("training", {})
    starting = payload.get("starting_artifact", {})
    if (
        digest != PROTOCOL_PAYLOAD_SHA256
        or receipt.get("payload_sha256") != digest
        or training.get("updates") != UPDATES
        or training.get("global_batch_size") != engine.GLOBAL_BATCH_SIZE
        or training.get("content_gate_parameter_tensors") != 126
        or starting.get("adapter_files_sha256") != STARTING_ADAPTER_FILES_SHA256
        or payload.get("protected_splits_opened_by_this_protocol") != []
    ):
        raise ValueError("Aligned precision protocol differs")
    return payload


def configure_engine() -> None:
    """Bind the generic negative-loss engine to this locked experiment."""
    bindings = {
        "SCHEMA": SCHEMA,
        "STEP_SCHEMA": STEP_SCHEMA,
        "PATCH_SCHEMA": PATCH_SCHEMA,
        "PROTOCOL": PROTOCOL,
        "PROTOCOL_PAYLOAD_SHA256": PROTOCOL_PAYLOAD_SHA256,
        "SEED": SEED,
        "SEEDS": (SEED,),
        "TRAIN_UPDATES": UPDATES,
    }
    for name, value in bindings.items():
        setattr(engine, name, value)
    # The generic engine constructs deterministic negatives during row loading.
    contrast.load_scene_rows = engine.load_scene_rows


def load_starting_model(
    base_model: Path,
    adapter_path: Path,
    *,
    device: torch.device,
) -> tuple[torch.nn.Module, Any, Mapping[str, Any]]:
    model, tokenizer, inherited = aligned.load_model(
        base_model,
        device=device,
        candidate=aligned.SELECTED_CANDIDATE,
    )
    files = contrast.gate.snapshot_directory_files(adapter_path)
    if contrast.gate._sha256_json(files) != STARTING_ADAPTER_FILES_SHA256:
        raise ValueError("Aligned precision starting adapter files differ")
    if sha256_file(adapter_path / "delta_mem_config.json") != STARTING_ADAPTER_CONFIG_SHA256:
        raise ValueError("Aligned precision starting adapter config differs")
    loaded_config = load_delta_mem_adapter(model, adapter_path)
    screen_helper.configure_candidate(model, aligned.SELECTED_CANDIDATE)
    modules = tuple(iter_delta_mem_modules(model))
    if len(modules) != 42 or any(
        module.rwkv_ms_hybrid_mode != "aligned_vector_gate"
        or module.memory_fusion_mode != "content_gated_add"
        for _, module in modules
    ):
        raise RuntimeError("Aligned precision runtime mode restoration failed")
    return model, tokenizer, {
        **dict(inherited),
        "starting_adapter_files_sha256": contrast.gate._sha256_json(files),
        "starting_adapter_config_sha256": STARTING_ADAPTER_CONFIG_SHA256,
        "loaded_delta_config": loaded_config.to_dict(),
        "runtime_hybrid_mode": "aligned_vector_gate",
        "wrapped_layers": len(modules),
    }


def run(
    *,
    context: distributed.DistributedTrainingContext,
    output_dir: Path,
    base_model: Path,
    adapter_path: Path,
    dataset_root: Path,
) -> Mapping[str, Any]:
    if context.world_size != WORLD_SIZE:
        raise ValueError("Aligned precision training requires exactly four ranks")
    if os.environ.get("HF_ENDPOINT") != "https://hf-mirror.com":
        raise ValueError("HF_ENDPOINT must be https://hf-mirror.com")
    configure_engine()
    protocol = validate_protocol()
    base_model = base_model.expanduser().resolve(strict=True)
    adapter_path = adapter_path.expanduser().resolve(strict=True)
    dataset_root = dataset_root.expanduser().resolve(strict=True)
    resolved_output = output_dir.expanduser().resolve()
    if resolved_output.exists():
        raise ValueError(f"Aligned precision output must be fresh: {resolved_output}")
    if engine.sha256_file(base_model / "config.json") != contrast.BASE_CONFIG_SHA256:
        raise ValueError("Aligned precision base configuration differs")
    model, tokenizer, model_audit = load_starting_model(
        base_model, adapter_path, device=context.device
    )
    named_trainable, trainable_audit = contrast.configure_gate_only_training(model)
    if trainable_audit.get("parameter_tensors") != 126:
        raise RuntimeError("Aligned precision selected a non-gate parameter")
    rows = contrast.load_scene_rows(tokenizer, dataset_root)
    mapping, donor_deltas, donor_payload = contrast.build_donor_mapping(rows)
    schedules, payloads = engine.build_schedules(rows, mapping, donor_deltas)
    schedule = schedules[SEED]
    schedule_payload = payloads[SEED]
    input_binding = {
        "schema": SCHEMA,
        "protocol_payload_sha256": PROTOCOL_PAYLOAD_SHA256,
        "seed": SEED,
        "updates": UPDATES,
        "selected_candidate": aligned.SELECTED_CANDIDATE,
        "world_size": context.world_size,
        "global_batch_size": engine.GLOBAL_BATCH_SIZE,
        "local_rows": engine.LOCAL_ROWS,
        "base_model": str(base_model),
        "base_config_sha256": contrast.BASE_CONFIG_SHA256,
        "starting_adapter": str(adapter_path),
        "starting_adapter_files_sha256": STARTING_ADAPTER_FILES_SHA256,
        "dataset_root": str(dataset_root),
        "scene_file_sha256": contrast.SCENE_FILE_SHA256,
        "donor_mapping_payload_sha256": canonical_sha256(donor_payload),
        "schedule_payload_sha256": canonical_sha256(schedule_payload),
        "model_audit": model_audit,
        "trainable_audit": trainable_audit,
        "hf_endpoint": os.environ.get("HF_ENDPOINT"),
        "protected_splits_opened": [],
    }
    distributed.require_consensus(
        context,
        canonical_sha256(input_binding),
        description="aligned precision input binding",
    )
    creation_error: BaseException | None = None
    if context.is_primary:
        try:
            resolved_output.mkdir(parents=True, exist_ok=False)
            contrast._write_json(resolved_output / "input_binding.json", input_binding)
        except BaseException as error:
            creation_error = error
    distributed.phase_consensus(
        context, phase="aligned-precision-output-creation", error=creation_error
    )
    training = engine.train(
        model,
        rows,
        schedule,
        seed=SEED,
        updates=UPDATES,
        context=context,
        pad_token_id=int(tokenizer.pad_token_id),
        dtype=torch.bfloat16,
        output_dir=resolved_output,
        named_trainable=named_trainable,
    )
    result: dict[str, Any] = {}
    save_error: BaseException | None = None
    if context.is_primary:
        try:
            adapter_dir = resolved_output / "adapter"
            save_delta_mem_adapter(model, adapter_dir, top2_screen.build_config(aligned.SELECTED_CANDIDATE))
            adapter_files = contrast.gate.snapshot_directory_files(adapter_dir)
            result = {
                "schema": SCHEMA,
                "status": "aligned_vector_gate_precision_unlikelihood_training_passed_generation_authorized",
                "passed": True,
                "training_passed": True,
                "open_native_generation_authorized": True,
                "protocol_payload_sha256": PROTOCOL_PAYLOAD_SHA256,
                "seed": SEED,
                "updates": UPDATES,
                "selected_candidate": aligned.SELECTED_CANDIDATE,
                "input_binding": input_binding,
                "training": training,
                "adapter_files": adapter_files,
                "adapter_files_sha256": contrast.gate._sha256_json(adapter_files),
                "protected_splits_opened": [],
                "code_bindings": {
                    "runner_sha256": sha256_file(Path(__file__)),
                    "engine_sha256": sha256_file(Path(engine.__file__)),
                    "distributed_sha256": sha256_file(Path(distributed.__file__)),
                },
            }
            result["receipt"] = {
                "algorithm": "sha256",
                "payload_scope": "canonical_result_without_receipt",
                "payload_sha256": canonical_sha256(result),
            }
            contrast._write_json(resolved_output / "result.json", result)
        except BaseException as error:
            save_error = error
    distributed.phase_consensus(
        context, phase="aligned-precision-result-save", error=save_error
    )
    del model
    gc.collect()
    torch.cuda.empty_cache()
    return result if context.is_primary else {"status": "worker_complete", "rank": context.process_rank}


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--base-model", type=Path, default=BASE_MODEL)
    parser.add_argument("--adapter-path", type=Path, default=STARTING_ADAPTER)
    parser.add_argument("--dataset-root", type=Path, default=DATASET_ROOT)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args(argv)


@record
def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    context = distributed.initialize_distributed_training(args.device)
    if context is None:
        raise ValueError("Aligned precision training requires torchrun with four ranks")
    try:
        result = run(
            context=context,
            output_dir=args.output_dir,
            base_model=args.base_model,
            adapter_path=args.adapter_path,
            dataset_root=args.dataset_root,
        )
    finally:
        distributed.destroy_distributed_training(context)
    print(json.dumps(result, ensure_ascii=True, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
