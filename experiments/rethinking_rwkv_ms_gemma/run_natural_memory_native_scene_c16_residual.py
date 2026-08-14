#!/usr/bin/env python3
"""Train one locked checkpoint-16-anchored native-scene residual seed."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

import torch
from torch.distributed.elastic.multiprocessing.errors import record


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = Path(__file__).resolve().parents[2]
for import_root in (SCRIPT_DIR, PROJECT_ROOT):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

from deltamem.core.delta import iter_delta_mem_modules, load_delta_mem_adapter  # noqa: E402
from experiments.rethinking_rwkv_ms_gemma import (  # noqa: E402
    natural_memory_distributed as distributed,
)
from experiments.rethinking_rwkv_ms_gemma import (  # noqa: E402
    run_natural_memory_gate as gate,
)
from experiments.rethinking_rwkv_ms_gemma import (  # noqa: E402
    run_natural_memory_native_evolution as evolution,
)
from experiments.rethinking_rwkv_ms_gemma import (  # noqa: E402
    run_natural_memory_native_scene_contrast_dropout as contrast,
)
from experiments.rethinking_rwkv_ms_gemma import (  # noqa: E402
    run_natural_memory_native_scene_contrast_probe as probe,
)
from experiments.rethinking_rwkv_ms_gemma import (  # noqa: E402
    run_natural_memory_native_scene_seed_ensemble as robust,
)
from experiments.rethinking_rwkv_ms_gemma import (  # noqa: E402
    run_synthetic_compositional_associative_retrieval_v3 as runtime,
)


SCHEMA = "rwkv_ms_natural_memory_native_scene_c16_residual_training_result.v1"
STEP_SCHEMA = "rwkv_ms_natural_memory_native_scene_c16_residual_training_step.v1"
PATCH_SCHEMA = "rwkv_ms_natural_memory_native_scene_c16_residual_training_patch.v1"
PROTOCOL = SCRIPT_DIR / "natural_memory_native_scene_c16_residual_protocol_v1.json"
PROTOCOL_PAYLOAD_SHA256 = "e0b797bda233266b8806e67202b4c30067d95d7a604f6a33607b6340383d55e3"
WORLD_SIZE = 4
GLOBAL_BATCH_SIZE = 16
LOCAL_ROWS = 4
TRAIN_UPDATES = 8
SEEDS = (71, 89, 107)
LEARNING_RATE = 2.5e-5
POST_STEP_DELTA_RETENTION = 0.995
TRAIN_SALT_PREFIX = "rwkv-ms-native-scene-c16-residual-v1:"
DROP_SALT_PREFIX = "rwkv-ms-native-scene-c16-residual-drop-v1:"
EXPECTED_EXCLUDED_ROWS = 692
EXPECTED_AVAILABLE_ROWS = 743
STARTING_STEP = 16
STARTING_GATE_STATE_SHA256 = "5b0670683046e9701c24171a5c9d8cfc58e1078f25b72806662732260bab7d4f"
STARTING_PATCH_SHA256 = "c7bd4f6a396c06404cd884f3f6ff92ad5d891621fe295a3e91e9b33d73d4834a"
SEED_BINDINGS = {
    71: {
        "selected_rows_payload_sha256": "5db998f2a5313246ea82475320f6fb1c6597cd19115217fa6d13d153b2114a43",
        "schedule_payload_sha256": "317e6ee0134ff517428e7f509d591b0598aa83939a5756bd932c7093354554f8",
    },
    89: {
        "selected_rows_payload_sha256": "fba18a72f9849a7a8a97d9164664a004d50ebb584fe137eef9583ba1bc19646d",
        "schedule_payload_sha256": "d5f38ccd1a49bd55526e269f80cb9338bdeace48a04a56bac0695a6b21459fd5",
    },
    107: {
        "selected_rows_payload_sha256": "a721e9c3090a023dff664a11e64a7c316015257c369b8d1ac63b2f25d5b35a31",
        "schedule_payload_sha256": "3dab9322b8eb2cd5add0b4958bb7ba108e4ad01d91dc3bb6e5173083b387d295",
    },
}


def canonical_sha256(value: Any) -> str:
    return robust.canonical_sha256(value)


def sha256_file(path: Path) -> str:
    return robust.sha256_file(path)


def validate_protocol() -> Mapping[str, Any]:
    value = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    receipt = value.get("receipt")
    if not isinstance(receipt, Mapping):
        raise ValueError("C16-residual protocol receipt is missing")
    unsigned = dict(value)
    unsigned.pop("receipt")
    digest = canonical_sha256(unsigned)
    if digest != PROTOCOL_PAYLOAD_SHA256 or receipt.get("payload_sha256") != digest:
        raise ValueError("C16-residual protocol hash differs")
    return value


def configure_training_engine() -> None:
    robust.PROTOCOL_PAYLOAD_SHA256 = PROTOCOL_PAYLOAD_SHA256
    robust.PATCH_SCHEMA = PATCH_SCHEMA
    robust.STEP_SCHEMA = STEP_SCHEMA
    robust.LEARNING_RATE = LEARNING_RATE
    robust.POST_STEP_DELTA_RETENTION = POST_STEP_DELTA_RETENTION


def build_schedules(
    rows: Sequence[contrast.SceneContrastRow],
    mapping: Mapping[int, int],
    deltas: Mapping[int, int],
) -> tuple[dict[int, tuple[robust.RobustScheduleStep, ...]], dict[int, list[dict[str, Any]]]]:
    original_schedule, _ = contrast.build_schedule(rows, mapping, deltas)
    excluded = {
        source_ordinal
        for step in original_schedule[:STARTING_STEP]
        for source_ordinal in step.source_ordinals
    }
    for prior_seed in robust.SEEDS:
        schedule, _ = robust.build_schedule(
            rows,
            mapping,
            deltas,
            seed=prior_seed,
        )
        excluded.update(
            source_ordinal for step in schedule for source_ordinal in step.source_ordinals
        )
    if len(excluded) != EXPECTED_EXCLUDED_ROWS:
        raise ValueError("C16-residual exclusion set differs")
    eligible = {
        source_ordinal
        for source_ordinal in range(len(rows))
        if deltas[source_ordinal] <= contrast.MAX_DONOR_TOKEN_DELTA
    }
    available = eligible - excluded
    if len(available) != EXPECTED_AVAILABLE_ROWS:
        raise ValueError("C16-residual available row set differs")
    schedules: dict[int, tuple[robust.RobustScheduleStep, ...]] = {}
    payloads: dict[int, list[dict[str, Any]]] = {}
    for seed in SEEDS:
        train_salt = f"{TRAIN_SALT_PREFIX}{seed}:"
        selected = sorted(
            available,
            key=lambda source_ordinal: (
                hashlib.sha256(
                    (train_salt + rows[source_ordinal].example.row_sha256).encode("utf-8")
                ).hexdigest(),
                source_ordinal,
            ),
        )[: TRAIN_UPDATES * GLOBAL_BATCH_SIZE]
        available.difference_update(selected)
        selected_hashes = [rows[index].example.row_sha256 for index in selected]
        if canonical_sha256(selected_hashes) != SEED_BINDINGS[seed][
            "selected_rows_payload_sha256"
        ]:
            raise ValueError(f"C16-residual selected row hash differs: {seed}")
        schedule: list[robust.RobustScheduleStep] = []
        payload: list[dict[str, Any]] = []
        for offset in range(0, len(selected), GLOBAL_BATCH_SIZE):
            step = offset // GLOBAL_BATCH_SIZE + 1
            group = selected[offset : offset + GLOBAL_BATCH_SIZE]
            no_state = frozenset(
                sorted(
                    group,
                    key=lambda source_ordinal: (
                        hashlib.sha256(
                            (
                                f"{DROP_SALT_PREFIX}{seed}:{step}:"
                                + rows[source_ordinal].example.row_sha256
                            ).encode("utf-8")
                        ).hexdigest(),
                        source_ordinal,
                    ),
                )[:4]
            )
            row_payload = [
                {
                    "source_ordinal": source_ordinal,
                    "source_row_sha256": rows[source_ordinal].example.row_sha256,
                    "donor_ordinal": mapping[source_ordinal],
                    "positive_condition": (
                        "no_state" if source_ordinal in no_state else "correct_state"
                    ),
                }
                for source_ordinal in group
            ]
            step_payload = {"step": step, "rows": row_payload}
            payload.append(step_payload)
            schedule.append(
                robust.RobustScheduleStep(
                    step=step,
                    source_ordinals=tuple(group),
                    donor_ordinals=tuple(mapping[index] for index in group),
                    no_state_ordinals=no_state,
                    payload_sha256=canonical_sha256(step_payload),
                )
            )
        if canonical_sha256(payload) != SEED_BINDINGS[seed]["schedule_payload_sha256"]:
            raise ValueError(f"C16-residual schedule hash differs: {seed}")
        schedules[seed] = tuple(schedule)
        payloads[seed] = payload
    if len(available) != 359:
        raise ValueError("C16-residual remaining row count differs")
    return schedules, payloads


def load_starting_checkpoint(
    model: torch.nn.Module,
    training_root: Path,
) -> Mapping[str, Any]:
    manifests = probe.validate_training_root(training_root)
    manifest = next(item for item in manifests if int(item["step"]) == STARTING_STEP)
    patch_path = training_root / f"checkpoint-{STARTING_STEP}" / "gate_patch.pt"
    if (
        manifest.get("gate_state_sha256") != STARTING_GATE_STATE_SHA256
        or manifest.get("patch_file", {}).get("sha256") != STARTING_PATCH_SHA256
        or sha256_file(patch_path) != STARTING_PATCH_SHA256
    ):
        raise ValueError("C16-residual starting checkpoint binding differs")
    payload = torch.load(patch_path, map_location="cpu", weights_only=True)
    if not isinstance(payload, Mapping) or not isinstance(payload.get("state_dict"), Mapping):
        raise ValueError("C16-residual starting checkpoint payload differs")
    state = payload["state_dict"]
    if runtime._state_dict_sha256(state) != STARTING_GATE_STATE_SHA256:
        raise ValueError("C16-residual starting gate state differs")
    parameters = dict(model.named_parameters())
    names = {
        name
        for name in parameters
        if any(name.endswith(f".{family}") for family in contrast.GATE_FAMILIES)
    }
    if set(state) != names or len(names) != 126:
        raise ValueError("C16-residual runtime gate names differ")
    with torch.no_grad():
        for name in sorted(names):
            target = parameters[name]
            source = state[name]
            if source.shape != target.shape:
                raise ValueError(f"C16-residual runtime shape differs: {name}")
            target.copy_(source.to(device=target.device, dtype=target.dtype))
    loaded = {name: parameters[name].detach().cpu().clone() for name in sorted(names)}
    runtime_sha256 = runtime._state_dict_sha256(loaded)
    expected = {
        name: state[name].to(dtype=parameters[name].dtype).detach().cpu().clone()
        for name in sorted(names)
    }
    if runtime_sha256 != runtime._state_dict_sha256(expected):
        raise ValueError("C16-residual runtime-cast checkpoint differs")
    return {
        "step": STARTING_STEP,
        "gate_state_sha256": STARTING_GATE_STATE_SHA256,
        "runtime_gate_state_sha256": runtime_sha256,
        "patch_sha256": STARTING_PATCH_SHA256,
    }


def run(
    *,
    context: distributed.DistributedTrainingContext,
    output_dir: Path,
    seed: int,
    base_model: Path,
    adapter_path: Path,
    dataset_root: Path,
    training_root: Path,
) -> Mapping[str, Any]:
    if context.world_size != WORLD_SIZE or seed not in SEEDS:
        raise ValueError("C16-residual requires a locked seed on exactly four ranks")
    gate.configure_hf_mirror()
    validate_protocol()
    configure_training_engine()
    base_model = base_model.expanduser().resolve(strict=True)
    adapter_path = adapter_path.expanduser().resolve(strict=True)
    dataset_root = dataset_root.expanduser().resolve(strict=True)
    training_root = training_root.expanduser().resolve(strict=True)
    resolved_output = output_dir.expanduser().resolve()
    if resolved_output.exists() or output_dir.expanduser().is_symlink():
        raise ValueError(f"C16-residual output must be fresh: {resolved_output}")
    if sha256_file(base_model / "config.json") != contrast.BASE_CONFIG_SHA256:
        raise ValueError("C16-residual base configuration differs")
    adapter_files = gate.snapshot_directory_files(adapter_path)
    if gate._sha256_json(adapter_files) != contrast.V9_ADAPTER_FILES_SHA256:
        raise ValueError("C16-residual V9 adapter differs")
    runtime.set_seed(seed)
    delta_config = evolution.build_evolution_delta_config(
        "shared_qo_content_gated_attention_output"
    )
    model, tokenizer, _, _, _ = gate._load_model_and_tokenizer(
        {"model": {"path": str(base_model)}},
        device=context.device,
        dtype=torch.bfloat16,
        attn_implementation="sdpa",
        delta_config=delta_config,
    )
    loaded_config = load_delta_mem_adapter(model, adapter_path)
    if loaded_config.to_dict() != delta_config.to_dict():
        raise ValueError("C16-residual adapter configuration differs")
    if len(list(iter_delta_mem_modules(model))) != 42:
        raise ValueError("C16-residual requires 42 wrapped layers")
    starting = load_starting_checkpoint(model, training_root)
    named_trainable, trainable_audit = contrast.configure_gate_only_training(model)
    rows = contrast.load_scene_rows(tokenizer, dataset_root)
    mapping, donor_deltas, donor_payload = contrast.build_donor_mapping(rows)
    if canonical_sha256(donor_payload) != contrast.DONOR_MAPPING_SHA256:
        raise ValueError("C16-residual donor mapping differs")
    schedules, payloads = build_schedules(rows, mapping, donor_deltas)
    schedule = schedules[seed]
    schedule_payload = payloads[seed]
    input_binding = {
        "schema": SCHEMA,
        "seed": seed,
        "updates": TRAIN_UPDATES,
        "protocol_payload_sha256": PROTOCOL_PAYLOAD_SHA256,
        "runner_sha256": sha256_file(Path(__file__)),
        "robust_engine_sha256": sha256_file(Path(robust.__file__)),
        "base_model": str(base_model),
        "base_config_sha256": contrast.BASE_CONFIG_SHA256,
        "adapter_path": str(adapter_path),
        "adapter_files_sha256": contrast.V9_ADAPTER_FILES_SHA256,
        "dataset_root": str(dataset_root),
        "scene_file_sha256": contrast.SCENE_FILE_SHA256,
        "training_root": str(training_root),
        "starting_checkpoint": starting,
        "schedule_payload_sha256": canonical_sha256(schedule_payload),
        "global_batch_size": GLOBAL_BATCH_SIZE,
        "local_rows": LOCAL_ROWS,
        "world_size": context.world_size,
        "learning_rate": LEARNING_RATE,
        "post_step_delta_retention": POST_STEP_DELTA_RETENTION,
        "trainable_audit": trainable_audit,
        "hf_endpoint": os.environ.get("HF_ENDPOINT"),
        "dtype": "bfloat16",
        "attn_implementation": "sdpa",
    }
    distributed.require_consensus(
        context,
        canonical_sha256(input_binding),
        description=f"C16-residual seed {seed} input binding",
    )
    creation_error: BaseException | None = None
    if context.is_primary:
        try:
            resolved_output.mkdir(parents=True, exist_ok=False)
            contrast._write_json(resolved_output / "input_binding.json", input_binding)
        except BaseException as error:
            creation_error = error
    distributed.phase_consensus(
        context,
        phase=f"c16-residual-seed-{seed}-output-creation",
        error=creation_error,
    )
    training = robust.train(
        model,
        rows,
        schedule,
        seed=seed,
        updates=TRAIN_UPDATES,
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
            result = {
                "schema": SCHEMA,
                "status": "training_complete_evaluation_pending",
                "input_binding": input_binding,
                "training": dict(training),
                "protected_splits_opened": [],
                "code_bindings": {
                    "runner_sha256": sha256_file(Path(__file__)),
                    "robust_engine_sha256": sha256_file(Path(robust.__file__)),
                    "contrast_helper_sha256": sha256_file(Path(contrast.__file__)),
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
        context,
        phase=f"c16-residual-seed-{seed}-result-save",
        error=save_error,
    )
    del model
    gc.collect()
    torch.cuda.empty_cache()
    return result if context.is_primary else {
        "status": "worker_complete",
        "rank": context.process_rank,
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, choices=SEEDS, required=True)
    parser.add_argument("--base-model", type=Path, default=contrast.BASE_MODEL)
    parser.add_argument("--adapter-path", type=Path, default=contrast.V9_ADAPTER)
    parser.add_argument("--dataset-root", type=Path, default=contrast.NATIVE_DATASET_ROOT)
    parser.add_argument("--training-root", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args(argv)


@record
def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    context = distributed.initialize_distributed_training(args.device)
    if context is None:
        raise ValueError("C16-residual training requires four-rank torchrun")
    try:
        result = run(
            context=context,
            output_dir=args.output_dir,
            seed=args.seed,
            base_model=args.base_model,
            adapter_path=args.adapter_path,
            dataset_root=args.dataset_root,
            training_root=args.training_root,
        )
    finally:
        distributed.destroy_distributed_training(context)
    print(json.dumps(result, ensure_ascii=True, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
