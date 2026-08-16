#!/usr/bin/env python3
"""Train top-2 RWKV abstention and gate on held-out causal CE."""

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

from deltamem.core.delta import reset_delta_mem_states, save_delta_mem_adapter  # noqa: E402
from experiments.rethinking_rwkv_ms_gemma import (  # noqa: E402
    natural_memory_distributed as distributed,
)
from experiments.rethinking_rwkv_ms_gemma import (  # noqa: E402
    run_natural_memory_native_evolution as evolution,
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
    run_natural_memory_native_rwkv_sharp_router_screen as screen_helper,
)
from experiments.rethinking_rwkv_ms_gemma import (  # noqa: E402
    run_natural_memory_native_rwkv_top2_abstention_contrast_calibration as calibration,
)
from experiments.rethinking_rwkv_ms_gemma import (  # noqa: E402
    run_natural_memory_native_rwkv_top2_abstention_screen as screen,
)
from experiments.rethinking_rwkv_ms_gemma import (  # noqa: E402
    run_natural_memory_native_scene_contrast_dropout as contrast,
)


SCHEMA = "rwkv_ms_natural_memory_native_top2_abstention_causal_train.v1"
STEP_SCHEMA = "rwkv_ms_natural_memory_native_top2_abstention_causal_train_step.v1"
INPUT_SCHEMA = "rwkv_ms_natural_memory_native_top2_abstention_causal_train_input.v1"
PROTOCOL = (
    SCRIPT_DIR / "natural_memory_native_rwkv_top2_abstention_causal_train_protocol_v1.json"
)
PROTOCOL_PAYLOAD_SHA256 = (
    "7c366a005d2f2a1fb6ad720d91f94eec58a363c354b90b94c02cdc4368c6703f"
)
CALIBRATION_RESULT = (
    SCRIPT_DIR
    / "local_artifacts/"
    "natural_memory_native_rwkv_top2_abstention_contrast_calibration_v1/"
    "result.json"
)
CALIBRATION_RESULT_FILE_SHA256 = (
    "f186d0286d10a4bc50118affcb3791dee5d648f5e9b7071d7c79a9eaee82db90"
)
CALIBRATION_RESULT_RECEIPT = (
    "7ffb6ec4796c761030a255fad77d440fa86cc2e16d6b5e5de2c421bfe36f884e"
)
SELECTED_CANDIDATE = calibration.SELECTED_CANDIDATE
HF_MIRROR_ENDPOINT = "https://hf-mirror.com"
BASE_MODEL = calibration.BASE_MODEL
DATASET_ROOT = calibration.DATASET_ROOT
WORLD_SIZE = 4
SEED = 68
UPDATES = 8
HELDOUT_ORDINALS = (
    731, 977, 1355, 327, 588, 261, 1375, 1051,
    1124, 416, 1118, 105, 64, 1227, 741, 1418,
    303, 698, 592, 694, 1000, 45, 164, 671,
    19, 208, 1364, 756, 137, 1067, 971, 473,
)
HELDOUT_PAYLOAD_SHA256 = (
    "f9a3bbd244c2e60528a5a84749c75ae8495335caf73bffe79c38c94d50059dcd"
)
TRAINING_PREFIX_SHA256 = (
    "108b83ce2f2dd590c6ad45c7d46affeb4fd01afddee08dce35b2d5d18219876d"
)
RUNNER_BINDING_PATH = Path(__file__)
PASS_STATUS = "heldout_causal_gate_passed_generation_authorized"
FAIL_STATUS = "heldout_causal_gate_failed_generation_blocked"
TRAINABLE_CONFIGURER = recurrent_train.configure_trainable_parameters
REQUIRE_RECURRENT_SUBSET_CHANGED = True
TRAINING_FUNCTION = causal_train.train
MODEL_LOADER = screen.load_model


def canonical_sha256(value: Any) -> str:
    return distributed.canonical_sha256(value)


def sha256_file(path: Path) -> str:
    return preflight.sha256_file(path)


def write_json(path: Path, value: Mapping[str, Any]) -> None:
    if path.exists():
        raise ValueError(f"Top-2 causal training output must be fresh: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=True, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def validate_protocol() -> Mapping[str, Any]:
    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    receipt = protocol.get("receipt")
    if not isinstance(receipt, Mapping):
        raise ValueError("Top-2 causal training protocol receipt is missing")
    unsigned = dict(protocol)
    unsigned.pop("receipt")
    digest = canonical_sha256(unsigned)
    if digest != PROTOCOL_PAYLOAD_SHA256 or receipt.get("payload_sha256") != digest:
        raise ValueError("Top-2 causal training protocol payload differs")
    required = {
        "calibration_protocol_payload_sha256": calibration.PROTOCOL_PAYLOAD_SHA256,
        "calibration_result_file": (
            "local_artifacts/"
            "natural_memory_native_rwkv_top2_abstention_contrast_calibration_v1/"
            "result.json"
        ),
        "calibration_result_file_sha256": CALIBRATION_RESULT_FILE_SHA256,
        "calibration_result_receipt": CALIBRATION_RESULT_RECEIPT,
        "calibration_status": "calibration_passed_contrastive_training_authorized",
        "selected_candidate": SELECTED_CANDIDATE,
    }
    if protocol.get("authorization_basis") != required:
        raise ValueError("Top-2 causal training authorization differs")
    endpoint = protocol.get("heldout_causal_endpoint", {})
    if (
        endpoint.get("source_ordinals") != list(HELDOUT_ORDINALS)
        or endpoint.get("source_donor_payload_sha256") != HELDOUT_PAYLOAD_SHA256
    ):
        raise ValueError("Top-2 causal held-out endpoint differs")
    if protocol.get("protected_splits_opened_by_this_protocol") != []:
        raise ValueError("Top-2 causal training may not open protected data")
    return protocol


def validate_calibration_result() -> Mapping[str, Any]:
    if sha256_file(CALIBRATION_RESULT) != CALIBRATION_RESULT_FILE_SHA256:
        raise ValueError("Top-2 contrast calibration result file hash differs")
    result = json.loads(CALIBRATION_RESULT.read_text(encoding="utf-8"))
    receipt = result.get("receipt")
    if not isinstance(receipt, Mapping):
        raise ValueError("Top-2 contrast calibration receipt is missing")
    unsigned = dict(result)
    unsigned.pop("receipt")
    digest = canonical_sha256(unsigned)
    required = {
        "schema": calibration.SCHEMA,
        "status": "calibration_passed_contrastive_training_authorized",
        "passed": True,
        "contrastive_training_authorized": True,
        "protected_splits_opened": [],
    }
    if (
        digest != CALIBRATION_RESULT_RECEIPT
        or receipt.get("payload_sha256") != digest
        or any(result.get(name) != value for name, value in required.items())
    ):
        raise ValueError("Top-2 contrast calibration did not authorize training")
    return result


def heldout_payload(
    rows: Sequence[contrast.SceneContrastRow],
    donor_mapping: Mapping[int, int],
) -> list[dict[str, Any]]:
    return [
        {
            "source_ordinal": ordinal,
            "source_row_id": rows[ordinal].example.row_id,
            "donor_ordinal": int(donor_mapping[ordinal]),
            "donor_row_id": rows[int(donor_mapping[ordinal])].example.row_id,
        }
        for ordinal in HELDOUT_ORDINALS
    ]


def evaluate_heldout_causal_endpoint(
    model: torch.nn.Module,
    rows: Sequence[contrast.SceneContrastRow],
    donor_mapping: Mapping[int, int],
    *,
    context: distributed.DistributedTrainingContext,
    pad_token_id: int,
) -> Mapping[str, Any]:
    model.eval()
    modules = causal_train.ordered_modules(model)
    local_rows: list[dict[str, Any]] = []
    for endpoint_index, source_ordinal in enumerate(HELDOUT_ORDINALS):
        if endpoint_index % WORLD_SIZE != context.process_rank:
            continue
        donor_ordinal = int(donor_mapping[source_ordinal])
        target_batch = evolution.collate_native_examples(
            [rows[source_ordinal].example],
            pad_token_id=pad_token_id,
            device=context.device,
        )
        donor_batch = contrast.build_donor_batch(
            target_batch,
            rows[donor_ordinal].example,
            device=context.device,
        )
        with torch.no_grad():
            evolution._native_write(model, target_batch, dtype=torch.bfloat16)
            correct_state = causal_train.capture_online_state_references(modules)
            correct_logits = evolution._native_read(
                model,
                target_batch,
                dtype=torch.bfloat16,
            )
            evolution._native_write(model, donor_batch, dtype=torch.bfloat16)
            donor_state = causal_train.capture_online_state_references(modules)
            donor_carrier_fixed = causal_train.install_intervened_state(
                modules,
                projected=correct_state,
                recurrent=donor_state,
                rotate_recurrent_layers=False,
            )
            donor_logits = evolution._native_read(
                model,
                target_batch,
                dtype=torch.bfloat16,
            )
            permuted_carrier_fixed = causal_train.install_intervened_state(
                modules,
                projected=correct_state,
                recurrent=correct_state,
                rotate_recurrent_layers=True,
            )
            permuted_logits = evolution._native_read(
                model,
                target_batch,
                dtype=torch.bfloat16,
            )
            reset_delta_mem_states(model)
            zero_logits = evolution._native_read(
                model,
                target_batch,
                dtype=torch.bfloat16,
            )
        condition_logits = {
            "correct": correct_logits,
            "zero": zero_logits,
            "donor": donor_logits,
            "layer_permuted": permuted_logits,
        }
        ce: dict[str, float] = {}
        token_counts: set[int] = set()
        for name, logits in condition_logits.items():
            value, tokens = contrast.detached_answer_ce(logits, target_batch.labels)
            ce[name] = value
            token_counts.add(tokens)
        if len(token_counts) != 1:
            raise RuntimeError("Top-2 held-out condition token counts differ")
        tokens = token_counts.pop()
        local_rows.append(
            {
                "source_ordinal": source_ordinal,
                "donor_ordinal": donor_ordinal,
                "answer_target_tokens": tokens,
                "condition_mean_ce": ce,
                "zero_minus_correct_ce": ce["zero"] - ce["correct"],
                "donor_minus_correct_ce": ce["donor"] - ce["correct"],
                "layer_permuted_minus_correct_ce": (
                    ce["layer_permuted"] - ce["correct"]
                ),
                "projected_carrier_fixed": bool(
                    donor_carrier_fixed and permuted_carrier_fixed
                ),
                "all_condition_logits_finite": all(
                    bool(torch.isfinite(logits).all().item())
                    for logits in condition_logits.values()
                ),
            }
        )
        del target_batch, donor_batch, correct_logits, zero_logits, donor_logits
        del permuted_logits
        reset_delta_mem_states(model)
        evolution.release_native_row_allocator_cache(context.device)
    gathered = distributed.gather_objects(context, local_rows)
    all_rows = [row for rank_rows in gathered for row in rank_rows]
    all_rows.sort(key=lambda value: HELDOUT_ORDINALS.index(value["source_ordinal"]))
    if len(all_rows) != len(HELDOUT_ORDINALS):
        raise RuntimeError("Top-2 held-out endpoint row count differs")
    total_tokens = sum(row["answer_target_tokens"] for row in all_rows)
    condition_ce = {
        name: sum(
            row["condition_mean_ce"][name] * row["answer_target_tokens"]
            for row in all_rows
        )
        / total_tokens
        for name in ("correct", "zero", "donor", "layer_permuted")
    }
    margins = {
        "zero_minus_correct": condition_ce["zero"] - condition_ce["correct"],
        "donor_minus_correct": condition_ce["donor"] - condition_ce["correct"],
        "layer_permuted_minus_correct": (
            condition_ce["layer_permuted"] - condition_ce["correct"]
        ),
    }
    checks = {
        "rows_complete": len(all_rows) == len(HELDOUT_ORDINALS),
        "projected_carrier_fixed_every_row": all(
            row["projected_carrier_fixed"] for row in all_rows
        ),
        "all_condition_logits_finite": all(
            row["all_condition_logits_finite"] for row in all_rows
        ),
        "zero_minus_correct_mean_ce_positive": margins["zero_minus_correct"] > 0.0,
        "donor_minus_correct_mean_ce_positive": margins["donor_minus_correct"] > 0.0,
        "layer_permuted_minus_correct_mean_ce_positive": (
            margins["layer_permuted_minus_correct"] > 0.0
        ),
    }
    return {
        "rows": len(all_rows),
        "answer_target_tokens": total_tokens,
        "condition_mean_ce": condition_ce,
        "mean_ce_margins": margins,
        "positive_row_fractions": {
            name: sum(row[f"{name}_ce"] > 0.0 for row in all_rows) / len(all_rows)
            for name in (
                "zero_minus_correct",
                "donor_minus_correct",
                "layer_permuted_minus_correct",
            )
        },
        "checks": checks,
        "passed": all(checks.values()),
        "rank_rows": list(gathered),
    }


def run(
    *,
    context: distributed.DistributedTrainingContext,
    output_dir: Path,
    base_model: Path = BASE_MODEL,
    dataset_root: Path = DATASET_ROOT,
) -> Mapping[str, Any]:
    if context.world_size != WORLD_SIZE:
        raise ValueError("Top-2 causal training requires exactly four ranks")
    if os.environ.get("HF_ENDPOINT") != HF_MIRROR_ENDPOINT:
        raise ValueError(f"HF_ENDPOINT must be exactly {HF_MIRROR_ENDPOINT}")
    protocol = validate_protocol()
    calibration_result = validate_calibration_result()
    base_model = base_model.expanduser().resolve(strict=True)
    dataset_root = dataset_root.expanduser().resolve(strict=True)
    if sha256_file(base_model / "config.json") != preflight.EXPECTED_BASE_CONFIG_SHA256:
        raise ValueError("Top-2 causal training pinned base config differs")
    resolved_output = output_dir.expanduser().resolve()
    freshness_error: BaseException | None = None
    if context.is_primary and resolved_output.exists():
        freshness_error = ValueError(
            f"Top-2 causal training output must be fresh: {resolved_output}"
        )
    distributed.phase_consensus(
        context,
        phase="top2-causal-training-output-freshness",
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
        phase="top2-causal-training-output-creation",
        error=creation_error,
    )

    set_seed(SEED)
    model, tokenizer, model_audit = MODEL_LOADER(
        base_model,
        device=context.device,
        candidate=SELECTED_CANDIDATE,
    )
    screen_helper.configure_candidate(model, SELECTED_CANDIDATE)
    named_trainable, trainable_audit = TRAINABLE_CONFIGURER(model)
    rows = contrast.load_scene_rows(tokenizer, dataset_root)
    donor_mapping, donor_deltas, donor_payload = contrast.build_donor_mapping(rows)
    schedule, schedule_payload = contrast.build_schedule(
        rows,
        donor_mapping,
        donor_deltas,
    )
    endpoint_payload = heldout_payload(rows, donor_mapping)
    training_used = {
        ordinal
        for step in schedule[:UPDATES]
        for ordinal in (*step.source_ordinals, *step.donor_ordinals)
    }
    endpoint_disjoint = all(
        row["source_ordinal"] not in training_used
        and row["donor_ordinal"] not in training_used
        for row in endpoint_payload
    )
    if (
        canonical_sha256(schedule_payload) != contrast.FULL_SCHEDULE_SHA256
        or canonical_sha256(donor_payload) != contrast.DONOR_MAPPING_SHA256
        or canonical_sha256(schedule_payload[:UPDATES]) != TRAINING_PREFIX_SHA256
        or canonical_sha256(endpoint_payload) != HELDOUT_PAYLOAD_SHA256
        or not endpoint_disjoint
    ):
        raise RuntimeError("Top-2 causal training or endpoint binding differs")
    input_binding = {
        "schema": INPUT_SCHEMA,
        "protocol_payload_sha256": PROTOCOL_PAYLOAD_SHA256,
        "protocol_objective": protocol["objective"],
        "calibration_result_file_sha256": CALIBRATION_RESULT_FILE_SHA256,
        "calibration_result_receipt": CALIBRATION_RESULT_RECEIPT,
        "calibration_status": calibration_result["status"],
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
        "heldout_payload_sha256": canonical_sha256(endpoint_payload),
        "heldout_rows": len(endpoint_payload),
        "heldout_disjoint_from_training": endpoint_disjoint,
        "hf_endpoint": os.environ.get("HF_ENDPOINT"),
        "rank_devices": list(context.rank_devices),
        "model_audit": model_audit,
        "trainable_audit": trainable_audit,
        "runner_sha256": sha256_file(RUNNER_BINDING_PATH),
        "protected_splits_opened": [],
    }
    distributed.require_consensus(
        context,
        canonical_sha256(input_binding),
        description="top-2 causal training input binding",
    )
    binding_error: BaseException | None = None
    if context.is_primary:
        try:
            write_json(resolved_output / "input_binding.json", input_binding)
        except BaseException as error:
            binding_error = error
    distributed.phase_consensus(
        context,
        phase="top2-causal-training-input-binding-save",
        error=binding_error,
    )

    previous_step_schema = causal_train.STEP_SCHEMA
    previous_protocol_hash = causal_train.PROTOCOL_PAYLOAD_SHA256
    causal_train.STEP_SCHEMA = STEP_SCHEMA
    causal_train.PROTOCOL_PAYLOAD_SHA256 = PROTOCOL_PAYLOAD_SHA256
    try:
        training = TRAINING_FUNCTION(
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
    endpoint = evaluate_heldout_causal_endpoint(
        model,
        rows,
        donor_mapping,
        context=context,
        pad_token_id=int(tokenizer.pad_token_id),
    )
    rank_runtime = distributed.gather_objects(
        context,
        {
            "rank": context.process_rank,
            "peak_cuda_memory_bytes": int(
                torch.cuda.max_memory_allocated(context.device)
            ),
        },
    )
    training_passed = (
        addressed_screen.four_distinct_a100s(context.rank_devices)
        and trainable_audit["projected_router_frozen_tensors"]
        == preflight.EXPECTED_LAYERS
        and training["initial_adapter_sha256"] != training["final_adapter_sha256"]
        and training["trainable_subset_changed"] is True
        and training["recurrent_subset_changed"]
        is REQUIRE_RECURRENT_SUBSET_CHANGED
        and training["maximum_global_inactive_parameter_tensors"] == 0
        and training["projected_carrier_fixed_every_row"] is True
        and training["first_update_gradient_audit"]["passed"] is True
    )
    passed = training_passed and endpoint["passed"] is True
    result: dict[str, Any] = {}
    save_error: BaseException | None = None
    if context.is_primary:
        try:
            adapter_dir = resolved_output / "adapter"
            save_delta_mem_adapter(
                model,
                adapter_dir,
                screen.build_config(SELECTED_CANDIDATE),
            )
            adapter_files = contrast.gate.snapshot_directory_files(adapter_dir)
            result = {
                "schema": SCHEMA,
                "status": (
                    PASS_STATUS
                    if passed
                    else FAIL_STATUS
                ),
                "passed": passed,
                "training_passed": training_passed,
                "protocol_payload_sha256": PROTOCOL_PAYLOAD_SHA256,
                "seed": SEED,
                "updates": UPDATES,
                "input_binding": input_binding,
                "training": training,
                "heldout_causal_endpoint": endpoint,
                "adapter_files": adapter_files,
                "adapter_files_sha256": contrast.gate._sha256_json(adapter_files),
                "rank_runtime": list(rank_runtime),
                "open_native_generation_authorized": passed,
                "protected_splits_opened": [],
                "code_bindings": {
                    "runner_sha256": sha256_file(RUNNER_BINDING_PATH),
                    "shared_training_runner_sha256": sha256_file(Path(__file__)),
                    "training_helper_sha256": sha256_file(Path(causal_train.__file__)),
                    "protocol_file_sha256": sha256_file(PROTOCOL),
                    "calibration_result_file_sha256": sha256_file(
                        CALIBRATION_RESULT
                    ),
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
        phase="top2-causal-training-result-save",
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
        raise ValueError("Top-2 causal training requires four-rank torchrun")
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
