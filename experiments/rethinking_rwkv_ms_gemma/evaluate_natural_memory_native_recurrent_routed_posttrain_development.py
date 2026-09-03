#!/usr/bin/env python3
"""Evaluate the locked recurrent-routed candidate on fresh development rows."""

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


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[1]
for import_root in (SCRIPT_DIR, PROJECT_ROOT):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

from deltamem.core.delta import (  # noqa: E402
    load_delta_mem_adapter,
    reset_delta_mem_states,
)
from experiments.rethinking_rwkv_ms_gemma import (  # noqa: E402
    natural_memory_distributed as distributed,
)
from experiments.rethinking_rwkv_ms_gemma import (  # noqa: E402
    recurrent_routed_posttrain_common as common,
)
from experiments.rethinking_rwkv_ms_gemma import (  # noqa: E402
    run_natural_memory_native_evolution as evolution,
)
from experiments.rethinking_rwkv_ms_gemma import (  # noqa: E402
    run_natural_memory_native_scene_contrast_dropout as contrast,
)
from experiments.rethinking_rwkv_ms_gemma import (  # noqa: E402
    run_natural_memory_native_recurrent_routed_posttrain as training,
)
from experiments.rethinking_rwkv_ms_gemma import (  # noqa: E402
    run_synthetic_compositional_associative_retrieval_v3 as runtime,
)


SCHEMA = "rwkv_ms_natural_memory_native_recurrent_routed_development.v1"
ROW_SCHEMA = "rwkv_ms_natural_memory_native_recurrent_routed_development_row.v1"
INPUT_SCHEMA = "rwkv_ms_natural_memory_native_recurrent_routed_development_input.v1"
WORLD_SIZE = 4
TRAINING_RESULT_RECEIPT = (
    "2c83c06a323b08d150e8b91263fffe3c9fb9a69f28bbf398085a352c353bf85f"
)
TRAINING_ROOT = (
    SCRIPT_DIR / "local_artifacts/recurrent_routed_posttrain_train32_v4"
)
EXPECTED_ROWS = len(common.TASKS) * 32 * 4
TRAINED_SUFFIXES = (
    ".rwkv_route_query_proj",
    ".rwkv_route_state_proj",
    ".hrm_rwkv7_core.output.weight",
)


def write_fresh_json(path: Path, value: Mapping[str, Any]) -> None:
    if path.exists():
        raise ValueError(f"Development output must be fresh: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=True, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def validate_training_result(
    training_root: Path,
    *,
    expected_receipt: str = TRAINING_RESULT_RECEIPT,
    expected_status: str = "training_complete_development_evaluation_authorized",
    diagnostic_only: bool = False,
) -> Mapping[str, Any]:
    result = common.validate_signed_json(
        training_root / "result.json",
        expected_receipt,
    )
    adapter_dir = training_root / "adapter"
    adapter_files = contrast.gate.snapshot_directory_files(adapter_dir)
    expected_passed = not diagnostic_only
    if (
        result.get("status") != expected_status
        or result.get("passed") is not expected_passed
        or result.get("open_development_evaluation_authorized")
        is not expected_passed
        or result.get("final_rows_opened") is not False
        or contrast.gate._sha256_json(adapter_files)
        != result.get("adapter_files_sha256")
    ):
        raise ValueError("Recurrent-routed training result does not authorize development")
    return result


def choose_donor(
    target: common.SourceRow,
    task_rows: Sequence[common.SourceRow],
) -> common.SourceRow:
    candidates = [
        row
        for row in task_rows
        if row.source_ordinal != target.source_ordinal
        and row.assistant_identity != target.assistant_identity
    ]
    if not candidates:
        raise ValueError(f"Development row has no different-answer donor: {target}")
    return min(
        candidates,
        key=lambda row: (
            abs(row.user_characters - target.user_characters),
            row.row_sha256,
            row.source_ordinal,
        ),
    )


def build_schedule(
    rows_by_task: Mapping[str, Sequence[common.SourceRow]],
) -> tuple[tuple[tuple[common.SourceRow, common.SourceRow, int], ...], list[dict[str, Any]]]:
    schedule = []
    payload = []
    for task in common.TASKS:
        rows = sorted(rows_by_task[task], key=lambda row: row.source_ordinal)
        if len(rows) != 32:
            raise ValueError(f"Development row count differs for {task}")
        for target in rows:
            donor = choose_donor(target, rows_by_task[task])
            for variant in range(4):
                schedule.append((target, donor, variant))
                payload.append(
                    {
                        "task": task,
                        "source_ordinal": target.source_ordinal,
                        "source_row_sha256": target.row_sha256,
                        "donor_source_ordinal": donor.source_ordinal,
                        "donor_row_sha256": donor.row_sha256,
                        "prompt_variant": variant,
                    }
                )
    if len(schedule) != EXPECTED_ROWS:
        raise RuntimeError("Development schedule size differs")
    return tuple(schedule), payload


def preserve_trained_precision(model: torch.nn.Module) -> None:
    selected = []
    for name, parameter in model.named_parameters():
        if not name.endswith(TRAINED_SUFFIXES):
            continue
        parameter.data = parameter.data.float()
        parameter.requires_grad_(False)
        selected.append(name)
    if len(selected) != common.EXPECTED_LAYERS * len(TRAINED_SUFFIXES):
        raise RuntimeError("Development trained-precision parameter count differs")


def projected_carrier_sha256(model: torch.nn.Module) -> str:
    state = {}
    for name, module in common.ordered_modules(model):
        for attribute in common.PROJECTED_ATTRIBUTES:
            tensor = getattr(module, attribute)
            if tensor is None:
                raise RuntimeError(f"Development carrier is missing: {name}.{attribute}")
            state[f"{name}.{attribute}"] = tensor.detach().cpu().clone()
    return runtime._state_dict_sha256(state)


def evaluate_row(
    model: torch.nn.Module,
    tokenizer: Any,
    target_row: common.SourceRow,
    donor_row: common.SourceRow,
    prompt_variant: int,
    *,
    device: torch.device,
) -> Mapping[str, Any]:
    target_example = common.encode_row(
        tokenizer,
        target_row,
        prompt_variant=prompt_variant,
    )
    donor_example = common.encode_row(
        tokenizer,
        donor_row,
        prompt_variant=prompt_variant,
    )
    target = evolution.collate_native_examples(
        [target_example],
        pad_token_id=int(tokenizer.pad_token_id),
        device=device,
    )
    donor_write = evolution.collate_native_examples(
        [donor_example],
        pad_token_id=int(tokenizer.pad_token_id),
        device=device,
    )
    donor = evolution.NativeFullRowBatch(
        examples=target.examples,
        write_input_ids=donor_write.write_input_ids,
        write_attention_mask=donor_write.write_attention_mask,
        read_input_ids=target.read_input_ids,
        read_attention_mask=target.read_attention_mask,
        labels=target.labels,
    )
    condition_ce = {}
    condition_tokens = {}
    carrier_hashes = {}
    audits = {}
    try:
        with torch.inference_mode():
            for condition in common.CONDITIONS:
                logits, audit = common.direct_condition_logits(
                    model,
                    target,
                    condition=condition,
                    donor=(
                        donor
                        if condition == "matched_donor_recurrent_state"
                        else None
                    ),
                    dtype=torch.bfloat16,
                )
                ce, tokens = contrast.detached_answer_ce(logits, target.labels)
                condition_ce[condition] = ce
                condition_tokens[condition] = tokens
                audits[condition] = dict(audit)
                carrier_hashes[condition] = projected_carrier_sha256(model)
                del logits
                reset_delta_mem_states(model)
                evolution.release_native_row_allocator_cache(device)
        if len(set(condition_tokens.values())) != 1:
            raise RuntimeError("Development condition token counts differ")
        correct = condition_ce["correct_recurrent_state"]
        margins = {
            condition: condition_ce[condition] - correct
            for condition in common.CONDITIONS[1:]
        }
        carrier_fixed = (
            len(set(carrier_hashes.values())) == 1
            and all(
                audit["projected_carrier_references_fixed"]
                and audit["projected_carrier_bytes_fixed"]
                for audit in audits.values()
            )
        )
        return {
            "schema": ROW_SCHEMA,
            "task": target_row.task,
            "source_ordinal": target_row.source_ordinal,
            "source_row_sha256": target_row.row_sha256,
            "donor_source_ordinal": donor_row.source_ordinal,
            "donor_row_sha256": donor_row.row_sha256,
            "prompt_variant": prompt_variant,
            "answer_tokens": next(iter(condition_tokens.values())),
            "condition_ce": condition_ce,
            "control_minus_correct_ce": margins,
            "projected_carrier_sha256": carrier_hashes,
            "projected_carrier_fixed": carrier_fixed,
            "condition_audits": audits,
        }
    finally:
        del target, donor, donor_write, target_example, donor_example
        reset_delta_mem_states(model)
        evolution.release_native_row_allocator_cache(device)


def mean_metrics(rows: Sequence[Mapping[str, Any]]) -> Mapping[str, Any]:
    count = len(rows)
    if count <= 0:
        raise ValueError("Development metric group is empty")
    return {
        "rows": count,
        "mean_condition_ce": {
            condition: sum(float(row["condition_ce"][condition]) for row in rows)
            / count
            for condition in common.CONDITIONS
        },
        "mean_control_minus_correct_ce": {
            condition: sum(
                float(row["control_minus_correct_ce"][condition]) for row in rows
            )
            / count
            for condition in common.CONDITIONS[1:]
        },
    }


def summarize(rows: Sequence[Mapping[str, Any]]) -> Mapping[str, Any]:
    ordered = sorted(
        rows,
        key=lambda row: (
            common.TASKS.index(str(row["task"])),
            int(row["source_ordinal"]),
            int(row["prompt_variant"]),
        ),
    )
    overall = mean_metrics(ordered)
    by_task = {
        task: mean_metrics([row for row in ordered if row["task"] == task])
        for task in common.TASKS
    }
    by_task_variant = {
        f"{task}:{variant}": mean_metrics(
            [
                row
                for row in ordered
                if row["task"] == task and row["prompt_variant"] == variant
            ]
        )
        for task in common.TASKS
        for variant in range(4)
    }
    controls = common.CONDITIONS[1:]
    overall_pass = all(
        overall["mean_control_minus_correct_ce"][condition] > 0.0
        for condition in controls
    )
    per_task_pass = all(
        by_task[task]["mean_control_minus_correct_ce"][condition] > 0.0
        for task in common.TASKS
        for condition in controls
    )
    variant_donor_pass = all(
        metrics["mean_control_minus_correct_ce"][
            "matched_donor_recurrent_state"
        ]
        > 0.0
        for metrics in by_task_variant.values()
    )
    carrier_pass = all(row["projected_carrier_fixed"] is True for row in ordered)
    row_count_pass = (
        len(ordered) == EXPECTED_ROWS
        and all(by_task[task]["rows"] == 128 for task in common.TASKS)
        and all(metrics["rows"] == 32 for metrics in by_task_variant.values())
    )
    passed = bool(
        overall_pass
        and per_task_pass
        and variant_donor_pass
        and carrier_pass
        and row_count_pass
    )
    return {
        "rows": len(ordered),
        "overall": overall,
        "by_task": by_task,
        "by_task_prompt_variant": by_task_variant,
        "causal_criteria": {
            "overall_correct_over_all_controls": overall_pass,
            "per_task_correct_over_all_controls": per_task_pass,
            "all_task_prompt_variants_correct_over_donor": variant_donor_pass,
            "projected_carriers_fixed": carrier_pass,
            "row_counts_exact": row_count_pass,
        },
        "passed": passed,
    }


def run(
    *,
    context: distributed.DistributedTrainingContext,
    output_dir: Path,
    training_root: Path,
    base_model: Path,
    training_result_receipt: str = TRAINING_RESULT_RECEIPT,
    training_status: str = "training_complete_development_evaluation_authorized",
    protocol_file: Path = common.PROTOCOL,
    protocol_receipt: str = common.PROTOCOL_PAYLOAD_SHA256,
    diagnostic_only: bool = False,
) -> Mapping[str, Any]:
    if context.world_size != WORLD_SIZE:
        raise ValueError("Development evaluation requires exactly four ranks")
    if os.environ.get("HF_ENDPOINT") != common.HF_MIRROR_ENDPOINT:
        raise ValueError(f"HF_ENDPOINT must be exactly {common.HF_MIRROR_ENDPOINT}")
    protocol = common.validate_signed_json(protocol_file, protocol_receipt)
    manifest, open_receipt = common.validate_split_artifacts()
    training_result = validate_training_result(
        training_root,
        expected_receipt=training_result_receipt,
        expected_status=training_status,
        diagnostic_only=diagnostic_only,
    )
    rows_by_task = common.load_open_rows("development", manifest=manifest)
    schedule, schedule_payload = build_schedule(rows_by_task)

    resolved_output = output_dir.expanduser().resolve()
    creation_error = None
    if context.is_primary:
        try:
            resolved_output.mkdir(parents=True, exist_ok=False)
        except BaseException as error:
            creation_error = error
    distributed.phase_consensus(
        context,
        phase="recurrent-routed-development-output-creation",
        error=creation_error,
    )

    model, tokenizer, delta_config, model_audit = common.load_model(
        base_model,
        device=context.device,
        trainable=False,
    )
    preserve_trained_precision(model)
    loaded_config = load_delta_mem_adapter(model, training_root / "adapter")
    if loaded_config.to_dict() != delta_config.to_dict():
        raise ValueError("Development candidate adapter configuration differs")
    model.eval()

    input_binding = {
        "schema": INPUT_SCHEMA,
        "protocol_payload_sha256": protocol_receipt,
        "protocol_objective": protocol["objective"],
        "training_result_receipt": training_result_receipt,
        "training_adapter_files_sha256": training_result["adapter_files_sha256"],
        "base_model": str(base_model.expanduser().resolve()),
        "base_model_revision": common.BASE_MODEL_REVISION,
        "split_manifest_receipt": common.SPLIT_MANIFEST_RECEIPT,
        "open_split_receipt": common.OPEN_SPLIT_RECEIPT,
        "development_files": {
            path: value
            for path, value in open_receipt["files"].items()
            if path.endswith("/development.jsonl")
        },
        "schedule_sha256": common.canonical_sha256(schedule_payload),
        "rows": EXPECTED_ROWS,
        "conditions": list(common.CONDITIONS),
        "prompt_variants": 4,
        "world_size": context.world_size,
        "rank_devices": list(context.rank_devices),
        "trained_parameter_suffixes": list(TRAINED_SUFFIXES),
        "model_audit": model_audit,
        "task_router": False,
        "template_matcher": False,
        "dual_pass_selector": False,
        "benchmark_specific_decoder": False,
        "baseline_fallback": False,
        "benchmark_time_parameter_override": False,
        "diagnostic_only": diagnostic_only,
        "evaluator_sha256": common.sha256_file(Path(__file__)),
        "common_helper_sha256": common.sha256_file(Path(common.__file__)),
        "final_rows_opened": False,
        "publisher_validation_opened": False,
        "publisher_test_opened": False,
    }
    distributed.require_consensus(
        context,
        common.canonical_sha256(input_binding),
        description="recurrent-routed development input binding",
    )
    if context.is_primary:
        write_fresh_json(resolved_output / "input_binding.json", input_binding)

    local_rows = []
    for index, (target, donor, variant) in enumerate(schedule):
        if index % context.world_size != context.process_rank:
            continue
        local_rows.append(
            evaluate_row(
                model,
                tokenizer,
                target,
                donor,
                variant,
                device=context.device,
            )
        )
        if context.is_primary and len(local_rows) % 8 == 0:
            print(
                json.dumps(
                    {
                        "development_local_rows": len(local_rows),
                        "development_total_local_rows": EXPECTED_ROWS // WORLD_SIZE,
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
    gathered = distributed.gather_objects(context, local_rows)
    result = {}
    save_error = None
    if context.is_primary:
        try:
            all_rows = [row for rank_rows in gathered for row in rank_rows]
            all_rows.sort(
                key=lambda row: (
                    common.TASKS.index(str(row["task"])),
                    int(row["source_ordinal"]),
                    int(row["prompt_variant"]),
                )
            )
            records_path = resolved_output / "records.jsonl"
            if records_path.exists():
                raise ValueError("Development records must be fresh")
            with records_path.open("w", encoding="utf-8") as handle:
                for row in all_rows:
                    handle.write(
                        json.dumps(
                            row,
                            ensure_ascii=True,
                            sort_keys=True,
                            separators=(",", ":"),
                        )
                        + "\n"
                    )
            summary = summarize(all_rows)
            causal_criteria_passed = summary["passed"]
            passed = bool(causal_criteria_passed and not diagnostic_only)
            result = {
                "schema": SCHEMA,
                "status": (
                    "development_diagnostic_complete_final_evaluation_blocked"
                    if diagnostic_only
                    else "development_passed_final_evaluation_authorized"
                    if passed
                    else "development_failed_final_evaluation_blocked"
                ),
                "passed": passed,
                "protocol_payload_sha256": protocol_receipt,
                "input_binding": input_binding,
                "summary": summary,
                "diagnostic_causal_criteria_passed": causal_criteria_passed,
                "records_sha256": common.sha256_file(records_path),
                "final_evaluation_authorized": False if diagnostic_only else passed,
                "final_rows_opened": False,
                "publisher_validation_opened": False,
                "publisher_test_opened": False,
            }
            result["receipt"] = {
                "algorithm": "sha256",
                "payload_scope": "canonical_result_without_receipt",
                "payload_sha256": common.canonical_sha256(result),
            }
            write_fresh_json(resolved_output / "result.json", result)
        except BaseException as error:
            save_error = error
    distributed.phase_consensus(
        context,
        phase="recurrent-routed-development-result-save",
        error=save_error,
    )
    pass_values = distributed.gather_objects(
        context,
        result["passed"] if context.is_primary else None,
    )
    passed = bool(pass_values[0])
    del model, tokenizer, rows_by_task
    gc.collect()
    torch.cuda.empty_cache()
    return result if context.is_primary else {
        "status": "worker_complete",
        "passed": bool(passed),
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--training-root", type=Path, default=TRAINING_ROOT)
    parser.add_argument(
        "--training-result-receipt",
        default=TRAINING_RESULT_RECEIPT,
    )
    parser.add_argument(
        "--training-status",
        default="training_complete_development_evaluation_authorized",
    )
    parser.add_argument("--protocol-file", type=Path, default=common.PROTOCOL)
    parser.add_argument(
        "--protocol-receipt",
        default=common.PROTOCOL_PAYLOAD_SHA256,
    )
    parser.add_argument("--base-model", type=Path, default=common.BASE_MODEL)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--diagnostic-only", action="store_true")
    return parser.parse_args(argv)


@record
def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    context = distributed.initialize_distributed_training(args.device)
    if context is None:
        raise ValueError("Development evaluation requires four-rank torchrun")
    try:
        result = run(
            context=context,
            output_dir=args.output_dir,
            training_root=args.training_root.expanduser().resolve(strict=True),
            base_model=args.base_model,
            training_result_receipt=args.training_result_receipt,
            training_status=args.training_status,
            protocol_file=args.protocol_file.expanduser().resolve(strict=True),
            protocol_receipt=args.protocol_receipt,
            diagnostic_only=args.diagnostic_only,
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
                    result.get("receipt", {}).get("payload_sha256")
                    if context.is_primary
                    else None
                ),
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
