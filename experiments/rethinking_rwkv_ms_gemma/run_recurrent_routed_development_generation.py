#!/usr/bin/env python3
"""Run signed free generation on the locked development-v2 rows."""

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
PROJECT_ROOT = SCRIPT_DIR.parents[1]
for import_root in (SCRIPT_DIR, PROJECT_ROOT):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

from experiments.rethinking_rwkv_ms_gemma import (  # noqa: E402
    natural_memory_distributed as distributed,
)
from experiments.rethinking_rwkv_ms_gemma import (  # noqa: E402
    recurrent_routed_posttrain_common as common,
)
from experiments.rethinking_rwkv_ms_gemma import (  # noqa: E402
    run_novel_agent_eval as generation,
)
from experiments.rethinking_rwkv_ms_gemma import (  # noqa: E402
    run_recurrent_routed_final as final,
)
from deltamem.core.delta import load_delta_mem_adapter  # noqa: E402


SCHEMA = "rwkv_ms_recurrent_routed_development_generation.v2"
INPUT_SCHEMA = "rwkv_ms_recurrent_routed_development_generation_input.v2"
V2_ROOT = SCRIPT_DIR / "local_artifacts/recurrent_routed_posttrain_development_v2"
V2_MANIFEST_RECEIPT = (
    "2236d1e3e980ce92787e34500a40a38634ea7017835e629759d9564ba99036d6"
)
DEFAULT_OUTPUT = (
    SCRIPT_DIR / "local_artifacts/recurrent_routed_development_generation_v2"
)
SYSTEMS = final.SYSTEMS
TASKS = final.TASKS
TASK_METRICS = {
    "attribution": "accuracy",
    "narrative": "unit_label_accuracy",
    "scene": "micro_f1",
}


def load_v2_manifest() -> Mapping[str, Any]:
    value = json.loads((V2_ROOT / "manifest.json").read_text(encoding="utf-8"))
    receipt = value.pop("receipt", None)
    if (
        not isinstance(receipt, Mapping)
        or receipt.get("payload_sha256") != V2_MANIFEST_RECEIPT
        or common.canonical_sha256(value) != V2_MANIFEST_RECEIPT
        or value.get("final_rows_opened") is not False
    ):
        raise ValueError("Development-v2 manifest differs")
    value["receipt"] = receipt
    return value


def read_v2_rows() -> dict[str, list[dict[str, Any]]]:
    split_manifest = common.validate_signed_json(
        common.SPLIT_ROOT / "manifest.json",
        common.SPLIT_MANIFEST_RECEIPT,
    )
    source_rows = common.load_open_rows("train", manifest=split_manifest)
    v2_manifest = load_v2_manifest()
    rows: dict[str, list[dict[str, Any]]] = {}
    for task in TASKS:
        wanted = tuple(
            int(value)
            for value in v2_manifest["development_source_ordinals"][task]
        )
        lookup = {row.source_ordinal: row for row in source_rows[task]}
        if any(source_ordinal not in lookup for source_ordinal in wanted):
            raise ValueError(f"Development-v2 row identity differs for {task}")
        converted: list[dict[str, Any]] = []
        for source_ordinal in wanted:
            source = lookup[source_ordinal]
            for prompt_variant in range(len(common.PROMPT_VARIANTS[task])):
                raw_line = common.paraphrased_raw_line(source, prompt_variant)
                value = json.loads(raw_line)
                gold = generation.extract_json(
                    str(value["messages"][-1]["content"])
                )
                if gold is None:
                    raise ValueError(
                        "Development-v2 gold JSON is invalid for "
                        f"{task}:{source_ordinal}:{prompt_variant}"
                    )
                converted.append(
                    {
                        "task": task,
                        "line_index": len(converted),
                        "source_ordinal": source_ordinal,
                        "source_row_sha256": source.row_sha256,
                        "row_sha256": hashlib.sha256(
                            raw_line.encode("utf-8")
                        ).hexdigest(),
                        "prompt_variant": prompt_variant,
                        "messages": value["messages"][:-1],
                        "gold": gold,
                    }
                )
        expected = len(wanted) * len(common.PROMPT_VARIANTS[task])
        if len(converted) != expected:
            raise RuntimeError(
                f"Development-v2 expanded row count differs for {task}"
            )
        rows[task] = converted
    return rows


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=True, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line
    ]


def append_jsonl(path: Path, row: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(
            json.dumps(
                row,
                ensure_ascii=True,
                sort_keys=True,
                separators=(",", ":"),
            )
            + "\n"
        )
        handle.flush()
        os.fsync(handle.fileno())


def write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(
                json.dumps(
                    row,
                    ensure_ascii=True,
                    sort_keys=True,
                    separators=(",", ":"),
                )
                + "\n"
            )


def _summarize_single_task(
    task: str,
    rows: Sequence[Mapping[str, Any]],
) -> Mapping[str, Any]:
    if task == "attribution":
        correct = sum(bool(row["score"]["correct"]) for row in rows)
        return {
            "rows": len(rows),
            "correct": correct,
            "accuracy": 0.0 if not rows else correct / len(rows),
            "uncertain_accuracy": (
                0.0
                if not rows
                else sum(
                    bool(row["score"].get("uncertain_correct", False))
                    for row in rows
                )
                / len(rows)
            ),
            "joint_accuracy": (
                0.0
                if not rows
                else sum(
                    bool(row["score"].get("joint_correct", False))
                    for row in rows
                )
                / len(rows)
            ),
            "recovered_accuracy_diagnostic": (
                0.0
                if not rows
                else sum(
                    bool(row["recovered_score"]["correct"])
                    for row in rows
                )
                / len(rows)
            ),
            "recovery_covered_diagnostic": sum(
                bool(row["recovered_score"]["covered"]) for row in rows
            ),
            "strict_schema_valid": sum(
                bool(row["score"]["schema_valid"]) for row in rows
            ),
        }
    if task == "narrative":
        correct = sum(
            int(row["score"]["correct_units"]) for row in rows
        )
        units = sum(int(row["score"]["gold_units"]) for row in rows)
        return {
            "rows": len(rows),
            "correct_units": correct,
            "gold_units": units,
            "unit_label_accuracy": 0.0 if units == 0 else correct / units,
            "recovered_unit_label_accuracy_diagnostic": (
                0.0
                if units == 0
                else sum(
                    int(row["recovered_score"]["correct_units"])
                    for row in rows
                )
                / units
            ),
            "recovery_covered_diagnostic": sum(
                bool(row["recovered_score"]["covered"]) for row in rows
            ),
            "strict_schema_valid": sum(
                bool(row["score"]["schema_valid"]) for row in rows
            ),
        }
    true_positive = sum(int(row["score"]["tp"]) for row in rows)
    false_positive = sum(int(row["score"]["fp"]) for row in rows)
    false_negative = sum(int(row["score"]["fn"]) for row in rows)
    recovered_true_positive = sum(
        int(row["recovered_score"]["tp"]) for row in rows
    )
    recovered_false_positive = sum(
        int(row["recovered_score"]["fp"]) for row in rows
    )
    recovered_false_negative = sum(
        int(row["recovered_score"]["fn"]) for row in rows
    )
    return {
        "rows": len(rows),
        "tp": true_positive,
        "fp": false_positive,
        "fn": false_negative,
        "micro_f1": generation.f1_from_counts(
            true_positive,
            false_positive,
            false_negative,
        ),
        "recovered_micro_f1_diagnostic": generation.f1_from_counts(
            recovered_true_positive,
            recovered_false_positive,
            recovered_false_negative,
        ),
        "recovery_covered_diagnostic": sum(
            bool(row["recovered_score"]["covered"]) for row in rows
        ),
        "strict_schema_valid": sum(
            bool(row["score"]["schema_valid"]) for row in rows
        ),
    }


def summarize(records: Sequence[Mapping[str, Any]]) -> Mapping[str, Any]:
    by_task: dict[str, Mapping[str, Any]] = {}
    by_task_prompt_variant: dict[str, Mapping[str, Any]] = {}
    for task in TASKS:
        task_rows = [row for row in records if row["task"] == task]
        by_task[task] = _summarize_single_task(task, task_rows)
        for prompt_variant in range(len(common.PROMPT_VARIANTS[task])):
            variant_rows = [
                row
                for row in task_rows
                if int(row["prompt_variant"]) == prompt_variant
            ]
            by_task_prompt_variant[f"{task}:{prompt_variant}"] = (
                _summarize_single_task(task, variant_rows)
            )
    return {
        "rows": len(records),
        "by_task": by_task,
        "by_task_prompt_variant": by_task_prompt_variant,
    }


def evaluate_promotion_criteria(
    systems: Mapping[str, Mapping[str, Any]],
    expected_rows: Mapping[str, int],
) -> Mapping[str, Any]:
    if set(systems) != set(SYSTEMS):
        return {"passed": False, "status": "all_three_systems_required"}
    metrics = {
        system: {
            task: float(summary["by_task"][task][TASK_METRICS[task]])
            for task in TASKS
        }
        for system, summary in systems.items()
    }
    strict_schema_rates = {
        system: {
            task: (
                float(summary["by_task"][task]["strict_schema_valid"])
                / float(expected_rows[task])
            )
            for task in TASKS
        }
        for system, summary in systems.items()
    }
    candidate = metrics["recurrent_routed_candidate"]
    stronger = {
        task: max(
            metrics["frozen_gemma_base"][task],
            metrics["v9_projected_slot_baseline"][task],
        )
        for task in TASKS
    }
    semantic_non_regression = all(
        candidate[task] >= stronger[task] for task in TASKS
    )
    strict_improvement = any(
        candidate[task] > metrics["frozen_gemma_base"][task]
        and candidate[task] > metrics["v9_projected_slot_baseline"][task]
        for task in TASKS
    )
    schema_non_regression = all(
        strict_schema_rates["recurrent_routed_candidate"][task]
        >= max(
            strict_schema_rates["frozen_gemma_base"][task],
            strict_schema_rates["v9_projected_slot_baseline"][task],
        )
        for task in TASKS
    )
    row_counts_exact = all(
        summary["rows"] == sum(expected_rows.values())
        and all(
            int(summary["by_task"][task]["rows"]) == expected_rows[task]
            for task in TASKS
        )
        for summary in systems.values()
    )
    passed = bool(
        semantic_non_regression
        and strict_improvement
        and schema_non_regression
        and row_counts_exact
    )
    return {
        "metrics": metrics,
        "strict_schema_rates": strict_schema_rates,
        "candidate_at_least_stronger_baseline_every_task": semantic_non_regression,
        "candidate_strictly_better_than_both_one_task": strict_improvement,
        "candidate_schema_not_worse_than_both_every_task": schema_non_regression,
        "row_counts_exact": row_counts_exact,
        "passed": passed,
    }


def adapter_binding(path: Path) -> Mapping[str, Any]:
    resolved = path.expanduser().resolve(strict=True)
    return {
        "path": str(resolved),
        "weights_sha256": common.sha256_file(
            resolved / "delta_mem_adapter.pt"
        ),
        "config_sha256": common.sha256_file(
            resolved / "delta_mem_config.json"
        ),
    }


def preserve_signed_candidate_precision(
    model: torch.nn.Module,
    adapter: Path,
    training_receipt: str,
) -> Mapping[str, Any]:
    training_result = common.validate_signed_json(
        adapter.parent / "result.json",
        training_receipt,
    )
    trainable_audit = training_result.get("input_binding", {}).get(
        "model_audit",
        {},
    ).get("trainables", {})
    suffixes = tuple(trainable_audit.get("trainable_parameter_suffixes", ()))
    expected_tensors = int(trainable_audit.get("parameter_tensors", 0))
    if not suffixes or expected_tensors <= 0:
        raise ValueError("Signed candidate trainable precision binding is missing")
    selected = []
    for name, parameter in model.named_parameters():
        if not name.endswith(suffixes):
            continue
        parameter.data = parameter.data.float()
        parameter.requires_grad_(False)
        selected.append(name)
    if len(selected) != expected_tensors:
        raise ValueError(
            "Signed candidate trainable precision tensor count differs: "
            f"expected={expected_tensors} actual={len(selected)}"
        )
    loaded_config = load_delta_mem_adapter(model, adapter)
    return {
        "parameter_tensors": len(selected),
        "parameter_suffixes": list(suffixes),
        "parameter_names_sha256": common.canonical_sha256(selected),
        "dtype": "float32",
        "weights_reloaded_after_promotion": True,
        "adapter_config_sha256": common.sha256_file(
            adapter / "delta_mem_config.json"
        ),
        "loaded_hybrid_mode": loaded_config.rwkv_ms_hybrid_mode,
        "loaded_hybrid_gain": loaded_config.rwkv_ms_hybrid_gain,
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--base-model", type=Path, default=final.BASE_MODEL)
    parser.add_argument("--v9-adapter", type=Path, default=final.V9_ADAPTER)
    parser.add_argument("--candidate-adapter", type=Path, required=True)
    parser.add_argument("--candidate-training-receipt", required=True)
    parser.add_argument("--candidate-protocol-receipt", required=True)
    parser.add_argument("--systems", default=",".join(SYSTEMS))
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--device", default="cuda")
    return parser.parse_args(argv)


@record
def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if os.environ.get("HF_ENDPOINT") != common.HF_MIRROR_ENDPOINT:
        raise ValueError(
            f"HF_ENDPOINT must be exactly {common.HF_MIRROR_ENDPOINT}"
        )
    systems = tuple(
        value.strip() for value in args.systems.split(",") if value.strip()
    )
    if (
        not systems
        or any(value not in SYSTEMS for value in systems)
        or len(set(systems)) != len(systems)
    ):
        raise ValueError(f"Unknown or repeated systems: {systems}")
    context = distributed.initialize_distributed_training(
        args.device,
        timeout_seconds=7200,
    )
    if context is None or context.world_size != 4:
        raise ValueError("Development generation requires exactly four ranks")
    passed = False
    try:
        rows_by_task = read_v2_rows()
        expected_rows = {task: len(rows_by_task[task]) for task in TASKS}
        output = args.output_dir.expanduser().resolve()
        creation_error = None
        if context.is_primary:
            try:
                if args.resume:
                    if not output.is_dir():
                        raise ValueError(
                            f"Resume output does not exist: {output}"
                        )
                else:
                    output.mkdir(parents=True, exist_ok=False)
            except BaseException as error:
                creation_error = error
        distributed.phase_consensus(
            context,
            phase="development-generation-output",
            error=creation_error,
        )
        summaries: dict[str, Mapping[str, Any]] = {}
        all_records: dict[str, list[dict[str, Any]]] = {}
        candidate_precision_audit = None
        for system in systems:
            base_model, adapter = final.model_paths(
                system,
                args.base_model.expanduser().resolve(strict=True),
                args.v9_adapter.expanduser().resolve(strict=True),
                args.candidate_adapter.expanduser().resolve(strict=True),
            )
            model, tokenizer = final.load_model(
                base_model,
                adapter,
                str(context.device),
            )
            if system == "recurrent_routed_candidate":
                candidate_precision_audit = preserve_signed_candidate_precision(
                    model,
                    args.candidate_adapter.expanduser().resolve(strict=True),
                    args.candidate_training_receipt,
                )
            model.eval()
            shard_path = (
                output
                / f"shard-{context.process_rank}"
                / f"{system}.jsonl"
            )
            local_records = (
                read_jsonl(shard_path) if shard_path.is_file() else []
            )
            completed = {
                (str(row["task"]), int(row["line_index"]))
                for row in local_records
            }
            if len(completed) != len(local_records):
                raise ValueError(f"Duplicate development rows in {shard_path}")
            flat_rows = [
                (task, row) for task in TASKS for row in rows_by_task[task]
            ]
            for index, (task, row) in enumerate(flat_rows):
                if index % context.world_size != context.process_rank:
                    continue
                key = (task, int(row["line_index"]))
                if key in completed:
                    continue
                evaluated = dict(
                    final.evaluate_row(
                        model,
                        tokenizer,
                        task,
                        row,
                        str(context.device),
                        system,
                    )
                )
                evaluated["prompt_variant"] = int(row["prompt_variant"])
                evaluated["source_row_sha256"] = row[
                    "source_row_sha256"
                ]
                local_records.append(evaluated)
                append_jsonl(shard_path, evaluated)
                if len(local_records) % 4 == 0:
                    print(
                        json.dumps(
                            {
                                "system": system,
                                "rank": context.process_rank,
                                "local_rows": len(local_records),
                            },
                            sort_keys=True,
                        ),
                        flush=True,
                    )
            torch.distributed.barrier(group=context.control_group)
            if context.is_primary:
                records: list[dict[str, Any]] = []
                for rank in range(context.world_size):
                    records.extend(
                        read_jsonl(
                            output
                            / f"shard-{rank}"
                            / f"{system}.jsonl"
                        )
                    )
                records.sort(
                    key=lambda row: (
                        TASKS.index(str(row["task"])),
                        int(row["line_index"]),
                    )
                )
                write_jsonl(output / f"{system}.jsonl", records)
                all_records[system] = records
                summaries[system] = summarize(records)
            del model, tokenizer
            gc.collect()
            torch.cuda.empty_cache()
            distributed.require_consensus(
                context,
                True,
                description=f"completed development generation {system}",
            )
        if context.is_primary:
            promotion_criteria = evaluate_promotion_criteria(
                summaries,
                expected_rows,
            )
            passed = promotion_criteria.get("passed") is True
            input_binding = {
                "schema": INPUT_SCHEMA,
                "development_manifest": str(
                    (V2_ROOT / "manifest.json").resolve()
                ),
                "development_manifest_receipt": V2_MANIFEST_RECEIPT,
                "rows_per_task": expected_rows,
                "prompt_variants_sha256": common.canonical_sha256(
                    common.PROMPT_VARIANTS
                ),
                "base_model": str(args.base_model.expanduser().resolve()),
                "base_model_revision": common.BASE_MODEL_REVISION,
                "base_model_weights_sha256": (
                    common.BASE_MODEL_WEIGHTS_SHA256
                ),
                "v9_adapter": adapter_binding(args.v9_adapter),
                "candidate_adapter": adapter_binding(
                    args.candidate_adapter
                ),
                "candidate_training_receipt": (
                    args.candidate_training_receipt
                ),
                "candidate_protocol_receipt": (
                    args.candidate_protocol_receipt
                ),
                "candidate_configuration_source": "signed_adapter_checkpoint",
                "candidate_precision_audit": candidate_precision_audit,
                "systems": list(systems),
                "world_size": context.world_size,
                "rank_devices": list(context.rank_devices),
                "generation": (
                    "single-pass greedy batch-one with one write then one "
                    "read for memory systems"
                ),
                "task_router": False,
                "template_matcher": False,
                "dual_pass_selector": False,
                "benchmark_specific_decoder": False,
                "final_rows_opened": False,
                "publisher_validation_opened": False,
                "publisher_test_opened": False,
                "hf_endpoint": os.environ.get("HF_ENDPOINT"),
                "runner_sha256": common.sha256_file(Path(__file__)),
            }
            result = {
                "schema": SCHEMA,
                "status": (
                    "development_generation_passed"
                    if passed
                    else "development_generation_failed"
                ),
                "passed": passed,
                "input_binding": input_binding,
                "summary": {
                    "systems": summaries,
                    "promotion_criteria": promotion_criteria,
                    "passed": passed,
                },
                "raw_prediction_files": {
                    system: {
                        "path": str(
                            (output / f"{system}.jsonl").resolve()
                        ),
                        "rows": len(all_records[system]),
                        "sha256": common.sha256_file(
                            output / f"{system}.jsonl"
                        ),
                    }
                    for system in systems
                },
                "final_rows_opened": False,
                "publisher_validation_opened": False,
                "publisher_test_opened": False,
            }
            result["receipt"] = {
                "algorithm": "sha256",
                "payload_scope": "canonical_result_without_receipt",
                "payload_sha256": common.canonical_sha256(result),
            }
            write_json(output / "summary.json", result["summary"])
            write_json(output / "result.json", result)
        return_codes = distributed.gather_objects(
            context,
            0 if not context.is_primary or passed else 1,
        )
        return int(return_codes[0])
    finally:
        distributed.destroy_distributed_training(context)


if __name__ == "__main__":
    raise SystemExit(main())
