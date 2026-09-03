#!/usr/bin/env python3
"""Measure fixed-row generation sensitivity under a model-side PLE gain scale."""

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


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[1]
for import_root in (SCRIPT_DIR, PROJECT_ROOT):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

from deltamem.core.delta import iter_delta_mem_modules  # noqa: E402
from experiments.rethinking_rwkv_ms_gemma import (  # noqa: E402
    diagnose_natural_memory_native_rwkv_direct_ple_generation as fixed,
)
from experiments.rethinking_rwkv_ms_gemma import (  # noqa: E402
    evaluate_natural_memory_native_rwkv_direct_ple_development as evaluator,
)
from experiments.rethinking_rwkv_ms_gemma import (  # noqa: E402
    recurrent_routed_posttrain_common as common,
)


SCHEMA = "rwkv_ms_natural_memory_native_rwkv_direct_ple_gain_diagnostic.v1"
ALLOWED_MULTIPLIERS = (2.0, 4.0, 8.0)


def apply_gain(model: torch.nn.Module, multiplier: float) -> Mapping[str, Any]:
    values = []
    for name, module in iter_delta_mem_modules(model):
        before = {
            "rwkv_ms_ple_gain": float(module.rwkv_ms_ple_gain),
            "rwkv_ms_ple_input_gain": float(module.rwkv_ms_ple_input_gain),
        }
        module.rwkv_ms_ple_gain = before["rwkv_ms_ple_gain"] * multiplier
        module.rwkv_ms_ple_input_gain = (
            before["rwkv_ms_ple_input_gain"] * multiplier
        )
        values.append(
            {
                "module": name,
                "before": before,
                "after": {
                    "rwkv_ms_ple_gain": float(module.rwkv_ms_ple_gain),
                    "rwkv_ms_ple_input_gain": float(module.rwkv_ms_ple_input_gain),
                },
            }
        )
    if len(values) != 42:
        raise ValueError(f"Expected 42 native PLE modules, found {len(values)}")
    before_values = {evaluator.canonical_sha256(value["before"]) for value in values}
    after_values = {evaluator.canonical_sha256(value["after"]) for value in values}
    if len(before_values) != 1 or len(after_values) != 1:
        raise ValueError("Native PLE gain values differ across layers")
    return {
        "multiplier": multiplier,
        "layers": len(values),
        "before": values[0]["before"],
        "after": values[0]["after"],
        "module_names_sha256": evaluator.canonical_sha256(
            [value["module"] for value in values]
        ),
    }


def run(args: argparse.Namespace) -> Mapping[str, Any]:
    if os.environ.get("HF_ENDPOINT") != evaluator.HF_MIRROR_ENDPOINT:
        raise ValueError(
            f"HF_ENDPOINT must be exactly {evaluator.HF_MIRROR_ENDPOINT}"
        )
    if args.ple_gain_multiplier not in ALLOWED_MULTIPLIERS:
        raise ValueError(f"PLE gain multiplier must be one of {ALLOWED_MULTIPLIERS}")
    output = args.output_dir.expanduser().resolve()
    if output.exists():
        raise ValueError(f"Gain diagnostic output must be fresh: {output}")
    output.mkdir(parents=True)
    training_result = evaluator.validate_training_result(
        args.training_result,
        args.candidate_adapter,
    )
    model, tokenizer, precision = evaluator.load_system(
        "direct_ple_candidate",
        base_model=args.base_model.expanduser().resolve(strict=True),
        v9_adapter=args.v9_adapter.expanduser().resolve(strict=True),
        candidate_adapter=args.candidate_adapter.expanduser().resolve(strict=True),
        candidate_result=training_result,
        device=args.device,
    )
    gain = apply_gain(model, args.ple_gain_multiplier)
    model.eval()
    selected = fixed.fixed_rows()
    module_names = tuple(name for name, _ in common.ordered_modules(model))
    records: list[Mapping[str, Any]] = []
    try:
        for task in evaluator.TASKS:
            row = selected[task]
            records.extend(
                evaluator.candidate_row_conditions(
                    model,
                    tokenizer,
                    row,
                    fixed.donor_for(task, row),
                    module_names=module_names,
                    device=args.device,
                )
            )
    finally:
        del model, tokenizer
        gc.collect()
        if torch.device(args.device).type == "cuda":
            torch.cuda.empty_cache()
    records_path = output / "records.jsonl"
    evaluator.write_jsonl(records_path, records)
    correct = [
        row for row in records if row["condition"] == "correct_recurrent_state"
    ]
    value: dict[str, Any] = {
        "schema": SCHEMA,
        "status": "gain_diagnostic_complete_not_final_evaluation",
        "input_binding": {
            "training_result": str(args.training_result.expanduser().resolve(strict=True)),
            "training_result_sha256": evaluator.sha256_file(
                args.training_result.expanduser().resolve(strict=True)
            ),
            "training_result_receipt": training_result["receipt"]["payload_sha256"],
            "candidate_adapter": str(
                args.candidate_adapter.expanduser().resolve(strict=True)
            ),
            "gain": gain,
            "rows": {
                task: {
                    "line_index": int(row["line_index"]),
                    "source_ordinal": int(row["source_ordinal"]),
                    "prompt_variant": int(row["prompt_variant"]),
                    "row_sha256": row["row_sha256"],
                }
                for task, row in selected.items()
            },
            "candidate_precision": precision,
            "task_router": False,
            "template_matcher": False,
            "dual_pass_selector": False,
            "benchmark_specific_decoder": False,
            "final_rows_opened": False,
            "runner_sha256": evaluator.sha256_file(Path(__file__)),
            "evaluator_sha256": evaluator.sha256_file(Path(evaluator.__file__)),
        },
        "summary": {
            "correct_state": evaluator.summarize_records(correct),
            "conditions": {
                condition: evaluator.summarize_records(
                    [row for row in records if row["condition"] == condition]
                )
                for condition in evaluator.CONDITIONS
            },
            "sensitivity": fixed.pairwise_sensitivity(records),
        },
        "raw_records": {
            "path": str(records_path),
            "rows": len(records),
            "sha256": evaluator.sha256_file(records_path),
        },
        "final_rows_opened": False,
        "publisher_validation_opened": False,
        "publisher_test_opened": False,
    }
    value["receipt"] = {
        "algorithm": "sha256",
        "payload_scope": "canonical_result_without_receipt",
        "payload_sha256": evaluator.canonical_sha256(value),
    }
    evaluator.write_json(output / "result.json", value)
    return value


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ple-gain-multiplier", type=float, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--training-result", type=Path, default=fixed.DEFAULT_TRAINING_RESULT)
    parser.add_argument("--candidate-adapter", type=Path, default=fixed.DEFAULT_CANDIDATE_ADAPTER)
    parser.add_argument("--base-model", type=Path, default=evaluator.DEFAULT_BASE_MODEL)
    parser.add_argument("--v9-adapter", type=Path, default=evaluator.DEFAULT_V9_ADAPTER)
    parser.add_argument("--device", default="cuda:0")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    value = run(parse_args(argv))
    print(
        json.dumps(
            {
                "status": value["status"],
                "receipt": value["receipt"]["payload_sha256"],
                "gain": value["input_binding"]["gain"],
                "sensitivity": value["summary"]["sensitivity"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
