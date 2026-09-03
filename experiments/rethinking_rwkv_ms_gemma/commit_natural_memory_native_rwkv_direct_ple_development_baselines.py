#!/usr/bin/env python3
"""Commit complete direct-PLE development baselines from a failed candidate run."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[1]
for import_root in (SCRIPT_DIR, PROJECT_ROOT):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

from experiments.rethinking_rwkv_ms_gemma import (  # noqa: E402
    evaluate_natural_memory_native_rwkv_direct_ple_development as evaluator,
)
from experiments.rethinking_rwkv_ms_gemma import (  # noqa: E402
    recurrent_routed_posttrain_common as common,
)


SCHEMA = "rwkv_ms_natural_memory_native_rwkv_direct_ple_baseline_commit.v1"
SOURCE_RUNNER_SHA256 = "5b0a3ee4da7169dc0f7f8c597bb019f3caf877a6b6755454bc0a4980660d4b5d"
SYSTEMS = evaluator.SYSTEMS[:2]
WORLD_SIZE = evaluator.WORLD_SIZE
DEFAULT_SOURCE = SCRIPT_DIR / "local_artifacts/natural_memory_native_rwkv_direct_ple_development_v2"
DEFAULT_LOG = SCRIPT_DIR / "local_artifacts/natural_memory_native_rwkv_direct_ple_development_v2_r3.log"
DEFAULT_OUTPUT = SCRIPT_DIR / "local_artifacts/natural_memory_native_rwkv_direct_ple_development_baselines_v1.json"
V9_ADAPTER = SCRIPT_DIR / "local_artifacts/natural_memory_native_shared_qo_gate_stage1_v9/adapter"


def expected_rows() -> Mapping[tuple[str, int], Mapping[str, Any]]:
    rows_by_task = evaluator.development.read_v2_rows()
    return {
        (task, int(row["line_index"])): row
        for task in evaluator.TASKS
        for row in rows_by_task[task]
    }


def validate_records(
    records: Sequence[Mapping[str, Any]],
    *,
    system: str,
    expected: Mapping[tuple[str, int], Mapping[str, Any]],
) -> None:
    actual = {(str(row["task"]), int(row["line_index"])): row for row in records}
    if len(records) != len(expected) or len(actual) != len(expected) or set(actual) != set(expected):
        raise ValueError(f"Baseline row identities differ for {system}")
    for key, row in actual.items():
        source = expected[key]
        if (
            row.get("system") != system
            or row.get("row_sha256") != source["row_sha256"]
            or int(row["prompt_variant"]) != int(source["prompt_variant"])
        ):
            raise ValueError(f"Baseline row binding differs for {system}: {key}")


def commit(source_root: Path, log_path: Path) -> Mapping[str, Any]:
    root = source_root.expanduser().resolve(strict=True)
    log = log_path.expanduser().resolve(strict=True)
    runner = Path(evaluator.__file__).resolve(strict=True)
    if evaluator.sha256_file(runner) != SOURCE_RUNNER_SHA256:
        raise ValueError("Source evaluator changed before baseline commitment")
    expected = expected_rows()
    files: dict[str, Any] = {}
    summaries: dict[str, Any] = {}
    for system in SYSTEMS:
        consolidated_path = (root / f"{system}.jsonl").resolve(strict=True)
        consolidated = evaluator.read_jsonl(consolidated_path)
        validate_records(consolidated, system=system, expected=expected)
        shard_records: list[Mapping[str, Any]] = []
        shards: dict[str, Any] = {}
        for rank in range(WORLD_SIZE):
            shard_path = (root / f"shard-{rank}/{system}.jsonl").resolve(strict=True)
            rows = evaluator.read_jsonl(shard_path)
            shard_records.extend(rows)
            shards[str(rank)] = {
                "path": str(shard_path),
                "rows": len(rows),
                "sha256": evaluator.sha256_file(shard_path),
            }
        shard_records.sort(
            key=lambda row: (
                evaluator.TASKS.index(str(row["task"])),
                int(row["line_index"]),
            )
        )
        if evaluator.canonical_sha256(shard_records) != evaluator.canonical_sha256(consolidated):
            raise ValueError(f"Consolidated baseline differs from shards: {system}")
        files[system] = {
            "path": str(consolidated_path),
            "rows": len(consolidated),
            "sha256": evaluator.sha256_file(consolidated_path),
            "shards": shards,
        }
        summaries[system] = evaluator.summarize_records(consolidated)
    candidate_files = sorted(str(path) for path in root.glob("**/direct_ple_candidate.jsonl"))
    control_files = sorted(str(path) for path in root.glob("**/direct_ple_controls.jsonl"))
    if candidate_files or control_files:
        raise ValueError("Failed source unexpectedly contains candidate predictions")
    value: dict[str, Any] = {
        "schema": SCHEMA,
        "status": "development_baselines_committed_candidate_not_evaluated",
        "passed": True,
        "source": {
            "root": str(root),
            "failure_log": str(log),
            "failure_log_sha256": evaluator.sha256_file(log),
            "source_runner": str(runner),
            "source_runner_sha256": SOURCE_RUNNER_SHA256,
            "failure": "candidate_fp32_parameter_storage_without_bf16_autocast",
            "candidate_prediction_files": candidate_files,
            "control_files": control_files,
        },
        "bindings": {
            "base_model": str(common.BASE_MODEL.expanduser().resolve(strict=True)),
            "base_model_revision": common.BASE_MODEL_REVISION,
            "base_model_weights_sha256": common.BASE_MODEL_WEIGHTS_SHA256,
            "v9_adapter": str(V9_ADAPTER.resolve(strict=True)),
            "v9_adapter_weights_sha256": evaluator.sha256_file(V9_ADAPTER / "delta_mem_adapter.pt"),
            "v9_adapter_config_sha256": evaluator.sha256_file(V9_ADAPTER / "delta_mem_config.json"),
            "development_manifest_receipt": evaluator.development.V2_MANIFEST_RECEIPT,
            "prompt_variants_sha256": evaluator.canonical_sha256(common.PROMPT_VARIANTS),
            "systems": list(SYSTEMS),
            "rows_per_task": {
                task: len(evaluator.development.read_v2_rows()[task])
                for task in evaluator.TASKS
            },
            "world_size": WORLD_SIZE,
        },
        "files": files,
        "summaries": summaries,
        "task_router": False,
        "template_matcher": False,
        "dual_pass_selector": False,
        "benchmark_specific_decoder": False,
        "final_rows_opened": False,
        "publisher_validation_opened": False,
        "publisher_test_opened": False,
        "runner_sha256": evaluator.sha256_file(Path(__file__).resolve(strict=True)),
    }
    value["receipt"] = {
        "algorithm": "sha256",
        "payload_scope": "canonical_result_without_receipt",
        "payload_sha256": evaluator.canonical_sha256(value),
    }
    return value


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--failure-log", type=Path, default=DEFAULT_LOG)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    output = args.output.expanduser().resolve()
    if output.exists():
        raise ValueError(f"Baseline commit output must be fresh: {output}")
    value = commit(args.source_root, args.failure_log)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(value, ensure_ascii=True, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "status": value["status"],
                "receipt": value["receipt"]["payload_sha256"],
                "files": {
                    system: metadata["sha256"]
                    for system, metadata in value["files"].items()
                },
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
