#!/usr/bin/env python3
"""Run the one-shot gated recurrent-routing publisher-test benchmark."""

from __future__ import annotations

import hashlib
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
    analyze_novel_agent_eval as recovery,
)
from experiments.rethinking_rwkv_ms_gemma import (  # noqa: E402
    recurrent_routed_posttrain_common as routed_common,
)
from experiments.rethinking_rwkv_ms_gemma import (  # noqa: E402
    run_natural_memory_native_recurrent_routed_gated_query_value as training,
)
from experiments.rethinking_rwkv_ms_gemma import (  # noqa: E402
    run_recurrent_routed_final as final,
)


COMMITMENT_ROOT = (
    SCRIPT_DIR
    / "local_artifacts/recurrent_routed_gated_query_value_publisher_test_commit_v1"
)
COMMITMENT_RECEIPT = (
    "f7761887ccb2d7f0462c113ba6654361f2a6b988f5d39ef9e4bc385eb49d7c15"
)
TRAINING_ROOT = (
    SCRIPT_DIR / "local_artifacts/recurrent_routed_gated_query_value_train20_v1"
)
TRAINING_RECEIPT = (
    "5596ade12c16b9ce5e0e6a1e27e4bc5ef860b0f788f595343f653f4c27a52d1a"
)
DEVELOPMENT_RESULT = (
    SCRIPT_DIR
    / "local_artifacts/recurrent_routed_gated_query_value_development_v1/result.json"
)
DEVELOPMENT_RECEIPT = (
    "5492272781cd46ebf0df6056e3b62936d7d1c2cef1fff4e80ad1623eba365123"
)
CANDIDATE_ADAPTER = TRAINING_ROOT / "adapter"
ROW_COUNTS = {"attribution": 30, "narrative": 39, "scene": 149}
FILE_SHA256 = {
    "attribution": "e7d04250ace946448023f603d49be40827e6a087abe28ed5dad04c6d882da6a9",
    "narrative": "5b9c7aa7e27aaa73eb874a6d1039cc3769de26282fcea9cd6bd3d0241cc15e8d",
    "scene": "d8b50ca3862bd40f023155bd14aa7b25d9d5dd3db4ea1c4d5a7e6f4f79cdfd6d",
}


def load_publisher_test_rows(
    final_root: Path = COMMITMENT_ROOT,
) -> tuple[dict[str, list[dict[str, Any]]], Mapping[str, Any]]:
    commitment = final.validate_signed(
        final_root / "commitment.json",
        COMMITMENT_RECEIPT,
    )
    if (
        commitment.get("repo_id")
        != "mikuhhn1239/novel-agent-sft-dataset"
        or commitment.get("revision")
        != "5d3040d21f51b3ce90b9396b058e552c47f43cd5"
        or commitment.get("hf_endpoint") != "https://hf-mirror.com"
        or commitment.get("training_result_receipt") != TRAINING_RECEIPT
        or commitment.get("development_result_receipt") != DEVELOPMENT_RECEIPT
        or commitment.get("semantic_content_parsed_before_commitment") is not False
        or commitment.get("publisher_test_opened") is not False
    ):
        raise ValueError("Publisher-test commitment binding differs")
    rows_by_task = {}
    for task in final.TASKS:
        metadata = commitment["files"][task]
        path = Path(str(metadata["cache_path"])).resolve(strict=True)
        if (
            final.sha256_file(path) != FILE_SHA256[task]
            or metadata.get("sha256") != FILE_SHA256[task]
        ):
            raise ValueError(f"Publisher-test file differs for {task}")
        raw_lines = [
            line for line in path.read_text(encoding="utf-8").splitlines() if line
        ]
        if len(raw_lines) != ROW_COUNTS[task]:
            raise ValueError(f"Publisher-test row count differs for {task}")
        rows = []
        for line_index, raw_line in enumerate(raw_lines):
            value = json.loads(raw_line)
            messages = value.get("messages")
            if (
                not isinstance(messages, list)
                or len(messages) != 3
                or [message.get("role") for message in messages]
                != ["system", "user", "assistant"]
            ):
                raise ValueError(
                    f"Publisher-test row messages differ for {task}:{line_index}"
                )
            gold = final.generation.extract_json(
                str(messages[-1].get("content", ""))
            )
            if gold is None:
                raise ValueError(
                    f"Publisher-test gold JSON differs for {task}:{line_index}"
                )
            rows.append(
                {
                    "line_index": line_index,
                    "source_ordinal": line_index,
                    "row_sha256": hashlib.sha256(
                        raw_line.encode("utf-8")
                    ).hexdigest(),
                    "messages": messages[:-1],
                    "gold": gold,
                }
            )
        rows_by_task[task] = rows
    return rows_by_task, commitment


def summarize_system(
    records: Sequence[Mapping[str, Any]],
    *,
    expected_rows_per_task: int = 64,
) -> Mapping[str, Any]:
    del expected_rows_per_task
    by_task = {}
    for task in final.TASKS:
        task_rows = [row for row in records if row["task"] == task]
        if len(task_rows) != ROW_COUNTS[task]:
            raise ValueError(f"Publisher-test {task} row count differs")
        if task == "attribution":
            correct = sum(
                bool(row["recovered_score"]["correct"]) for row in task_rows
            )
            by_task[task] = {
                "rows": len(task_rows),
                "correct": correct,
                "accuracy": correct / len(task_rows),
                "covered": sum(
                    bool(row["recovered_score"]["covered"]) for row in task_rows
                ),
                "strict_schema_valid": sum(
                    bool(row["score"]["schema_valid"]) for row in task_rows
                ),
            }
        elif task == "narrative":
            correct = sum(
                int(row["recovered_score"]["correct_units"])
                for row in task_rows
            )
            units = sum(
                int(row["recovered_score"]["gold_units"]) for row in task_rows
            )
            by_task[task] = {
                "rows": len(task_rows),
                "correct_units": correct,
                "gold_units": units,
                "unit_label_accuracy": 0.0 if units == 0 else correct / units,
                "covered": sum(
                    bool(row["recovered_score"]["covered"]) for row in task_rows
                ),
                "strict_schema_valid": sum(
                    bool(row["score"]["schema_valid"]) for row in task_rows
                ),
            }
        else:
            tp = sum(int(row["recovered_score"]["tp"]) for row in task_rows)
            fp = sum(int(row["recovered_score"]["fp"]) for row in task_rows)
            fn = sum(int(row["recovered_score"]["fn"]) for row in task_rows)
            by_task[task] = {
                "rows": len(task_rows),
                "tp": tp,
                "fp": fp,
                "fn": fn,
                "micro_f1": final.generation.f1_from_counts(tp, fp, fn),
                "covered": sum(
                    bool(row["recovered_score"]["covered"]) for row in task_rows
                ),
                "strict_schema_valid": sum(
                    bool(row["score"]["schema_valid"]) for row in task_rows
                ),
            }
    return {"rows": len(records), "by_task": by_task}


def control_summary(records: Sequence[Mapping[str, Any]]) -> Mapping[str, Any]:
    if len(records) != sum(ROW_COUNTS.values()):
        raise ValueError("Publisher-test causal control row count differs")
    by_task = {}
    for task in final.TASKS:
        task_rows = [row for row in records if row["task"] == task]
        if len(task_rows) != ROW_COUNTS[task]:
            raise ValueError(f"Publisher-test control count differs for {task}")
        by_task[task] = {
            "rows": len(task_rows),
            "mean_condition_ce": {
                condition: sum(
                    float(row["condition_ce"][condition]) for row in task_rows
                )
                / len(task_rows)
                for condition in routed_common.CONDITIONS
            },
            "mean_control_minus_correct_ce": {
                condition: sum(
                    float(row["control_minus_correct_ce"][condition])
                    for row in task_rows
                )
                / len(task_rows)
                for condition in routed_common.CONDITIONS[1:]
            },
            "projected_carrier_fixed": all(
                row["projected_carrier_fixed"] is True for row in task_rows
            ),
        }
    overall = {
        condition: sum(
            float(row["control_minus_correct_ce"][condition]) for row in records
        )
        / len(records)
        for condition in routed_common.CONDITIONS[1:]
    }
    passed = (
        all(value > 0.0 for value in overall.values())
        and all(
            value > 0.0
            for task in final.TASKS
            for value in by_task[task][
                "mean_control_minus_correct_ce"
            ].values()
        )
        and all(row["projected_carrier_fixed"] is True for row in records)
    )
    return {
        "rows": len(records),
        "overall_mean_control_minus_correct_ce": overall,
        "by_task": by_task,
        "passed": passed,
    }


def configure() -> None:
    routed_common.HYBRID_MODE = training.MODE
    routed_common.HYBRID_GAIN = training.HYBRID_GAIN
    final.SCHEMA = "rwkv_ms_recurrent_routed_gated_publisher_test.v1"
    final.CANDIDATE_ADAPTER = CANDIDATE_ADAPTER
    final.FINAL_ROOT = COMMITMENT_ROOT
    final.FINAL_OPENING_RECEIPT = COMMITMENT_RECEIPT
    final.FINAL_OPENING_FILENAME = "commitment.json"
    final.PROTOCOL_FILE = training.PROTOCOL
    final.PROTOCOL_RECEIPT = training.PROTOCOL_PAYLOAD_SHA256
    final.TRAINING_ROOT = TRAINING_ROOT
    final.TRAINING_RESULT_RECEIPT = TRAINING_RECEIPT
    final.DEVELOPMENT_RESULT = DEVELOPMENT_RESULT
    final.DEVELOPMENT_RESULT_RECEIPT = DEVELOPMENT_RECEIPT
    final.PUBLISHER_TEST_OPENED = True
    final.load_final_rows = load_publisher_test_rows
    final.summarize_system = summarize_system
    final.control_summary = control_summary


def main(argv: Sequence[str] | None = None) -> int:
    configure()
    return final.main(argv)


if __name__ == "__main__":
    raise SystemExit(main())
