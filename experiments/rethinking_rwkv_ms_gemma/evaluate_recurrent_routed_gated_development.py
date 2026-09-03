#!/usr/bin/env python3
"""Evaluate gated recurrent routing on committed train-derived development rows."""

from __future__ import annotations

import argparse
from difflib import SequenceMatcher
import hashlib
import json
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

from torch.distributed.elastic.multiprocessing.errors import record


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[1]
for import_root in (SCRIPT_DIR, PROJECT_ROOT):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

from experiments.rethinking_rwkv_ms_gemma import (  # noqa: E402
    evaluate_natural_memory_native_recurrent_routed_posttrain_development as evaluator,
)
from experiments.rethinking_rwkv_ms_gemma import (  # noqa: E402
    natural_memory_distributed as distributed,
)
from experiments.rethinking_rwkv_ms_gemma import (  # noqa: E402
    recurrent_routed_posttrain_common as common,
)
from experiments.rethinking_rwkv_ms_gemma import (  # noqa: E402
    run_natural_memory_native_recurrent_routed_gated_query_value as training,
)


DEVELOPMENT_ROOT = (
    SCRIPT_DIR
    / "local_artifacts/recurrent_routed_gated_query_value_development_split_v1"
)
DEVELOPMENT_MANIFEST_RECEIPT = (
    "40e6390f338cac7fdf7dd047499fe2b2b50050a3a3b8b25d84250f33e1716e42"
)
ROWS_PER_TASK = 32
EXPECTED_ROWS = len(common.TASKS) * ROWS_PER_TASK * 4
TRAINED_SUFFIXES = (
    ".rwkv_route_query_proj",
    ".rwkv_route_state_proj",
    ".hrm_rwkv7_core.output.weight",
    ".rwkv_pair_value_proj",
    ".rwkv_pair_gate_weight",
    ".rwkv_pair_gate_bias",
)


def development_manifest() -> Mapping[str, Any]:
    return common.validate_signed_json(
        DEVELOPMENT_ROOT / "manifest.json",
        DEVELOPMENT_MANIFEST_RECEIPT,
    )


def validate_development_files(
    manifest: Mapping[str, Any],
) -> dict[str, Mapping[str, Any]]:
    files = {}
    for task in common.TASKS:
        metadata = manifest["tasks"][task]
        relative_path = str(metadata["materialized_relative_path"])
        path = DEVELOPMENT_ROOT / relative_path
        if (
            not path.is_file()
            or path.stat().st_size != int(metadata["materialized_bytes"])
            or common.sha256_file(path) != metadata["materialized_sha256"]
        ):
            raise ValueError(f"Gated development file differs for {task}")
        files[relative_path] = {
            "rows": ROWS_PER_TASK,
            "bytes": path.stat().st_size,
            "sha256": metadata["materialized_sha256"],
            "row_payload_sha256": metadata["selected_rows_sha256"],
        }
    return files


def fake_split_artifacts() -> tuple[Mapping[str, Any], Mapping[str, Any]]:
    manifest = development_manifest()
    files = validate_development_files(manifest)
    return manifest, {
        "schema": "rwkv_ms_recurrent_routed_gated_development_open.v1",
        "manifest_receipt": DEVELOPMENT_MANIFEST_RECEIPT,
        "materialized_splits": ["train_derived_development"],
        "final_files_written": [],
        "files": files,
    }


def load_development_rows(
    split: str,
    *,
    manifest: Mapping[str, Any],
) -> dict[str, tuple[common.SourceRow, ...]]:
    if split != "development":
        raise ValueError("Gated evaluation exposes only committed development")
    rows_by_task = {}
    for task in common.TASKS:
        metadata = manifest["tasks"][task]
        path = DEVELOPMENT_ROOT / str(metadata["materialized_relative_path"])
        raw_lines = tuple(
            line for line in path.read_text(encoding="utf-8").splitlines() if line
        )
        committed = metadata["selected_rows"]
        if len(raw_lines) != ROWS_PER_TASK or len(committed) != ROWS_PER_TASK:
            raise ValueError(f"Gated development row count differs for {task}")
        loaded = []
        for raw_line, row_metadata in zip(raw_lines, committed):
            digest = hashlib.sha256(raw_line.encode("utf-8")).hexdigest()
            value = json.loads(raw_line)
            messages = value.get("messages")
            if (
                digest != row_metadata["row_sha256"]
                or not isinstance(messages, list)
                or len(messages) != 3
                or [message.get("role") for message in messages]
                != ["system", "user", "assistant"]
            ):
                raise ValueError(f"Gated development row differs for {task}")
            loaded.append(
                common.SourceRow(
                    task=task,
                    source_ordinal=int(row_metadata["source_ordinal"]),
                    row_sha256=digest,
                    raw_line=raw_line,
                    assistant_identity=str(messages[-1]["content"]),
                    user_characters=len(str(messages[1]["content"])),
                )
            )
        rows_by_task[task] = tuple(loaded)
    return rows_by_task


def row_user_content(row: common.SourceRow) -> str:
    return str(json.loads(row.raw_line)["messages"][1]["content"])


def choose_hard_donor(
    target: common.SourceRow,
    rows: Sequence[common.SourceRow],
) -> common.SourceRow:
    candidates = [
        row
        for row in rows
        if row.source_ordinal != target.source_ordinal
        and row.assistant_identity != target.assistant_identity
    ]
    if not candidates:
        raise ValueError(f"Gated development row has no donor: {target}")
    return max(
        candidates,
        key=lambda row: (
            SequenceMatcher(
                None,
                row_user_content(target),
                row_user_content(row),
            ).ratio(),
            -abs(row.user_characters - target.user_characters),
            row.row_sha256,
        ),
    )


def build_schedule(
    rows_by_task: Mapping[str, Sequence[common.SourceRow]],
) -> tuple[
    tuple[tuple[common.SourceRow, common.SourceRow, int], ...],
    list[dict[str, Any]],
]:
    schedule = []
    payload = []
    for task in common.TASKS:
        rows = sorted(rows_by_task[task], key=lambda row: row.source_ordinal)
        if len(rows) != ROWS_PER_TASK:
            raise ValueError(f"Gated development count differs for {task}")
        for target in rows:
            donor = choose_hard_donor(target, rows)
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
        raise RuntimeError("Gated development schedule size differs")
    return tuple(schedule), payload


def configure() -> None:
    common.HYBRID_MODE = training.MODE
    common.HYBRID_GAIN = training.HYBRID_GAIN
    common.SPLIT_MANIFEST_RECEIPT = DEVELOPMENT_MANIFEST_RECEIPT
    common.OPEN_SPLIT_RECEIPT = DEVELOPMENT_MANIFEST_RECEIPT
    common.validate_split_artifacts = fake_split_artifacts
    common.load_open_rows = load_development_rows
    evaluator.EXPECTED_ROWS = EXPECTED_ROWS
    evaluator.TRAINED_SUFFIXES = TRAINED_SUFFIXES
    evaluator.build_schedule = build_schedule


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--training-root", type=Path, required=True)
    parser.add_argument("--training-result-receipt", required=True)
    parser.add_argument(
        "--training-status",
        default=(
            "gated_query_value_training_complete_"
            "development_evaluation_authorized"
        ),
    )
    parser.add_argument("--base-model", type=Path, default=common.BASE_MODEL)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args(argv)


@record
def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    configure()
    context = distributed.initialize_distributed_training(args.device)
    if context is None:
        raise ValueError("Gated development evaluation requires four ranks")
    try:
        result = evaluator.run(
            context=context,
            output_dir=args.output_dir,
            training_root=args.training_root.expanduser().resolve(strict=True),
            base_model=args.base_model,
            training_result_receipt=args.training_result_receipt,
            training_status=args.training_status,
            protocol_file=training.PROTOCOL,
            protocol_receipt=training.PROTOCOL_PAYLOAD_SHA256,
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
