#!/usr/bin/env python3
"""Evaluate the authorized recurrent-routed candidate on the opened final rows."""

from __future__ import annotations

import argparse
import gc
import hashlib
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

from experiments.rethinking_rwkv_ms_gemma import (  # noqa: E402
    analyze_novel_agent_eval as recovery,
)
from experiments.rethinking_rwkv_ms_gemma import (  # noqa: E402
    evaluate_natural_memory_native_recurrent_routed_posttrain_development as development,
)
from experiments.rethinking_rwkv_ms_gemma import (  # noqa: E402
    natural_memory_distributed as distributed,
)
from experiments.rethinking_rwkv_ms_gemma import (  # noqa: E402
    recurrent_routed_posttrain_common as routed_common,
)
from experiments.rethinking_rwkv_ms_gemma import (  # noqa: E402
    run_natural_memory_native_scene_contrast_dropout as contrast,
)
from experiments.rethinking_rwkv_ms_gemma import (  # noqa: E402
    run_novel_agent_eval as generation,
)


SCHEMA = "rwkv_ms_natural_memory_native_recurrent_routed_final.v1"
ROW_SCHEMA = "rwkv_ms_natural_memory_native_recurrent_routed_final_row.v1"
CONTROL_SCHEMA = "rwkv_ms_natural_memory_native_recurrent_routed_final_control.v1"
WORLD_SIZE = 4
TASKS = ("attribution", "narrative", "scene")
MAX_NEW_TOKENS = {"attribution": 1024, "narrative": 1024, "scene": 128}
SYSTEMS = (
    "frozen_gemma_base",
    "v9_projected_slot_baseline",
    "recurrent_routed_candidate",
)
BASE_MODEL = Path("/root/x/models/gemma-4-E4B-it")
V9_ADAPTER = SCRIPT_DIR / "local_artifacts/natural_memory_native_shared_qo_gate_stage1_v9/adapter"
CANDIDATE_ADAPTER = SCRIPT_DIR / "local_artifacts/recurrent_routed_query_value_gain_one_train20_v1/adapter"
FINAL_ROOT = SCRIPT_DIR / "local_artifacts/recurrent_routed_query_value_gain_one_final_split_v1"
FINAL_OPENING_RECEIPT = "6098d203369f9d9d7851f48b19732923fb6aae46fc4a6b533ed41a10aa67c3a1"
PROTOCOL_FILE = SCRIPT_DIR / "natural_memory_native_recurrent_routed_posttrain_protocol_v1.json"
PROTOCOL_RECEIPT = "576537822ca7079d15fc6d0ce618a94b8631286c47008f136ef1b6ed725d191d"
TRAINING_ROOT = SCRIPT_DIR / "local_artifacts/recurrent_routed_query_value_gain_one_train20_v1"
TRAINING_RESULT_RECEIPT = "74939af17b1296e04cd37a67f17d73b8865de69e6e29cded5d3a77a05259e858"
DEVELOPMENT_RESULT = SCRIPT_DIR / "local_artifacts/recurrent_routed_query_value_gain_one_development_v2/result.json"
DEVELOPMENT_RESULT_RECEIPT = "e6e7c9a58e12f497cc2ab09fc2a5ee68382e6a028ac91e8f910953d144b021ef"
OPEN_SPLIT_RECEIPT = "159cf93c913715f0c90e03ca659bf3bd4f1deb9d3e12c64f923d7b5b71340ad8"
SPLIT_MANIFEST_RECEIPT = "05314bfcaa3f4c6febe860f33bf7867af8d57a80e9e1b9020b1cc318bceebc96"
FINAL_COMMITMENT_RECEIPT = "c8c106a00e1379e26bbae5b774f0fe831de2c2527c3195a1e42881a33b2b2fae"
FINAL_OPENING_FILENAME = "final_opening.json"
PUBLISHER_TEST_OPENED = False


def canonical_sha256(value: Any) -> str:
    return distributed.canonical_sha256(value)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_signed(path: Path, receipt: str) -> Mapping[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    signed = value.get("receipt")
    unsigned = dict(value)
    unsigned.pop("receipt", None)
    if not isinstance(signed, Mapping) or signed.get("payload_sha256") != receipt:
        raise ValueError(f"Receipt differs: {path}")
    if canonical_sha256(unsigned) != receipt:
        raise ValueError(f"Signed payload differs: {path}")
    return value


def load_final_rows(final_root: Path = FINAL_ROOT) -> tuple[dict[str, list[dict[str, Any]]], Mapping[str, Any]]:
    opening = validate_signed(final_root / "final_opening.json", FINAL_OPENING_RECEIPT)
    if (
        opening.get("final_rows_opened") is not True
        or opening.get("publisher_validation_opened") is not False
        or opening.get("publisher_test_opened") is not False
        or opening.get("manifest_receipt") != SPLIT_MANIFEST_RECEIPT
        or opening.get("final_commitment_receipt") != FINAL_COMMITMENT_RECEIPT
        or opening.get("open_split_receipt") != OPEN_SPLIT_RECEIPT
        or opening.get("rows_per_task") != 64
    ):
        raise ValueError("Final opening binding differs")
    manifest = validate_signed(
        routed_common.SPLIT_ROOT / "manifest.json", SPLIT_MANIFEST_RECEIPT
    )
    rows_by_task: dict[str, list[dict[str, Any]]] = {}
    for task in TASKS:
        path = final_root / "final" / task / "final.jsonl"
        relative = f"final/{task}/final.jsonl"
        metadata = opening.get("files", {}).get(relative)
        if not isinstance(metadata, Mapping):
            raise ValueError(f"Final opening omits {relative}")
        if not path.is_file() or sha256_file(path) != metadata.get("sha256"):
            raise ValueError(f"Final file hash differs: {path}")
        committed = manifest["tasks"][task]["splits"]["final"]["rows"]
        raw_lines = [line for line in path.read_text(encoding="utf-8").splitlines() if line]
        if len(raw_lines) != 64:
            raise ValueError(f"Final row count differs for {task}")
        rows: list[dict[str, Any]] = []
        for line_index, (raw_line, committed_row) in enumerate(zip(raw_lines, committed, strict=True)):
            row_hash = hashlib.sha256(raw_line.encode("utf-8")).hexdigest()
            if row_hash != committed_row["row_sha256"]:
                raise ValueError(f"Final row hash differs for {task}:{line_index}")
            value = json.loads(raw_line)
            messages = value.get("messages")
            if (
                not isinstance(messages, list)
                or len(messages) != 3
                or [message.get("role") for message in messages]
                != ["system", "user", "assistant"]
            ):
                raise ValueError(f"Final row messages differ for {task}:{line_index}")
            gold = generation.extract_json(str(messages[-1].get("content", "")))
            if gold is None:
                raise ValueError(f"Final gold JSON is invalid for {task}:{line_index}")
            rows.append(
                {
                    "line_index": line_index,
                    "source_ordinal": int(committed_row["source_ordinal"]),
                    "row_sha256": row_hash,
                    "messages": messages[:-1],
                    "gold": gold,
                }
            )
        rows_by_task[task] = rows
    return rows_by_task, opening


def model_paths(system: str, base_model: Path, v9_adapter: Path, candidate_adapter: Path) -> tuple[Path, Path | None]:
    if system == "frozen_gemma_base":
        return base_model, None
    if system == "v9_projected_slot_baseline":
        return base_model, v9_adapter
    if system == "recurrent_routed_candidate":
        return base_model, candidate_adapter
    raise ValueError(f"Unknown system: {system}")


def prediction_for(task: str, parsed: Any, messages: Sequence[Mapping[str, str]]) -> Any:
    if task == "attribution":
        candidates = recovery.parse_candidates(str(messages[-1].get("content", "")))
        return recovery.recover_attribution(parsed, candidates)
    if task == "narrative":
        return recovery.recover_narrative(parsed)
    if task == "scene":
        recovered = recovery.recover_scene(parsed)
        return None if recovered is None else sorted(recovered)
    raise ValueError(f"Unknown task: {task}")


def strict_task_json(task: str, raw_generation: str) -> Any | None:
    try:
        value = json.loads(raw_generation.strip())
    except (json.JSONDecodeError, TypeError):
        return None
    if task == "attribution":
        if (
            not isinstance(value, dict)
            or set(value) != {"best_candidate", "uncertain"}
            or not isinstance(value["best_candidate"], str)
            or not isinstance(value["uncertain"], bool)
        ):
            return None
        return value
    if task == "narrative":
        if not isinstance(value, dict) or set(value) != {"labels"}:
            return None
        labels = value["labels"]
        if not isinstance(labels, list):
            return None
        unit_ids: list[str] = []
        for label in labels:
            if (
                not isinstance(label, dict)
                or set(label) != {"unit_id", "type"}
                or not isinstance(label["unit_id"], str)
                or label["type"] not in recovery.NARRATIVE_TYPES
            ):
                return None
            unit_ids.append(label["unit_id"])
        if len(unit_ids) != len(set(unit_ids)):
            return None
        return value
    if task == "scene":
        if not isinstance(value, dict) or set(value) != {"boundaries"}:
            return None
        boundaries = value["boundaries"]
        if (
            not isinstance(boundaries, list)
            or any(
                isinstance(boundary, bool) or not isinstance(boundary, int)
                for boundary in boundaries
            )
            or len(boundaries) != len(set(boundaries))
        ):
            return None
        return value
    raise ValueError(f"Unknown task: {task}")


def evaluate_row(model: Any, tokenizer: Any, task: str, row: Mapping[str, Any], device: str, system: str) -> Mapping[str, Any]:
    memory_protocol = "legacy_write_only" if system == "frozen_gemma_base" else "write_then_read"
    result = generation.generate_one(
        model=model,
        tokenizer=tokenizer,
        messages=list(row["messages"]),
        max_new_tokens=MAX_NEW_TOKENS[task],
        device=device,
        online_memory_protocol=memory_protocol,
    )
    parsed = result["parsed_json"]
    strict_parsed = strict_task_json(task, result["raw_generation"])
    prediction = prediction_for(task, parsed, row["messages"])
    score = generation.score_prediction(task, strict_parsed, row["gold"])
    if task == "attribution":
        score["uncertain_correct"] = bool(
            strict_parsed is not None
            and strict_parsed["uncertain"] == row["gold"].get("uncertain")
        )
        score["joint_correct"] = bool(
            score["correct"] and score["uncertain_correct"]
        )
        recovered_score = {
            "covered": prediction is not None,
            "correct": bool(
                prediction is not None
                and prediction == row["gold"].get("best_candidate")
            ),
        }
    elif task == "narrative":
        gold_labels = recovery.gold_label_map(row["gold"])
        recovered_score = {
            "covered": prediction is not None,
            "correct_units": sum(
                prediction.get(unit_id) == label_type
                for unit_id, label_type in gold_labels.items()
            )
            if isinstance(prediction, Mapping)
            else 0,
            "gold_units": len(gold_labels),
        }
    else:
        predicted = prediction if isinstance(prediction, list) else []
        gold_boundaries = recovery.strict_gold_boundaries(row["gold"])
        predicted_set = set(predicted)
        recovered_score = {
            "covered": prediction is not None,
            "tp": len(predicted_set & gold_boundaries),
            "fp": len(predicted_set - gold_boundaries),
            "fn": len(gold_boundaries - predicted_set),
        }
    return {
        "schema": ROW_SCHEMA,
        "system": system,
        "task": task,
        "line_index": int(row["line_index"]),
        "source_ordinal": int(row["source_ordinal"]),
        "row_sha256": row["row_sha256"],
        "gold": row["gold"],
        "prediction": prediction,
        "score": score,
        "recovered_score": recovered_score,
        "max_new_tokens": MAX_NEW_TOKENS[task],
        "online_memory_protocol": memory_protocol,
        "raw_generation": result["raw_generation"],
        "parsed_json": parsed,
        "strict_parsed_json": strict_parsed,
        "input_tokens": result["input_tokens"],
        "output_tokens": result["output_tokens"],
        "hit_max_new_tokens": result["hit_max_new_tokens"],
        "elapsed_seconds": result["elapsed_seconds"],
        "peak_cuda_memory_bytes": result["peak_cuda_memory_bytes"],
    }


def summarize_system(
    records: Sequence[Mapping[str, Any]],
    *,
    expected_rows_per_task: int = 64,
) -> Mapping[str, Any]:
    by_task: dict[str, Mapping[str, Any]] = {}
    for task in TASKS:
        task_rows = [row for row in records if row["task"] == task]
        if len(task_rows) != expected_rows_per_task:
            raise ValueError(f"Final {task} row count differs")
        if task == "attribution":
            correct = sum(bool(row["score"]["correct"]) for row in task_rows)
            by_task[task] = {
                "rows": len(task_rows),
                "correct": correct,
                "accuracy": correct / len(task_rows),
                "uncertain_accuracy": sum(
                    bool(row["score"]["uncertain_correct"])
                    for row in task_rows
                ) / len(task_rows),
                "joint_accuracy": sum(
                    bool(row["score"]["joint_correct"])
                    for row in task_rows
                ) / len(task_rows),
                "recovered_accuracy_diagnostic": sum(
                    bool(row["recovered_score"]["correct"])
                    for row in task_rows
                ) / len(task_rows),
                "recovery_covered_diagnostic": sum(
                    bool(row["recovered_score"]["covered"])
                    for row in task_rows
                ),
                "strict_schema_valid": sum(bool(row["score"]["schema_valid"]) for row in task_rows),
            }
        elif task == "narrative":
            correct = sum(int(row["score"]["correct_units"]) for row in task_rows)
            units = sum(int(row["score"]["gold_units"]) for row in task_rows)
            by_task[task] = {
                "rows": len(task_rows),
                "correct_units": correct,
                "gold_units": units,
                "unit_label_accuracy": 0.0 if units == 0 else correct / units,
                "recovered_unit_label_accuracy_diagnostic": (
                    sum(
                        int(row["recovered_score"]["correct_units"])
                        for row in task_rows
                    )
                    / units
                    if units
                    else 0.0
                ),
                "recovery_covered_diagnostic": sum(
                    bool(row["recovered_score"]["covered"])
                    for row in task_rows
                ),
                "strict_schema_valid": sum(bool(row["score"]["schema_valid"]) for row in task_rows),
            }
        else:
            tp = sum(int(row["score"]["tp"]) for row in task_rows)
            fp = sum(int(row["score"]["fp"]) for row in task_rows)
            fn = sum(int(row["score"]["fn"]) for row in task_rows)
            recovered_tp = sum(
                int(row["recovered_score"]["tp"]) for row in task_rows
            )
            recovered_fp = sum(
                int(row["recovered_score"]["fp"]) for row in task_rows
            )
            recovered_fn = sum(
                int(row["recovered_score"]["fn"]) for row in task_rows
            )
            by_task[task] = {
                "rows": len(task_rows),
                "tp": tp,
                "fp": fp,
                "fn": fn,
                "micro_f1": generation.f1_from_counts(tp, fp, fn),
                "recovered_micro_f1_diagnostic": generation.f1_from_counts(
                    recovered_tp,
                    recovered_fp,
                    recovered_fn,
                ),
                "recovery_covered_diagnostic": sum(
                    bool(row["recovered_score"]["covered"])
                    for row in task_rows
                ),
                "strict_schema_valid": sum(bool(row["score"]["schema_valid"]) for row in task_rows),
            }
    return {"rows": len(records), "by_task": by_task}


def as_source_row(task: str, row: Mapping[str, Any]) -> routed_common.SourceRow:
    raw_value = {"messages": list(row["messages"]) + [{"role": "assistant", "content": json.dumps(row["gold"], ensure_ascii=False)}]}
    raw_line = json.dumps(raw_value, ensure_ascii=False, separators=(",", ":"))
    return routed_common.SourceRow(
        task=task,
        source_ordinal=int(row["source_ordinal"]),
        row_sha256=str(row["row_sha256"]),
        raw_line=raw_line,
        assistant_identity=json.dumps(row["gold"], ensure_ascii=False, sort_keys=True),
        user_characters=len(str(row["messages"][1].get("content", ""))),
    )


def choose_control_donor(target: routed_common.SourceRow, rows: Sequence[routed_common.SourceRow]) -> routed_common.SourceRow:
    candidates = [
        row for row in rows
        if row.source_ordinal != target.source_ordinal
        and row.assistant_identity != target.assistant_identity
    ]
    if not candidates:
        raise ValueError(f"No causal donor for {target.task}:{target.source_ordinal}")
    return min(candidates, key=lambda row: (abs(row.user_characters - target.user_characters), row.row_sha256))


def evaluate_controls(
    model: Any,
    tokenizer: Any,
    rows_by_task: Mapping[str, Sequence[Mapping[str, Any]]],
    device: torch.device,
    *,
    process_rank: int,
    world_size: int,
) -> list[Mapping[str, Any]]:
    source_rows = {
        task: tuple(as_source_row(task, row) for row in rows)
        for task, rows in rows_by_task.items()
    }
    records: list[Mapping[str, Any]] = []
    flat_rows = [
        (task, row, target)
        for task in TASKS
        for row, target in zip(rows_by_task[task], source_rows[task], strict=True)
    ]
    for flat_index, (task, row, target) in enumerate(flat_rows):
        if flat_index % world_size != process_rank:
            continue
        donor = choose_control_donor(target, source_rows[task])
        target_example = routed_common.encode_row(tokenizer, target, prompt_variant=0)
        donor_example = routed_common.encode_row(tokenizer, donor, prompt_variant=0)
        target_batch = routed_common.evolution.collate_native_examples(
            [target_example], pad_token_id=int(tokenizer.pad_token_id), device=device
        )
        donor_write = routed_common.evolution.collate_native_examples(
            [donor_example], pad_token_id=int(tokenizer.pad_token_id), device=device
        )
        donor_batch = routed_common.evolution.NativeFullRowBatch(
            examples=target_batch.examples,
            write_input_ids=donor_write.write_input_ids,
            write_attention_mask=donor_write.write_attention_mask,
            read_input_ids=target_batch.read_input_ids,
            read_attention_mask=target_batch.read_attention_mask,
            labels=target_batch.labels,
        )
        condition_ce: dict[str, float] = {}
        condition_tokens: dict[str, int] = {}
        audits: dict[str, Mapping[str, Any]] = {}
        try:
            with torch.inference_mode():
                for condition in routed_common.CONDITIONS:
                    logits, audit = routed_common.direct_condition_logits(
                        model,
                        target_batch,
                        condition=condition,
                        donor=donor_batch if condition == "matched_donor_recurrent_state" else None,
                        dtype=torch.bfloat16,
                    )
                    ce, tokens = contrast.detached_answer_ce(logits, target_batch.labels)
                    condition_ce[condition] = float(ce)
                    condition_tokens[condition] = int(tokens)
                    audits[condition] = dict(audit)
                    del logits
                    routed_common.reset_delta_mem_states(model)
                    routed_common.evolution.release_native_row_allocator_cache(device)
            if len(set(condition_tokens.values())) != 1:
                raise RuntimeError("Final control token counts differ")
            correct = condition_ce["correct_recurrent_state"]
            records.append(
                {
                    "schema": CONTROL_SCHEMA,
                    "task": task,
                    "line_index": int(row["line_index"]),
                    "source_ordinal": int(row["source_ordinal"]),
                    "row_sha256": row["row_sha256"],
                    "donor_source_ordinal": donor.source_ordinal,
                    "donor_row_sha256": donor.row_sha256,
                    "answer_tokens": next(iter(condition_tokens.values())),
                    "condition_ce": condition_ce,
                    "control_minus_correct_ce": {
                        condition: condition_ce[condition] - correct
                        for condition in routed_common.CONDITIONS[1:]
                    },
                    "condition_audits": audits,
                    "projected_carrier_fixed": all(
                        audit["projected_carrier_references_fixed"]
                        and audit["projected_carrier_bytes_fixed"]
                        for audit in audits.values()
                    ),
                }
            )
        finally:
            del target_batch, donor_batch, donor_write, target_example, donor_example
            routed_common.reset_delta_mem_states(model)
            routed_common.evolution.release_native_row_allocator_cache(device)
    return records


def control_summary(records: Sequence[Mapping[str, Any]]) -> Mapping[str, Any]:
    if len(records) != 192:
        raise ValueError("Final causal control row count differs")
    by_task = {}
    for task in TASKS:
        task_rows = [row for row in records if row["task"] == task]
        by_task[task] = {
            "rows": len(task_rows),
            "mean_condition_ce": {
                condition: sum(float(row["condition_ce"][condition]) for row in task_rows) / len(task_rows)
                for condition in routed_common.CONDITIONS
            },
            "mean_control_minus_correct_ce": {
                condition: sum(float(row["control_minus_correct_ce"][condition]) for row in task_rows) / len(task_rows)
                for condition in routed_common.CONDITIONS[1:]
            },
            "projected_carrier_fixed": all(row["projected_carrier_fixed"] is True for row in task_rows),
        }
    overall = {
        condition: sum(float(row["control_minus_correct_ce"][condition]) for row in records) / len(records)
        for condition in routed_common.CONDITIONS[1:]
    }
    passed = all(value > 0.0 for value in overall.values()) and all(
        value > 0.0
        for task in TASKS
        for value in by_task[task]["mean_control_minus_correct_ce"].values()
    ) and all(row["projected_carrier_fixed"] is True for row in records)
    return {"rows": len(records), "overall_mean_control_minus_correct_ce": overall, "by_task": by_task, "passed": passed}


def load_model(base_model: Path, adapter: Path | None, device: str):
    from common import load_model_and_tokenizer

    kwargs = {
        "base_model": str(base_model),
        "device": device,
        "dtype": "bfloat16",
        "attn_implementation": "sdpa",
        "delta_mem_root": PROJECT_ROOT,
    }
    if adapter is not None:
        kwargs["memory_dir"] = str(adapter)
    return load_model_and_tokenizer(**kwargs)


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=True, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(
                json.dumps(row, ensure_ascii=True, sort_keys=True, separators=(",", ":"))
                + "\n"
            )


def read_jsonl(path: Path) -> list[Mapping[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line
    ]


def append_jsonl(path: Path, row: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(
            json.dumps(row, ensure_ascii=True, sort_keys=True, separators=(",", ":"))
            + "\n"
        )
        handle.flush()
        os.fsync(handle.fileno())


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--base-model", type=Path, default=BASE_MODEL)
    parser.add_argument("--v9-adapter", type=Path, default=V9_ADAPTER)
    parser.add_argument("--candidate-adapter", type=Path, default=CANDIDATE_ADAPTER)
    parser.add_argument("--final-root", type=Path, default=FINAL_ROOT)
    parser.add_argument("--systems", default=",".join(SYSTEMS))
    parser.add_argument("--limit-per-task", type=int)
    parser.add_argument("--skip-controls", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--device", default="cuda")
    return parser.parse_args(argv)


@record
def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if os.environ.get("HF_ENDPOINT") != "https://hf-mirror.com":
        raise ValueError("HF_ENDPOINT must be exactly https://hf-mirror.com")
    requested_systems = tuple(item.strip() for item in args.systems.split(",") if item.strip())
    if not requested_systems or any(system not in SYSTEMS for system in requested_systems):
        raise ValueError(f"Unknown systems: {requested_systems}")
    if len(set(requested_systems)) != len(requested_systems):
        raise ValueError("Systems must not repeat")
    if args.limit_per_task is not None and args.limit_per_task <= 0:
        raise ValueError("--limit-per-task must be positive")
    context = distributed.initialize_distributed_training(
        args.device,
        timeout_seconds=7200,
    )
    if context is None:
        raise ValueError("Final evaluation requires four torchrun ranks")
    try:
        rows_by_task, opening = load_final_rows(
            args.final_root.expanduser().resolve(strict=True)
        )
        if args.limit_per_task is not None:
            rows_by_task = {
                task: rows[: args.limit_per_task] for task, rows in rows_by_task.items()
            }
        resolved_output = args.output_dir.expanduser().resolve()
        creation_error = None
        if context.is_primary:
            try:
                if args.resume:
                    if not resolved_output.is_dir():
                        raise ValueError(f"Resume output does not exist: {resolved_output}")
                else:
                    resolved_output.mkdir(parents=True, exist_ok=False)
            except BaseException as error:
                creation_error = error
        distributed.phase_consensus(context, phase="recurrent-routed-final-output-creation", error=creation_error)
        all_records: dict[str, list[Mapping[str, Any]]] = {}
        all_controls: list[Mapping[str, Any]] = []
        for system in requested_systems:
            base_model, adapter = model_paths(
                system,
                args.base_model.expanduser().resolve(strict=True),
                args.v9_adapter.expanduser().resolve(strict=True),
                args.candidate_adapter.expanduser().resolve(strict=True),
            )
            model, tokenizer = load_model(base_model, adapter, str(context.device))
            model.eval()
            shard_path = resolved_output / f"shard-{context.process_rank}" / f"{system}.jsonl"
            local_records = read_jsonl(shard_path) if shard_path.is_file() else []
            completed_keys = {
                (str(row["task"]), int(row["line_index"])) for row in local_records
            }
            flat_rows = [(task, row) for task in TASKS for row in rows_by_task[task]]
            for index, (task, row) in enumerate(flat_rows):
                if index % context.world_size != context.process_rank:
                    continue
                if (task, int(row["line_index"])) in completed_keys:
                    continue
                record = evaluate_row(model, tokenizer, task, row, str(context.device), system)
                local_records.append(record)
                append_jsonl(shard_path, record)
                if len(local_records) % 4 == 0:
                    print(json.dumps({"system": system, "rank": context.process_rank, "local_rows": len(local_records)}, sort_keys=True), flush=True)
            torch.distributed.barrier(group=context.control_group)
            if context.is_primary:
                records = []
                for rank in range(context.world_size):
                    records.extend(read_jsonl(resolved_output / f"shard-{rank}" / f"{system}.jsonl"))
                records.sort(key=lambda row: (TASKS.index(str(row["task"])), int(row["line_index"])))
                all_records[system] = records
            if system == "recurrent_routed_candidate" and not args.skip_controls:
                local_controls = evaluate_controls(
                    model,
                    tokenizer,
                    rows_by_task,
                    context.device,
                    process_rank=context.process_rank,
                    world_size=context.world_size,
                )
                control_shard_path = resolved_output / f"shard-{context.process_rank}" / "recurrent_controls.jsonl"
                write_jsonl(control_shard_path, local_controls)
                torch.distributed.barrier(group=context.control_group)
                if context.is_primary:
                    all_controls = []
                    for rank in range(context.world_size):
                        all_controls.extend(
                            read_jsonl(
                                resolved_output / f"shard-{rank}" / "recurrent_controls.jsonl"
                            )
                        )
                    all_controls.sort(key=lambda row: (TASKS.index(str(row["task"])), int(row["line_index"])))
            del model, tokenizer
            gc.collect()
            torch.cuda.empty_cache()
            distributed.require_consensus(context, True, description=f"completed final system {system}")
        if context.is_primary:
            input_binding = {
                "schema": "rwkv_ms_natural_memory_native_recurrent_routed_final_input.v1",
                "protocol_file": str(PROTOCOL_FILE.resolve()),
                "protocol_receipt": PROTOCOL_RECEIPT,
                "final_opening": str(
                    (args.final_root / FINAL_OPENING_FILENAME).expanduser().resolve()
                ),
                "final_opening_receipt": FINAL_OPENING_RECEIPT,
                "training_root": str(TRAINING_ROOT.resolve()),
                "training_result_receipt": TRAINING_RESULT_RECEIPT,
                "development_result": str(DEVELOPMENT_RESULT.resolve()),
                "development_result_receipt": DEVELOPMENT_RESULT_RECEIPT,
                "base_model": str(args.base_model.expanduser().resolve()),
                "v9_adapter": str(args.v9_adapter.expanduser().resolve()),
                "candidate_adapter": str(args.candidate_adapter.expanduser().resolve()),
                "systems": list(requested_systems),
                "rows_per_task": {task: len(rows_by_task[task]) for task in TASKS},
                "world_size": context.world_size,
                "rank_devices": list(context.rank_devices),
                "generation": "single-pass greedy batch-one with one write then one read for memory systems",
                "task_router": False,
                "template_matcher": False,
                "dual_pass_selector": False,
                "benchmark_specific_decoder": False,
                "final_rows_opened": True,
                "publisher_validation_opened": False,
                "publisher_test_opened": PUBLISHER_TEST_OPENED,
            }
            write_json(resolved_output / "input_binding.json", input_binding)
            for system, records in all_records.items():
                with (resolved_output / f"{system}.jsonl").open("w", encoding="utf-8") as handle:
                    for row in records:
                        handle.write(json.dumps(row, ensure_ascii=True, sort_keys=True, separators=(",", ":")) + "\n")
            if all_controls:
                with (resolved_output / "recurrent_controls.jsonl").open("w", encoding="utf-8") as handle:
                    for row in all_controls:
                        handle.write(json.dumps(row, ensure_ascii=True, sort_keys=True, separators=(",", ":")) + "\n")
            expected_rows_per_task = (
                args.limit_per_task if args.limit_per_task is not None else 64
            )
            summary = {
                "systems": {
                    system: summarize_system(
                        records,
                        expected_rows_per_task=expected_rows_per_task,
                    )
                    for system, records in all_records.items()
                }
            }
            summary["controls"] = control_summary(all_controls) if all_controls else {"status": "not_run"}
            if set(requested_systems) == set(SYSTEMS) and not args.limit_per_task:
                metrics = {
                    system: {task: summary["systems"][system]["by_task"][task][metric]
                             for task, metric in (("attribution", "accuracy"), ("narrative", "unit_label_accuracy"), ("scene", "micro_f1"))}
                    for system in SYSTEMS
                }
                candidate = metrics["recurrent_routed_candidate"]
                stronger = {
                    task: max(metrics["frozen_gemma_base"][task], metrics["v9_projected_slot_baseline"][task])
                    for task in candidate
                }
                non_regressive = all(candidate[task] >= stronger[task] for task in candidate)
                strict_better = any(
                    candidate[task] > metrics["frozen_gemma_base"][task]
                    and candidate[task] > metrics["v9_projected_slot_baseline"][task]
                    for task in candidate
                )
                control_pass = summary["controls"].get("passed") is True
                summary["metrics"] = metrics
                summary["promotion_criteria"] = {
                    "candidate_at_least_stronger_baseline_every_task": non_regressive,
                    "candidate_strictly_better_than_both_one_task": strict_better,
                    "candidate_correct_state_beats_controls": control_pass,
                }
                passed = bool(non_regressive and strict_better and control_pass)
            else:
                passed = False
            summary["passed"] = passed
            result = {
                "schema": SCHEMA,
                "status": "final_pass" if passed else "final_failure",
                "passed": passed,
                "protocol_receipt": PROTOCOL_RECEIPT,
                "input_binding": input_binding,
                "summary": summary,
                "raw_prediction_files": {
                    system: {
                        "path": str((resolved_output / f"{system}.jsonl").resolve()),
                        "rows": len(records),
                        "sha256": sha256_file(resolved_output / f"{system}.jsonl"),
                    }
                    for system, records in all_records.items()
                },
                "causal_control_file": (
                    {
                        "path": str((resolved_output / "recurrent_controls.jsonl").resolve()),
                        "rows": len(all_controls),
                        "sha256": sha256_file(resolved_output / "recurrent_controls.jsonl"),
                    }
                    if all_controls
                    else None
                ),
                "final_rows_opened": True,
                "publisher_validation_opened": False,
                "publisher_test_opened": PUBLISHER_TEST_OPENED,
            }
            result["receipt"] = {"algorithm": "sha256", "payload_scope": "canonical_result_without_receipt", "payload_sha256": canonical_sha256(result)}
            write_json(resolved_output / "summary.json", summary)
            write_json(resolved_output / "result.json", result)
        distributed.gather_objects(context, True)
        if context.is_primary:
            return_code = 0 if passed else 1
        else:
            return_code = 0
        values = distributed.gather_objects(context, return_code)
        return int(values[0])
    finally:
        distributed.destroy_distributed_training(context)


if __name__ == "__main__":
    raise SystemExit(main())
