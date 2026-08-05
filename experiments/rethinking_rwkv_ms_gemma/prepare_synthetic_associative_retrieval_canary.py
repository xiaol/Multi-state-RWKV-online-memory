#!/usr/bin/env python3
"""Build the locked four-row associative projected-KV retrieval canary."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from datasets import Dataset
from transformers import AutoTokenizer

from deltamem.train.delta_sft_experimental import (
    build_episode_training_examples,
    materialize_scene_state_identity_pairs,
)
from experiments.rethinking_rwkv_ms_gemma.prepare_synthetic_state_identity_canary import (
    MODEL_ARTIFACT_NAMES,
    atomic_write,
    bind_model_artifacts,
    canonical_json_bytes,
    canonical_sha256,
    sha256_bytes,
    sha256_file,
)


SOURCE_SCHEMA = "rwkv_ms_synthetic_associative_retrieval_source.v1"
ROW_SCHEMA = "rwkv_ms_synthetic_associative_retrieval_row.v1"
TASK_NAME = "synthetic-query-selected-associative-retrieval"
SOURCE_PURPOSE = "train_only_projected_kv_addressing_canary_v1"
SYSTEM_PROMPT = (
    "You are a deterministic associative lookup engine. Use the two memory "
    "records to answer the final query. Reply with exactly "
    '{"boundaries":[N]} and no other text.'
)
RECORD_TEMPLATE = "MEMORY RECORD. KEY {key}. VALUE {value}."
QUERY_TEMPLATE = "QUERY KEY {key}. Return its stored value."
RESPONSE_TEMPLATE = '{{"boundaries":[{value}]}}'
KEYS = ("ALPHA", "BETA")
MAPPINGS = ((1, 2), (2, 1))
CASE_LAYOUT = (
    (0, "ALPHA"),
    (1, "ALPHA"),
    (0, "BETA"),
    (1, "BETA"),
)
DONOR_INDICES = (1, 0, 3, 2)
EPISODE_CONTRACT = {
    "episode_recent_messages": 1,
    "write_phase": "system + two record messages",
    "read_supervision": "system + query + assistant",
}
ATOMIC_BATCH_SIZE = 4
RWKV_MS_NUM_STATES = 2
PROJECTED_KV_KEY_DIM = 32
MAX_READ_LENGTH = 128
MAX_WRITE_LENGTH = 128
GATE0_MIN_PAIR_LOGIT_MARGIN = 5.0


def _case_id(mapping_index: int, query_key: str) -> str:
    return f"mapping_{mapping_index}_{query_key.lower()}"


def canary_spec() -> dict[str, Any]:
    cases = []
    for mapping_index, query_key in CASE_LAYOUT:
        alpha_value, beta_value = MAPPINGS[mapping_index]
        answer = alpha_value if query_key == "ALPHA" else beta_value
        cases.append(
            {
                "case_id": _case_id(mapping_index, query_key),
                "mapping_index": mapping_index,
                "mapping": {"ALPHA": alpha_value, "BETA": beta_value},
                "query_key": query_key,
                "answer": answer,
                "donor_index": DONOR_INDICES[len(cases)],
            }
        )
    return {
        "schema": SOURCE_SCHEMA,
        "task": TASK_NAME,
        "purpose": SOURCE_PURPOSE,
        "system_prompt": SYSTEM_PROMPT,
        "cases": cases,
        "episode_contract": dict(EPISODE_CONTRACT),
        "objective": {
            "memory_loss_mode": "scene_state_identity_ce",
            "target_mode": "first_distinguishing_semantic_token",
            "target_span_tokens": 1,
            "atomic_batch_size": ATOMIC_BATCH_SIZE,
            "locked_donor_indices": list(DONOR_INDICES),
        },
        "memory_topology": {
            "backend": "rwkv_ms",
            "readout": "projected_kv_slots",
            "write_granularity": "message_mean",
            "num_states": RWKV_MS_NUM_STATES,
            "projected_kv_key_dim": PROJECTED_KV_KEY_DIM,
            "expected_record_proposals": 2,
        },
        "gate0": {
            "model_role": "frozen_full_context_no_adapter",
            "required_exact_generations": len(CASE_LAYOUT),
            "minimum_pair_logit_margin": GATE0_MIN_PAIR_LOGIT_MARGIN,
        },
    }


CANARY_SPEC_SHA256 = canonical_sha256(canary_spec())


def build_rows() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for mapping_index, query_key in CASE_LAYOUT:
        alpha_value, beta_value = MAPPINGS[mapping_index]
        answer = alpha_value if query_key == "ALPHA" else beta_value
        rows.append(
            {
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {
                        "role": "user",
                        "content": RECORD_TEMPLATE.format(
                            key="ALPHA",
                            value=alpha_value,
                        ),
                    },
                    {
                        "role": "assistant",
                        "content": RECORD_TEMPLATE.format(
                            key="BETA",
                            value=beta_value,
                        ),
                    },
                    {
                        "role": "user",
                        "content": QUERY_TEMPLATE.format(key=query_key),
                    },
                    {
                        "role": "assistant",
                        "content": RESPONSE_TEMPLATE.format(value=answer),
                    },
                ]
            }
        )
    return rows


def _active_message_ids(episode: dict[str, Any]) -> list[int]:
    return sorted(
        {
            int(message_id)
            for message_id in episode["write_message_ids"]
            if int(message_id) >= 0
        }
    )


def _tokenize_and_audit(
    tokenizer: Any,
    rows: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    episodes = [
        build_episode_training_examples(
            tokenizer,
            row["messages"],
            MAX_READ_LENGTH,
            assistant_loss_mode="final_assistant_only",
            episode_recent_messages=1,
            max_write_length=MAX_WRITE_LENGTH,
            include_sentence_ids=True,
            require_scene_state_semantic_mask=True,
        )[0]
        for row in rows
    ]
    paired, pairing_manifest = materialize_scene_state_identity_pairs(
        Dataset.from_list(episodes),
        split_name="train",
        locked_donor_indices=DONOR_INDICES,
    )
    if paired["scene_state_donor_index"] != list(DONOR_INDICES):
        raise ValueError("Trainer donor map differs from the locked associative map")
    if pairing_manifest.get("pairing_locked") is not True:
        raise ValueError("Trainer did not mark the associative donor map as locked")

    pair_records = pairing_manifest.get("pairs")
    if not isinstance(pair_records, list) or len(pair_records) != len(rows):
        raise ValueError("Trainer pairing audit does not cover all four rows")

    target_positions: list[int] = []
    target_token_ids: list[int] = []
    row_audits: list[dict[str, Any]] = []
    for index, (row, episode, pair_record) in enumerate(
        zip(rows, episodes, pair_records, strict=True)
    ):
        mapping_index, query_key = CASE_LAYOUT[index]
        active_message_ids = _active_message_ids(episode)
        if active_message_ids != [0, 1]:
            raise ValueError(
                f"Associative row {index} must produce exactly two record proposals; "
                f"got message IDs {active_message_ids}"
            )
        read_text = tokenizer.decode(
            episode["input_ids"],
            skip_special_tokens=False,
        )
        write_text = tokenizer.decode(
            episode["write_input_ids"],
            skip_special_tokens=False,
        )
        query_text = str(row["messages"][3]["content"])
        record_texts = [
            str(row["messages"][1]["content"]),
            str(row["messages"][2]["content"]),
        ]
        if query_text not in read_text or any(text in read_text for text in record_texts):
            raise ValueError(f"Associative row {index} read phase is not query-only")
        if query_text in write_text or any(text not in write_text for text in record_texts):
            raise ValueError(f"Associative row {index} query leaked into its writes")

        positions = pair_record.get("target_label_positions")
        source_ids = pair_record.get("target_token_ids")
        donor_ids = pair_record.get("donor_token_ids")
        if (
            not isinstance(positions, list)
            or len(positions) != 1
            or not isinstance(source_ids, list)
            or len(source_ids) != 1
            or not isinstance(donor_ids, list)
            or len(donor_ids) != 1
            or source_ids == donor_ids
        ):
            raise ValueError(f"Associative row {index} lacks a one-token donor distinction")
        target_position = int(positions[0])
        target_token_id = int(source_ids[0])
        if int(episode["labels"][target_position]) != target_token_id:
            raise ValueError(f"Associative row {index} target token is not supervised")
        target_positions.append(target_position)
        target_token_ids.append(target_token_id)
        row_audits.append(
            {
                "case_id": _case_id(mapping_index, query_key),
                "mapping_index": mapping_index,
                "query_key": query_key,
                "answer": canary_spec()["cases"][index]["answer"],
                "donor_index": DONOR_INDICES[index],
                "write_token_count": len(episode["write_input_ids"]),
                "write_token_ids_sha256": canonical_sha256(
                    [int(value) for value in episode["write_input_ids"]]
                ),
                "write_message_ids_sha256": canonical_sha256(
                    [int(value) for value in episode["write_message_ids"]]
                ),
                "record_proposal_count": len(active_message_ids),
                "query_visible_in_read": True,
                "query_excluded_from_write": True,
                "state_only_read_token_count": len(episode["input_ids"]),
                "state_only_read_token_ids_sha256": canonical_sha256(
                    [int(value) for value in episode["input_ids"]]
                ),
                "target_label_position": target_position,
                "target_token_id": target_token_id,
                "donor_target_token_id": int(donor_ids[0]),
                "target_token_text": tokenizer.decode([target_token_id]),
            }
        )

    for source_index, donor_index in enumerate(DONOR_INDICES):
        source_position = target_positions[source_index]
        donor_position = target_positions[donor_index]
        if source_position != donor_position:
            raise ValueError("Associative donor target positions differ")
        source_prefix = episodes[source_index]["input_ids"][:source_position]
        donor_prefix = episodes[donor_index]["input_ids"][:donor_position]
        if source_prefix != donor_prefix:
            raise ValueError("Associative donor causal read prefixes differ")
        if CASE_LAYOUT[source_index][1] != CASE_LAYOUT[donor_index][1]:
            raise ValueError("Associative donor pairing crossed query keys")

    return episodes, {
        "tokenizer_class": type(tokenizer).__name__,
        "max_read_length": MAX_READ_LENGTH,
        "max_write_length": MAX_WRITE_LENGTH,
        "episode_recent_messages": 1,
        "write_granularity": "message_mean",
        "record_proposals_per_row": 2,
        "minimum_required_occupied_slots": 2,
        "query_visible_in_every_read": True,
        "query_excluded_from_every_write": True,
        "identical_donor_causal_prefixes": True,
        "target_span_tokens": 1,
        "target_label_positions": target_positions,
        "target_token_ids": target_token_ids,
        "identity_donor_indices": list(DONOR_INDICES),
        "pairing_manifest_sha256": pairing_manifest["manifest_sha256"],
        "rows": row_audits,
    }


def build_bundle(model_path: Path, output_dir: Path) -> dict[str, Any]:
    resolved_output = output_dir.expanduser().resolve()
    if resolved_output.exists() and any(resolved_output.iterdir()):
        raise ValueError(
            f"Associative canary output directory must be fresh or empty: {resolved_output}"
        )
    model = bind_model_artifacts(model_path)
    tokenizer = AutoTokenizer.from_pretrained(model["path"], local_files_only=True)
    rows = build_rows()
    _, token_audit = _tokenize_and_audit(tokenizer, rows)

    train_lines = [
        json.dumps(row, ensure_ascii=True, sort_keys=True, separators=(",", ":"))
        for row in rows
    ]
    train_payload = ("\n".join(train_lines) + "\n").encode("utf-8")
    train_path = (resolved_output / "train.jsonl").resolve()

    row_records = []
    for index, (line, audit) in enumerate(
        zip(train_lines, token_audit["rows"], strict=True)
    ):
        row_records.append(
            {
                "schema": ROW_SCHEMA,
                "source_split": "train",
                "train_row_ordinal": index,
                "case_id": audit["case_id"],
                "mapping_index": audit["mapping_index"],
                "query_key": audit["query_key"],
                "answer": audit["answer"],
                "donor_index": audit["donor_index"],
                "row_sha256": sha256_bytes(line.encode("utf-8")),
                "token_metadata": {
                    key: audit[key]
                    for key in (
                        "write_token_count",
                        "write_token_ids_sha256",
                        "write_message_ids_sha256",
                        "record_proposal_count",
                        "query_visible_in_read",
                        "query_excluded_from_write",
                        "state_only_read_token_count",
                        "state_only_read_token_ids_sha256",
                        "target_label_position",
                        "target_token_id",
                        "donor_target_token_id",
                        "target_token_text",
                    )
                },
            }
        )
    rows_payload = b"".join(
        canonical_json_bytes(record) + b"\n" for record in row_records
    )
    rows_path = (resolved_output / "source_rows.jsonl").resolve()

    atomic_write(train_path, train_payload)
    atomic_write(rows_path, rows_payload)
    contract = {
        "episode_contract": dict(EPISODE_CONTRACT),
        "identity_donor_indices": list(DONOR_INDICES),
        "query_visible_during_read": True,
        "query_excluded_from_writes": True,
        "protected_evaluation_included": False,
        "synthetic_data_only": True,
    }
    manifest: dict[str, Any] = {
        "schema": SOURCE_SCHEMA,
        "task": TASK_NAME,
        "purpose": SOURCE_PURPOSE,
        "spec_sha256": CANARY_SPEC_SHA256,
        "spec": canary_spec(),
        "contract": contract,
        "model": model,
        "partitions": {
            "train": {
                "rows": len(rows),
                "source_split": "train",
                "data": {
                    "path": str(train_path),
                    "bytes": len(train_payload),
                    "sha256": sha256_bytes(train_payload),
                },
                "row_manifest": {
                    "path": str(rows_path),
                    "bytes": len(rows_payload),
                    "sha256": sha256_bytes(rows_payload),
                },
            }
        },
        "canary": token_audit,
    }
    manifest["manifest_sha256"] = canonical_sha256(manifest)
    manifest_path = (resolved_output / "source_manifest.json").resolve()
    atomic_write(
        manifest_path,
        json.dumps(manifest, ensure_ascii=True, indent=2, sort_keys=True).encode(
            "utf-8"
        )
        + b"\n",
    )
    return {
        "output_dir": str(resolved_output),
        "train_file": str(train_path),
        "train_file_sha256": sha256_file(train_path),
        "source_rows": str(rows_path),
        "source_rows_sha256": sha256_file(rows_path),
        "source_manifest": str(manifest_path),
        "source_manifest_file_sha256": sha256_file(manifest_path),
        "source_manifest_sha256": manifest["manifest_sha256"],
        "model_identity_sha256": model["identity_sha256"],
        "rows": len(rows),
    }


def load_source_bundle(
    source_manifest: Path,
    *,
    model_path: Path | None = None,
    verify_model_hashes: bool = False,
) -> dict[str, Any]:
    manifest_path = source_manifest.expanduser().resolve()
    if not manifest_path.is_file() or manifest_path.is_symlink():
        raise ValueError(f"Associative source manifest is invalid: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(manifest, dict):
        raise ValueError("Associative source manifest must be a JSON object")
    unsigned = dict(manifest)
    declared_hash = unsigned.pop("manifest_sha256", None)
    actual_hash = canonical_sha256(unsigned)
    expected_contract = {
        "episode_contract": dict(EPISODE_CONTRACT),
        "identity_donor_indices": list(DONOR_INDICES),
        "query_visible_during_read": True,
        "query_excluded_from_writes": True,
        "protected_evaluation_included": False,
        "synthetic_data_only": True,
    }
    if declared_hash != actual_hash:
        raise ValueError("Associative source-manifest canonical SHA-256 differs")
    if (
        manifest.get("schema") != SOURCE_SCHEMA
        or manifest.get("task") != TASK_NAME
        or manifest.get("purpose") != SOURCE_PURPOSE
        or manifest.get("spec_sha256") != CANARY_SPEC_SHA256
        or manifest.get("spec") != canary_spec()
        or manifest.get("contract") != expected_contract
    ):
        raise ValueError("Associative source-manifest identity differs")

    partitions = manifest.get("partitions")
    train = partitions.get("train") if isinstance(partitions, dict) else None
    if (
        not isinstance(partitions, dict)
        or set(partitions) != {"train"}
        or not isinstance(train, dict)
        or train.get("rows") != len(CASE_LAYOUT)
        or train.get("source_split") != "train"
    ):
        raise ValueError("Associative source must contain the exact four-row train partition")

    def bound_artifact(record: Any, description: str) -> tuple[Path, str]:
        if not isinstance(record, dict):
            raise ValueError(f"Associative source omits {description}")
        path = Path(str(record.get("path", ""))).expanduser().resolve()
        if not path.is_file() or path.is_symlink():
            raise ValueError(f"Associative {description} is invalid: {path}")
        digest = sha256_file(path)
        if record.get("sha256") != digest or record.get("bytes") != path.stat().st_size:
            raise ValueError(f"Associative {description} binding differs")
        return path, digest

    train_path, train_sha256 = bound_artifact(train.get("data"), "train data")
    rows_path, rows_sha256 = bound_artifact(
        train.get("row_manifest"),
        "row manifest",
    )
    raw_train = train_path.read_text(encoding="utf-8").splitlines()
    raw_rows = rows_path.read_text(encoding="utf-8").splitlines()
    parsed_train = [json.loads(line) for line in raw_train]
    row_records = [json.loads(line) for line in raw_rows]
    if parsed_train != build_rows() or len(row_records) != len(CASE_LAYOUT):
        raise ValueError("Associative source rows differ from the locked prompt spec")
    for index, (raw_line, record) in enumerate(
        zip(raw_train, row_records, strict=True)
    ):
        mapping_index, query_key = CASE_LAYOUT[index]
        if (
            not isinstance(record, dict)
            or record.get("schema") != ROW_SCHEMA
            or record.get("source_split") != "train"
            or record.get("train_row_ordinal") != index
            or record.get("case_id") != _case_id(mapping_index, query_key)
            or record.get("mapping_index") != mapping_index
            or record.get("query_key") != query_key
            or record.get("donor_index") != DONOR_INDICES[index]
            or record.get("row_sha256") != sha256_bytes(raw_line.encode("utf-8"))
        ):
            raise ValueError(f"Associative row identity differs at index {index}")

    canary = manifest.get("canary")
    if (
        not isinstance(canary, dict)
        or canary.get("episode_recent_messages") != 1
        or canary.get("write_granularity") != "message_mean"
        or canary.get("record_proposals_per_row") != 2
        or canary.get("minimum_required_occupied_slots") != 2
        or canary.get("query_visible_in_every_read") is not True
        or canary.get("query_excluded_from_every_write") is not True
        or canary.get("identical_donor_causal_prefixes") is not True
        or canary.get("identity_donor_indices") != list(DONOR_INDICES)
    ):
        raise ValueError("Associative tokenizer audit differs")

    model_record = manifest.get("model")
    if not isinstance(model_record, dict):
        raise ValueError("Associative source manifest omits model provenance")
    resolved_model_path = Path(str(model_record.get("path", ""))).expanduser().resolve()
    artifacts = model_record.get("artifacts")
    if (
        not resolved_model_path.is_dir()
        or resolved_model_path.is_symlink()
        or not isinstance(artifacts, dict)
        or set(artifacts) != set(MODEL_ARTIFACT_NAMES)
        or model_record.get("identity_sha256") != canonical_sha256(artifacts)
    ):
        raise ValueError("Associative model identity differs")
    if model_path is not None and resolved_model_path != model_path.expanduser().resolve():
        raise ValueError("Associative model path differs from source provenance")
    if verify_model_hashes and bind_model_artifacts(resolved_model_path) != model_record:
        raise ValueError("Current model artifacts differ from associative provenance")

    return {
        "manifest": manifest,
        "manifest_path": manifest_path,
        "manifest_file_sha256": sha256_file(manifest_path),
        "manifest_sha256": actual_hash,
        "train_path": train_path,
        "train_sha256": train_sha256,
        "rows_path": rows_path,
        "rows_sha256": rows_sha256,
        "rows": parsed_train,
        "row_records": row_records,
        "model": model_record,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = build_bundle(args.model_path, args.output_dir)
    print(json.dumps(result, ensure_ascii=True, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
