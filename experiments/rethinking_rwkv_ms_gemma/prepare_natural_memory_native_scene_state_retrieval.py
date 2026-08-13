#!/usr/bin/env python3
"""Build the label-free state-retrieval mapping for the native scene study."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path
import sys
from typing import Any, Iterable, Mapping, Sequence
import unicodedata

from transformers import AutoTokenizer


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = Path(__file__).resolve().parents[2]
for import_root in (SCRIPT_DIR, PROJECT_ROOT):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

from deltamem.chat_templates import apply_chat_template  # noqa: E402


SCHEMA = "rwkv_ms_natural_memory_native_scene_state_retrieval_mapping.v1"
PROTOCOL = SCRIPT_DIR / "natural_memory_native_scene_state_retrieval_protocol_v1.json"
AMENDMENT = (
    SCRIPT_DIR
    / "natural_memory_native_scene_state_retrieval_protocol_v1_amendment1.json"
)
PROTOCOL_PAYLOAD_SHA256 = (
    "efca688c361cfb28e8f9a0cc107bd2d1714032e61c2eec903348ad3d9b19f3cf"
)
AMENDMENT_PAYLOAD_SHA256 = (
    "db2c4e32d9aaf7a0b6a66314f3071dbb63176716fc1c0e590beb2465fe6c28d1"
)
TARGET_RELATIVE_PATH = "v4-scene-boundary-detection/train_derived_development.jsonl"
BANK_RELATIVE_PATH = "v4-scene-boundary-detection/train_derived_fit.jsonl"
TARGET_SHA256 = "b383625cee07e6a7565142e38bb0b0a4d4a2468b2c91171570115b7b311e1e68"
BANK_SHA256 = "8b0552cf1ddd39230896ce1ed6a3842aef94212e70bbc9e76ee8f13c546e6e57"
TOKENIZER_SHA256 = "cc8d3a0ce36466ccc1278bf987df5f71db1719b9ca6b4118264f45cb627bfe0f"
EXPECTED_TARGET_ROWS = 361
EXPECTED_BANK_ROWS = 1443
EXCLUDED_TARGET_ROWS = 4
NGRAM_SIZES = (2, 3, 4)
HYBRID_LENGTH_PENALTY = 0.15
RANDOM_NAMESPACE = "rwkv-ms-state-retrieval-v1:"
CANDIDATE_METHODS = (
    "length_nearest",
    "char_tfidf_nearest",
    "hybrid_char_length",
    "hash_random",
)
FIT_PARTITION_SHA256 = "81e0eae9807ebc82fe4b739820e6233d81e5f87c02afdb8a64bc9c99c93a6957"
HOLDOUT_PARTITION_SHA256 = "0167fe4bc2b532db4c097de480e54e7acb424c844142c9a730e5576f4dd8935f"
ALL_PARTITION_SHA256 = "ce6a4684e8858dd6e799229dd0d2ae3c9abf7e94f337e3237513bf17a4ce2e7d"


def canonical_sha256(value: Any) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_receipt(path: Path, expected: str) -> Mapping[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    receipt = value.get("receipt")
    if not isinstance(receipt, Mapping):
        raise ValueError(f"Receipt missing: {path}")
    unsigned = dict(value)
    unsigned.pop("receipt")
    digest = canonical_sha256(unsigned)
    if digest != expected or receipt.get("payload_sha256") != digest:
        raise ValueError(f"Receipt differs: {path}")
    return value


def validate_protocol() -> tuple[Mapping[str, Any], Mapping[str, Any]]:
    return (
        validate_receipt(PROTOCOL, PROTOCOL_PAYLOAD_SHA256),
        validate_receipt(AMENDMENT, AMENDMENT_PAYLOAD_SHA256),
    )


def load_prompt_rows(
    path: Path,
    *,
    expected_sha256: str,
    expected_rows: int,
) -> list[dict[str, Any]]:
    if sha256_file(path) != expected_sha256:
        raise ValueError(f"State-retrieval dataset hash differs: {path}")
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            if not raw_line.strip():
                continue
            value = json.loads(raw_line)
            messages = value.get("messages")
            if not isinstance(messages, list) or len(messages) < 3:
                raise ValueError(f"Invalid state-retrieval row: {path}")
            if messages[-1].get("role") != "assistant":
                raise ValueError(f"State-retrieval row lacks final assistant: {path}")
            rows.append(
                {
                    "source_index": len(rows),
                    "messages": messages[:-1],
                    "row_sha256": hashlib.sha256(
                        raw_line.rstrip("\n").encode("utf-8")
                    ).hexdigest(),
                }
            )
    if len(rows) != expected_rows:
        raise ValueError(
            f"Expected {expected_rows} state-retrieval rows, found {len(rows)}: {path}"
        )
    return rows


def normalize_user_text(messages: Sequence[Mapping[str, Any]]) -> str:
    parts = [
        str(message.get("content", ""))
        for message in messages
        if message.get("role") == "user"
    ]
    normalized = unicodedata.normalize("NFKC", " ".join(parts)).casefold()
    return " ".join(normalized.split())


def ngram_counts(text: str) -> Counter[tuple[int, str]]:
    return Counter(
        (size, text[offset : offset + size])
        for size in NGRAM_SIZES
        for offset in range(max(0, len(text) - size + 1))
    )


def fit_tfidf_index(
    texts: Sequence[str],
) -> tuple[
    dict[tuple[int, str], float],
    dict[tuple[int, str], list[tuple[int, float]]],
]:
    document_frequency: Counter[tuple[int, str]] = Counter()
    counts_by_document: list[Counter[tuple[int, str]]] = []
    for text in texts:
        counts = ngram_counts(text)
        counts_by_document.append(counts)
        document_frequency.update(counts.keys())
    document_count = len(texts)
    idf = {
        feature: math.log((1 + document_count) / (1 + frequency)) + 1.0
        for feature, frequency in document_frequency.items()
    }
    inverted: dict[tuple[int, str], list[tuple[int, float]]] = defaultdict(list)
    for document_index, counts in enumerate(counts_by_document):
        weighted = {
            feature: (1.0 + math.log(count)) * idf[feature]
            for feature, count in counts.items()
        }
        norm = math.sqrt(sum(value * value for value in weighted.values()))
        if norm == 0.0:
            continue
        for feature, value in weighted.items():
            inverted[feature].append((document_index, value / norm))
    return idf, dict(inverted)


def cosine_scores(
    text: str,
    *,
    idf: Mapping[tuple[int, str], float],
    inverted: Mapping[tuple[int, str], Sequence[tuple[int, float]]],
    document_count: int,
) -> list[float]:
    counts = ngram_counts(text)
    weighted = {
        feature: (1.0 + math.log(count)) * idf[feature]
        for feature, count in counts.items()
        if feature in idf
    }
    norm = math.sqrt(sum(value * value for value in weighted.values()))
    scores = [0.0] * document_count
    if norm == 0.0:
        return scores
    for feature, value in weighted.items():
        target_weight = value / norm
        for document_index, document_weight in inverted[feature]:
            scores[document_index] += target_weight * document_weight
    return scores


def write_prompt_token_counts(tokenizer, rows: Sequence[Mapping[str, Any]]) -> list[int]:
    counts: list[int] = []
    for row in rows:
        rendered = apply_chat_template(
            tokenizer,
            list(row["messages"]),
            tokenize=False,
            add_generation_prompt=False,
        )
        counts.append(len(tokenizer(rendered, add_special_tokens=False).input_ids))
    return counts


def partition_for_hash(row_sha256: str) -> str:
    return "holdout" if int(row_sha256[:8], 16) % 5 == 0 else "fit"


def partition_payload(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "source_index": int(row["source_index"]),
            "row_sha256": str(row["row_sha256"]),
            "partition": partition_for_hash(str(row["row_sha256"])),
        }
        for row in rows
        if int(row["source_index"]) >= EXCLUDED_TARGET_ROWS
    ]


def validate_partitions(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    payload = partition_payload(rows)
    fit = [record for record in payload if record["partition"] == "fit"]
    holdout = [record for record in payload if record["partition"] == "holdout"]
    expected = (
        (payload, ALL_PARTITION_SHA256, 357),
        (fit, FIT_PARTITION_SHA256, 289),
        (holdout, HOLDOUT_PARTITION_SHA256, 68),
    )
    for partition, digest, count in expected:
        if len(partition) != count or canonical_sha256(partition) != digest:
            raise ValueError("State-retrieval partition binding differs")
    return payload


def best_index(scores: Sequence[float]) -> int:
    return min(range(len(scores)), key=lambda index: (-scores[index], index))


def mapping_records(
    target_rows: Sequence[Mapping[str, Any]],
    bank_rows: Sequence[Mapping[str, Any]],
    *,
    target_tokens: Sequence[int],
    bank_tokens: Sequence[int],
    idf: Mapping[tuple[int, str], float],
    inverted: Mapping[tuple[int, str], Sequence[tuple[int, float]]],
) -> list[dict[str, Any]]:
    bank_by_index = {int(row["source_index"]): row for row in bank_rows}
    records: list[dict[str, Any]] = []
    for row in target_rows:
        source_index = int(row["source_index"])
        if source_index < EXCLUDED_TARGET_ROWS:
            continue
        similarities = cosine_scores(
            normalize_user_text(row["messages"]),
            idf=idf,
            inverted=inverted,
            document_count=len(bank_rows),
        )
        length_index = min(
            range(len(bank_rows)),
            key=lambda index: (
                abs(target_tokens[source_index] - bank_tokens[index]),
                index,
            ),
        )
        char_index = best_index(similarities)
        hybrid_scores = [
            similarity
            - HYBRID_LENGTH_PENALTY
            * abs(target_tokens[source_index] - bank_tokens[index])
            / max(target_tokens[source_index], bank_tokens[index], 1)
            for index, similarity in enumerate(similarities)
        ]
        hybrid_index = best_index(hybrid_scores)
        random_index = int(
            hashlib.sha256(
                f"{RANDOM_NAMESPACE}{row['row_sha256']}".encode("ascii")
            ).hexdigest()[:16],
            16,
        ) % len(bank_rows)
        selected = {
            "length_nearest": (
                length_index,
                -float(abs(target_tokens[source_index] - bank_tokens[length_index])),
            ),
            "char_tfidf_nearest": (char_index, similarities[char_index]),
            "hybrid_char_length": (hybrid_index, hybrid_scores[hybrid_index]),
            "hash_random": (random_index, None),
        }
        methods: dict[str, Any] = {}
        for method, (bank_index, score) in selected.items():
            bank_row = bank_by_index[bank_index]
            methods[method] = {
                "bank_source_index": bank_index,
                "bank_row_sha256": bank_row["row_sha256"],
                "bank_write_tokens": bank_tokens[bank_index],
                "absolute_write_token_delta": abs(
                    target_tokens[source_index] - bank_tokens[bank_index]
                ),
                "char_tfidf_cosine": similarities[bank_index],
                "selection_score": score,
            }
        records.append(
            {
                "target_source_index": source_index,
                "target_row_sha256": row["row_sha256"],
                "partition": partition_for_hash(str(row["row_sha256"])),
                "target_write_tokens": target_tokens[source_index],
                "methods": methods,
            }
        )
    return records


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-model", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    validate_protocol()
    base_model = args.base_model.expanduser().resolve(strict=True)
    dataset_root = args.dataset_root.expanduser().resolve(strict=True)
    output = args.output.expanduser().resolve()
    if output.exists():
        raise ValueError(f"State-retrieval mapping output must be fresh: {output}")
    if sha256_file(base_model / "tokenizer.json") != TOKENIZER_SHA256:
        raise ValueError("State-retrieval tokenizer hash differs")
    target_path = dataset_root / TARGET_RELATIVE_PATH
    bank_path = dataset_root / BANK_RELATIVE_PATH
    target_rows = load_prompt_rows(
        target_path,
        expected_sha256=TARGET_SHA256,
        expected_rows=EXPECTED_TARGET_ROWS,
    )
    bank_rows = load_prompt_rows(
        bank_path,
        expected_sha256=BANK_SHA256,
        expected_rows=EXPECTED_BANK_ROWS,
    )
    partitions = validate_partitions(target_rows)
    tokenizer = AutoTokenizer.from_pretrained(base_model, local_files_only=True)
    target_tokens = write_prompt_token_counts(tokenizer, target_rows)
    bank_tokens = write_prompt_token_counts(tokenizer, bank_rows)
    bank_texts = [normalize_user_text(row["messages"]) for row in bank_rows]
    idf, inverted = fit_tfidf_index(bank_texts)
    records = mapping_records(
        target_rows,
        bank_rows,
        target_tokens=target_tokens,
        bank_tokens=bank_tokens,
        idf=idf,
        inverted=inverted,
    )
    result: dict[str, Any] = {
        "schema": SCHEMA,
        "protocol_payload_sha256": PROTOCOL_PAYLOAD_SHA256,
        "amendment_payload_sha256": AMENDMENT_PAYLOAD_SHA256,
        "target_file": str(target_path),
        "target_file_sha256": TARGET_SHA256,
        "bank_file": str(bank_path),
        "bank_file_sha256": BANK_SHA256,
        "tokenizer_sha256": TOKENIZER_SHA256,
        "mapper_sha256": sha256_file(Path(__file__)),
        "target_rows": len(records),
        "bank_rows": len(bank_rows),
        "fit_rows": sum(record["partition"] == "fit" for record in records),
        "holdout_rows": sum(record["partition"] == "holdout" for record in records),
        "candidate_methods": list(CANDIDATE_METHODS),
        "tfidf": {
            "fit_corpus": "state_bank_only",
            "ngram_sizes": list(NGRAM_SIZES),
            "vocabulary_features": len(idf),
            "sublinear_tf": True,
            "l2_normalized": True,
        },
        "partition_payload_sha256": canonical_sha256(partitions),
        "records": records,
    }
    result["receipt"] = {
        "algorithm": "sha256",
        "payload_scope": "canonical_mapping_without_receipt",
        "payload_sha256": canonical_sha256(result),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        f"STATE_RETRIEVAL_MAPPING_COMPLETE targets={len(records)} "
        f"bank={len(bank_rows)} features={len(idf)}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
