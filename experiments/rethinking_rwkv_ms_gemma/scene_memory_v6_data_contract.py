#!/usr/bin/env python3
"""Validate the frozen scene-memory V6 identity-proof data contract.

Training consumes 32 frozen-base failures copied exactly from the official
scene-v4 train split. Checkpoint selection is restricted to a predeclared 32-row
official-validation subset. The test split is opened only for provenance and
overlap auditing and is never emitted by this module.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
import re
import sys
import unicodedata
from typing import Iterable, Mapping, Sequence


SCHEMA = "rwkv_ms_scene_v6_identity_data.v1"
TASK = "scene-v4-current"
DATASET_ROOT = Path(
    "/run/media/xiaol/B214449214445C0B/datasets/novel-agent-sft-dataset"
)
TASK_DIR = DATASET_ROOT / "training/v4-scene-boundary-detection"
SPLIT_ORDER = ("train", "val", "test")
PARAGRAPH_ANCHOR = re.compile(r"^\[P(\d+)\]\s*", re.MULTILINE)
PARAGRAPH_SPLIT_ANCHOR = re.compile(r"^\[P\d+\]\s*", re.MULTILINE)
EXPECTED_ROLES = ("system", "user", "assistant")
PAIR_ROOT = Path(
    "/run/media/xiaol/B214449214445C0B/delta_mem_data/scene_failure_state/"
    "pairs_candidate64_failure32_holdout32_v1"
)
PAIR_MANIFEST = PAIR_ROOT / "manifest.json"
PAIR_MANIFEST_SHA256 = "2ceb291b9c21063164e30ca0b8b052798f8ba42d9a089a5abc78d1cb321dc008"
PROOF_TRAIN = PAIR_ROOT / "train.jsonl"
PROOF_TRAIN_SHA256 = "5f35f6ed41a2edaf88afee83626f17c34da38f5cb61cf4b6796a03eaae38f897"
PROOF_TRAIN_ROW_MANIFEST = PAIR_ROOT / "train_manifest.jsonl"
PROOF_TRAIN_ROW_MANIFEST_SHA256 = (
    "d112056a80b9dc13728b021646c0fbe3da5c3c41641fb28bb8c5448b1f8427fa"
)
HARD32_DATA = PAIR_ROOT / "holdout.jsonl"
HARD32_DATA_SHA256 = "b5b1137de89f82eee4b3ae3e3c7b5305240699ec7b65e84b61cb415a7a000d4a"
HARD32_ROW_MANIFEST = PAIR_ROOT / "holdout_manifest.jsonl"
HARD32_ROW_MANIFEST_SHA256 = (
    "6802d992805164342ea4ed16b9113814ee472ad363aa76eaf5298147e7a0d1cc"
)
HARD32_INDICES = PAIR_ROOT / "holdout_source_indices.json"
HARD32_INDICES_SHA256 = "76d510e5f02c30f2cf3a0262cc4c97d69ef8f52861e9ced855d530b233d916db"
HARD32_SOURCE_INDICES = (
    3, 6, 16, 21, 24, 30, 33, 47, 50, 56, 59, 63, 64, 66, 67, 70,
    71, 74, 75, 79, 87, 88, 102, 112, 113, 128, 132, 141, 144, 151,
    159, 166,
)
PROOF_TRAIN_ROW_HASHES_SHA256 = (
    "bc07dbe4cb16e0d40284c7120fab2dd6c36f10032dc306d46cc59b960793e2c1"
)
HARD32_ROW_HASHES_SHA256 = (
    "0909e6f436dedd4c78c77e459d05f561d4ba93d110c8c86c84a65f7d97420660"
)
PROOF_TRAIN_PROMPT_HASHES_SHA256 = (
    "8c6f10689b4be6688d67c4bcd75c19b295531837696dfd7252af5270e15cfb9f"
)
HARD32_PROMPT_HASHES_SHA256 = (
    "99e4551f351939ef44587f84ec1cbb9ef12d53cfa4e91a7df1a583e6e434288c"
)


@dataclass(frozen=True)
class SplitLock:
    rows: int
    file_sha256: str
    row_hashes_sha256: str
    prompt_hashes_sha256: str
    paragraph_rows_sha256: str
    unique_paragraph_hashes_sha256: str
    paragraph_instances: int
    unique_paragraphs: int


@dataclass(frozen=True)
class OverlapLock:
    shared_prompts: int
    shared_paragraphs: int
    left_rows_exposed: int
    right_rows_exposed: int
    shared_paragraph_hashes_sha256: str


SPLIT_LOCKS: Mapping[str, SplitLock] = {
    "train": SplitLock(
        rows=1804,
        file_sha256="785fe54c0a4e5c64e33f64f9bc88d64719576407c21eb0d520f9dec5a59b8e22",
        row_hashes_sha256="194ea6dced92378aca08408964923da0890c5fe41559fc647e4f9c05d182a78b",
        prompt_hashes_sha256="86294e663790e87aef35615f6465c741fdaef680c8cb587e6caed99d64ab8a31",
        paragraph_rows_sha256="9a1a4f888631e796be2314abb7732ef29d40c20b4abd9aaa1578b62414206c5e",
        unique_paragraph_hashes_sha256="f83dc66f279a7cb0d07415f374bcf63111b5f6804d7aa55e43956bad60776b8e",
        paragraph_instances=24183,
        unique_paragraphs=14374,
    ),
    "val": SplitLock(
        rows=170,
        file_sha256="61e94bcc536a124b07aef2c38ba285d7073d94a223866b58ddc7e5e1f509d513",
        row_hashes_sha256="cc6618aafcde5fbd411c264bec096ab599ef646752430015f2722796ea76d3f8",
        prompt_hashes_sha256="eb06a84436f07d9141ca4634694a7229e59a6cf3b9382a0bf0e8a3b8cf4b001e",
        paragraph_rows_sha256="923a1f7e6642c97657592ed91dbd913e22a95f28519ff6fe1dc3acf24641c02c",
        unique_paragraph_hashes_sha256="e5bf0577c2a4774b715a95c5424895a39b670774b2568df6404ace6a3da872ca",
        paragraph_instances=2261,
        unique_paragraphs=2176,
    ),
    "test": SplitLock(
        rows=149,
        file_sha256="d8b50ca3862bd40f023155bd14aa7b25d9d5dd3db4ea1c4d5a7e6f4f79cdfd6d",
        row_hashes_sha256="7d55a84f0c07132af74eca8115fe80efeca0380849ad90e2ed4c0e1e59d948ec",
        prompt_hashes_sha256="d23763a292f4aa53e4088e83c4a28219dd658897d19331ed4f16a848cc0b1700",
        paragraph_rows_sha256="ded3d69dc41bb98dea339314358eb6df45734ea5db7f44282ddcf53e4bdaf58d",
        unique_paragraph_hashes_sha256="7535f9b7f6520fa9924ca68da75bfb3a171ab501c6869a4e2d011b3dd3c039db",
        paragraph_instances=1992,
        unique_paragraphs=1878,
    ),
}

OVERLAP_LOCKS: Mapping[tuple[str, str], OverlapLock] = {
    ("train", "val"): OverlapLock(
        shared_prompts=0,
        shared_paragraphs=1451,
        left_rows_exposed=257,
        right_rows_exposed=131,
        shared_paragraph_hashes_sha256="463c6f1ca60ab75d27f9ffb1a694849f0af377ea9fef48f64e61321a1ab7ae3b",
    ),
    ("train", "test"): OverlapLock(
        shared_prompts=0,
        shared_paragraphs=1307,
        left_rows_exposed=227,
        right_rows_exposed=118,
        shared_paragraph_hashes_sha256="a2cd49e7e220b037857ac0e07438125a60c0d47c61921ca15e3ff12d076645ea",
    ),
    ("val", "test"): OverlapLock(
        shared_prompts=0,
        shared_paragraphs=223,
        left_rows_exposed=25,
        right_rows_exposed=26,
        shared_paragraph_hashes_sha256="cd7be63cb10e4d53d18c4a2288181a2fc3977763c49cfe784ab97c1f588ec2c5",
    ),
}

PROOF_SPLIT_LOCKS: Mapping[str, SplitLock] = {
    "failure32_train": SplitLock(
        rows=32,
        file_sha256=PROOF_TRAIN_SHA256,
        row_hashes_sha256=PROOF_TRAIN_ROW_HASHES_SHA256,
        prompt_hashes_sha256="9bbf921c639972a79ca185de95f8a6d1e3005e1f1ebc57e97b5ef0d6e095ee23",
        paragraph_rows_sha256="69cc46065213dfd9bbc0f08061b11a7512bb38c31d903b9687c2498f08a0181c",
        unique_paragraph_hashes_sha256="988d08886f5749848553809989746effdcfb4088064c256ac806554063713bbc",
        paragraph_instances=433,
        unique_paragraphs=426,
    ),
    "fixed_val32": SplitLock(
        rows=32,
        file_sha256=HARD32_DATA_SHA256,
        row_hashes_sha256=HARD32_ROW_HASHES_SHA256,
        prompt_hashes_sha256="f7eaed74f7449df2f8751be3e9ff59522a9c636dccd0c070fc3aa4d66ccd4993",
        paragraph_rows_sha256="d897ef2a6b4d42b14e4e7ede53177fcdf5c9d3e8242ac6f21c55791883a0bc6e",
        unique_paragraph_hashes_sha256="e01ee3766bdc0093cec804101bdf0bf8b83355bdd26725772b4990a375044f3c",
        paragraph_instances=412,
        unique_paragraphs=412,
    ),
}

PROOF_OVERLAP_LOCK = OverlapLock(
    shared_prompts=0,
    shared_paragraphs=0,
    left_rows_exposed=0,
    right_rows_exposed=0,
    shared_paragraph_hashes_sha256=(
        "4f53cda18c2baa0c0354bb5f9a3ecbe5ed12ab4d8e11ba873c2f11161202b945"
    ),
)


class ContractError(ValueError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ContractError(message)


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_text(payload: str) -> str:
    return sha256_bytes(payload.encode("utf-8"))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_sha256(payload: object) -> str:
    encoded = json.dumps(
        payload,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return sha256_bytes(encoded)


def normalize_overlap_text(value: str) -> str:
    """Apply the locked NFKC plus Unicode-whitespace removal rule."""

    normalized = unicodedata.normalize("NFKC", value)
    return "".join(character for character in normalized if not character.isspace())


def normalized_paragraphs(prompt: str) -> list[str]:
    """Extract anchored passage paragraphs, excluding any prompt preamble."""

    anchors = list(PARAGRAPH_SPLIT_ANCHOR.finditer(prompt))
    segments = (
        normalize_overlap_text(
            prompt[anchor.end() : anchors[index + 1].start()]
            if index + 1 < len(anchors)
            else prompt[anchor.end() :]
        )
        for index, anchor in enumerate(anchors)
    )
    return list(dict.fromkeys(segment for segment in segments if segment))


def _read_jsonl(path: Path) -> list[tuple[str, object]]:
    records: list[tuple[str, object]] = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        for line_number, line in enumerate(handle, start=1):
            raw_line = line.rstrip("\r\n")
            require(bool(raw_line.strip()), f"blank JSONL row at {path}:{line_number}")
            try:
                payload = json.loads(raw_line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSON at {path}:{line_number}") from exc
            records.append((raw_line, payload))
    return records


def _validate_row(payload: object, *, split: str, row_number: int) -> tuple[str, str]:
    prefix = f"{split} row {row_number}"
    require(isinstance(payload, dict), f"{prefix} must be an object")
    require(set(payload) == {"messages"}, f"{prefix} must be messages-only")
    messages = payload["messages"]
    require(isinstance(messages, list) and len(messages) == 3, f"{prefix} must contain three messages")
    require(all(isinstance(message, dict) for message in messages), f"{prefix} has a non-object message")
    require(
        tuple(message.get("role") for message in messages) == EXPECTED_ROLES,
        f"{prefix} roles must be system,user,assistant",
    )
    require(
        all(isinstance(message.get("content"), str) and message["content"] for message in messages),
        f"{prefix} message content must be non-empty text",
    )
    user_prompt = messages[1]["content"]
    paragraph_numbers = [int(match.group(1)) for match in PARAGRAPH_ANCHOR.finditer(user_prompt)]
    require(
        paragraph_numbers == list(range(1, len(paragraph_numbers) + 1)),
        f"{prefix} paragraph anchors must be contiguous from P1",
    )
    require(len(paragraph_numbers) >= 2, f"{prefix} must contain at least two paragraphs")
    try:
        assistant_payload = json.loads(messages[2]["content"])
    except json.JSONDecodeError as exc:
        raise ValueError(f"{prefix} assistant content must be JSON") from exc
    require(
        isinstance(assistant_payload, dict) and set(assistant_payload) == {"boundaries"},
        f"{prefix} assistant payload must contain only boundaries",
    )
    boundaries = assistant_payload["boundaries"]
    require(isinstance(boundaries, list), f"{prefix} boundaries must be a list")
    require(
        all(isinstance(boundary, int) and not isinstance(boundary, bool) for boundary in boundaries),
        f"{prefix} boundaries must contain integers",
    )
    require(boundaries == sorted(set(boundaries)), f"{prefix} boundaries must be sorted and unique")
    require(
        all(1 <= boundary < len(paragraph_numbers) for boundary in boundaries),
        f"{prefix} contains an out-of-range boundary",
    )
    return user_prompt, messages[2]["content"]


def audit_split(path: Path, *, split: str) -> dict[str, object]:
    records = _read_jsonl(path)
    row_hashes: list[str] = []
    prompt_hashes: list[str] = []
    paragraph_hash_rows: list[list[str]] = []
    for row_number, (raw_line, payload) in enumerate(records, start=1):
        prompt, _ = _validate_row(payload, split=split, row_number=row_number)
        row_hashes.append(sha256_text(raw_line))
        prompt_hashes.append(sha256_text(normalize_overlap_text(prompt)))
        paragraph_hash_rows.append(
            [sha256_text(paragraph) for paragraph in normalized_paragraphs(prompt)]
        )
    unique_paragraph_hashes = sorted(
        {paragraph_hash for row in paragraph_hash_rows for paragraph_hash in row}
    )
    return {
        "source_split": split,
        "path": str(path.resolve()),
        "sha256": sha256_file(path),
        "rows": len(records),
        "row_hashes_sha256": canonical_sha256(row_hashes),
        "normalized_full_prompt_hashes": prompt_hashes,
        "normalized_full_prompt_hashes_sha256": canonical_sha256(prompt_hashes),
        "unique_normalized_full_prompts": len(set(prompt_hashes)),
        "normalized_paragraphs": {
            "per_row_unique_instances": sum(len(row) for row in paragraph_hash_rows),
            "unique": len(unique_paragraph_hashes),
            "per_row_hashes_sha256": canonical_sha256(paragraph_hash_rows),
            "unique_hashes_sha256": canonical_sha256(unique_paragraph_hashes),
        },
        "_paragraph_hash_rows": paragraph_hash_rows,
    }


def audit_overlap(left: Mapping[str, object], right: Mapping[str, object]) -> dict[str, object]:
    left_prompt_hashes = set(left["normalized_full_prompt_hashes"])
    right_prompt_hashes = set(right["normalized_full_prompt_hashes"])
    left_rows: Sequence[Sequence[str]] = left["_paragraph_hash_rows"]  # type: ignore[assignment]
    right_rows: Sequence[Sequence[str]] = right["_paragraph_hash_rows"]  # type: ignore[assignment]
    left_paragraphs = {item for row in left_rows for item in row}
    right_paragraphs = {item for row in right_rows for item in row}
    shared_paragraphs = sorted(left_paragraphs & right_paragraphs)
    shared_set = set(shared_paragraphs)
    return {
        "left_split": left["source_split"],
        "right_split": right["source_split"],
        "exact_normalized_full_prompts_shared": len(left_prompt_hashes & right_prompt_hashes),
        "exact_normalized_paragraphs_shared": len(shared_paragraphs),
        "left_rows_with_shared_paragraph": sum(bool(set(row) & shared_set) for row in left_rows),
        "right_rows_with_shared_paragraph": sum(bool(set(row) & shared_set) for row in right_rows),
        "shared_paragraph_hashes_sha256": canonical_sha256(shared_paragraphs),
    }


def _assert_split_lock(split: str, record: Mapping[str, object], lock: SplitLock) -> None:
    paragraph_record = record["normalized_paragraphs"]
    require(isinstance(paragraph_record, dict), f"{split} paragraph audit is malformed")
    actual = {
        "rows": record["rows"],
        "file_sha256": record["sha256"],
        "row_hashes_sha256": record["row_hashes_sha256"],
        "prompt_hashes_sha256": record["normalized_full_prompt_hashes_sha256"],
        "paragraph_rows_sha256": paragraph_record["per_row_hashes_sha256"],
        "unique_paragraph_hashes_sha256": paragraph_record["unique_hashes_sha256"],
        "paragraph_instances": paragraph_record["per_row_unique_instances"],
        "unique_paragraphs": paragraph_record["unique"],
    }
    expected = {
        "rows": lock.rows,
        "file_sha256": lock.file_sha256,
        "row_hashes_sha256": lock.row_hashes_sha256,
        "prompt_hashes_sha256": lock.prompt_hashes_sha256,
        "paragraph_rows_sha256": lock.paragraph_rows_sha256,
        "unique_paragraph_hashes_sha256": lock.unique_paragraph_hashes_sha256,
        "paragraph_instances": lock.paragraph_instances,
        "unique_paragraphs": lock.unique_paragraphs,
    }
    require(actual == expected, f"{split} split differs from locked official data: expected={expected} actual={actual}")


def _assert_overlap_lock(pair: tuple[str, str], record: Mapping[str, object], lock: OverlapLock) -> None:
    actual = {
        "shared_prompts": record["exact_normalized_full_prompts_shared"],
        "shared_paragraphs": record["exact_normalized_paragraphs_shared"],
        "left_rows_exposed": record["left_rows_with_shared_paragraph"],
        "right_rows_exposed": record["right_rows_with_shared_paragraph"],
        "shared_paragraph_hashes_sha256": record["shared_paragraph_hashes_sha256"],
    }
    expected = {
        "shared_prompts": lock.shared_prompts,
        "shared_paragraphs": lock.shared_paragraphs,
        "left_rows_exposed": lock.left_rows_exposed,
        "right_rows_exposed": lock.right_rows_exposed,
        "shared_paragraph_hashes_sha256": lock.shared_paragraph_hashes_sha256,
    }
    require(actual == expected, f"overlap audit {pair} differs from lock: expected={expected} actual={actual}")


def _load_json_object(path: Path, *, description: str) -> dict[str, object]:
    require(path.is_file() and not path.is_symlink(), f"{description} is missing: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"{description} is invalid JSON: {path}") from exc
    require(isinstance(payload, dict), f"{description} must be an object: {path}")
    return payload


def _partition_row_hashes(path: Path, *, split: str) -> tuple[list[str], list[str]]:
    row_hashes: list[str] = []
    prompt_hashes: list[str] = []
    for row_number, (raw_line, payload) in enumerate(_read_jsonl(path), start=1):
        prompt, _ = _validate_row(payload, split=split, row_number=row_number)
        row_hashes.append(sha256_text(raw_line))
        prompt_hashes.append(sha256_text(prompt))
    return row_hashes, prompt_hashes


def _validate_frozen_pair_bundle(
    official_records: Mapping[str, Sequence[tuple[str, object]]],
) -> dict[str, object]:
    expected_files = {
        PAIR_MANIFEST: PAIR_MANIFEST_SHA256,
        PROOF_TRAIN: PROOF_TRAIN_SHA256,
        PROOF_TRAIN_ROW_MANIFEST: PROOF_TRAIN_ROW_MANIFEST_SHA256,
        HARD32_DATA: HARD32_DATA_SHA256,
        HARD32_ROW_MANIFEST: HARD32_ROW_MANIFEST_SHA256,
        HARD32_INDICES: HARD32_INDICES_SHA256,
    }
    for path, expected_sha256 in expected_files.items():
        require(path.is_file() and not path.is_symlink(), f"frozen pair artifact is missing: {path}")
        require(
            sha256_file(path) == expected_sha256,
            f"frozen pair artifact SHA-256 differs: {path}",
        )

    pair_manifest = _load_json_object(PAIR_MANIFEST, description="pair manifest")
    require(pair_manifest.get("schema") == "rwkv_ms_scene_failure_pairs.v1", "pair manifest schema differs")
    require(pair_manifest.get("task") == TASK, "pair manifest task differs")
    config = pair_manifest.get("config")
    require(
        isinstance(config, dict)
        and config.get("train_failure_count") == 32
        and config.get("holdout_count") == 32,
        "pair manifest partition sizes differ",
    )
    sources = pair_manifest.get("sources")
    require(isinstance(sources, dict), "pair manifest source records are missing")
    for split, split_lock in SPLIT_LOCKS.items():
        source = sources.get(split)
        require(isinstance(source, dict), f"pair manifest source is missing: {split}")
        require(
            source.get("path") == str(TASK_DIR / f"{split}.jsonl")
            and source.get("sha256") == split_lock.file_sha256
            and source.get("rows") == split_lock.rows,
            f"pair manifest official source differs: {split}",
        )
    require(
        sources["test"].get("emitted_for_training") is False
        and sources["test"].get("emitted_for_holdout") is False,
        "pair manifest must leave test untouched",
    )

    partitions = pair_manifest.get("partitions")
    require(isinstance(partitions, dict), "pair manifest partitions are missing")
    train_record = partitions.get("train")
    holdout_record = partitions.get("holdout")
    require(isinstance(train_record, dict) and isinstance(holdout_record, dict), "pair partitions are malformed")
    require(
        train_record.get("rows") == 32
        and train_record.get("source_split") == "train"
        and train_record.get("row_hashes_sha256") == PROOF_TRAIN_ROW_HASHES_SHA256
        and train_record.get("prompt_hashes_sha256") == PROOF_TRAIN_PROMPT_HASHES_SHA256,
        "proof-train manifest partition differs",
    )
    require(
        holdout_record.get("rows") == 32
        and holdout_record.get("source_split") == "val"
        and holdout_record.get("row_hashes_sha256") == HARD32_ROW_HASHES_SHA256
        and holdout_record.get("prompt_hashes_sha256") == HARD32_PROMPT_HASHES_SHA256,
        "hard32 manifest partition differs",
    )

    train_rows, train_prompts = _partition_row_hashes(PROOF_TRAIN, split="proof_train")
    hard_rows, hard_prompts = _partition_row_hashes(HARD32_DATA, split="hard32")
    require(len(train_rows) == 32 and len(hard_rows) == 32, "proof partitions must contain 32 rows")
    require(canonical_sha256(train_rows) == PROOF_TRAIN_ROW_HASHES_SHA256, "proof-train row digest differs")
    require(canonical_sha256(hard_rows) == HARD32_ROW_HASHES_SHA256, "hard32 row digest differs")
    require(canonical_sha256(train_prompts) == PROOF_TRAIN_PROMPT_HASHES_SHA256, "proof-train prompt digest differs")
    require(canonical_sha256(hard_prompts) == HARD32_PROMPT_HASHES_SHA256, "hard32 prompt digest differs")

    official_hashes = {
        split: [sha256_text(raw_line) for raw_line, _ in records]
        for split, records in official_records.items()
    }
    require(set(train_rows) <= set(official_hashes["train"]), "proof-train contains a non-official-train row")
    require(set(hard_rows) <= set(official_hashes["val"]), "hard32 contains a non-official-val row")
    require(not (set(train_rows) & set(hard_rows)), "proof-train and hard32 overlap")
    require(not (set(train_rows) & set(official_hashes["test"])), "test row entered proof training")
    require(not (set(hard_rows) & set(official_hashes["test"])), "test row entered hard32")

    proof_split_records = {
        "failure32_train": audit_split(PROOF_TRAIN, split="failure32_train"),
        "fixed_val32": audit_split(HARD32_DATA, split="fixed_val32"),
    }
    for split, lock in PROOF_SPLIT_LOCKS.items():
        _assert_split_lock(split, proof_split_records[split], lock)
    proof_overlap = audit_overlap(
        proof_split_records["failure32_train"],
        proof_split_records["fixed_val32"],
    )
    _assert_overlap_lock(
        ("failure32_train", "fixed_val32"),
        proof_overlap,
        PROOF_OVERLAP_LOCK,
    )
    require(
        proof_overlap["exact_normalized_full_prompts_shared"] == 0
        and proof_overlap["exact_normalized_paragraphs_shared"] == 0
        and proof_overlap["left_rows_with_shared_paragraph"] == 0
        and proof_overlap["right_rows_with_shared_paragraph"] == 0,
        "failure32 and fixed-val32 must remain exact-prompt and passage disjoint",
    )
    for record in proof_split_records.values():
        record.pop("_paragraph_hash_rows")

    indices = _load_json_object(HARD32_INDICES, description="hard32 selection")
    require(indices.get("schema") == "rwkv_ms_scene_eval_selection.v1", "hard32 selection schema differs")
    selection_dataset = indices.get("dataset")
    require(
        isinstance(selection_dataset, dict)
        and selection_dataset.get("split") == "val"
        and selection_dataset.get("path") == str(TASK_DIR / "val.jsonl")
        and selection_dataset.get("sha256") == SPLIT_LOCKS["val"].file_sha256,
        "hard32 selection dataset differs",
    )
    selection_rows = indices.get("rows")
    require(isinstance(selection_rows, list), "hard32 selection rows are missing")
    require(
        [record.get("source_index") for record in selection_rows if isinstance(record, dict)]
        == list(HARD32_SOURCE_INDICES),
        "hard32 source indices differ",
    )
    require(
        [record.get("row_sha256") for record in selection_rows if isinstance(record, dict)]
        == hard_rows,
        "hard32 selected row hashes differ",
    )
    require(
        [official_hashes["val"][index] for index in HARD32_SOURCE_INDICES] == hard_rows,
        "hard32 data order differs from the frozen official-val selection",
    )

    return {
        "pair_manifest": {
            "path": str(PAIR_MANIFEST),
            "sha256": PAIR_MANIFEST_SHA256,
            "schema": pair_manifest["schema"],
        },
        "training_partition": {
            "source_split": "train",
            "selection_rule": "frozen_base_failure32_from_predeclared_official_train_candidate64",
            "rows": 32,
            "path": str(PROOF_TRAIN),
            "sha256": PROOF_TRAIN_SHA256,
            "row_hashes_sha256": PROOF_TRAIN_ROW_HASHES_SHA256,
            "prompt_hashes_sha256": PROOF_TRAIN_PROMPT_HASHES_SHA256,
            "row_manifest": {
                "path": str(PROOF_TRAIN_ROW_MANIFEST),
                "sha256": PROOF_TRAIN_ROW_MANIFEST_SHA256,
            },
            "val_or_test_rows_emitted_for_training": 0,
        },
        "hard_evaluation_selection": {
            "name": "scene_v6_identity_hard32",
            "source_split": "val",
            "selection_rule": "lowest_predeclared_prompt_hash_ranks_without_labels_or_model_outputs",
            "rows": 32,
            "path": str(HARD32_DATA),
            "sha256": HARD32_DATA_SHA256,
            "row_hashes_sha256": HARD32_ROW_HASHES_SHA256,
            "prompt_hashes_sha256": HARD32_PROMPT_HASHES_SHA256,
            "row_manifest": {
                "path": str(HARD32_ROW_MANIFEST),
                "sha256": HARD32_ROW_MANIFEST_SHA256,
            },
            "source_indices": list(HARD32_SOURCE_INDICES),
            "source_indices_file": {
                "path": str(HARD32_INDICES),
                "sha256": HARD32_INDICES_SHA256,
            },
            "checkpoint_selection_only": True,
        },
        "selected_slice_overlap_audit": {
            "passage_disjoint": True,
            "normalization": "anchored [P#] passage text only; prompt preamble excluded",
            "training_partition": proof_split_records["failure32_train"],
            "evaluation_partition": proof_split_records["fixed_val32"],
            "comparison": proof_overlap,
        },
    }


def build_official_contract() -> dict[str, object]:
    require(TASK_DIR.resolve() == (DATASET_ROOT / "training/v4-scene-boundary-detection").resolve(), "task path lock changed")
    official_records = {
        split: _read_jsonl(TASK_DIR / f"{split}.jsonl") for split in SPLIT_ORDER
    }
    split_records = {split: audit_split(TASK_DIR / f"{split}.jsonl", split=split) for split in SPLIT_ORDER}
    for split, lock in SPLIT_LOCKS.items():
        _assert_split_lock(split, split_records[split], lock)

    overlap_records: dict[str, object] = {}
    for pair, lock in OVERLAP_LOCKS.items():
        record = audit_overlap(split_records[pair[0]], split_records[pair[1]])
        _assert_overlap_lock(pair, record, lock)
        overlap_records[f"{pair[0]}__{pair[1]}"] = record

    for record in split_records.values():
        record.pop("_paragraph_hash_rows")

    proof_bundle = _validate_frozen_pair_bundle(official_records)
    payload: dict[str, object] = {
        "schema": SCHEMA,
        "experiment": "scene_memory_v6_identity_proof",
        "task": TASK,
        "dataset_root": str(DATASET_ROOT),
        "task_dir": str(TASK_DIR),
        "pair_manifest": proof_bundle["pair_manifest"],
        "training_partition": proof_bundle["training_partition"],
        "hard_evaluation_selection": proof_bundle["hard_evaluation_selection"],
        "selected_slice_overlap_audit": proof_bundle[
            "selected_slice_overlap_audit"
        ],
        "split_policy": {
            "train": "only the frozen failure32 subset is passed to the trainer",
            "val": "only the frozen hard32 subset may select checkpoints",
            "full_val": "forbidden until a hard32 pass receipt exists",
            "test": "untouched; final evaluation only after validation authorization",
        },
        "normalization": {
            "full_prompt": "Unicode NFKC, then remove every Unicode whitespace character",
            "paragraph_split_regex": r"(?m)^\[P\d+\]\s*",
            "paragraph_segments": (
                "extract text after each [P#] anchor up to the next anchor, exclude "
                "prompt preamble, normalize, discard empty segments, and deduplicate "
                "within each row"
            ),
            "comparison": "exact normalized strings represented by SHA-256",
        },
        "splits": split_records,
        "overlap_audit": {
            "passage_disjoint": False,
            "warning": (
                "Official benchmark splits have no exact normalized full-prompt overlap, "
                "but they have substantial exact normalized paragraph overlap. Do not claim "
                "passage-level or semantic disjointness."
            ),
            "pairs": overlap_records,
        },
        "test_policy": {
            "rows_emitted_for_training": 0,
            "rows_emitted_for_checkpoint_selection": 0,
            "full_validation_before_hard32_pass": "forbidden",
            "test_before_validation_selection_receipt": "forbidden",
        },
    }
    payload["manifest_sha256"] = canonical_sha256(payload)
    return payload


def _summary(contract: Mapping[str, object]) -> dict[str, object]:
    overlaps = contract["overlap_audit"]
    require(isinstance(overlaps, dict), "overlap audit missing")
    return {
        "schema": contract["schema"],
        "manifest_sha256": contract["manifest_sha256"],
        "train_rows": contract["training_partition"]["rows"],  # type: ignore[index]
        "train_sha256": contract["training_partition"]["sha256"],  # type: ignore[index]
        "hard32_rows": contract["hard_evaluation_selection"]["rows"],  # type: ignore[index]
        "hard32_sha256": contract["hard_evaluation_selection"]["sha256"],  # type: ignore[index]
        "pair_manifest_sha256": contract["pair_manifest"]["sha256"],  # type: ignore[index]
        "passage_disjoint": overlaps["passage_disjoint"],
        "overlap_pairs": overlaps["pairs"],
        "selected_slice_overlap_audit": contract[
            "selected_slice_overlap_audit"
        ],
    }


def write_json_exclusive(path: Path, payload: object) -> None:
    path = path.expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    try:
        with temporary.open("x", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
        os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--summary", action="store_true")
    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        contract = build_official_contract()
        rendered = _summary(contract) if args.summary else contract
        if args.output is not None:
            require(not args.summary, "--output and --summary are mutually exclusive")
            require(not args.output.exists(), f"output already exists: {args.output}")
            write_json_exclusive(args.output, rendered)
        else:
            json.dump(rendered, sys.stdout, ensure_ascii=False, sort_keys=True)
            sys.stdout.write("\n")
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
