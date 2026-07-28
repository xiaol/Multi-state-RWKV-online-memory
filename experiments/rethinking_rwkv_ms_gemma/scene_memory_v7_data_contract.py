#!/usr/bin/env python3
"""Validate the frozen-Hard32-aligned V7 Train32 and tiny2 bundle."""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from experiments.rethinking_rwkv_ms_gemma.prepare_scene_memory_v7_data import (
    DEFAULT_OUTPUT_DIR,
    FAILURE_STRATA,
    GOLD_CARDINALITY_QUOTAS,
    HARD32_FILE,
    HARD32_FILE_SHA256,
    PAIRING_BINDING_SCHEMA,
    PAIRING_SCHEMA,
    SCHEMA,
    SOURCE_SCHEMA,
    TARGET_STRATA,
    TINY_DIRECTED_PAIR_QUOTAS,
    TRAIN_DIRECTED_PAIR_QUOTAS,
    VAL_FILE,
    TEST_FILE,
    canonical_sha256,
    load_json_object,
    normalized_paragraph_hashes,
    read_jsonl,
    require,
    sha256_file,
    sha256_text,
    strict_failure_stratum,
    validate_self_hash,
)
from experiments.rethinking_rwkv_ms_gemma.run_novel_agent_eval import score_prediction


SOURCE_LOCK = Path(__file__).with_name("scene_memory_v7_source_lock.json")
SOURCE_LOCK_SCHEMA = "rwkv_ms_scene_memory_v7_source_lock.v1"
ARTIFACT_FILENAMES = {
    "bundle_manifest": "manifest.json",
    "train32": "train32.jsonl",
    "train32_rows": "train32_rows.jsonl",
    "train32_pair_manifest": "train32_pair_manifest.json",
    "train32_source_manifest": "train32_source_manifest.json",
    "tiny2": "tiny2.jsonl",
    "tiny2_rows": "tiny2_rows.jsonl",
    "tiny2_pair_manifest": "tiny2_pair_manifest.json",
    "tiny2_source_manifest": "tiny2_source_manifest.json",
}


def load_source_lock() -> dict[str, Any]:
    lock = load_json_object(SOURCE_LOCK, description="V7 source lock")
    require(lock.get("schema") == SOURCE_LOCK_SCHEMA, "V7 source-lock schema differs")
    unsigned = dict(lock)
    recorded = unsigned.pop("lock_sha256", None)
    require(recorded == canonical_sha256(unsigned), "V7 source-lock checksum differs")
    require(
        lock.get("fixed_hard32")
        == {
            "data_sha256": HARD32_FILE_SHA256,
            "selection_sha256": (
                "76d510e5f02c30f2cf3a0262cc4c97d69ef8f52861e9ced855d530b233d916db"
            ),
            "base_records_sha256": (
                "4740695691bad3ba6c808cc29d734022bf25b0a58ce301a02d551905ec27b4a1"
            ),
        },
        "V7 source-lock Hard32 binding differs",
    )
    require(
        lock.get("pairing_quotas")
        == {
            "train32": dict(TRAIN_DIRECTED_PAIR_QUOTAS),
            "tiny2": dict(TINY_DIRECTED_PAIR_QUOTAS),
        },
        "V7 source-lock pairing quotas differ",
    )
    return lock


def _file_binding(path: Path, expected_sha256: str) -> dict[str, Any]:
    require(path.is_file() and not path.is_symlink(), f"V7 artifact is missing: {path}")
    digest = sha256_file(path)
    require(digest == expected_sha256, f"V7 artifact SHA-256 differs: {path.name}")
    return {"path": str(path.resolve()), "sha256": digest, "bytes": path.stat().st_size}


def _validate_row_manifests(
    *,
    data_path: Path,
    rows_path: Path,
    expected_rows: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    data_records = read_jsonl(data_path)
    row_records = read_jsonl(rows_path)
    require(len(data_records) == len(row_records) == expected_rows, "V7 data and row manifest counts differ")
    data_rows = [payload for _, payload in data_records]
    manifests = [payload for _, payload in row_records]
    for ordinal, ((raw_line, data), manifest) in enumerate(zip(data_records, manifests)):
        validate_self_hash(manifest, field="record_sha256")
        require(manifest.get("train_row_ordinal") == ordinal, "V7 row ordinal differs")
        require(manifest.get("source_split") == "train", "V7 row source split differs")
        require(manifest.get("row_sha256") == sha256_text(raw_line), "V7 row SHA-256 differs")
        messages = data.get("messages")
        require(isinstance(messages, list) and len(messages) == 3, "V7 row must contain three messages")
        require(
            [message.get("role") for message in messages] == ["system", "user", "assistant"],
            "V7 row roles differ",
        )
        gold = json.loads(messages[2]["content"])
        perfect = score_prediction("scene", gold, gold)
        require(bool(perfect["schema_valid"]) and not perfect["fp"] and not perfect["fn"], "V7 gold differs")
        require(manifest.get("gold_boundaries") == perfect["gold_boundaries"], "V7 row gold binding differs")
        strict = manifest.get("strict_score")
        require(isinstance(strict, dict), "V7 strict score is missing")
        require(
            strict_failure_stratum(strict) == manifest.get("strict_failure_stratum"),
            "V7 strict failure stratum differs",
        )
        require(manifest.get("strict_failure_stratum") in FAILURE_STRATA, "V7 row is not a strict failure")
        token_metadata = manifest.get("token_metadata")
        require(isinstance(token_metadata, dict), "V7 row token metadata is missing")
        required_token_fields = {
            "write_rendered_sha256",
            "write_input_ids_sha256",
            "write_token_count",
            "generation_prefix_rendered_sha256",
            "generation_prefix_input_ids_sha256",
            "generation_prefix_token_count",
            "gold_content_sha256",
            "canonical_read_rendered_sha256",
            "canonical_read_input_ids_sha256",
            "canonical_read_token_count",
            "semantic_target_positions",
            "semantic_target_token_ids",
            "semantic_target_mask_sha256",
        }
        require(required_token_fields <= set(token_metadata), "V7 row token metadata is incomplete")
        require(int(token_metadata["write_token_count"]) > 0, "V7 row write token count is invalid")
        require(
            len(token_metadata["semantic_target_positions"])
            == len(token_metadata["semantic_target_token_ids"])
            > 0,
            "V7 semantic target metadata differs",
        )
    return data_rows, manifests


def _validate_pair_manifest(
    *,
    pair_path: Path,
    data_path: Path,
    row_manifests: Sequence[Mapping[str, Any]],
    expected_quotas: Mapping[str, int],
) -> dict[str, Any]:
    pair_manifest = load_json_object(pair_path, description="V7 pair manifest")
    validate_self_hash(pair_manifest)
    require(pair_manifest.get("schema") == PAIRING_SCHEMA, "V7 pair manifest schema differs")
    dataset = pair_manifest.get("dataset")
    require(isinstance(dataset, dict), "V7 pair dataset binding is missing")
    require(
        Path(str(dataset.get("path"))).resolve() == data_path.resolve()
        and dataset.get("sha256") == sha256_file(data_path)
        and dataset.get("rows") == len(row_manifests),
        "V7 pair dataset binding differs",
    )
    ordered_hashes = [str(row["row_sha256"]) for row in row_manifests]
    require(dataset.get("ordered_row_sha256") == canonical_sha256(ordered_hashes), "V7 pair row order digest differs")
    require(pair_manifest.get("quotas") == dict(expected_quotas), "V7 directed pair quotas differ")
    directed = pair_manifest.get("directed_pairs")
    require(isinstance(directed, list) and len(directed) == len(row_manifests), "V7 directed pairs do not cover rows")
    require(pair_manifest.get("entries_sha256") == canonical_sha256(directed), "V7 directed pair digest differs")
    strata = Counter()
    by_ordinal: dict[int, dict[str, Any]] = {}
    for entry in directed:
        require(isinstance(entry, dict), "V7 directed pair entry is invalid")
        validate_self_hash(entry, field="entry_sha256")
        ordinal = entry.get("train_row_ordinal")
        donor_ordinal = entry.get("donor_train_row_ordinal")
        require(
            isinstance(ordinal, int)
            and not isinstance(ordinal, bool)
            and isinstance(donor_ordinal, int)
            and not isinstance(donor_ordinal, bool)
            and 0 <= ordinal < len(row_manifests)
            and 0 <= donor_ordinal < len(row_manifests)
            and ordinal != donor_ordinal,
            "V7 directed pair ordinals differ",
        )
        require(ordinal not in by_ordinal, "V7 directed pair source ordinal is duplicated")
        source = row_manifests[ordinal]
        donor = row_manifests[donor_ordinal]
        source_tokens = source["token_metadata"]
        donor_tokens = donor["token_metadata"]
        require(entry.get("official_source_index") == source["official_source_index"], "V7 source index differs")
        require(entry.get("donor_official_source_index") == donor["official_source_index"], "V7 donor index differs")
        require(entry.get("source_row_sha256") == source["row_sha256"], "V7 source row hash differs")
        require(entry.get("donor_row_sha256") == donor["row_sha256"], "V7 donor row hash differs")
        require(entry.get("source_label_sha256") == source["label_sha256"], "V7 source label hash differs")
        require(entry.get("donor_label_sha256") == donor["label_sha256"], "V7 donor label hash differs")
        require(entry.get("source_label_sha256") != entry.get("donor_label_sha256"), "V7 donor label is not distinct")
        require(entry.get("source_base_record_sha256") == source["base_record_sha256"], "V7 source base record differs")
        require(entry.get("donor_base_record_sha256") == donor["base_record_sha256"], "V7 donor base record differs")
        require(entry.get("source_strict_failure_stratum") == source["strict_failure_stratum"], "V7 source strict outcome differs")
        require(entry.get("donor_strict_failure_stratum") == donor["strict_failure_stratum"], "V7 donor strict outcome differs")
        require(entry.get("source_strict_score_sha256") == canonical_sha256(source["strict_score"]), "V7 source strict score digest differs")
        require(entry.get("donor_strict_score_sha256") == canonical_sha256(donor["strict_score"]), "V7 donor strict score digest differs")
        require(entry.get("source_write_sha256") == source_tokens["write_input_ids_sha256"], "V7 source write hash differs")
        require(entry.get("donor_write_sha256") == donor_tokens["write_input_ids_sha256"], "V7 donor write hash differs")
        require(entry.get("source_generation_prefix_sha256") == source_tokens["generation_prefix_input_ids_sha256"], "V7 source generation prefix differs")
        require(entry.get("donor_generation_prefix_sha256") == donor_tokens["generation_prefix_input_ids_sha256"], "V7 donor generation prefix differs")
        require(
            entry.get("write_token_count_delta")
            == abs(int(source_tokens["write_token_count"]) - int(donor_tokens["write_token_count"])),
            "V7 write length delta differs",
        )
        positions = entry.get("selected_target_positions")
        predictors = entry.get("selected_target_predictor_positions")
        target_ids = entry.get("selected_target_token_ids")
        donor_ids = entry.get("donor_target_token_ids")
        require(
            isinstance(positions, list)
            and len(positions) == len(predictors) == len(target_ids) == len(donor_ids) == 1
            and predictors == [positions[0] - 1]
            and target_ids != donor_ids,
            "V7 distinguishing target metadata differs",
        )
        stratum = entry.get("target_stratum")
        require(stratum in TARGET_STRATA, "V7 target stratum differs")
        strata[stratum] += 1
        by_ordinal[ordinal] = entry
    require(set(by_ordinal) == set(range(len(row_manifests))), "V7 directed pair ordinals are incomplete")
    require({stratum: strata[stratum] for stratum in TARGET_STRATA} == dict(expected_quotas), "V7 observed pair quotas differ")
    for ordinal, entry in by_ordinal.items():
        donor_ordinal = int(entry["donor_train_row_ordinal"])
        reverse = by_ordinal[donor_ordinal]
        require(reverse["donor_train_row_ordinal"] == ordinal, "V7 donor map is not symmetric")
        require(reverse["source_row_sha256"] == entry["donor_row_sha256"], "V7 reverse source hash differs")
        require(reverse["donor_row_sha256"] == entry["source_row_sha256"], "V7 reverse donor hash differs")
    optimization = pair_manifest.get("optimization")
    require(
        isinstance(optimization, dict)
        and optimization.get("global_minimum_after_exact_quotas") is True,
        "V7 pair optimization proof is missing",
    )
    return pair_manifest


def _validate_source_manifest(
    *,
    source_path: Path,
    data_path: Path,
    rows_path: Path,
    pair_path: Path,
    pair_manifest: Mapping[str, Any],
    expected_rows: int,
    expected_quotas: Mapping[str, int],
) -> dict[str, Any]:
    source = load_json_object(source_path, description="V7 source manifest")
    validate_self_hash(source)
    require(source.get("schema") == SOURCE_SCHEMA, "V7 source manifest schema differs")
    contract = source.get("contract")
    require(isinstance(contract, dict), "V7 source contract is missing")
    require(contract.get("source_split") == "train", "V7 source split differs")
    require(contract.get("val_rows") == contract.get("test_rows") == 0, "V7 val or test rows entered training")
    require(
        contract.get("episode_contract")
        == {
            "episode_recent_messages": 0,
            "write_phase": "system + user",
            "read_supervision": "system + assistant",
        },
        "V7 episode contract differs",
    )
    train = source.get("partitions", {}).get("train")
    require(isinstance(train, dict) and train.get("rows") == expected_rows, "V7 source train partition differs")
    require(
        train.get("data")
        == {"path": str(data_path.resolve()), "sha256": sha256_file(data_path)},
        "V7 source train data binding differs",
    )
    require(
        train.get("row_manifest")
        == {"path": str(rows_path.resolve()), "sha256": sha256_file(rows_path)},
        "V7 source row manifest binding differs",
    )
    binding = source.get("v7_pairing")
    require(isinstance(binding, dict), "V7 pairing binding is missing")
    require(binding.get("schema") == PAIRING_BINDING_SCHEMA, "V7 pairing binding schema differs")
    require(binding.get("dataset_sha256") == sha256_file(data_path), "V7 pairing dataset digest differs")
    require(binding.get("directed_entry_count") == expected_rows, "V7 pairing entry count differs")
    require(binding.get("quotas") == dict(expected_quotas), "V7 pairing binding quotas differ")
    require(binding.get("entries_sha256") == pair_manifest["entries_sha256"], "V7 pairing entries binding differs")
    require(
        binding.get("pair_manifest")
        == {
            "path": str(pair_path.resolve()),
            "sha256": sha256_file(pair_path),
            "manifest_sha256": pair_manifest["manifest_sha256"],
        },
        "V7 pair manifest binding differs",
    )
    return source


def validate_bundle(root: Path = DEFAULT_OUTPUT_DIR, *, require_frozen_hashes: bool = True) -> dict[str, Any]:
    root = root.expanduser().resolve()
    source_lock = load_source_lock()
    locked_artifacts = source_lock.get("artifacts")
    require(isinstance(locked_artifacts, dict), "V7 source-lock artifacts are missing")
    bindings = {}
    for artifact_name, filename in ARTIFACT_FILENAMES.items():
        path = root / filename
        lock_record = locked_artifacts.get(artifact_name)
        require(isinstance(lock_record, dict), f"V7 source-lock artifact is missing: {artifact_name}")
        require(
            Path(str(lock_record.get("path"))).resolve() == path.resolve(),
            f"V7 source-lock artifact path differs: {artifact_name}",
        )
        expected_sha256 = str(lock_record.get("sha256"))
        if require_frozen_hashes:
            bindings[filename] = _file_binding(path, expected_sha256)
        else:
            require(path.is_file() and not path.is_symlink(), f"V7 artifact is missing: {path}")
            bindings[filename] = {
                "path": str(path.resolve()),
                "sha256": sha256_file(path),
                "bytes": path.stat().st_size,
            }
    bundle = load_json_object(root / "manifest.json", description="V7 bundle manifest")
    validate_self_hash(bundle)
    require(bundle.get("schema") == SCHEMA, "V7 bundle schema differs")
    require(bundle.get("leakage") == {
        "normalization": "anchored paragraphs; Unicode NFKC; remove Unicode whitespace",
        "train32_hard32_shared_normalized_paragraphs": 0,
        "val_rows_in_training": 0,
        "test_rows_in_training": 0,
    }, "V7 leakage declaration differs")
    selection = bundle.get("selection")
    require(isinstance(selection, dict), "V7 selection audit is missing")
    require(selection.get("gold_cardinality_quotas") == {str(key): value for key, value in GOLD_CARDINALITY_QUOTAS.items()} or selection.get("gold_cardinality_quotas") == GOLD_CARDINALITY_QUOTAS, "V7 cardinality quotas differ")
    require(selection.get("minimum_cellwise_l1") == 2, "V7 minimum failure-stratum L1 differs")
    require(selection.get("overall_failure_count_l1") == 2, "V7 failure count L1 differs")
    require(selection.get("relaxed_silently") is False, "V7 selection silently relaxed")

    train_data, train_rows = _validate_row_manifests(
        data_path=root / "train32.jsonl",
        rows_path=root / "train32_rows.jsonl",
        expected_rows=32,
    )
    cardinalities = Counter(int(row["gold_boundary_count"]) for row in train_rows)
    require(dict(sorted(cardinalities.items())) == GOLD_CARDINALITY_QUOTAS, "V7 Train32 observed cardinalities differ")
    pair_manifest = _validate_pair_manifest(
        pair_path=root / "train32_pair_manifest.json",
        data_path=root / "train32.jsonl",
        row_manifests=train_rows,
        expected_quotas=TRAIN_DIRECTED_PAIR_QUOTAS,
    )
    _validate_source_manifest(
        source_path=root / "train32_source_manifest.json",
        data_path=root / "train32.jsonl",
        rows_path=root / "train32_rows.jsonl",
        pair_path=root / "train32_pair_manifest.json",
        pair_manifest=pair_manifest,
        expected_rows=32,
        expected_quotas=TRAIN_DIRECTED_PAIR_QUOTAS,
    )

    tiny_data, tiny_rows = _validate_row_manifests(
        data_path=root / "tiny2.jsonl",
        rows_path=root / "tiny2_rows.jsonl",
        expected_rows=2,
    )
    train_hashes = {str(row["row_sha256"]) for row in train_rows}
    tiny_hashes = {str(row["row_sha256"]) for row in tiny_rows}
    require(len(tiny_hashes) == 2 and tiny_hashes <= train_hashes, "V7 tiny2 is not a Train32 subset")
    require(
        all(int(row["gold_boundary_count"]) > 0 for row in tiny_rows)
        and len({int(row["gold_boundary_count"]) for row in tiny_rows}) == 1
        and len({str(row["label_sha256"]) for row in tiny_rows}) == 2,
        "V7 tiny2 is not one positive same-cardinality distinct-label pair",
    )
    tiny_pair_manifest = _validate_pair_manifest(
        pair_path=root / "tiny2_pair_manifest.json",
        data_path=root / "tiny2.jsonl",
        row_manifests=tiny_rows,
        expected_quotas=TINY_DIRECTED_PAIR_QUOTAS,
    )
    tiny_source = _validate_source_manifest(
        source_path=root / "tiny2_source_manifest.json",
        data_path=root / "tiny2.jsonl",
        rows_path=root / "tiny2_rows.jsonl",
        pair_path=root / "tiny2_pair_manifest.json",
        pair_manifest=tiny_pair_manifest,
        expected_rows=2,
        expected_quotas=TINY_DIRECTED_PAIR_QUOTAS,
    )
    require(tiny_source.get("parent_train32_sha256") == sha256_file(root / "train32.jsonl"), "V7 tiny2 parent binding differs")

    val_hashes = {sha256_text(raw_line) for raw_line, _ in read_jsonl(VAL_FILE)}
    test_hashes = {sha256_text(raw_line) for raw_line, _ in read_jsonl(TEST_FILE)}
    require(not (train_hashes & val_hashes) and not (train_hashes & test_hashes), "V7 Train32 contains val or test rows")
    hard_paragraphs = {
        paragraph
        for _, row in read_jsonl(HARD32_FILE)
        for paragraph in normalized_paragraph_hashes(row["messages"][1]["content"])
    }
    train_paragraphs = {
        paragraph
        for row in train_data
        for paragraph in normalized_paragraph_hashes(row["messages"][1]["content"])
    }
    require(not (train_paragraphs & hard_paragraphs), "V7 Train32 overlaps fixed Hard32 paragraphs")
    require(sha256_file(HARD32_FILE) == HARD32_FILE_SHA256, "fixed Hard32 changed")
    return {
        "status": "pass",
        "root": str(root),
        "bundle_manifest_sha256": bundle["manifest_sha256"],
        "train32_sha256": sha256_file(root / "train32.jsonl"),
        "tiny2_sha256": sha256_file(root / "tiny2.jsonl"),
        "train32_pair_manifest_sha256": pair_manifest["manifest_sha256"],
        "tiny2_pair_manifest_sha256": tiny_pair_manifest["manifest_sha256"],
        "minimum_failure_stratum_l1": selection["minimum_cellwise_l1"],
        "train32_pair_quotas": dict(TRAIN_DIRECTED_PAIR_QUOTAS),
        "tiny2_pair_quotas": dict(TINY_DIRECTED_PAIR_QUOTAS),
        "source_lock": {
            "path": str(SOURCE_LOCK.resolve()),
            "sha256": sha256_file(SOURCE_LOCK),
            "lock_sha256": source_lock["lock_sha256"],
        },
        "artifacts": bindings,
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--no-frozen-hashes", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        result = validate_bundle(
            args.root,
            require_frozen_hashes=not args.no_frozen_hashes,
        )
    except Exception as exc:
        print(f"ERROR: {exc}", file=__import__("sys").stderr)
        return 2
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
