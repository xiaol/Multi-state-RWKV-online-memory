from __future__ import annotations

import contextlib
import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from experiments.rethinking_rwkv_ms_gemma import run_scene_state_eval as evaluator
from experiments.rethinking_rwkv_ms_gemma import (
    scene_hard_failure_train_contract as hard_failure_contract,
)
from experiments.rethinking_rwkv_ms_gemma import (
    select_scene_hard_failure_checkpoint as hard_failure_selector,
)


def test_focused_evaluator_source_lock_matches_training_curriculum() -> None:
    source = json.loads(
        hard_failure_contract.SOURCE_MANIFEST.read_text(encoding="utf-8")
    )

    assert evaluator.SCENE_HARD_FAILURE_SOURCE_MANIFEST_FILE_SHA256 == (
        hard_failure_contract.FILE_SHA256["source_manifest.json"]
    )
    assert evaluator.sha256_file(hard_failure_contract.SOURCE_MANIFEST) == (
        evaluator.SCENE_HARD_FAILURE_SOURCE_MANIFEST_FILE_SHA256
    )
    assert source["manifest_sha256"] == (
        evaluator.SCENE_HARD_FAILURE_SOURCE_MANIFEST_SHA256
    )


def test_parse_row_indices_accepts_json_and_delimited_text() -> None:
    assert evaluator.parse_row_indices("[142, 25, 28]") == [142, 25, 28]
    assert evaluator.parse_row_indices("142,25\n28") == [142, 25, 28]


def test_parse_selection_manifest_accepts_and_validates_row_hashes() -> None:
    row_hash = "a" * 64
    indices, expected_hashes = evaluator.parse_selection_manifest(
        json.dumps(
            {
                "schema": "scene_state_selection.v1",
                "rows": [{"source_index": 142, "row_sha256": row_hash}],
            }
        )
    )

    assert indices == [142]
    assert expected_hashes == {142: row_hash}

    with pytest.raises(ValueError, match="invalid row_sha256"):
        evaluator.parse_selection_manifest(
            json.dumps(
                {"rows": [{"source_index": 142, "row_sha256": "not-a-hash"}]}
            )
        )


def test_selection_dataset_contract_binds_path_and_hash(tmp_path: Path) -> None:
    dataset = tmp_path / "val.jsonl"
    dataset.write_text("{}\n", encoding="utf-8")
    digest = evaluator.sha256_file(dataset)
    raw = json.dumps(
        {
            "schema": "rwkv_ms_scene_eval_selection.v1",
            "dataset": {"split": "val", "path": str(dataset), "sha256": digest},
            "rows": [{"source_index": 0, "row_sha256": "a" * 64}],
        }
    )

    contract = evaluator.parse_selection_dataset_contract(raw)
    evaluator.validate_selection_dataset_contract(dataset.resolve(), contract)

    with pytest.raises(ValueError, match="dataset path differs"):
        evaluator.validate_selection_dataset_contract(
            (tmp_path / "other" / "val.jsonl").resolve(), contract
        )
    dataset.write_text("changed\n", encoding="utf-8")
    with pytest.raises(ValueError, match="dataset SHA-256 differs"):
        evaluator.validate_selection_dataset_contract(dataset.resolve(), contract)


def test_scene_selection_manifest_requires_dataset_contract() -> None:
    with pytest.raises(ValueError, match="missing 'dataset'"):
        evaluator.parse_selection_dataset_contract(
            json.dumps({"schema": "rwkv_ms_scene_eval_selection.v1", "rows": []})
        )


def test_focused_evaluator_accepts_only_validation_filename(tmp_path: Path) -> None:
    val_path = tmp_path / "val.jsonl"
    assert evaluator.resolve_validation_dataset_file(val_path) == val_path.resolve()

    for filename in ("train.jsonl", "test.jsonl", "holdout.jsonl"):
        with pytest.raises(ValueError, match="requires the official val.jsonl"):
            evaluator.resolve_validation_dataset_file(tmp_path / filename)


def test_focused_train_source_manifest_binds_all_rows_and_reciprocal_pairs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "curriculum"
    root.mkdir()
    dataset = root / "train.jsonl"
    rows_path = root / "train_rows.jsonl"
    pair_path = root / "pair_manifest.json"
    source_path = root / "source_manifest.json"
    raw_rows = [
        json.dumps(scene_row([1] if index % 2 == 0 else [2]), sort_keys=True)
        for index in range(32)
    ]
    dataset.write_text("\n".join(raw_rows) + "\n", encoding="utf-8")
    row_hashes = [evaluator.sha256_text(raw) for raw in raw_rows]
    row_manifests = []
    for index, row_sha256 in enumerate(row_hashes):
        row = {
            "schema": evaluator.SCENE_HARD_FAILURE_ROW_SCHEMA,
            "train_row_ordinal": index,
            "source_split": "train",
            "row_sha256": row_sha256,
            "label_sha256": "a" * 64 if index % 2 == 0 else "b" * 64,
        }
        row["record_sha256"] = evaluator._canonical_sha256(row)
        row_manifests.append(row)
    rows_path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in row_manifests),
        encoding="utf-8",
    )
    directed = []
    for index, row in enumerate(row_manifests):
        donor_index = index ^ 1
        donor = row_manifests[donor_index]
        entry = {
            "train_row_ordinal": index,
            "donor_train_row_ordinal": donor_index,
            "source_row_sha256": row["row_sha256"],
            "donor_row_sha256": donor["row_sha256"],
            "source_label_sha256": row["label_sha256"],
            "donor_label_sha256": donor["label_sha256"],
        }
        entry["entry_sha256"] = evaluator._canonical_sha256(entry)
        directed.append(entry)
    entries_sha256 = evaluator._canonical_sha256(directed)
    pair_manifest = {
        "schema": evaluator.SCENE_HARD_FAILURE_PAIR_SCHEMA,
        "dataset": {
            "path": str(dataset.resolve()),
            "sha256": evaluator.sha256_file(dataset),
            "rows": 32,
            "ordered_row_sha256": evaluator._canonical_sha256(row_hashes),
        },
        "directed_pairs": directed,
        "entries_sha256": entries_sha256,
    }
    pair_manifest["manifest_sha256"] = evaluator._canonical_sha256(pair_manifest)
    pair_path.write_text(json.dumps(pair_manifest, sort_keys=True), encoding="utf-8")
    source = {
        "schema": evaluator.SCENE_HARD_FAILURE_SOURCE_SCHEMA,
        "task": evaluator.TASK_NAME,
        "purpose": evaluator.SCENE_HARD_FAILURE_SOURCE_PURPOSE,
        "contract": {
            "source_split": "train",
            "val_rows": 0,
            "test_rows": 0,
            "episode_contract": {
                "episode_recent_messages": 0,
                "write_phase": "system + user",
                "read_supervision": "system + assistant",
            },
        },
        "partitions": {
            "train": {
                "source_split": "train",
                "rows": 32,
                "data": {
                    "path": str(dataset.resolve()),
                    "sha256": evaluator.sha256_file(dataset),
                },
                "row_manifest": {
                    "path": str(rows_path.resolve()),
                    "sha256": evaluator.sha256_file(rows_path),
                },
            }
        },
        "v7_pairing": {
            "dataset_sha256": evaluator.sha256_file(dataset),
            "directed_entry_count": 32,
            "entries_sha256": entries_sha256,
            "pair_manifest": {
                "path": str(pair_path.resolve()),
                "sha256": evaluator.sha256_file(pair_path),
                "manifest_sha256": pair_manifest["manifest_sha256"],
            },
        },
    }
    source["manifest_sha256"] = evaluator._canonical_sha256(source)
    source_path.write_text(json.dumps(source, sort_keys=True), encoding="utf-8")
    for name, value in {
        "SCENE_HARD_FAILURE_SOURCE_MANIFEST_FILE_SHA256": evaluator.sha256_file(
            source_path
        ),
        "SCENE_HARD_FAILURE_SOURCE_MANIFEST_SHA256": source["manifest_sha256"],
        "SCENE_HARD_FAILURE_TRAIN_FILE_SHA256": evaluator.sha256_file(dataset),
        "SCENE_HARD_FAILURE_ROW_MANIFEST_FILE_SHA256": evaluator.sha256_file(
            rows_path
        ),
        "SCENE_HARD_FAILURE_PAIR_MANIFEST_FILE_SHA256": evaluator.sha256_file(
            pair_path
        ),
        "SCENE_HARD_FAILURE_PAIR_MANIFEST_SHA256": pair_manifest[
            "manifest_sha256"
        ],
        "SCENE_HARD_FAILURE_PAIR_ENTRIES_SHA256": entries_sha256,
    }.items():
        monkeypatch.setattr(evaluator, name, value)

    validated = evaluator.validate_focused_train_source_manifest(
        source_path,
        dataset_file=dataset,
    )

    assert validated["dataset"]["split"] == "train"
    assert validated["expected_row_hashes"] == dict(enumerate(row_hashes))
    assert validated["donor_by_source_index"] == {
        index: index ^ 1 for index in range(32)
    }


@pytest.mark.parametrize("raw", ["", "1,1", "-1", "1.5", "[true]"])
def test_parse_row_indices_rejects_unsafe_selections(raw: str) -> None:
    with pytest.raises(ValueError):
        evaluator.parse_row_indices(raw)


def scene_row(boundaries: list[int]) -> dict:
    return {
        "messages": [
            {"role": "system", "content": "scene system"},
            {"role": "user", "content": "[P1] A\n[P2] B"},
            {"role": "assistant", "content": json.dumps({"boundaries": boundaries})},
        ]
    }


def test_load_selected_rows_preserves_explicit_order_and_hashes(tmp_path: Path) -> None:
    dataset = tmp_path / "val.jsonl"
    raw_rows = [json.dumps(scene_row([index + 1])) for index in range(3)]
    dataset.write_text("\n".join(raw_rows) + "\n", encoding="utf-8")

    rows = evaluator.load_selected_rows(dataset, [2, 0])

    assert [row["source_index"] for row in rows] == [2, 0]
    assert rows[0]["gold"] == {"boundaries": [3]}
    assert rows[0]["row_sha256"] == evaluator.sha256_text(raw_rows[2])
    assert [message["role"] for message in rows[0]["messages"]] == ["system", "user"]
    assert rows[0]["prime_messages_sha256"] == evaluator.fingerprint_payload_sha256(
        {"messages": rows[0]["messages"]}
    )


def test_load_selected_rows_rejects_missing_index(tmp_path: Path) -> None:
    dataset = tmp_path / "val.jsonl"
    dataset.write_text(json.dumps(scene_row([1])) + "\n", encoding="utf-8")

    with pytest.raises(IndexError, match="outside the dataset"):
        evaluator.load_selected_rows(dataset, [1])


def test_load_selected_rows_rejects_selection_hash_drift(tmp_path: Path) -> None:
    dataset = tmp_path / "val.jsonl"
    dataset.write_text(json.dumps(scene_row([1])) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="Selection row hash differs"):
        evaluator.load_selected_rows(
            dataset,
            [0],
            expected_hashes={0: "0" * 64},
        )


def test_cyclic_donor_mapping_uses_only_fixed_selected_order() -> None:
    samples = [
        {
            "source_index": 142,
            "row_sha256": "a" * 64,
            "prime_messages_sha256": "1" * 64,
        },
        {
            "source_index": 25,
            "row_sha256": "b" * 64,
            "prime_messages_sha256": "2" * 64,
        },
        {
            "source_index": 28,
            "row_sha256": "c" * 64,
            "prime_messages_sha256": "3" * 64,
        },
    ]

    donors = evaluator.build_cyclic_donor_mapping(samples)

    assert donors == {142: samples[1], 25: samples[2], 28: samples[0]}
    assert set(map(id, donors.values())) == set(map(id, samples))
    assert evaluator.donor_mapping_fingerprint_rows(samples, donors) == [
        {
            "source_index": 142,
            "row_sha256": "a" * 64,
            "prime_messages_sha256": "1" * 64,
            "donor_source_index": 25,
            "donor_row_sha256": "b" * 64,
            "donor_prime_messages_sha256": "2" * 64,
        },
        {
            "source_index": 25,
            "row_sha256": "b" * 64,
            "prime_messages_sha256": "2" * 64,
            "donor_source_index": 28,
            "donor_row_sha256": "c" * 64,
            "donor_prime_messages_sha256": "3" * 64,
        },
        {
            "source_index": 28,
            "row_sha256": "c" * 64,
            "prime_messages_sha256": "3" * 64,
            "donor_source_index": 142,
            "donor_row_sha256": "a" * 64,
            "donor_prime_messages_sha256": "1" * 64,
        },
    ]


def test_state_only_donor_rejects_one_row_selection() -> None:
    with pytest.raises(ValueError, match="requires at least two selected rows"):
        evaluator.build_cyclic_donor_mapping(
            [
                {
                    "source_index": 142,
                    "row_sha256": "a" * 64,
                    "prime_messages_sha256": "1" * 64,
                }
            ]
        )


def test_state_only_donor_rejects_duplicate_priming_prompts() -> None:
    with pytest.raises(ValueError, match="unique priming prompts"):
        evaluator.build_cyclic_donor_mapping(
            [
                {
                    "source_index": 1,
                    "row_sha256": "a" * 64,
                    "prime_messages_sha256": "1" * 64,
                },
                {
                    "source_index": 2,
                    "row_sha256": "b" * 64,
                    "prime_messages_sha256": "1" * 64,
                },
            ]
        )


def test_length_matched_donors_are_label_distinct_symmetric_pairs() -> None:
    samples = [
        {
            "source_index": 1,
            "row_sha256": "a" * 64,
            "prime_messages_sha256": "1" * 64,
            "gold": {"boundaries": [1]},
            "write_token_count": 100,
        },
        {
            "source_index": 2,
            "row_sha256": "b" * 64,
            "prime_messages_sha256": "2" * 64,
            "gold": {"boundaries": [1]},
            "write_token_count": 101,
        },
        {
            "source_index": 3,
            "row_sha256": "c" * 64,
            "prime_messages_sha256": "3" * 64,
            "gold": {"boundaries": []},
            "write_token_count": 102,
        },
        {
            "source_index": 4,
            "row_sha256": "d" * 64,
            "prime_messages_sha256": "4" * 64,
            "gold": {"boundaries": [2]},
            "write_token_count": 150,
        },
    ]

    donors = evaluator.build_length_matched_label_distinct_donor_mapping(samples)

    assert set(donors) == {1, 2, 3, 4}
    assert {donor["source_index"] for donor in donors.values()} == {1, 2, 3, 4}
    for sample in samples:
        donor = donors[sample["source_index"]]
        assert donor["gold"] != sample["gold"]
        assert donors[donor["source_index"]] is sample
    rows = evaluator.donor_mapping_fingerprint_rows(samples, donors)
    assert all(row["absolute_write_token_difference"] >= 0 for row in rows)
    assert rows[0]["write_token_count"] == 100


def test_deterministic_shuffled_mapping_is_label_distinct_bijective_and_not_donor() -> None:
    samples = [
        {
            "source_index": index,
            "row_sha256": f"{index + 1:064x}",
            "prime_messages_sha256": f"{index + 101:064x}",
            "gold": {"boundaries": [1] if index % 2 == 0 else [2]},
        }
        for index in range(8)
    ]
    donors = {index: samples[index ^ 1] for index in range(len(samples))}

    first = evaluator.build_deterministic_shuffled_mapping(samples, donors)
    second = evaluator.build_deterministic_shuffled_mapping(samples, donors)

    assert {source: row["source_index"] for source, row in first.items()} == {
        source: row["source_index"] for source, row in second.items()
    }
    assert {row["source_index"] for row in first.values()} == set(range(8))
    for source_index, shuffled in first.items():
        assert shuffled["source_index"] != source_index
        assert shuffled["source_index"] != donors[source_index]["source_index"]
        assert shuffled["gold"] != samples[source_index]["gold"]
    fingerprint = evaluator.shuffled_mapping_fingerprint_rows(samples, first)
    assert [row["source_index"] for row in fingerprint] == list(range(8))
    assert all("shuffled_row_sha256" in row for row in fingerprint)


def test_length_matched_donors_reject_unpairable_labels() -> None:
    samples = [
        {
            "source_index": index,
            "row_sha256": str(index) * 64,
            "prime_messages_sha256": str(index + 4) * 64,
            "gold": {"boundaries": [1]},
            "write_token_count": 100 + index,
        }
        for index in range(1, 3)
    ]

    with pytest.raises(ValueError, match="No complete label-distinct donor pairing"):
        evaluator.build_length_matched_label_distinct_donor_mapping(samples)


def test_hard32_frozen_global_pairing_identity_is_locked() -> None:
    expected_pairs = (
        (3, 112),
        (6, 33),
        (16, 141),
        (21, 88),
        (24, 47),
        (30, 102),
        (50, 56),
        (59, 70),
        (63, 67),
        (64, 71),
        (66, 74),
        (75, 79),
        (87, 128),
        (113, 166),
        (132, 151),
        (144, 159),
    )

    assert evaluator.HARD32_FROZEN_DONOR_PAIRS == expected_pairs
    assert tuple(sorted(index for pair in expected_pairs for index in pair)) == (
        evaluator.HARD32_ROW_INDICES
    )
    assert evaluator.sha256_text(
        json.dumps(expected_pairs, separators=(",", ":"))
    ) == "e772e8c77210537234df4b584b7bf5f762a228362d56eb644baffd33d16c9aea"
    assert evaluator.HARD32_FROZEN_DONOR_MAPPING_ROWS_SHA256 == (
        "a531552ef876479a7462fe290dc61f50168fb01926be47727d177337ad13b0cf"
    )


def test_resolved_condition_protocol_records_matched_donor_rule() -> None:
    protocols = evaluator.resolved_condition_protocols(
        ["state_only", "state_only_donor"],
        donor_rule=evaluator.DONOR_RULE_LENGTH_MATCHED,
    )

    assert protocols["state_only_donor"]["donor_rule"] == evaluator.DONOR_RULE_LENGTH_MATCHED
    assert "nearest write-token length" in protocols["state_only_donor"]["description"]
    assert (
        evaluator.CONDITION_PROTOCOLS["state_only_donor"]["donor_rule"]
        == evaluator.DONOR_RULE_CYCLIC
    )


def _focused_memory_dir(tmp_path: Path) -> Path:
    memory_dir = tmp_path / "run" / "trainer" / "checkpoint-64"
    memory_dir.mkdir(parents=True)
    (memory_dir / "delta_mem_adapter.pt").write_bytes(b"adapter")
    (memory_dir / "delta_mem_config.json").write_text("{}\n", encoding="utf-8")
    return memory_dir


def test_focused_train_contract_requires_exact_train32_and_emits_train_split(
    tmp_path: Path,
) -> None:
    memory_dir = _focused_memory_dir(tmp_path)
    row_hashes = {index: f"{index + 1:064x}" for index in range(32)}
    train_source = {
        "expected_row_hashes": row_hashes,
        "selection": [
            {"source_index": index, "row_sha256": row_hashes[index]}
            for index in range(32)
        ],
    }

    contract = evaluator.validate_scene_hard_failure_contract(
        contract=evaluator.SCENE_HARD_FAILURE_TRAIN_OVERFIT_CONTRACT,
        row_indices=list(range(32)),
        expected_hashes=row_hashes,
        selection_dataset_contract=None,
        conditions=list(evaluator.SCENE_FOCUSED_CONDITIONS),
        donor_rule=evaluator.DONOR_RULE_LENGTH_MATCHED,
        max_new_tokens=evaluator.DEFAULT_MAX_NEW_TOKENS,
        normal_fusion_profile="native",
        expected_memory_layer_count=42,
        memory_target_layers=list(range(42)),
        memory_delta_heads=["q", "o"],
        memory_rank=4,
        rwkv_ms_semantics_version=2,
        memory_backend="rwkv_ms",
        memory_dir=memory_dir,
        selection_manifest_sha256=None,
        train_source=train_source,
        train_selection_authorization=None,
    )

    assert contract["split"] == "train"
    assert contract["conditions"] == list(evaluator.SCENE_FOCUSED_CONDITIONS)
    assert contract["hard32_authorized"] is False


def test_focused_hard32_contract_requires_selected_checkpoint_and_seven_controls(
    tmp_path: Path,
) -> None:
    memory_dir = _focused_memory_dir(tmp_path)
    authorization = {
        "selected_checkpoint": {
            "path": str(memory_dir.resolve()),
            "artifacts": {
                "delta_mem_adapter.pt": evaluator.sha256_file(
                    memory_dir / "delta_mem_adapter.pt"
                ),
                "delta_mem_config.json": evaluator.sha256_file(
                    memory_dir / "delta_mem_config.json"
                ),
            },
        },
        "hard32_output_dir": str((tmp_path / "focused-hard32").absolute()),
        "authorization_consumption": {
            "schema": (
                evaluator.SCENE_HARD_FAILURE_HARD32_CONSUMPTION_MARKER_SCHEMA
            ),
            "path": str(tmp_path / "hard32_authorization_consumed.json"),
            "file_sha256": "a" * 64,
            "claim_sha256": "b" * 64,
        },
    }

    contract = evaluator.validate_scene_hard_failure_contract(
        contract=evaluator.SCENE_HARD_FAILURE_HARD32_CONTRACT,
        row_indices=list(evaluator.HARD32_ROW_INDICES),
        expected_hashes=dict(evaluator.HARD32_ROW_HASHES),
        selection_dataset_contract={
            "split": "val",
            "path": "/official/val.jsonl",
            "sha256": evaluator.OFFICIAL_SCENE_V4_VAL_SHA256,
        },
        conditions=list(evaluator.SCENE_FOCUSED_CONDITIONS),
        donor_rule=evaluator.DONOR_RULE_LENGTH_MATCHED,
        max_new_tokens=evaluator.DEFAULT_MAX_NEW_TOKENS,
        normal_fusion_profile="native",
        expected_memory_layer_count=42,
        memory_target_layers=list(range(42)),
        memory_delta_heads=["q", "o"],
        memory_rank=4,
        rwkv_ms_semantics_version=2,
        memory_backend="rwkv_ms",
        memory_dir=memory_dir,
        selection_manifest_sha256=evaluator.HARD32_SELECTION_SHA256,
        train_source=None,
        train_selection_authorization=authorization,
    )

    assert contract["split"] == "val"
    assert contract["rows"] == 32
    assert contract["conditions"] == list(evaluator.SCENE_FOCUSED_CONDITIONS)
    assert contract["full170_authorized"] is False


def _focused_exactly_once_fixture(
    tmp_path: Path,
    memory_dir: Path,
) -> tuple[Path, dict[str, object]]:
    audit_path = memory_dir / hard_failure_selector.run_audit.AUDIT_FILENAME
    audit_path.write_text("{}\n", encoding="utf-8")
    run_root = memory_dir.parent.parent
    receipt_dir = run_root / "train32_endpoint_screen"
    receipt_dir.mkdir()
    receipt_path = receipt_dir / "train32_checkpoint_selection_receipt.json"
    receipt_path.write_text("{}\n", encoding="utf-8")
    return receipt_path, {
        "authorization_kind": "scene_hard_failure_train_overfit_selection",
        "scope": evaluator.SCENE_HARD_FAILURE_AUTHORIZATION_SCOPE,
        "hard32_authorized": True,
        "full170_authorized": False,
        "test_authorized": False,
        "other_benchmarks_authorized": False,
        "receipt": {
            "path": str(receipt_path.resolve()),
            "file_sha256": evaluator.sha256_file(receipt_path),
            "receipt_sha256": "1" * 64,
        },
        "selected_checkpoint": {
            "path": str(memory_dir.resolve()),
            "global_step": 64,
            "artifacts": {
                "delta_mem_adapter.pt": evaluator.sha256_file(
                    memory_dir / "delta_mem_adapter.pt"
                ),
                "delta_mem_config.json": evaluator.sha256_file(
                    memory_dir / "delta_mem_config.json"
                ),
                hard_failure_selector.run_audit.AUDIT_FILENAME: (
                    evaluator.sha256_file(audit_path)
                ),
            },
        },
        "evaluation_fingerprint": "2" * 64,
        "gate": {
            "path": str(receipt_dir / "focused_gate.json"),
            "file_sha256": "3" * 64,
            "canonical_sha256": "4" * 64,
        },
    }


def _focused_exactly_once_kwargs(
    *,
    receipt_path: Path,
    memory_dir: Path,
    output_dir: Path,
    dataset_file: Path,
    selection_file: Path,
) -> dict[str, object]:
    return {
        "selection_receipt_path": receipt_path,
        "memory_dir": memory_dir,
        "output_dir": output_dir,
        "overwrite": False,
        "dataset_file": dataset_file,
        "selection_file": selection_file,
        "base_model": evaluator.HISTORICAL_V6_BASE_MODEL,
        "device": "cuda:0",
        "dtype": "bfloat16",
        "attn_implementation": "sdpa",
        "delta_mem_root": evaluator.PROJECT_ROOT,
        "conditions": list(evaluator.SCENE_FOCUSED_CONDITIONS),
        "donor_rule": evaluator.DONOR_RULE_LENGTH_MATCHED,
        "max_new_tokens": evaluator.DEFAULT_MAX_NEW_TOKENS,
        "normal_fusion_profile": "native",
        "expected_memory_layer_count": 42,
        "inline_row_indices": None,
        "preflight_only": False,
    }


def _valid_focused_v2_selection_receipt(
    tmp_path: Path,
) -> tuple[Path, Path, dict[str, object]]:
    step = 16
    run_root = tmp_path / "run"
    memory_dir = run_root / "trainer" / f"checkpoint-{step}"
    initial_dir = run_root / "initial_adapter"
    receipt_dir = run_root / "train32_endpoint_screen"
    memory_dir.mkdir(parents=True)
    initial_dir.mkdir()
    receipt_dir.mkdir()

    initial_manifest = initial_dir / "initial_adapter_manifest.json"
    initial_adapter = initial_dir / "delta_mem_adapter.pt"
    checkpoint_adapter = memory_dir / "delta_mem_adapter.pt"
    checkpoint_config = memory_dir / "delta_mem_config.json"
    checkpoint_protocol = memory_dir / "training_protocol.json"
    checkpoint_audit = memory_dir / hard_failure_selector.run_audit.AUDIT_FILENAME
    initial_manifest.write_text(
        json.dumps({"topology": {"fixture": True}}, sort_keys=True),
        encoding="utf-8",
    )
    initial_adapter.write_bytes(b"initial-adapter")
    checkpoint_adapter.write_bytes(b"checkpoint-adapter")
    checkpoint_config.write_text(
        json.dumps(
            {
                "target_layers": list(range(42)),
                "delta_heads": ["q", "o"],
                "rank": 4,
                "rwkv_ms_semantics_version": 2,
                "memory_backend": "rwkv_ms",
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    training_protocol = {
        "schema_version": hard_failure_contract.OBJECTIVE_SCHEMA_VERSION,
        "memory_objective_version": hard_failure_contract.OBJECTIVE_VERSION,
        "gradient_accumulation_steps": (
            hard_failure_contract.GRADIENT_ACCUMULATION_STEPS
        ),
        "max_steps": hard_failure_contract.TOTAL_OPTIMIZER_STEPS,
        "save_steps": hard_failure_contract.SAVE_STEPS,
        "save_total_limit": hard_failure_contract.SAVE_TOTAL_LIMIT,
        "scene_generation_hard_failure_run_mode": (
            hard_failure_contract.PRODUCTION_RUN_MODE
        ),
        "scene_generation_hard_failure_production_eligible": True,
        "scene_generation_row_objective_audit_filename": (
            hard_failure_contract.ROW_OBJECTIVE_AUDIT_FILENAME
        ),
        "scene_generation_row_objective_audit_schema": (
            hard_failure_contract.ROW_OBJECTIVE_AUDIT_SCHEMA
        ),
        "train_schedule": {
            "checkpoint_steps": list(hard_failure_contract.CHECKPOINT_STEPS),
            "optimizer_checkpoint_steps": list(
                hard_failure_contract.CHECKPOINT_STEPS
            ),
            "generation_endpoint_steps": list(
                hard_failure_contract.GENERATION_ENDPOINT_STEPS
            ),
            "microbatch_cycle_size": 1,
            "continuation_policy": "forbidden_fresh_only",
        },
    }
    hard_failure_selector.run_audit._validate_protocol(
        training_protocol,
        smoke=False,
    )
    checkpoint_protocol.write_text(
        json.dumps(training_protocol, sort_keys=True),
        encoding="utf-8",
    )

    family_coverage = {
        suffix: 42
        for suffix in hard_failure_selector.run_audit.TRAINABLE_ADAPTER_TENSOR_SUFFIXES
    }
    adapter_change = {
        "changed_trainable_tensor_count": 1134,
        "changed_nontrainable_tensor_count": 0,
        "expected_trainable_tensor_count": 1134,
        "trainable_tensor_family_count": 27,
        "target_layer_count": 42,
        "trainable_family_layer_coverage": family_coverage,
        "missing_trainable_family_layers": {},
        "full_trainable_family_coverage": True,
        "full_trainable_family_coverage_required": False,
        "frozen_adapter_tensors_unchanged": True,
    }
    audit = {
        "schema": hard_failure_selector.run_audit.AUDIT_SCHEMA,
        "run_root": str(run_root),
        "checkpoint": str(memory_dir),
        "checkpoint_optimizer_step": step,
        "run_mode": hard_failure_contract.PRODUCTION_RUN_MODE,
        "objective_version": hard_failure_contract.OBJECTIVE_VERSION,
        "source_lock_sha256": hard_failure_contract.validate_source_lock()[
            "lock_sha256"
        ],
        "nontrainable_adapter_tensors_unchanged": True,
        "optimizer_contains_only_declared_trainable_adapter_state_count": True,
        "base_model_parameter_values_not_materialized_in_adapter_checkpoint": True,
        "row_audit_complete": True,
        "trainable_tensor_family_count": 27,
        "target_layer_count": 42,
        "trainable_family_layer_coverage": family_coverage,
        "full_trainable_family_coverage": True,
        "adapter_change": adapter_change,
    }
    audit["receipt_sha256"] = evaluator._canonical_sha256(audit)
    checkpoint_audit.write_text(
        json.dumps(audit, sort_keys=True),
        encoding="utf-8",
    )
    completion_audits = []
    for checkpoint_step in hard_failure_contract.CHECKPOINT_STEPS:
        audit_path = (
            run_root
            / "trainer"
            / f"checkpoint-{checkpoint_step}"
            / hard_failure_selector.run_audit.AUDIT_FILENAME
        )
        audit_path.parent.mkdir(parents=True, exist_ok=True)
        if checkpoint_step != step:
            audit_path.write_text(
                json.dumps({"checkpoint_optimizer_step": checkpoint_step}),
                encoding="utf-8",
            )
        completion_audits.append(
            {
                "step": checkpoint_step,
                "path": str(audit_path.resolve()),
                "sha256": evaluator.sha256_file(audit_path),
            }
        )
    completion = {
        "schema": "rwkv_ms_scene_hard_failure_completion.v1",
        "run_mode": "production",
        "run_root": str(run_root.resolve()),
        "global_step": hard_failure_contract.TOTAL_OPTIMIZER_STEPS,
        "checkpoint_steps": list(hard_failure_contract.CHECKPOINT_STEPS),
        "checkpoint_audits": completion_audits,
        "training_complete": True,
        "evaluation_accessed": False,
    }
    completion["receipt_sha256"] = evaluator._canonical_sha256(completion)
    completion_path = run_root.parent / "logs" / f"{run_root.name}.completion.json"
    completion_path.parent.mkdir()
    completion_path.write_text(
        json.dumps(completion, sort_keys=True),
        encoding="utf-8",
    )

    adapter_binding = hard_failure_selector.artifact_binding(
        checkpoint_adapter,
        description="checkpoint adapter",
    )
    coverage = hard_failure_selector.validate_recomputed_adapter_change(
        adapter_change,
        step=step,
    )
    checkpoint = {
        "path": str(memory_dir),
        "global_step": step,
        "artifacts": {
            "delta_mem_adapter.pt": adapter_binding,
            "delta_mem_config.json": hard_failure_selector.artifact_binding(
                checkpoint_config,
                description="checkpoint config",
            ),
            "training_protocol.json": hard_failure_selector.artifact_binding(
                checkpoint_protocol,
                description="checkpoint training protocol",
            ),
            hard_failure_selector.run_audit.AUDIT_FILENAME: (
                hard_failure_selector.artifact_binding(
                    checkpoint_audit,
                    description="checkpoint audit",
                )
            ),
        },
        **coverage,
        "checkpoint_audit_receipt_sha256": audit["receipt_sha256"],
        "current_adapter_validation": {
            "initial_adapter_manifest": hard_failure_selector.artifact_binding(
                initial_manifest,
                description="initial adapter manifest",
            ),
            "initial_adapter": hard_failure_selector.artifact_binding(
                initial_adapter,
                description="initial adapter",
            ),
            "checkpoint_adapter": adapter_binding,
            "recomputed_adapter_change_canonical_sha256": (
                evaluator._canonical_sha256(adapter_change)
            ),
            **coverage,
            "frozen_adapter_tensors_unchanged": True,
        },
    }
    source_lock = hard_failure_contract.validate_source_lock()
    source = {
        "source_lock_path": str(hard_failure_contract.SOURCE_LOCK.resolve()),
        "source_lock_file_sha256": evaluator.sha256_file(
            hard_failure_contract.SOURCE_LOCK
        ),
        "source_lock_sha256": source_lock["lock_sha256"],
        "source_manifest_path": str(hard_failure_contract.SOURCE_MANIFEST.resolve()),
        "source_manifest_file_sha256": (
            evaluator.SCENE_HARD_FAILURE_SOURCE_MANIFEST_FILE_SHA256
        ),
        "source_manifest_sha256": (
            evaluator.SCENE_HARD_FAILURE_SOURCE_MANIFEST_SHA256
        ),
        "train_file_sha256": evaluator.SCENE_HARD_FAILURE_TRAIN_FILE_SHA256,
        "pair_manifest_file_sha256": (
            evaluator.SCENE_HARD_FAILURE_PAIR_MANIFEST_FILE_SHA256
        ),
        "pair_manifest_sha256": (
            evaluator.SCENE_HARD_FAILURE_PAIR_MANIFEST_SHA256
        ),
        "entries_sha256": evaluator.SCENE_HARD_FAILURE_PAIR_ENTRIES_SHA256,
        "protected_evaluation_accessed": False,
    }
    results_dir = (run_root / f"train32_checkpoint_{step}").resolve()
    results_dir.mkdir()
    dataset_file = hard_failure_contract.TRAIN_FILE.resolve()
    validated_train_source = evaluator.validate_focused_train_source_manifest(
        hard_failure_contract.SOURCE_MANIFEST,
        dataset_file=dataset_file,
    )
    samples = evaluator.load_selected_rows(
        dataset_file,
        list(range(32)),
        expected_hashes=dict(validated_train_source["expected_row_hashes"]),
    )
    train_row_manifests = [
        json.loads(line)
        for line in hard_failure_contract.TRAIN_ROWS.read_text(
            encoding="utf-8"
        ).splitlines()
        if line.strip()
    ]
    assert len(train_row_manifests) == len(samples)
    for sample, row_manifest in zip(samples, train_row_manifests, strict=True):
        source_index = int(sample["source_index"])
        assert row_manifest["train_row_ordinal"] == source_index
        assert row_manifest["row_sha256"] == sample["row_sha256"]
        sample["write_token_count"] = row_manifest["token_metadata"][
            "write_token_count"
        ]
    candidate_lineage = evaluator.build_focused_train_checkpoint_lineage(
        memory_dir,
        train_source=validated_train_source,
    )
    candidate_lineage_record_binding = (
        evaluator.build_candidate_lineage_record_binding(candidate_lineage)
    )
    assert candidate_lineage_record_binding is not None
    donors = evaluator.materialize_focused_train_donor_mapping(
        samples,
        validated_train_source["donor_by_source_index"],
    )
    shuffled = evaluator.build_deterministic_shuffled_mapping(samples, donors)
    train_source = json.loads(json.dumps(validated_train_source, sort_keys=True))
    selection = train_source["selection"]
    donor_mapping = evaluator.donor_mapping_fingerprint_rows(samples, donors)
    shuffled_mapping = evaluator.shuffled_mapping_fingerprint_rows(
        samples,
        shuffled,
    )
    protocols = evaluator.resolved_condition_protocols(
        list(evaluator.SCENE_FOCUSED_CONDITIONS),
        donor_rule=evaluator.DONOR_RULE_LENGTH_MATCHED,
    )
    expected_hashes = {
        int(index): str(digest)
        for index, digest in validated_train_source["expected_row_hashes"].items()
    }
    evaluation_contract = evaluator.validate_scene_hard_failure_contract(
        contract=evaluator.SCENE_HARD_FAILURE_TRAIN_OVERFIT_CONTRACT,
        row_indices=list(range(32)),
        expected_hashes=expected_hashes,
        selection_dataset_contract=None,
        conditions=list(evaluator.SCENE_FOCUSED_CONDITIONS),
        donor_rule=evaluator.DONOR_RULE_LENGTH_MATCHED,
        max_new_tokens=evaluator.DEFAULT_MAX_NEW_TOKENS,
        normal_fusion_profile="native",
        expected_memory_layer_count=42,
        memory_target_layers=list(range(42)),
        memory_delta_heads=["q", "o"],
        memory_rank=4,
        rwkv_ms_semantics_version=2,
        memory_backend="rwkv_ms",
        memory_dir=memory_dir,
        selection_manifest_sha256=None,
        train_source=train_source,
        train_selection_authorization=None,
    )
    prompt_files = [
        {
            "relative_path": name,
            "bytes": int(identity["bytes"]),
            "sha256": str(identity["sha256"]),
        }
        for name, identity in sorted(
            hard_failure_contract.BASE_MODEL_ARTIFACTS.items()
        )
    ]
    base_model_prompt_artifacts = {
        "files": prompt_files,
        "combined_sha256": evaluator.sha256_text(
            json.dumps(prompt_files, sort_keys=True, separators=(",", ":"))
        ),
    }
    weight_files = [
        {
            "relative_path": hard_failure_contract.BASE_MODEL_WEIGHT_FILE,
            "bytes": hard_failure_contract.BASE_MODEL_WEIGHT_BYTES,
            "sha256": "0" * 64,
        }
    ]
    base_model_weights = {
        "layout": "unsharded",
        "files": weight_files,
        "combined_sha256": evaluator.sha256_text(
            json.dumps(weight_files, sort_keys=True, separators=(",", ":"))
        ),
    }
    profile_fields = evaluator.normal_fusion_fingerprint_fields("native", 42)
    fingerprint_payload = {
        "schema_version": 1,
        "task": evaluator.TASK_NAME,
        "split": "train",
        "code": evaluator.scene_state_code_fingerprint(evaluator.PROJECT_ROOT),
        "runtime_packages": evaluator.runtime_package_versions(),
        "delta_mem_root": str(evaluator.PROJECT_ROOT),
        "base_model": str(hard_failure_contract.PINNED_BASE_MODEL.resolve()),
        "base_model_weights": base_model_weights,
        "base_model_prompt_artifacts": base_model_prompt_artifacts,
        "memory_dir": str(memory_dir),
        "memory_config_sha256": checkpoint["artifacts"][
            "delta_mem_config.json"
        ]["sha256"],
        "memory_adapter_sha256": adapter_binding["sha256"],
        "dataset_file": str(dataset_file),
        "dataset_sha256": source["train_file_sha256"],
        "selection_source": {"kind": "inline_indices"},
        "selection": selection,
        "conditions": list(evaluator.SCENE_FOCUSED_CONDITIONS),
        "condition_protocols": protocols,
        "donor_rule": evaluator.DONOR_RULE_LENGTH_MATCHED,
        "state_only_donor_mapping": donor_mapping,
        "state_only_shuffled_mapping": shuffled_mapping,
        "evaluation_contract": evaluation_contract,
        "candidate_lineage": candidate_lineage,
        "candidate_lineage_record_binding": candidate_lineage_record_binding,
        "hard32_receipt_authorization": None,
        "scene_v7_train32_authorization": None,
        "scene_v14_candidate_authorization": None,
        "scene_v15_candidate_authorization": None,
        "focused_train_source": train_source,
        "focused_train_selection_authorization": None,
        "historical_v6_preflight": None,
        "max_new_tokens": evaluator.DEFAULT_MAX_NEW_TOKENS,
        "device": "cuda:0",
        "dtype": "bfloat16",
        "attn_implementation": "sdpa",
        **profile_fields,
    }
    evaluation_fingerprint = evaluator.fingerprint_payload_sha256(
        fingerprint_payload
    )
    manifest = {
        "fingerprint": evaluation_fingerprint,
        "fingerprint_payload": fingerprint_payload,
        "evaluation_contract": evaluation_contract,
        "candidate_lineage": candidate_lineage,
        "candidate_lineage_record_binding": candidate_lineage_record_binding,
    }
    (results_dir / "manifest.json").write_text(
        json.dumps(manifest, sort_keys=True),
        encoding="utf-8",
    )

    summary_conditions = {}
    samples_by_index = {
        int(sample["source_index"]): sample for sample in samples
    }
    for condition in evaluator.SCENE_FOCUSED_CONDITIONS:
        rows = []
        for index, identity in enumerate(selection):
            sample = samples_by_index[index]
            donor = donors[index]
            shuffled_sample = shuffled[index]
            gold = sample["gold"]
            if condition in {"normal_full", "state_only"}:
                parsed = gold
            elif condition == "state_only_shuffled":
                parsed = shuffled_sample["gold"]
            else:
                parsed = donor["gold"]
            row = {
                "status": "ok",
                "fingerprint": evaluation_fingerprint,
                "completed_at": "2026-08-01T00:00:00+00:00",
                "condition": condition,
                "condition_protocol": protocols[condition],
                "task": evaluator.TASK_NAME,
                "task_kind": "scene",
                "split": "train",
                "key": f"{evaluator.TASK_NAME}:{index}",
                "line_index": index,
                "source_index": index,
                "selection_ordinal": index,
                "row_sha256": identity["row_sha256"],
                "write_token_count": sample["write_token_count"],
                "candidate_lineage": candidate_lineage_record_binding,
                "gold": gold,
                "raw_generation": json.dumps(parsed, sort_keys=True),
                "parsed_json": parsed,
                "input_tokens": 10,
                "output_tokens": 2,
                "hit_max_new_tokens": False,
                "elapsed_seconds": 0.25,
                "input_rendered_sha256": "c" * 64,
                "peak_cuda_memory_bytes": None,
                "memory_trace": [],
                "online_state_after_generation": {},
                "prime": (
                    {
                        "tokens": 8,
                        "rendered_sha256": "d" * 64,
                        "kv_cache_retained": False,
                        "online_state": {},
                    }
                    if condition
                    in {
                        "state_only",
                        "state_only_donor",
                        "state_only_shuffled",
                        "state_only_no_write",
                    }
                    else None
                ),
                "donor_source_index": None,
                "donor_row_sha256": None,
                "semantic_decision_nll": None,
                "score_strict": evaluator.score_prediction("scene", parsed, gold),
                "score_recovered": evaluator.recovered_scene_score(parsed, gold),
            }
            if condition == "state_only_donor":
                row.update(
                    donor_source_index=donor_mapping[index]["donor_source_index"],
                    donor_row_sha256=donor_mapping[index]["donor_row_sha256"],
                )
            if condition == "state_only_shuffled":
                row.update(
                    shuffled_source_index=shuffled_mapping[index][
                        "shuffled_source_index"
                    ],
                    shuffled_row_sha256=shuffled_mapping[index][
                        "shuffled_row_sha256"
                    ],
                )
            rows.append(row)
        (results_dir / f"{condition}.jsonl").write_text(
            "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
            encoding="utf-8",
        )
        true_positives = sum(row["score_strict"]["tp"] for row in rows)
        false_positives = sum(row["score_strict"]["fp"] for row in rows)
        false_negatives = sum(row["score_strict"]["fn"] for row in rows)
        denominator = 2 * true_positives + false_positives + false_negatives
        summary_conditions[condition] = {
            "strict": {
                "primary_metric": (
                    0.0
                    if denominator == 0
                    else 2 * true_positives / denominator
                )
            }
        }
    summary = {
        "complete": True,
        "task": evaluator.TASK_NAME,
        "split": "train",
        "selected_source_indices": list(range(32)),
        "conditions": summary_conditions,
    }
    (results_dir / "summary.json").write_text(
        json.dumps(summary, sort_keys=True),
        encoding="utf-8",
    )
    expected_records = 32 * len(evaluator.SCENE_FOCUSED_CONDITIONS)
    (results_dir / "progress.json").write_text(
        json.dumps(
            {
                "fingerprint": evaluation_fingerprint,
                "completed": expected_records,
                "expected": expected_records,
                "complete": True,
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    gate = hard_failure_selector.focused_gate.analyze_results_dir(
        results_dir,
        stage="train_overfit",
    )
    assert gate["all_gates_passed"] is True
    gate_path = results_dir / "focused_recovery_gate.json"
    gate_path.write_text(json.dumps(gate, sort_keys=True), encoding="utf-8")
    artifact_names = (
        "manifest.json",
        "summary.json",
        "progress.json",
        *(f"{condition}.jsonl" for condition in evaluator.SCENE_FOCUSED_CONDITIONS),
    )
    evidence = hard_failure_selector.EndpointEvidence(
        step=step,
        checkpoint=checkpoint,
        evaluation={
            "results_dir": str(results_dir),
            "fingerprint": evaluation_fingerprint,
            "contract_canonical_sha256": evaluator._canonical_sha256(
                evaluation_contract
            ),
            "artifacts": {
                name: hard_failure_selector.artifact_binding(
                    results_dir / name,
                    description=f"checkpoint-{step} evaluation {name}",
                )
                for name in artifact_names
            },
        },
        gate={
            "focused_gate_path": str(gate_path),
            "file_sha256": evaluator.sha256_file(gate_path),
            "canonical_sha256": evaluator._canonical_sha256(gate),
            "evaluation_fingerprint": evaluation_fingerprint,
        },
        report=gate,
        fallback_rank=hard_failure_selector._fallback_rank(gate, step=step),
    )
    receipt = hard_failure_selector.build_selection_receipt(
        source=source,
        evaluated=[evidence],
        selected=evidence,
        passed=True,
        created_at="2026-08-01T00:00:00+00:00",
    )
    receipt_path = receipt_dir / "train32_checkpoint_selection_receipt.json"
    hard_failure_selector.atomic_write_json(receipt_path, receipt)
    return receipt_path, memory_dir, adapter_change


def _focused_v2_results_dir(memory_dir: Path) -> Path:
    step = int(memory_dir.name.removeprefix("checkpoint-"))
    return memory_dir.parent.parent / f"train32_checkpoint_{step}"


def _read_jsonl_records(path: Path) -> list[dict[str, object]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _write_jsonl_records(path: Path, records: list[dict[str, object]]) -> None:
    path.write_text(
        "".join(json.dumps(record, sort_keys=True) + "\n" for record in records),
        encoding="utf-8",
    )


def _rebind_focused_v2_selection_receipt(
    receipt_path: Path,
    memory_dir: Path,
) -> str:
    """Refresh every outer hash after a semantically adversarial result edit."""

    results_dir = _focused_v2_results_dir(memory_dir)
    manifest_path = results_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    fingerprint_payload = manifest["fingerprint_payload"]
    fingerprint = evaluator.fingerprint_payload_sha256(fingerprint_payload)
    manifest["fingerprint"] = fingerprint
    manifest["evaluation_contract"] = fingerprint_payload["evaluation_contract"]
    manifest_path.write_text(
        json.dumps(manifest, sort_keys=True),
        encoding="utf-8",
    )

    progress_path = results_dir / "progress.json"
    progress = json.loads(progress_path.read_text(encoding="utf-8"))
    progress["fingerprint"] = fingerprint
    progress_path.write_text(
        json.dumps(progress, sort_keys=True),
        encoding="utf-8",
    )

    gate_path = results_dir / "focused_recovery_gate.json"
    gate = json.loads(gate_path.read_text(encoding="utf-8"))
    gate["input"] = {
        "results_dir": str(results_dir.resolve()),
        "evaluation_fingerprint": fingerprint,
        "evaluation_contract": manifest["evaluation_contract"],
    }
    gate_path.write_text(json.dumps(gate, sort_keys=True), encoding="utf-8")

    artifact_names = (
        "manifest.json",
        "summary.json",
        "progress.json",
        *(f"{condition}.jsonl" for condition in evaluator.SCENE_FOCUSED_CONDITIONS),
    )
    artifacts = {
        name: hard_failure_selector.artifact_binding(
            results_dir / name,
            description=f"adversarial evaluation {name}",
        )
        for name in artifact_names
    }
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    endpoint = receipt["evaluated_endpoints"][-1]
    evaluation = dict(endpoint["evaluation"])
    evaluation.update(
        {
            "fingerprint": fingerprint,
            "contract_canonical_sha256": evaluator._canonical_sha256(
                manifest["evaluation_contract"]
            ),
            "artifacts": artifacts,
        }
    )
    gate_binding = {
        "focused_gate_path": str(gate_path),
        "file_sha256": evaluator.sha256_file(gate_path),
        "canonical_sha256": evaluator._canonical_sha256(gate),
        "evaluation_fingerprint": fingerprint,
    }
    endpoint["evaluation"] = evaluation
    endpoint["gate"] = gate_binding
    receipt["evaluation"] = json.loads(json.dumps(evaluation, sort_keys=True))
    receipt["gate"] = dict(gate_binding)
    receipt.pop("receipt_sha256", None)
    receipt["receipt_sha256"] = evaluator._canonical_sha256(receipt)
    hard_failure_selector.atomic_write_json(receipt_path, receipt)
    return fingerprint


def _assert_focused_v2_selection_rejected_without_claim(
    *,
    receipt_path: Path,
    memory_dir: Path,
    adapter_change: dict[str, object],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        evaluator,
        "_recompute_focused_adapter_change",
        lambda **kwargs: dict(adapter_change),
    )
    with pytest.raises(ValueError):
        evaluator.validate_focused_hard32_exactly_once_authorization(
            **_focused_exactly_once_kwargs(
                receipt_path=receipt_path,
                memory_dir=memory_dir,
                output_dir=memory_dir.parent.parent / "hard32_once",
                dataset_file=evaluator.HISTORICAL_V6_OFFICIAL_VAL,
                selection_file=evaluator.HISTORICAL_V6_HARD32_SELECTION,
            )
        )
    assert not evaluator.scene_hard_failure_consumption_marker_path(
        receipt_path
    ).exists()


def test_focused_v2_selection_authorization_reaches_atomic_one_shot_claim(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    receipt_path, memory_dir, adapter_change = _valid_focused_v2_selection_receipt(
        tmp_path
    )
    recomputed: list[Path] = []

    def recompute_adapter_change(*, checkpoint_adapter_path: Path, **kwargs):
        recomputed.append(checkpoint_adapter_path)
        return dict(adapter_change)

    monkeypatch.setattr(
        evaluator,
        "_recompute_focused_adapter_change",
        recompute_adapter_change,
    )
    output_dir = memory_dir.parent.parent / "hard32_once"
    authorization = evaluator.validate_focused_hard32_exactly_once_authorization(
        **_focused_exactly_once_kwargs(
            receipt_path=receipt_path,
            memory_dir=memory_dir,
            output_dir=output_dir,
            dataset_file=evaluator.HISTORICAL_V6_OFFICIAL_VAL,
            selection_file=evaluator.HISTORICAL_V6_HARD32_SELECTION,
        )
    )

    marker_path = evaluator.scene_hard_failure_consumption_marker_path(receipt_path)
    marker_bytes = marker_path.read_bytes()
    marker_file_sha256 = evaluator.sha256_file(marker_path)
    marker = json.loads(marker_bytes)
    unsigned = dict(marker)
    claim_sha256 = unsigned.pop("claim_sha256")
    assert recomputed == [memory_dir / "delta_mem_adapter.pt"]
    assert authorization["hard32_authorized"] is True
    assert authorization["selected_checkpoint"]["global_step"] == 16
    assert authorization["authorization_consumption"]["path"] == str(marker_path)
    assert marker["selection_receipt"] == authorization["receipt"]
    assert marker["selected_checkpoint"] == authorization["selected_checkpoint"]
    assert marker["hard32_output_dir"] == str(output_dir.absolute())
    assert marker["schema"] == (
        evaluator.SCENE_HARD_FAILURE_HARD32_CONSUMPTION_MARKER_SCHEMA
    )
    assert claim_sha256 == evaluator.fingerprint_payload_sha256(unsigned)
    assert marker_path.stat().st_mode & 0o777 == 0o400

    with pytest.raises(ValueError, match="already consumed"):
        evaluator.validate_focused_hard32_exactly_once_authorization(
            **_focused_exactly_once_kwargs(
                receipt_path=receipt_path,
                memory_dir=memory_dir,
                output_dir=output_dir,
                dataset_file=evaluator.HISTORICAL_V6_OFFICIAL_VAL,
                selection_file=evaluator.HISTORICAL_V6_HARD32_SELECTION,
            )
        )
    assert recomputed == [memory_dir / "delta_mem_adapter.pt"] * 2
    assert marker_path.read_bytes() == marker_bytes
    assert evaluator.sha256_file(marker_path) == marker_file_sha256


def test_focused_v2_rejects_nonexistent_train32_bundle_claiming_canonical_hashes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    receipt_path, memory_dir, adapter_change = _valid_focused_v2_selection_receipt(
        tmp_path
    )
    results_dir = _focused_v2_results_dir(memory_dir)
    manifest_path = results_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    payload = manifest["fingerprint_payload"]
    forged_dataset = results_dir.parent / "synthetic_train32" / "train.jsonl"
    assert not forged_dataset.exists()
    payload["dataset_file"] = str(forged_dataset)
    payload["focused_train_source"]["dataset"]["path"] = str(forged_dataset)
    payload["evaluation_contract"]["train_source"]["dataset"]["path"] = str(
        forged_dataset
    )
    assert payload["dataset_sha256"] == (
        evaluator.SCENE_HARD_FAILURE_TRAIN_FILE_SHA256
    )
    forged_fingerprint = evaluator.fingerprint_payload_sha256(payload)
    manifest_path.write_text(
        json.dumps(manifest, sort_keys=True),
        encoding="utf-8",
    )
    for condition in evaluator.SCENE_FOCUSED_CONDITIONS:
        path = results_dir / f"{condition}.jsonl"
        records = _read_jsonl_records(path)
        for record in records:
            record["fingerprint"] = forged_fingerprint
        _write_jsonl_records(path, records)
    assert (
        _rebind_focused_v2_selection_receipt(receipt_path, memory_dir)
        == forged_fingerprint
    )

    _assert_focused_v2_selection_rejected_without_claim(
        receipt_path=receipt_path,
        memory_dir=memory_dir,
        adapter_change=adapter_change,
        monkeypatch=monkeypatch,
    )


def test_focused_v2_rejects_condition_relabeling_with_wrong_protocol(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    receipt_path, memory_dir, adapter_change = _valid_focused_v2_selection_receipt(
        tmp_path
    )
    results_dir = _focused_v2_results_dir(memory_dir)
    normal_path = results_dir / "normal_full.jsonl"
    state_path = results_dir / "state_only.jsonl"
    normal_records = _read_jsonl_records(normal_path)
    state_records = _read_jsonl_records(state_path)
    relabeled_normal = []
    relabeled_state = []
    for normal_record, state_record in zip(
        normal_records,
        state_records,
        strict=True,
    ):
        forged_normal = dict(state_record)
        forged_normal["condition"] = "normal_full"
        forged_state = dict(normal_record)
        forged_state["condition"] = "state_only"
        relabeled_normal.append(forged_normal)
        relabeled_state.append(forged_state)
    assert relabeled_normal[0]["condition_protocol"] == (
        state_records[0]["condition_protocol"]
    )
    assert relabeled_state[0]["condition_protocol"] == (
        normal_records[0]["condition_protocol"]
    )
    _write_jsonl_records(normal_path, relabeled_normal)
    _write_jsonl_records(state_path, relabeled_state)
    _rebind_focused_v2_selection_receipt(receipt_path, memory_dir)

    _assert_focused_v2_selection_rejected_without_claim(
        receipt_path=receipt_path,
        memory_dir=memory_dir,
        adapter_change=adapter_change,
        monkeypatch=monkeypatch,
    )


def test_focused_v2_rejects_gate_minimal_records(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    receipt_path, memory_dir, adapter_change = _valid_focused_v2_selection_receipt(
        tmp_path
    )
    results_dir = _focused_v2_results_dir(memory_dir)
    gate_fields = {
        "status",
        "condition",
        "task",
        "split",
        "source_index",
        "row_sha256",
        "gold",
        "parsed_json",
        "score_strict",
        "score_recovered",
    }
    for condition in evaluator.SCENE_FOCUSED_CONDITIONS:
        path = results_dir / f"{condition}.jsonl"
        records = _read_jsonl_records(path)
        if condition == "state_only_donor":
            required = gate_fields | {"donor_source_index", "donor_row_sha256"}
        elif condition == "state_only_shuffled":
            required = gate_fields | {
                "shuffled_source_index",
                "shuffled_row_sha256",
            }
        else:
            required = gate_fields
        minimal = [
            {field: record[field] for field in required}
            for record in records
        ]
        _write_jsonl_records(path, minimal)
    _rebind_focused_v2_selection_receipt(receipt_path, memory_dir)

    _assert_focused_v2_selection_rejected_without_claim(
        receipt_path=receipt_path,
        memory_dir=memory_dir,
        adapter_change=adapter_change,
        monkeypatch=monkeypatch,
    )


def test_focused_v2_rejects_raw_generation_that_disagrees_with_parsed_json(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    receipt_path, memory_dir, adapter_change = _valid_focused_v2_selection_receipt(
        tmp_path
    )
    path = _focused_v2_results_dir(memory_dir) / "base_full.jsonl"
    records = _read_jsonl_records(path)
    records[0]["raw_generation"] = "{}"
    assert evaluator.extract_json(records[0]["raw_generation"]) != records[0][
        "parsed_json"
    ]
    _write_jsonl_records(path, records)
    _rebind_focused_v2_selection_receipt(receipt_path, memory_dir)

    _assert_focused_v2_selection_rejected_without_claim(
        receipt_path=receipt_path,
        memory_dir=memory_dir,
        adapter_change=adapter_change,
        monkeypatch=monkeypatch,
    )


@pytest.mark.parametrize(
    "mutation",
    ("condition_protocol", "fingerprint", "candidate_lineage"),
)
def test_focused_v2_rejects_record_identity_mutation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mutation: str,
) -> None:
    receipt_path, memory_dir, adapter_change = _valid_focused_v2_selection_receipt(
        tmp_path
    )
    path = _focused_v2_results_dir(memory_dir) / "base_full.jsonl"
    records = _read_jsonl_records(path)
    if mutation == "condition_protocol":
        records[0][mutation] = evaluator.CONDITION_PROTOCOLS["state_only"]
    elif mutation == "fingerprint":
        records[0][mutation] = "0" * 64
    else:
        records[0][mutation] = {
            "lineage_kind": "forged_candidate",
            "lineage_sha256": "0" * 64,
        }
    _write_jsonl_records(path, records)
    _rebind_focused_v2_selection_receipt(receipt_path, memory_dir)

    _assert_focused_v2_selection_rejected_without_claim(
        receipt_path=receipt_path,
        memory_dir=memory_dir,
        adapter_change=adapter_change,
        monkeypatch=monkeypatch,
    )


def test_focused_v2_rejects_adapter_mutation_after_gate_before_claim(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    receipt_path, memory_dir, adapter_change = _valid_focused_v2_selection_receipt(
        tmp_path
    )
    checkpoint_adapter = memory_dir / "delta_mem_adapter.pt"
    monkeypatch.setattr(
        evaluator,
        "_recompute_focused_adapter_change",
        lambda **kwargs: dict(adapter_change),
    )
    real_analyze = hard_failure_selector.focused_gate.analyze_results_dir
    mutation_count = 0

    def mutate_adapter_after_gate(*args, **kwargs):
        nonlocal mutation_count
        report = real_analyze(*args, **kwargs)
        mutation_count += 1
        checkpoint_adapter.write_bytes(
            checkpoint_adapter.read_bytes() + b"-mutated-after-gate"
        )
        return report

    monkeypatch.setattr(
        hard_failure_selector.focused_gate,
        "analyze_results_dir",
        mutate_adapter_after_gate,
    )
    with pytest.raises(ValueError):
        evaluator.validate_focused_hard32_exactly_once_authorization(
            **_focused_exactly_once_kwargs(
                receipt_path=receipt_path,
                memory_dir=memory_dir,
                output_dir=memory_dir.parent.parent / "hard32_once",
                dataset_file=evaluator.HISTORICAL_V6_OFFICIAL_VAL,
                selection_file=evaluator.HISTORICAL_V6_HARD32_SELECTION,
            )
        )

    assert mutation_count >= 1
    assert not evaluator.scene_hard_failure_consumption_marker_path(
        receipt_path
    ).exists()


def test_focused_hard32_claim_is_atomic_bound_and_never_opens_protected_inputs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    memory_dir = _focused_memory_dir(tmp_path)
    receipt_path, selection_authorization = _focused_exactly_once_fixture(
        tmp_path,
        memory_dir,
    )
    dataset_file = evaluator.HISTORICAL_V6_OFFICIAL_VAL
    selection_file = evaluator.HISTORICAL_V6_HARD32_SELECTION
    output_dir = memory_dir.parent.parent / "hard32_once"
    kwargs = _focused_exactly_once_kwargs(
        receipt_path=receipt_path,
        memory_dir=memory_dir,
        output_dir=output_dir,
        dataset_file=dataset_file,
        selection_file=selection_file,
    )
    monkeypatch.setattr(
        evaluator,
        "validate_focused_train_selection_authorization",
        lambda *args, **kwargs: selection_authorization,
    )
    monkeypatch.setattr(evaluator, "resolved_memory_layer_count", lambda *args: 42)
    monkeypatch.setattr(
        evaluator,
        "memory_architecture_contract",
        lambda *args: {
            "target_layers": list(range(42)),
            "delta_heads": ["q", "o"],
            "rank": 4,
            "rwkv_ms_semantics_version": 2,
            "memory_backend": "rwkv_ms",
        },
    )

    protected = {
        str(dataset_file.absolute()),
        str(selection_file.absolute()),
        str(evaluator.HISTORICAL_V6_BASE_MODEL.absolute()),
    }
    protected_accesses: list[tuple[str, str]] = []
    real_stat = os.stat
    real_lstat = os.lstat
    real_open = Path.open

    def guarded_stat(path, *args, **kwargs):
        if str(Path(path).absolute()) in protected:
            protected_accesses.append(("stat", str(path)))
            raise AssertionError("protected Hard32 stat before exclusive claim")
        return real_stat(path, *args, **kwargs)

    def guarded_lstat(path, *args, **kwargs):
        if str(Path(path).absolute()) in protected:
            protected_accesses.append(("lstat", str(path)))
            raise AssertionError("protected Hard32 lstat before exclusive claim")
        return real_lstat(path, *args, **kwargs)

    def guarded_open(path: Path, *args, **kwargs):
        if str(path.absolute()) in protected:
            protected_accesses.append(("open", str(path)))
            raise AssertionError("protected Hard32 open before exclusive claim")
        return real_open(path, *args, **kwargs)

    monkeypatch.setattr(os, "stat", guarded_stat)
    monkeypatch.setattr(os, "lstat", guarded_lstat)
    monkeypatch.setattr(Path, "open", guarded_open)

    authorization = evaluator.validate_focused_hard32_exactly_once_authorization(
        **kwargs
    )
    marker_path = evaluator.scene_hard_failure_consumption_marker_path(receipt_path)
    marker = json.loads(marker_path.read_text(encoding="utf-8"))
    unsigned = dict(marker)
    claim_sha256 = unsigned.pop("claim_sha256")

    assert protected_accesses == []
    assert marker["schema"] == (
        evaluator.SCENE_HARD_FAILURE_HARD32_CONSUMPTION_MARKER_SCHEMA
    )
    assert claim_sha256 == evaluator.fingerprint_payload_sha256(unsigned)
    assert marker["selection_receipt"] == selection_authorization["receipt"]
    assert marker["selected_checkpoint"] == selection_authorization[
        "selected_checkpoint"
    ]
    assert marker["hard32_output_dir"] == str(output_dir.absolute())
    assert marker["base_model_path"] == str(
        evaluator.HISTORICAL_V6_BASE_MODEL.absolute()
    )
    assert marker["runtime"] == {
        "device": "cuda:0",
        "dtype": "bfloat16",
        "attn_implementation": "sdpa",
    }
    assert marker["delta_mem_root"] == str(evaluator.PROJECT_ROOT.absolute())
    assert marker["frozen_donor_pairs_sha256"] == (
        evaluator.HARD32_FROZEN_DONOR_PAIRS_SHA256
    )
    assert marker["retry_authorized"] is False
    assert marker_path.stat().st_mode & 0o777 == 0o400
    assert authorization["authorization_consumption"]["path"] == str(marker_path)
    assert authorization["hard32_output_dir"] == str(output_dir.absolute())

    with pytest.raises(ValueError, match="already consumed"):
        evaluator.validate_focused_hard32_exactly_once_authorization(
            **kwargs
        )
    assert json.loads(marker_path.read_text(encoding="utf-8")) == marker


@pytest.mark.parametrize(
    ("mutation", "message"),
    (
        ("dataset_file", "official val path"),
        ("selection_file", "Hard32 selection path"),
        ("base_model", "base-model path"),
        ("device", "CUDA 0, bfloat16, and SDPA"),
        ("dtype", "CUDA 0, bfloat16, and SDPA"),
        ("attn_implementation", "CUDA 0, bfloat16, and SDPA"),
        ("delta_mem_root", "this Delta-Mem checkout"),
    ),
)
def test_focused_hard32_rejects_noncanonical_paths_and_runtime_before_receipt(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mutation: str,
    message: str,
) -> None:
    memory_dir = _focused_memory_dir(tmp_path)
    receipt_path, _ = _focused_exactly_once_fixture(tmp_path, memory_dir)
    kwargs = _focused_exactly_once_kwargs(
        receipt_path=receipt_path,
        memory_dir=memory_dir,
        output_dir=memory_dir.parent.parent / "hard32_once",
        dataset_file=evaluator.HISTORICAL_V6_OFFICIAL_VAL,
        selection_file=evaluator.HISTORICAL_V6_HARD32_SELECTION,
    )
    wrong_values = {
        "dataset_file": tmp_path / "other" / "val.jsonl",
        "selection_file": tmp_path / "other" / "holdout_source_indices.json",
        "base_model": tmp_path / "other" / "base-model",
        "device": "cuda:1",
        "dtype": "float16",
        "attn_implementation": "flash_attention_2",
        "delta_mem_root": tmp_path / "other" / "delta-mem",
    }
    kwargs[mutation] = wrong_values[mutation]
    monkeypatch.setattr(
        evaluator,
        "validate_focused_train_selection_authorization",
        lambda *args, **kwargs: pytest.fail(
            "noncanonical binding must fail before receipt validation"
        ),
    )

    with pytest.raises(ValueError, match=message):
        evaluator.validate_focused_hard32_exactly_once_authorization(**kwargs)

    assert not evaluator.scene_hard_failure_consumption_marker_path(
        receipt_path
    ).exists()


def test_focused_hard32_recomputes_donor_pair_hash_before_receipt(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    memory_dir = _focused_memory_dir(tmp_path)
    receipt_path, _ = _focused_exactly_once_fixture(tmp_path, memory_dir)
    kwargs = _focused_exactly_once_kwargs(
        receipt_path=receipt_path,
        memory_dir=memory_dir,
        output_dir=memory_dir.parent.parent / "hard32_once",
        dataset_file=evaluator.HISTORICAL_V6_OFFICIAL_VAL,
        selection_file=evaluator.HISTORICAL_V6_HARD32_SELECTION,
    )
    monkeypatch.setattr(evaluator, "HARD32_FROZEN_DONOR_PAIRS_SHA256", "0" * 64)
    monkeypatch.setattr(
        evaluator,
        "validate_focused_train_selection_authorization",
        lambda *args, **kwargs: pytest.fail(
            "donor hash must fail before receipt validation"
        ),
    )

    with pytest.raises(RuntimeError, match="donor pair-list hash differs"):
        evaluator.validate_focused_hard32_exactly_once_authorization(**kwargs)

    assert not evaluator.scene_hard_failure_consumption_marker_path(
        receipt_path
    ).exists()


@pytest.mark.parametrize(
    ("mutation", "message"),
    (
        ("copied_receipt", "canonical checkpoint and Train32 selection receipt"),
        ("noncanonical_checkpoint", "canonical checkpoint and Train32 selection receipt"),
        ("alternate_output", "canonical hard32_once output directory"),
    ),
)
def test_focused_hard32_rejects_copied_receipt_or_alternate_output(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mutation: str,
    message: str,
) -> None:
    memory_dir = _focused_memory_dir(tmp_path)
    receipt_path, selection_authorization = _focused_exactly_once_fixture(
        tmp_path,
        memory_dir,
    )
    kwargs = _focused_exactly_once_kwargs(
        receipt_path=receipt_path,
        memory_dir=memory_dir,
        output_dir=memory_dir.parent.parent / "hard32_once",
        dataset_file=evaluator.HISTORICAL_V6_OFFICIAL_VAL,
        selection_file=evaluator.HISTORICAL_V6_HARD32_SELECTION,
    )
    if mutation == "copied_receipt":
        copied_receipt = tmp_path / "copied" / receipt_path.name
        copied_receipt.parent.mkdir()
        copied_receipt.write_bytes(receipt_path.read_bytes())
        kwargs["selection_receipt_path"] = copied_receipt
    elif mutation == "noncanonical_checkpoint":
        selection_authorization["selected_checkpoint"]["path"] = str(
            memory_dir.parent.parent / "other" / "checkpoint-64"
        )
    else:
        kwargs["output_dir"] = memory_dir.parent.parent / "hard32_second_run"
    monkeypatch.setattr(
        evaluator,
        "validate_focused_train_selection_authorization",
        lambda *args, **kwargs: selection_authorization,
    )

    with pytest.raises(ValueError, match=message):
        evaluator.validate_focused_hard32_exactly_once_authorization(**kwargs)

    assert not evaluator.scene_hard_failure_consumption_marker_path(
        receipt_path
    ).exists()


@pytest.mark.parametrize(
    ("overwrite", "existing_output", "message"),
    (
        (True, False, "overwrite"),
        (False, True, "cannot resume"),
    ),
)
def test_focused_hard32_rejects_overwrite_or_resume_before_receipt_validation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    overwrite: bool,
    existing_output: bool,
    message: str,
) -> None:
    memory_dir = _focused_memory_dir(tmp_path)
    receipt_path, _ = _focused_exactly_once_fixture(tmp_path, memory_dir)
    output_dir = memory_dir.parent.parent / "hard32_once"
    if existing_output:
        output_dir.mkdir()
    kwargs = _focused_exactly_once_kwargs(
        receipt_path=receipt_path,
        memory_dir=memory_dir,
        output_dir=output_dir,
        dataset_file=evaluator.HISTORICAL_V6_OFFICIAL_VAL,
        selection_file=evaluator.HISTORICAL_V6_HARD32_SELECTION,
    )
    kwargs["overwrite"] = overwrite
    monkeypatch.setattr(
        evaluator,
        "validate_focused_train_selection_authorization",
        lambda *args, **kwargs: pytest.fail(
            "receipt validation must not run for an invalid one-shot output"
        ),
    )

    with pytest.raises(ValueError, match=message):
        evaluator.validate_focused_hard32_exactly_once_authorization(**kwargs)

    assert not evaluator.scene_hard_failure_consumption_marker_path(
        receipt_path
    ).exists()


def test_focused_hard32_main_claims_before_protected_dataset_resolution(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    memory_dir = tmp_path / "checkpoint-64"
    memory_dir.mkdir()
    selection_receipt = tmp_path / "train-selection" / "selection_receipt.json"
    protected_selection = tmp_path / "protected" / "selection.json"
    protected_dataset = tmp_path / "protected" / "val.jsonl"
    events: list[str] = []
    args = SimpleNamespace(
        preflight_only=False,
        evaluation_contract=evaluator.SCENE_HARD_FAILURE_HARD32_CONTRACT,
        conditions=",".join(evaluator.SCENE_FOCUSED_CONDITIONS),
        max_new_tokens=evaluator.DEFAULT_MAX_NEW_TOKENS,
        memory_dir=memory_dir,
        output_dir=tmp_path / "focused-hard32",
        row_indices=None,
        row_indices_file=protected_selection,
        hard32_receipt=None,
        scene_v7_train32_receipt=None,
        scene_v8_train32_receipt=None,
        scene_v14_value14_receipt=None,
        scene_v14_candidate_lock=None,
        scene_v14_launch_receipt=None,
        scene_v14_completion_receipt=None,
        scene_v15_selection_receipt=None,
        scene_v15_candidate_lock=None,
        scene_v15_launch_receipt=None,
        scene_v15_completion_receipt=None,
        focused_source_manifest=None,
        focused_train_selection_receipt=selection_receipt,
        overwrite=False,
        dataset_file=protected_dataset,
        base_model=str(evaluator.HISTORICAL_V6_BASE_MODEL),
        device="cuda:0",
        dtype="bfloat16",
        attn_implementation="sdpa",
        delta_mem_root=str(evaluator.PROJECT_ROOT),
        donor_rule=evaluator.DONOR_RULE_LENGTH_MATCHED,
        normal_fusion_profile="native",
        expected_memory_layer_count=42,
    )

    monkeypatch.setattr(evaluator, "parse_args", lambda: args)

    def claim_authorization(**kwargs):
        events.append("authorization_claimed")
        return {"authorization_consumption": {"schema": "claimed"}}

    def reject_dataset_resolution(*args, **kwargs):
        assert events == ["authorization_claimed"]
        raise RuntimeError("protected dataset resolution reached after claim")

    monkeypatch.setattr(
        evaluator,
        "validate_focused_hard32_exactly_once_authorization",
        claim_authorization,
    )
    monkeypatch.setattr(
        evaluator,
        "resolve_scene_dataset_file",
        reject_dataset_resolution,
    )

    with pytest.raises(RuntimeError, match="reached after claim"):
        evaluator.main()

    assert events == ["authorization_claimed"]


def test_focused_train_main_uses_hard_failure_checkpoint_lineage(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    memory_dir = tmp_path / "run" / "trainer" / "checkpoint-16"
    memory_dir.mkdir(parents=True)
    (memory_dir / "delta_mem_config.json").write_text(
        json.dumps(
            {
                "target_layers": list(range(42)),
                "delta_heads": ["q", "o"],
                "rank": 4,
                "rwkv_ms_semantics_version": 2,
                "memory_backend": "rwkv_ms",
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    train_source = {
        "expected_row_hashes": {index: f"{index:064x}" for index in range(32)},
    }
    args = SimpleNamespace(
        preflight_only=False,
        evaluation_contract=evaluator.SCENE_HARD_FAILURE_TRAIN_OVERFIT_CONTRACT,
        conditions=",".join(evaluator.SCENE_FOCUSED_CONDITIONS),
        max_new_tokens=evaluator.DEFAULT_MAX_NEW_TOKENS,
        memory_dir=memory_dir,
        output_dir=tmp_path / "train32",
        row_indices=json.dumps(list(range(32))),
        row_indices_file=None,
        hard32_receipt=None,
        scene_v7_train32_receipt=None,
        scene_v8_train32_receipt=None,
        scene_v14_value14_receipt=None,
        scene_v14_candidate_lock=None,
        scene_v14_launch_receipt=None,
        scene_v14_completion_receipt=None,
        scene_v15_selection_receipt=None,
        scene_v15_candidate_lock=None,
        scene_v15_launch_receipt=None,
        scene_v15_completion_receipt=None,
        focused_source_manifest=tmp_path / "source_manifest.json",
        focused_train_selection_receipt=None,
        overwrite=False,
        dataset_file=tmp_path / "train.jsonl",
        base_model=str(evaluator.HISTORICAL_V6_BASE_MODEL),
        device="cuda:0",
        dtype="bfloat16",
        attn_implementation="sdpa",
        delta_mem_root=str(evaluator.PROJECT_ROOT),
        donor_rule=evaluator.DONOR_RULE_LENGTH_MATCHED,
        normal_fusion_profile="native",
        expected_memory_layer_count=42,
    )
    lineage = {
        "lineage_kind": (
            evaluator.SCENE_HARD_FAILURE_TRAIN_CHECKPOINT_LINEAGE_KIND
        ),
        "global_step": 16,
    }
    calls: list[tuple[Path, dict[str, object]]] = []

    monkeypatch.setattr(evaluator, "parse_args", lambda: args)
    monkeypatch.setattr(
        evaluator,
        "validate_focused_train_source_manifest",
        lambda *args, **kwargs: train_source,
    )

    def build_lineage(path: Path, *, train_source: dict[str, object]):
        calls.append((path, train_source))
        return lineage

    monkeypatch.setattr(
        evaluator,
        "build_focused_train_checkpoint_lineage",
        build_lineage,
    )
    monkeypatch.setattr(
        evaluator,
        "scene_v6_training_lineage",
        lambda path: pytest.fail("focused Train32 used legacy V6 lineage"),
    )

    def stop_after_lineage(**kwargs):
        raise RuntimeError("focused Train32 lineage dispatch reached")

    monkeypatch.setattr(
        evaluator,
        "validate_scene_v6_matched_donor_contract",
        stop_after_lineage,
    )

    with pytest.raises(RuntimeError, match="lineage dispatch reached"):
        evaluator.main()

    assert calls == [(memory_dir.resolve(), train_source)]


def test_scene_v6_matched_donor_contract_requires_all_170_hashed_val_rows() -> None:
    row_indices = list(range(170))
    expected_hashes = {index: f"{index:064x}"[-64:] for index in row_indices}
    dataset_contract = {
        "split": "val",
        "path": "/official/val.jsonl",
        "sha256": evaluator.OFFICIAL_SCENE_V4_VAL_SHA256,
    }

    contract = evaluator.validate_scene_v6_matched_donor_contract(
        contract="scene_v6_matched_donor_validation",
        row_indices=row_indices,
        expected_hashes=expected_hashes,
        selection_dataset_contract=dataset_contract,
        conditions=["state_only", "state_only_donor"],
        donor_rule=evaluator.DONOR_RULE_LENGTH_MATCHED,
        max_new_tokens=evaluator.DEFAULT_MAX_NEW_TOKENS,
        normal_fusion_profile="native",
        expected_memory_layer_count=42,
        memory_target_layers=list(range(42)),
        memory_delta_heads=["q", "o"],
        memory_rank=4,
        rwkv_ms_semantics_version=2,
        memory_backend="rwkv_ms",
        hard32_receipt_authorization={"status": "pass"},
    )

    assert contract["rows"] == 170
    assert contract["split"] == "val"
    assert contract["donor_rule"] == (
        "length_matched_label_distinct_symmetric_pair_v1"
    )
    assert contract["test_selection_forbidden"] is True


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ({"row_indices": list(range(169))}, "all 170 official"),
        ({"expected_hashes": {}}, "source hash for every row"),
        ({"selection_dataset_contract": None}, "dataset-bound selection"),
        ({"conditions": ["state_only"]}, "exact order"),
        ({"donor_rule": evaluator.DONOR_RULE_CYCLIC}, "requires donor_rule"),
        ({"max_new_tokens": 64}, "requires max_new_tokens=128"),
        ({"normal_fusion_profile": "native_gate_open"}, "requires normal_fusion_profile=native"),
        ({"expected_memory_layer_count": 41}, "requires expected_memory_layer_count=42"),
        ({"memory_target_layers": list(range(41))}, "requires checkpoint target_layers=0..41"),
        ({"memory_delta_heads": ["o"]}, "requires checkpoint delta_heads=q,o"),
        ({"memory_rank": 8}, "requires checkpoint rank=4"),
        ({"rwkv_ms_semantics_version": 1}, "requires checkpoint rwkv_ms_semantics_version=2"),
        ({"memory_backend": "delta"}, "requires checkpoint memory_backend=rwkv_ms"),
        (
            {"selection_dataset_contract": {"split": "val", "path": "/official/val.jsonl", "sha256": "b" * 64}},
            "requires the official scene-v4 val file",
        ),
    ],
)
def test_scene_v6_matched_donor_contract_fails_closed(mutation, message) -> None:
    row_indices = list(range(170))
    values = {
        "contract": "scene_v6_matched_donor_validation",
        "row_indices": row_indices,
        "expected_hashes": {index: "a" * 64 for index in row_indices},
        "selection_dataset_contract": {
            "split": "val",
            "path": "/official/val.jsonl",
            "sha256": evaluator.OFFICIAL_SCENE_V4_VAL_SHA256,
        },
        "conditions": ["state_only", "state_only_donor"],
        "donor_rule": evaluator.DONOR_RULE_LENGTH_MATCHED,
        "max_new_tokens": evaluator.DEFAULT_MAX_NEW_TOKENS,
        "normal_fusion_profile": "native",
        "expected_memory_layer_count": 42,
        "memory_target_layers": list(range(42)),
        "memory_delta_heads": ["q", "o"],
        "memory_rank": 4,
        "rwkv_ms_semantics_version": 2,
        "memory_backend": "rwkv_ms",
        "hard32_receipt_authorization": {"status": "pass"},
    }
    values.update(mutation)

    with pytest.raises(ValueError, match=message):
        evaluator.validate_scene_v6_matched_donor_contract(**values)


def test_scene_v6_identity_hard32_contract_binds_authoritative_selection() -> None:
    dataset_contract = {
        "split": "val",
        "path": "/official/val.jsonl",
        "sha256": evaluator.OFFICIAL_SCENE_V4_VAL_SHA256,
    }

    contract = evaluator.validate_scene_v6_matched_donor_contract(
        contract="scene_v6_identity_hard32",
        row_indices=list(evaluator.HARD32_ROW_INDICES),
        expected_hashes=dict(evaluator.HARD32_ROW_HASHES),
        selection_dataset_contract=dataset_contract,
        conditions=list(evaluator.CONDITIONS),
        donor_rule=evaluator.DONOR_RULE_LENGTH_MATCHED,
        max_new_tokens=evaluator.DEFAULT_MAX_NEW_TOKENS,
        normal_fusion_profile="native",
        expected_memory_layer_count=42,
        memory_target_layers=list(range(42)),
        memory_delta_heads=["q", "o"],
        memory_rank=4,
        rwkv_ms_semantics_version=2,
        memory_backend="rwkv_ms",
        selection_manifest_sha256=evaluator.HARD32_SELECTION_SHA256,
    )

    assert contract["name"] == "scene_v6_identity_hard32"
    assert contract["rows"] == 32
    assert contract["conditions"] == list(evaluator.CONDITIONS)
    assert contract["gate_requirements"] == evaluator.HARD32_GATE_REQUIREMENTS

    with pytest.raises(ValueError, match="authoritative fixed row hash"):
        evaluator.validate_scene_v6_matched_donor_contract(
            contract="scene_v6_identity_hard32",
            row_indices=list(evaluator.HARD32_ROW_INDICES),
            expected_hashes={**evaluator.HARD32_ROW_HASHES, 3: "0" * 64},
            selection_dataset_contract=dataset_contract,
            conditions=list(evaluator.CONDITIONS),
            donor_rule=evaluator.DONOR_RULE_LENGTH_MATCHED,
            max_new_tokens=evaluator.DEFAULT_MAX_NEW_TOKENS,
            normal_fusion_profile="native",
            expected_memory_layer_count=42,
            memory_target_layers=list(range(42)),
            memory_delta_heads=["q", "o"],
            memory_rank=4,
            rwkv_ms_semantics_version=2,
            memory_backend="rwkv_ms",
            selection_manifest_sha256=evaluator.HARD32_SELECTION_SHA256,
        )


def historical_v6_contract_values() -> dict:
    return {
        "row_indices": list(evaluator.HARD32_ROW_INDICES),
        "expected_hashes": dict(evaluator.HARD32_ROW_HASHES),
        "selection_dataset_contract": {
            "split": "val",
            "path": str(evaluator.HISTORICAL_V6_OFFICIAL_VAL),
            "sha256": evaluator.OFFICIAL_SCENE_V4_VAL_SHA256,
        },
        "conditions": list(evaluator.HISTORICAL_V6_HARD32_CONDITIONS),
        "donor_rule": evaluator.DONOR_RULE_CYCLIC,
        "max_new_tokens": evaluator.DEFAULT_MAX_NEW_TOKENS,
        "normal_fusion_profile": "native",
        "expected_memory_layer_count": 24,
        "memory_target_layers": list(range(24)),
        "memory_delta_heads": ["q", "o"],
        "memory_rank": 4,
        "rwkv_ms_semantics_version": 2,
        "memory_backend": "rwkv_ms",
        "selection_manifest_sha256": evaluator.HARD32_SELECTION_SHA256,
    }


def test_historical_v6_hard32_contract_is_three_condition_and_non_authorizing() -> None:
    contract = evaluator.validate_historical_v6_hard32_contract(
        **historical_v6_contract_values()
    )

    assert contract["conditions"] == [
        "base_full",
        "no_write_full",
        "normal_full",
    ]
    assert contract["rows"] == 32
    assert contract["expected_memory_layer_count"] == 24
    assert contract["checkpoint_artifact_sha256"] == (
        evaluator.HISTORICAL_V6_CHECKPOINT_ARTIFACT_SHA256
    )
    assert contract["full170_authorized"] is False
    assert contract["test_authorized"] is False
    assert contract["checkpoint_selection_authorized"] is False
    assert "predates commit-bound source locks" in contract["lineage_limitation"]


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ({"row_indices": list(evaluator.HARD32_ROW_INDICES[:-1])}, "source indices"),
        ({"expected_hashes": {}}, "row hash"),
        ({"selection_manifest_sha256": "0" * 64}, "selection file"),
        ({"conditions": ["base_full", "normal_full"]}, "exact order"),
        ({"expected_memory_layer_count": 42}, "expected_memory_layer_count=24"),
        ({"memory_target_layers": list(range(23))}, "target_layers=0..23"),
        ({"memory_delta_heads": ["o"]}, "delta_heads=q,o"),
        ({"normal_fusion_profile": "native_gate_open"}, "profile=native"),
        (
            {
                "selection_dataset_contract": {
                    "split": "val",
                    "path": str(evaluator.HISTORICAL_V6_OFFICIAL_VAL),
                    "sha256": "0" * 64,
                }
            },
            "official scene-v4 val",
        ),
    ],
)
def test_historical_v6_hard32_contract_fails_closed(
    mutation: dict,
    message: str,
) -> None:
    values = historical_v6_contract_values()
    values.update(mutation)

    with pytest.raises(ValueError, match=message):
        evaluator.validate_historical_v6_hard32_contract(**values)


def test_historical_v6_artifact_binding_rejects_hash_drift_and_symlink(
    tmp_path: Path,
) -> None:
    artifact = tmp_path / "adapter.pt"
    artifact.write_bytes(b"adapter")
    digest = evaluator.sha256_file(artifact)
    assert evaluator._historical_artifact_binding(
        artifact,
        description="adapter",
        expected_sha256=digest,
    )["sha256"] == digest

    with pytest.raises(ValueError, match="SHA-256 differs"):
        evaluator._historical_artifact_binding(
            artifact,
            description="adapter",
            expected_sha256="0" * 64,
        )
    symlink = tmp_path / "adapter-link.pt"
    symlink.symlink_to(artifact)
    with pytest.raises(ValueError, match="forbids symlinks"):
        evaluator._historical_artifact_binding(
            symlink,
            description="adapter",
            expected_sha256=digest,
        )


def test_historical_v6_contract_requires_clean_tracked_worktree(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        evaluator.subprocess,
        "check_output",
        lambda *args, **kwargs: "",
    )
    assert evaluator.require_historical_tracked_worktree_clean()[
        "tracked_worktree_clean"
    ] is True

    monkeypatch.setattr(
        evaluator.subprocess,
        "check_output",
        lambda *args, **kwargs: " M deltamem/core/delta.py\n",
    )
    with pytest.raises(ValueError, match="clean tracked worktree"):
        evaluator.require_historical_tracked_worktree_clean()


def test_historical_v6_preflight_rejects_overwrite_and_output_collision(
    tmp_path: Path,
) -> None:
    args = SimpleNamespace(
        evaluation_contract=evaluator.HISTORICAL_V6_HARD32_CONTRACT,
        overwrite=True,
        row_indices=None,
        row_indices_file=tmp_path / "selection.json",
        output_dir=tmp_path / "fresh-eval",
        base_model=str(tmp_path / "model"),
        memory_dir=tmp_path / "checkpoint-672",
        dataset_file=tmp_path / "val.jsonl",
    )
    with pytest.raises(ValueError, match="forbids --overwrite"):
        evaluator.validate_historical_v6_run_preflight(
            args,
            conditions=list(evaluator.HISTORICAL_V6_HARD32_CONDITIONS),
        )

    args.overwrite = False
    args.output_dir.mkdir()
    with pytest.raises(ValueError, match="fresh, nonexistent output"):
        evaluator.validate_historical_v6_run_preflight(
            args,
            conditions=list(evaluator.HISTORICAL_V6_HARD32_CONDITIONS),
        )


def test_full170_contract_requires_passed_hard32_authorization() -> None:
    row_indices = list(range(170))
    with pytest.raises(ValueError, match="passed hard32 receipt"):
        evaluator.validate_scene_v6_matched_donor_contract(
            contract="scene_v6_matched_donor_validation",
            row_indices=row_indices,
            expected_hashes={index: f"{index:064x}" for index in row_indices},
            selection_dataset_contract={
                "split": "val",
                "path": "/official/val.jsonl",
                "sha256": evaluator.OFFICIAL_SCENE_V4_VAL_SHA256,
            },
            conditions=["state_only", "state_only_donor"],
            donor_rule=evaluator.DONOR_RULE_LENGTH_MATCHED,
            max_new_tokens=evaluator.DEFAULT_MAX_NEW_TOKENS,
            normal_fusion_profile="native",
            expected_memory_layer_count=42,
            memory_target_layers=list(range(42)),
            memory_delta_heads=["q", "o"],
            memory_rank=4,
            rwkv_ms_semantics_version=2,
            memory_backend="rwkv_ms",
        )


def test_scene_v6_matched_donor_gate_requires_positive_identity_delta() -> None:
    contract = {
        "name": "scene_v6_matched_donor_validation",
        "rows": 170,
    }
    passed = evaluator.build_scene_v6_matched_donor_gate(
        {"state_only_minus_state_only_donor": {"delta": 0.01}},
        contract,
    )
    failed = evaluator.build_scene_v6_matched_donor_gate(
        {"state_only_minus_state_only_donor": {"delta": 0.0}},
        contract,
    )

    assert passed["status"] == "pass"
    assert passed["gate"]["operator"] == ">"
    assert failed["status"] == "fail"
    assert failed["test_selection_forbidden"] is True


def test_base_model_weight_identity_hashes_index_and_every_referenced_shard(
    tmp_path: Path,
) -> None:
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    index = {
        "weight_map": {
            "layer.0": "model-00001-of-00002.safetensors",
            "layer.1": "model-00002-of-00002.safetensors",
        }
    }
    (model_dir / "model.safetensors.index.json").write_text(
        json.dumps(index), encoding="utf-8"
    )
    (model_dir / "model-00001-of-00002.safetensors").write_bytes(b"first")
    (model_dir / "model-00002-of-00002.safetensors").write_bytes(b"second")
    (model_dir / "unreferenced.safetensors").write_bytes(b"ignore")

    identity = evaluator.base_model_weight_identity(model_dir)

    assert identity["layout"] == "sharded"
    assert identity["index"] == "model.safetensors.index.json"
    assert {row["relative_path"] for row in identity["files"]} == {
        "model.safetensors.index.json",
        "model-00001-of-00002.safetensors",
        "model-00002-of-00002.safetensors",
    }
    assert all(len(row["sha256"]) == 64 for row in identity["files"])
    assert len(identity["combined_sha256"]) == 64


def test_base_model_weight_identity_rejects_unsafe_shard_path(tmp_path: Path) -> None:
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    (model_dir / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {"layer.0": "../outside.safetensors"}}),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="Unsafe model artifact path"):
        evaluator.base_model_weight_identity(model_dir)


def test_base_model_prompt_identity_hashes_tokenizer_and_template(tmp_path: Path) -> None:
    for name, content in (
        ("config.json", "{}"),
        ("tokenizer.json", "{}"),
        ("chat_template.jinja", "template"),
    ):
        (tmp_path / name).write_text(content, encoding="utf-8")

    identity = evaluator.base_model_prompt_identity(tmp_path)

    assert {row["relative_path"] for row in identity["files"]} == {
        "config.json",
        "tokenizer.json",
        "chat_template.jinja",
    }


def test_state_only_primes_full_prompt_then_queries_system_only(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[tuple] = []
    sample = {
        "messages": [
            {"role": "system", "content": "system"},
            {"role": "user", "content": "private scene"},
        ],
        "gold": {"boundaries": [2]},
    }

    monkeypatch.setattr(evaluator, "reset_delta_state", lambda model: events.append(("reset",)))

    def fake_prime(**kwargs):
        events.append(("prime", [message["role"] for message in kwargs["messages"]]))
        return {"tokens": 10, "online_state": {"nonzero_state_modules": 1}}

    def fake_generate(**kwargs):
        events.append(("generate", [message["role"] for message in kwargs["messages"]]))
        return {"parsed_json": {"boundaries": [2]}}

    @contextlib.contextmanager
    def fake_condition(model, condition):
        events.append(("enter", condition))
        yield
        events.append(("exit", condition))

    monkeypatch.setattr(evaluator, "prime_online_state", fake_prime)
    monkeypatch.setattr(evaluator, "generate_messages", fake_generate)
    monkeypatch.setattr(evaluator, "memory_condition", fake_condition)

    result = evaluator.evaluate_condition(
        model=object(),
        tokenizer=object(),
        sample=sample,
        condition="state_only",
        max_new_tokens=8,
        device="cpu",
    )

    assert events == [
        ("reset",),
        ("prime", ["system", "user"]),
        ("enter", "no_write"),
        ("generate", ["system"]),
        ("exit", "no_write"),
        ("reset",),
    ]
    assert result["score_recovered"]["sample_f1"] == 1.0


def test_state_only_shuffled_primes_explicit_shuffle_and_records_identity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[tuple[str, str]] = []
    current = {
        "source_index": 1,
        "row_sha256": "a" * 64,
        "messages": [
            {"role": "system", "content": "current system"},
            {"role": "user", "content": "current scene"},
        ],
        "gold": {"boundaries": [2]},
    }
    donor = {
        "source_index": 2,
        "row_sha256": "b" * 64,
        "messages": [
            {"role": "system", "content": "donor system"},
            {"role": "user", "content": "donor scene"},
        ],
        "gold": {"boundaries": [1]},
    }
    shuffled = {
        "source_index": 3,
        "row_sha256": "c" * 64,
        "messages": [
            {"role": "system", "content": "shuffle system"},
            {"role": "user", "content": "shuffle scene"},
        ],
        "gold": {"boundaries": [4]},
    }
    monkeypatch.setattr(evaluator, "reset_delta_state", lambda model: None)
    monkeypatch.setattr(
        evaluator,
        "prime_online_state",
        lambda **kwargs: events.append(("prime", kwargs["messages"][1]["content"]))
        or {"tokens": 10},
    )
    monkeypatch.setattr(
        evaluator,
        "generate_messages",
        lambda **kwargs: events.append(("generate", kwargs["messages"][0]["content"]))
        or {"parsed_json": {"boundaries": [2]}},
    )

    @contextlib.contextmanager
    def no_write(model, condition):
        yield

    monkeypatch.setattr(evaluator, "memory_condition", no_write)

    result = evaluator.evaluate_condition(
        model=object(),
        tokenizer=object(),
        sample=current,
        donor_sample=donor,
        shuffled_sample=shuffled,
        condition="state_only_shuffled",
        max_new_tokens=8,
        device="cpu",
    )

    assert events == [
        ("prime", "shuffle scene"),
        ("generate", "current system"),
    ]
    assert result["shuffled_source_index"] == 3
    assert result["shuffled_row_sha256"] == "c" * 64
    assert result["score_strict"]["sample_f1"] == 1.0


def test_state_only_donor_primes_donor_then_queries_and_scores_current_row(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[tuple] = []
    current = {
        "source_index": 142,
        "row_sha256": "a" * 64,
        "messages": [
            {"role": "system", "content": "current system"},
            {"role": "user", "content": "current scene"},
        ],
        "gold": {"boundaries": [2]},
    }
    donor = {
        "source_index": 25,
        "row_sha256": "b" * 64,
        "messages": [
            {"role": "system", "content": "donor system"},
            {"role": "user", "content": "donor scene"},
        ],
        "gold": {"boundaries": [1]},
    }
    monkeypatch.setattr(
        evaluator, "reset_delta_state", lambda model: events.append(("reset",))
    )

    def fake_prime(**kwargs):
        events.append(
            (
                "prime",
                [
                    (message["role"], message["content"])
                    for message in kwargs["messages"]
                ],
            )
        )
        return {"tokens": 10}

    def fake_generate(**kwargs):
        events.append(
            (
                "generate",
                [
                    (message["role"], message["content"])
                    for message in kwargs["messages"]
                ],
            )
        )
        return {"parsed_json": {"boundaries": [2]}}

    @contextlib.contextmanager
    def fake_condition(model, condition):
        events.append(("enter", condition))
        yield
        events.append(("exit", condition))

    monkeypatch.setattr(evaluator, "prime_online_state", fake_prime)
    monkeypatch.setattr(evaluator, "generate_messages", fake_generate)
    monkeypatch.setattr(evaluator, "memory_condition", fake_condition)

    result = evaluator.evaluate_condition(
        model=object(),
        tokenizer=object(),
        sample=current,
        donor_sample=donor,
        condition="state_only_donor",
        max_new_tokens=8,
        device="cpu",
    )

    assert events == [
        ("reset",),
        (
            "prime",
            [("system", "donor system"), ("user", "donor scene")],
        ),
        ("enter", "no_write"),
        ("generate", [("system", "current system")]),
        ("exit", "no_write"),
        ("reset",),
    ]
    assert result["donor_source_index"] == 25
    assert result["donor_row_sha256"] == "b" * 64
    assert result["score_recovered"]["gold_boundaries"] == [2]
    assert result["score_recovered"]["sample_f1"] == 1.0


def test_no_write_disables_writes_during_prime_and_query(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[str] = []
    sample = {
        "messages": [
            {"role": "system", "content": "system"},
            {"role": "user", "content": "private scene"},
        ],
        "gold": {"boundaries": []},
    }
    monkeypatch.setattr(evaluator, "reset_delta_state", lambda model: events.append("reset"))
    monkeypatch.setattr(
        evaluator,
        "prime_online_state",
        lambda **kwargs: events.append("prime") or {"tokens": 10},
    )
    monkeypatch.setattr(
        evaluator,
        "generate_messages",
        lambda **kwargs: events.append("generate") or {"parsed_json": {"boundaries": []}},
    )

    @contextlib.contextmanager
    def fake_condition(model, condition):
        events.append("writes_off")
        yield
        events.append("writes_restored")

    monkeypatch.setattr(evaluator, "memory_condition", fake_condition)

    evaluator.evaluate_condition(
        model=object(),
        tokenizer=object(),
        sample=sample,
        condition="state_only_no_write",
        max_new_tokens=8,
        device="cpu",
    )

    assert events == [
        "reset",
        "writes_off",
        "prime",
        "generate",
        "writes_restored",
        "reset",
    ]


def test_no_write_full_disables_writes_for_full_prompt_generation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[tuple] = []
    sample = {
        "messages": [
            {"role": "system", "content": "system"},
            {"role": "user", "content": "scene"},
        ],
        "gold": {"boundaries": []},
    }
    monkeypatch.setattr(
        evaluator, "reset_delta_state", lambda model: events.append(("reset",))
    )

    def fake_generate(**kwargs):
        events.append(("generate", [message["role"] for message in kwargs["messages"]]))
        return {"parsed_json": {"boundaries": []}}

    @contextlib.contextmanager
    def fake_condition(model, condition):
        events.append(("enter", condition))
        yield
        events.append(("exit", condition))

    monkeypatch.setattr(evaluator, "generate_messages", fake_generate)
    monkeypatch.setattr(evaluator, "memory_condition", fake_condition)

    evaluator.evaluate_condition(
        model=object(),
        tokenizer=object(),
        sample=sample,
        condition="no_write_full",
        max_new_tokens=8,
        device="cpu",
    )

    assert events == [
        ("reset",),
        ("enter", "no_write"),
        ("generate", ["system", "user"]),
        ("exit", "no_write"),
        ("reset",),
    ]


def test_online_state_stats_cover_complete_rwkv_carrier(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = SimpleNamespace(
        layer_idx=7,
        delta_state=torch.tensor([[[[[0.0, 2.0]]]]]),
        rwkv_ms_positions=torch.tensor([19]),
        rwkv_ms_previous_source=torch.tensor([[3.0, 4.0]]),
    )
    monkeypatch.setattr(
        evaluator,
        "iter_delta_modules",
        lambda model: iter([("model.layers.7.self_attn", module)]),
    )

    stats = evaluator.online_state_stats(object())

    assert stats["nonzero_matrix_modules"] == 1
    assert stats["nonzero_previous_source_modules"] == 1
    assert stats["max_position"] == 19
    assert stats["max_previous_source_norm"] == pytest.approx(5.0)
    assert stats["by_layer"][0]["rwkv_ms_positions"] == [19]


def make_record(
    tp: int,
    fp: int,
    fn: int,
    *,
    strict_counts: tuple[int, int, int] | None = None,
) -> dict:
    strict_tp, strict_fp, strict_fn = strict_counts or (tp, fp, fn)
    recovered_predicted = list(range(1, tp + fp + 1))
    strict_predicted = list(range(1, strict_tp + strict_fp + 1))
    return {
        "score_recovered": {
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "schema_recovered": True,
            "gold_boundaries": [1] if tp + fn else [],
            "predicted_boundaries": recovered_predicted,
        },
        "score_strict": {
            "tp": strict_tp,
            "fp": strict_fp,
            "fn": strict_fn,
            "schema_valid": True,
            "gold_boundaries": [1] if strict_tp + strict_fn else [],
            "predicted_boundaries": strict_predicted,
        },
        "parsed_json": {"boundaries": strict_predicted},
        "hit_max_new_tokens": False,
        "input_tokens": 10,
        "output_tokens": 2,
        "elapsed_seconds": 0.5,
    }


def test_summary_uses_micro_f1_and_reports_causal_control_delta() -> None:
    state = evaluator.summarize_records([make_record(2, 1, 0), make_record(0, 0, 1)])
    no_write = evaluator.summarize_records([make_record(0, 1, 2), make_record(0, 0, 1)])
    donor = evaluator.summarize_records([make_record(1, 2, 0), make_record(0, 0, 1)])
    summaries = {
        "state_only": state,
        "state_only_no_write": no_write,
        "state_only_donor": donor,
    }

    assert state["format_recovered"]["primary_metric"] == pytest.approx(4 / 6)
    assert state["strict"]["precision"] == pytest.approx(2 / 3)
    assert state["strict"]["recall"] == pytest.approx(2 / 3)
    assert state["strict"]["predicted_boundary_count"] == 3
    assert state["strict"]["gold_boundary_count"] == 3
    assert state["strict"]["predicted_to_gold_boundary_ratio"] == pytest.approx(1.0)
    comparison = evaluator.build_comparisons(summaries)[
        "state_only_minus_state_only_no_write"
    ]
    assert comparison["metric_name"] == evaluator.BENCHMARK_SCENE_METRIC_NAME
    assert comparison["delta"] == pytest.approx(4 / 6)
    donor_comparison = evaluator.build_comparisons(summaries)[
        "state_only_minus_state_only_donor"
    ]
    assert donor_comparison["delta"] == pytest.approx((4 / 6) - (2 / 5))


def test_progress_log_labels_strict_and_recovered_scores() -> None:
    line = evaluator.progress_log_line(
        condition="state_only",
        source_index=3,
        record={
            "score_strict": {"sample_f1": 0.25},
            "score_recovered": {"sample_f1": 0.75},
        },
    )

    assert line == (
        "SCENE_STATE_EVAL condition=state_only source_index=3 "
        "strict_f1=0.2500 recovered_f1=0.7500"
    )


def test_strict_scene_score_does_not_coerce_numeric_boundary_strings() -> None:
    score = evaluator.score_prediction(
        "scene",
        {"boundaries": [1, "2"]},
        {"boundaries": [1, 2]},
    )

    assert score["schema_valid"] is True
    assert (score["tp"], score["fp"], score["fn"]) == (1, 1, 1)
    assert score["sample_f1"] == pytest.approx(0.5)


def test_comparisons_ignore_recovered_alias_scores() -> None:
    state = evaluator.summarize_records(
        [make_record(3, 0, 0, strict_counts=(0, 0, 3))]
    )
    donor = evaluator.summarize_records(
        [make_record(0, 0, 3, strict_counts=(1, 0, 2))]
    )

    comparison = evaluator.build_comparisons(
        {"state_only": state, "state_only_donor": donor}
    )["state_only_minus_state_only_donor"]

    assert state["format_recovered"]["primary_metric"] == 1.0
    assert donor["format_recovered"]["primary_metric"] == 0.0
    assert comparison["delta"] == pytest.approx(-0.5)


def test_empty_list_evidence_requires_canonical_strict_schema() -> None:
    record = make_record(0, 0, 0)
    record["parsed_json"] = {"scene_boundaries": []}
    record["score_strict"]["schema_valid"] = False

    summary = evaluator.summarize_records([record])

    assert summary["decision_quality"]["recovered_empty_list_exact"] == 1
    assert summary["decision_quality"]["canonical_empty_list_exact"] == 0


def test_scene_semantic_mask_keeps_empty_branch_and_drops_standalone_space() -> None:
    multi = '{"boundaries": [4, 7]}'
    multi_offsets = [
        (0, 14),
        (14, 16),
        (16, 17),
        (17, 18),
        (18, 19),
        (19, 20),
        (20, 22),
    ]
    assert evaluator.scene_boundary_decision_token_mask_from_offsets(
        content=multi,
        content_start=0,
        offsets=multi_offsets,
    ) == [False, True, True, True, False, True, True]

    empty = '{"boundaries": []}'
    assert evaluator.scene_boundary_decision_token_mask_from_offsets(
        content=empty,
        content_start=0,
        offsets=[(0, 14), (14, 17), (17, 18)],
    ) == [False, True, False]


def test_teacher_forced_semantic_nll_scores_only_decision_targets(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    prefix = "SYSTEM|MODEL|"
    suffix = "|END"

    def fake_template(tokenizer, messages, *, tokenize, add_generation_prompt):
        assert tokenize is False
        if len(messages) == 1:
            assert add_generation_prompt is True
            return prefix
        assert add_generation_prompt is False
        return prefix + messages[-1]["content"] + suffix

    class FakeTokenizer:
        def __call__(
            self,
            rendered,
            *,
            add_special_tokens,
            return_offsets_mapping,
            return_tensors,
        ):
            assert add_special_tokens is False
            assert return_offsets_mapping is True
            assert return_tensors == "pt"
            token_ids = [ord(character) for character in rendered]
            return SimpleNamespace(
                input_ids=torch.tensor([token_ids]),
                attention_mask=torch.ones((1, len(token_ids)), dtype=torch.long),
                offset_mapping=torch.tensor(
                    [[[index, index + 1] for index in range(len(token_ids))]]
                ),
            )

    class FakeModel:
        def __call__(self, *, input_ids, attention_mask, use_cache):
            assert use_cache is False
            vocab_size = 256
            logits = torch.zeros((1, input_ids.size(1), vocab_size))
            for predictor in range(input_ids.size(1) - 1):
                logits[0, predictor, input_ids[0, predictor + 1]] = 8.0
            return SimpleNamespace(logits=logits)

    monkeypatch.setattr(evaluator, "apply_chat_template", fake_template)
    donor_sample = {
        "source_index": 4,
        "row_sha256": "d" * 64,
        "messages": [
            {"role": "system", "content": "system"},
            {"role": "user", "content": "other scene"},
        ],
        "gold_content": '{"boundaries": [3]}',
    }
    result = evaluator.semantic_decision_nll_from_current_state(
        model=FakeModel(),
        tokenizer=FakeTokenizer(),
        sample={
            "messages": [
                {"role": "system", "content": "system"},
                {"role": "user", "content": "scene"},
            ],
            "gold_content": '{"boundaries": [2]}',
        },
        donor_sample=donor_sample,
        device="cpu",
    )

    assert result["all_semantic"]["token_count"] == 3
    assert result["all_semantic"]["selected_target_token_ids"] == [
        ord("["),
        ord("2"),
        ord("]"),
    ]
    assert result["pair_target"]["selected_target_token_ids"] == [ord("2")]
    assert result["pair_target"]["donor_target_token_ids"] == [ord("3")]
    assert result["pair_target"]["alternative_target_token_ids"] == [ord("3")]
    assert result["pair_target"]["first_differing_semantic_ordinal"] == 1
    assert result["all_semantic"]["mean_nll"] < 0.1
    assert result["pair_target"]["mean_nll"] < 0.1
    assert result["pair_target"]["alternative_target_mean_nll"] > 7.0
    assert (
        result["pair_target"]["selected_over_alternative_logprob_margin"] > 7.0
    )


def test_pair_target_rejects_an_earlier_nonsemantic_prefix_difference() -> None:
    with pytest.raises(ValueError, match="full causal prefix"):
        evaluator.first_pair_distinguishing_scene_target(
            source_features={
                "input_ids": torch.tensor([[1, 2, 3, 4]]),
                "selected_positions": [3],
            },
            donor_features={
                "input_ids": torch.tensor([[1, 9, 3, 5]]),
                "selected_positions": [3],
            },
            donor_sample={"source_index": 7, "row_sha256": "d" * 64},
        )


def semantic_record(
    source_index: int,
    all_semantic_mean_nll: float,
    *,
    donor_source_index: int | None = None,
    pair_donor_source_index: int | None = None,
    pair_target_nll: float | None = None,
) -> dict:
    pair_donor_source_index = (
        source_index ^ 1
        if pair_donor_source_index is None
        else pair_donor_source_index
    )
    pair_target_nll = (
        all_semantic_mean_nll if pair_target_nll is None else pair_target_nll
    )
    return {
        "source_index": source_index,
        "donor_source_index": donor_source_index,
        "row_sha256": f"{source_index:064x}",
        "gold": {"boundaries": [] if source_index < 9 else [1]},
        "semantic_decision_nll": {
            "all_semantic": {
                "mask_mode": evaluator.SEMANTIC_DECISION_MASK_MODE,
                "normalization": evaluator.SEMANTIC_DECISION_NLL_NORMALIZATION,
                "selected_target_positions": [10, 11],
                "selected_target_token_ids": [100, 101],
                "token_count": 2,
                "nll_sum": all_semantic_mean_nll * 2,
                "mean_nll": all_semantic_mean_nll,
                "read_rendered_sha256": "a" * 64,
            },
            "pair_target": {
                "mask_mode": evaluator.PAIR_TARGET_DECISION_MASK_MODE,
                "normalization": (
                    evaluator.PAIR_TARGET_DECISION_NLL_NORMALIZATION
                ),
                "target_mode": evaluator.PAIR_TARGET_DECISION_MASK_MODE,
                "selected_target_positions": [10],
                "selected_target_token_ids": [100],
                "donor_target_token_ids": [102],
                "first_differing_semantic_ordinal": 0,
                "causal_prefix_sha256": "c" * 64,
                "donor_source_index": pair_donor_source_index,
                "donor_row_sha256": f"{pair_donor_source_index:064x}",
                "token_count": 1,
                "nll_sum": pair_target_nll,
                "mean_nll": pair_target_nll,
                "read_rendered_sha256": "a" * 64,
            },
        },
    }


def test_semantic_decision_evidence_uses_comparator_minus_correct_sign() -> None:
    records = {
        "state_only": [
            semantic_record(index, 1.0, pair_target_nll=1.0)
            for index in range(32)
        ],
        "state_only_donor": [
            semantic_record(
                index,
                1.25,
                donor_source_index=index ^ 1,
                pair_target_nll=1.5,
            )
            for index in range(32)
        ],
        "state_only_no_write": [
            semantic_record(index, 2.0, pair_target_nll=1.1)
            for index in range(32)
        ],
    }

    evidence = evaluator.build_semantic_decision_evidence(records)

    assert evidence["donor_pair_target_minus_correct"]["positive_rows"] == 32
    assert evidence["donor_pair_target_minus_correct"]["mean_gap"] == pytest.approx(
        0.5
    )
    assert evidence["donor_all_semantic_minus_correct_diagnostic"][
        "mean_gap"
    ] == pytest.approx(0.25)
    assert evidence["zero_all_semantic_minus_correct"]["positive_rows"] == 32
    assert evidence["zero_all_semantic_minus_correct"]["mean_gap"] == pytest.approx(
        1.0
    )
    assert evidence["rows"][0]["donor_minus_correct_pair_target_nll_gap"] == 0.5
    assert (
        evidence["rows"][0]["zero_minus_correct_all_semantic_nll_gap"] == 1.0
    )
    same_cardinality = evidence["donor_label_cardinality"]["strata"][
        "nonempty_same_cardinality"
    ]
    assert same_cardinality["rows"] == 22
    assert same_cardinality["source_donor_pairs"][0] == {
        "source_index": 10,
        "donor_source_index": 11,
    }


def hard32_summary(
    metric: float,
    *,
    tp: int = 8,
    empty_exact: int = 6,
    recovered: int = 32,
    canonical: int = 32,
    density: float = 1.0,
) -> dict:
    gold_count = 32
    predicted_count = int(gold_count * density)
    return {
        "format_recovered": {"primary_metric": metric},
        "strict": {
            "metric_name": evaluator.BENCHMARK_SCENE_METRIC_NAME,
            "primary_metric": metric,
            "precision": 0.5,
            "recall": 0.5,
            "tp": tp,
            "fp": max(predicted_count - tp, 0),
            "fn": max(gold_count - tp, 0),
            "predicted_boundary_count": predicted_count,
            "gold_boundary_count": gold_count,
            "predicted_boundaries_per_sample": predicted_count / 32,
            "gold_boundaries_per_sample": 1.0,
            "predicted_to_gold_boundary_ratio": density,
            "schema_valid_rate": 1.0,
        },
        "decision_quality": {
            "recovered_gold_positives": tp,
            "canonical_empty_list_exact": empty_exact,
            "empty_list_rows": 9,
            "recovered_outputs": recovered,
            "canonical_outputs": canonical,
        },
    }


def hard32_semantic_evidence(
    *,
    donor_positive: int = 20,
    zero_positive: int = 21,
    same_cardinality_positive: int = 8,
    same_cardinality_rows: int = 10,
) -> dict:
    return {
        "donor_pair_target_minus_correct": {"positive_rows": donor_positive},
        "zero_all_semantic_minus_correct": {"positive_rows": zero_positive},
        "donor_label_cardinality": {
            "strata": {
                "nonempty_same_cardinality": {
                    "rows": same_cardinality_rows,
                    "positive_rows": same_cardinality_positive,
                }
            }
        },
    }


def historical_v6_record(
    source_index: int,
    *,
    condition: str,
    gold_boundaries: list[int],
    predicted_boundaries: list[int],
) -> dict:
    gold = {"boundaries": gold_boundaries}
    parsed = {"boundaries": predicted_boundaries}
    return {
        "source_index": source_index,
        "row_sha256": f"{source_index:064x}",
        "gold": gold,
        "condition": condition,
        "raw_generation": json.dumps(parsed, sort_keys=True),
        "parsed_json": parsed,
        "score_strict": evaluator.score_prediction("scene", parsed, gold),
        "score_recovered": evaluator.recovered_scene_score(parsed, gold),
        "hit_max_new_tokens": False,
        "input_tokens": 1,
        "output_tokens": 1,
        "elapsed_seconds": 0.01,
    }


def test_historical_v6_evidence_reports_stratum_uplift_and_differences() -> None:
    gold_by_index: dict[int, list[int]] = {}
    for pair_ordinal, (left, right) in enumerate(evaluator.HARD32_FROZEN_DONOR_PAIRS):
        if pair_ordinal < 9:
            gold_by_index[left] = []
            gold_by_index[right] = [2]
        elif pair_ordinal < 14:
            gold_by_index[left] = [2]
            gold_by_index[right] = [3]
        else:
            gold_by_index[left] = [2]
            gold_by_index[right] = [2, 4]

    records: dict[str, list[dict]] = {}
    for condition in evaluator.HISTORICAL_V6_HARD32_CONDITIONS:
        records[condition] = [
            historical_v6_record(
                index,
                condition=condition,
                gold_boundaries=gold_by_index[index],
                predicted_boundaries=(
                    gold_by_index[index] if condition == "normal_full" else []
                ),
            )
            for index in evaluator.HARD32_ROW_INDICES
        ]

    evidence = evaluator.build_historical_v6_hard32_evidence(records)

    assert evidence["observed_stratum_rows"] == {
        "presence": 18,
        "same_cardinality_value": 10,
        "cross_cardinality_value": 4,
    }
    assert evidence["overall"]["conditions"]["normal_full"]["strict_exact_rows"] == 32
    assert evidence["overall"]["strict_uplift"][
        "normal_full_minus_strongest_control"
    ] == pytest.approx(1.0)
    assert evidence["overall"]["generation_differences"][
        "normal_full_vs_no_write_full"
    ]["raw_generation_different_rows"] == 23
    assert evidence["strata"]["same_cardinality_value"]["rows"] == 10
    assert evidence["full170_authorized"] is False


def test_hard32_gate_requires_identity_causality_task_and_format() -> None:
    summaries = {
        "base_full": hard32_summary(0.10),
        "normal_full": hard32_summary(0.25),
        "no_write_full": hard32_summary(0.15),
        "state_only": hard32_summary(0.30),
        "state_only_donor": hard32_summary(0.20),
        "state_only_no_write": hard32_summary(0.10),
    }
    comparisons = evaluator.build_comparisons(summaries)
    semantic_evidence = hard32_semantic_evidence()
    contract = {"name": "scene_v6_identity_hard32", "rows": 32}

    passed = evaluator.build_scene_v6_identity_hard32_gate(
        summaries=summaries,
        comparisons=comparisons,
        semantic_evidence=semantic_evidence,
        contract=contract,
    )
    failed = evaluator.build_scene_v6_identity_hard32_gate(
        summaries={**summaries, "state_only": hard32_summary(0.30, empty_exact=5)},
        comparisons=comparisons,
        semantic_evidence=semantic_evidence,
        contract=contract,
    )
    density_failed = evaluator.build_scene_v6_identity_hard32_gate(
        summaries={**summaries, "state_only": hard32_summary(0.30, density=2.1)},
        comparisons=comparisons,
        semantic_evidence=semantic_evidence,
        contract=contract,
    )
    stratum_failed = evaluator.build_scene_v6_identity_hard32_gate(
        summaries=summaries,
        comparisons=comparisons,
        semantic_evidence=hard32_semantic_evidence(same_cardinality_positive=7),
        contract=contract,
    )
    weak_normal = {**summaries, "normal_full": hard32_summary(0.19)}
    normal_failed = evaluator.build_scene_v6_identity_hard32_gate(
        summaries=weak_normal,
        comparisons=evaluator.build_comparisons(weak_normal),
        semantic_evidence=semantic_evidence,
        contract=contract,
    )

    assert passed["status"] == "pass"
    assert passed["full170_authorized_for_bound_checkpoint"] is True
    assert passed["benchmark_metric_evidence"]["state_only"]["precision"] == 0.5
    assert passed["gates"][
        "correct_better_than_same_cardinality_nonempty_donor_rows"
    ]["passed"] is True
    assert failed["status"] == "fail"
    assert failed["gates"]["state_only_empty_list_exact"]["passed"] is False
    assert density_failed["gates"][
        "state_only_predicted_boundary_density"
    ]["passed"] is False
    assert stratum_failed["gates"][
        "correct_better_than_same_cardinality_nonempty_donor_rows"
    ]["passed"] is False
    assert normal_failed["gates"][
        "normal_full_minus_strongest_control_f1"
    ]["passed"] is False


def test_historical_step128_is_a_negative_identity_gate_fixture() -> None:
    summaries = {
        "base_full": hard32_summary(0.13793103448275862),
        "normal_full": hard32_summary(0.18681318681318682),
        "no_write_full": hard32_summary(0.13793103448275862),
        "state_only": hard32_summary(0.0851063829787234, tp=2),
        "state_only_donor": hard32_summary(0.0851063829787234, tp=2),
        "state_only_no_write": hard32_summary(0.0, tp=0),
    }
    gate = evaluator.build_scene_v6_identity_hard32_gate(
        summaries=summaries,
        comparisons=evaluator.build_comparisons(summaries),
        semantic_evidence=hard32_semantic_evidence(zero_positive=20),
        contract={"name": "scene_v6_identity_hard32", "rows": 32},
    )

    assert gate["status"] == "fail"
    assert gate["gates"]["normal_full_minus_strongest_control_f1"]["passed"] is False
    assert gate["gates"]["state_only_minus_zero_f1"]["passed"] is True
    assert gate["gates"]["state_only_minus_donor_f1"]["passed"] is False


def make_resume_record(
    sample: dict,
    *,
    condition: str = "state_only",
    donor_sample: dict | None = None,
    fingerprint: str = "f" * 64,
    condition_protocol: dict | None = None,
) -> dict:
    source_index = sample["source_index"]
    parsed_json = {"boundaries": [2]}
    return {
        "status": "ok",
        "condition": condition,
        "condition_protocol": (
            evaluator.CONDITION_PROTOCOLS[condition]
            if condition_protocol is None
            else condition_protocol
        ),
        "task": evaluator.TASK_NAME,
        "task_kind": "scene",
        "split": "val",
        "key": f"{evaluator.TASK_NAME}:{source_index}",
        "line_index": source_index,
        "source_index": source_index,
        "selection_ordinal": 0,
        "row_sha256": sample["row_sha256"],
        "write_token_count": sample.get("write_token_count"),
        "fingerprint": fingerprint,
        "gold": sample["gold"],
        "donor_source_index": (
            None if donor_sample is None else donor_sample["source_index"]
        ),
        "donor_row_sha256": (
            None if donor_sample is None else donor_sample["row_sha256"]
        ),
        "raw_generation": json.dumps(parsed_json),
        "parsed_json": parsed_json,
        "input_tokens": 10,
        "input_rendered_sha256": "c" * 64,
        "output_tokens": 2,
        "hit_max_new_tokens": False,
        "elapsed_seconds": 0.25,
        "peak_cuda_memory_bytes": None,
        "memory_trace": [],
        "online_state_after_generation": {},
        "prime": (
            {
                "tokens": 8,
                "rendered_sha256": "d" * 64,
                "kv_cache_retained": False,
                "online_state": {},
            }
            if condition
            in {"state_only", "state_only_donor", "state_only_no_write"}
            else None
        ),
        "score_strict": evaluator.score_prediction(
            "scene", parsed_json, sample["gold"]
        ),
        "score_recovered": evaluator.recovered_scene_score(
            parsed_json, sample["gold"]
        ),
    }


def test_validate_resume_records_accepts_full_contract_and_rejects_duplicates() -> None:
    sample = {
        "source_index": 3,
        "row_sha256": "abc",
        "gold": {"boundaries": [2]},
    }
    selected = {3: sample}
    valid = make_resume_record(sample)
    assert evaluator.validate_resume_records(
        [valid],
        condition="state_only",
        condition_protocol=evaluator.CONDITION_PROTOCOLS["state_only"],
        selected_by_index=selected,
        fingerprint="f" * 64,
    ) == {3: valid}

    with pytest.raises(ValueError, match="Duplicate"):
        evaluator.validate_resume_records(
            [valid, valid],
            condition="state_only",
            condition_protocol=evaluator.CONDITION_PROTOCOLS["state_only"],
            selected_by_index=selected,
            fingerprint="f" * 64,
        )


def test_validate_resume_records_preserves_train_split_for_focused_overfit() -> None:
    sample = {
        "source_index": 3,
        "row_sha256": "a" * 64,
        "gold": {"boundaries": [2]},
    }
    record = {**make_resume_record(sample), "split": "train"}

    validated = evaluator.validate_resume_records(
        [record],
        condition="state_only",
        condition_protocol=evaluator.CONDITION_PROTOCOLS["state_only"],
        selected_by_index={3: sample},
        fingerprint="f" * 64,
        split="train",
    )

    assert validated[3]["split"] == "train"


def test_validate_resume_records_rejects_resource_and_state_schema_drift() -> None:
    sample = {
        "source_index": 3,
        "row_sha256": "a" * 64,
        "gold": {"boundaries": [2]},
        "write_token_count": 120,
    }
    record = make_resume_record(sample)
    mutations = (
        ({"selection_ordinal": 1}, "selection_ordinal differs"),
        ({"write_token_count": 121}, "write_token_count differs"),
        ({"input_tokens": True}, "input_tokens is invalid"),
        ({"input_tokens": 0}, "input_tokens is invalid"),
        ({"output_tokens": -1}, "output_tokens is invalid"),
        ({"output_tokens": 129}, "output_tokens is invalid"),
        ({"output_tokens": 128}, "hit_max_new_tokens is inconsistent"),
        ({"hit_max_new_tokens": "false"}, "hit_max_new_tokens is inconsistent"),
        ({"elapsed_seconds": -0.1}, "elapsed_seconds is invalid"),
        ({"elapsed_seconds": float("inf")}, "elapsed_seconds is invalid"),
        ({"input_rendered_sha256": "bad"}, "input_rendered_sha256 is invalid"),
        ({"peak_cuda_memory_bytes": -1}, "peak_cuda_memory_bytes is invalid"),
        ({"memory_trace": {}}, "memory_trace is invalid"),
        (
            {"online_state_after_generation": []},
            "online_state_after_generation is invalid",
        ),
        ({"prime": None}, "prime is invalid"),
        ({"prime": {"tokens": 0}}, "prime is invalid"),
    )
    for mutation, message in mutations:
        with pytest.raises(ValueError, match=message):
            evaluator.validate_resume_records(
                [{**record, **mutation}],
                condition="state_only",
                condition_protocol=evaluator.CONDITION_PROTOCOLS["state_only"],
                selected_by_index={3: sample},
                fingerprint="f" * 64,
            )


def test_validate_resume_records_bind_donor_identity() -> None:
    current = {
        "source_index": 3,
        "row_sha256": "a" * 64,
        "gold": {"boundaries": [2]},
    }
    donor = {
        "source_index": 7,
        "row_sha256": "b" * 64,
        "gold": {"boundaries": [1]},
    }
    record = make_resume_record(
        current,
        condition="state_only_donor",
        donor_sample=donor,
    )

    assert evaluator.validate_resume_records(
        [record],
        condition="state_only_donor",
        condition_protocol=evaluator.CONDITION_PROTOCOLS["state_only_donor"],
        selected_by_index={3: current, 7: donor},
        donor_by_index={3: donor, 7: current},
        fingerprint="f" * 64,
    ) == {3: record}

    for field, value in (
        ("donor_source_index", 8),
        ("donor_row_sha256", "c" * 64),
    ):
        with pytest.raises(ValueError, match=f"{field} differs"):
            evaluator.validate_resume_records(
                [{**record, field: value}],
                condition="state_only_donor",
                condition_protocol=evaluator.CONDITION_PROTOCOLS[
                    "state_only_donor"
                ],
                selected_by_index={3: current, 7: donor},
                donor_by_index={3: donor, 7: current},
                fingerprint="f" * 64,
            )


def test_validate_resume_records_accepts_resolved_matched_donor_protocol() -> None:
    current = {
        "source_index": 3,
        "row_sha256": "a" * 64,
        "gold": {"boundaries": [2]},
    }
    donor = {
        "source_index": 7,
        "row_sha256": "b" * 64,
        "gold": {"boundaries": [1]},
    }
    protocol = evaluator.resolved_condition_protocols(
        ["state_only_donor"],
        donor_rule=evaluator.DONOR_RULE_LENGTH_MATCHED,
    )["state_only_donor"]
    record = make_resume_record(
        current,
        condition="state_only_donor",
        donor_sample=donor,
        condition_protocol=protocol,
    )

    assert evaluator.validate_resume_records(
        [record],
        condition="state_only_donor",
        condition_protocol=protocol,
        selected_by_index={3: current, 7: donor},
        donor_by_index={3: donor, 7: current},
        fingerprint="f" * 64,
    ) == {3: record}

    with pytest.raises(ValueError, match="condition_protocol differs"):
        evaluator.validate_resume_records(
            [record],
            condition="state_only_donor",
            condition_protocol=evaluator.CONDITION_PROTOCOLS["state_only_donor"],
            selected_by_index={3: current, 7: donor},
            donor_by_index={3: donor, 7: current},
            fingerprint="f" * 64,
        )


def test_hard32_resume_requires_finite_semantic_nll() -> None:
    sample = {
        "source_index": 3,
        "row_sha256": "a" * 64,
        "gold": {"boundaries": [2]},
    }
    donor = {
        "source_index": 4,
        "row_sha256": f"{4:064x}",
        "gold": {"boundaries": [3]},
    }
    record = make_resume_record(sample)
    with pytest.raises(ValueError, match="semantic_decision_nll is missing"):
        evaluator.validate_resume_records(
            [record],
            condition="state_only",
            condition_protocol=evaluator.CONDITION_PROTOCOLS["state_only"],
            selected_by_index={3: sample},
            fingerprint="f" * 64,
            require_semantic_nll=True,
        )

    semantic_nll = semantic_record(
        3,
        1.25,
        pair_donor_source_index=4,
    )["semantic_decision_nll"]
    assert evaluator.validate_resume_records(
        [{**record, "semantic_decision_nll": semantic_nll}],
        condition="state_only",
        condition_protocol=evaluator.CONDITION_PROTOCOLS["state_only"],
        selected_by_index={3: sample},
        donor_by_index={3: donor},
        fingerprint="f" * 64,
        require_semantic_nll=True,
    )[3]["semantic_decision_nll"]["all_semantic"]["mean_nll"] == 1.25


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ({"source_index": "3"}, "source_index must be an integer"),
        ({"source_index": True}, "source_index must be an integer"),
        ({"status": "failed"}, "status differs"),
        ({"condition": "normal_full"}, "condition differs"),
        ({"condition_protocol": {}}, "condition_protocol differs"),
        ({"task": "other"}, "task differs"),
        ({"task_kind": "narrative"}, "task_kind differs"),
        ({"split": "train"}, "split differs"),
        ({"key": "wrong"}, "key differs"),
        ({"row_sha256": "different"}, "row_sha256 differs"),
        ({"fingerprint": "0" * 64}, "fingerprint differs"),
        ({"gold": {"boundaries": [1]}}, "gold differs"),
        ({"raw_generation": "{}"}, "does not reproduce parsed_json"),
        ({"score_strict": {}}, "score_strict differs"),
        ({"score_recovered": {}}, "score_recovered differs"),
    ],
)
def test_validate_resume_records_rejects_contract_drift(
    mutation: dict,
    message: str,
) -> None:
    sample = {
        "source_index": 3,
        "row_sha256": "abc",
        "gold": {"boundaries": [2]},
    }
    record = {**make_resume_record(sample), **mutation}

    with pytest.raises(ValueError, match=message):
        evaluator.validate_resume_records(
            [record],
            condition="state_only",
            condition_protocol=evaluator.CONDITION_PROTOCOLS["state_only"],
            selected_by_index={3: sample},
            fingerprint="f" * 64,
        )


def test_validate_resume_record_requires_parsed_and_raw_generation() -> None:
    sample = {
        "source_index": 3,
        "row_sha256": "abc",
        "gold": {"boundaries": [2]},
    }
    valid = make_resume_record(sample)
    for missing, message in (
        ("parsed_json", "parsed_json is missing"),
        ("raw_generation", "raw_generation is invalid"),
    ):
        record = dict(valid)
        record.pop(missing)
        with pytest.raises(ValueError, match=message):
            evaluator.validate_resume_records(
                [record],
                condition="state_only",
                condition_protocol=evaluator.CONDITION_PROTOCOLS["state_only"],
                selected_by_index={3: sample},
                fingerprint="f" * 64,
            )


def test_existing_manifest_payload_must_hash_to_recorded_fingerprint() -> None:
    payload = {"task": evaluator.TASK_NAME, "split": "val", "rows": [3]}
    fingerprint = evaluator.fingerprint_payload_sha256(payload)
    manifest = {"fingerprint": fingerprint, "fingerprint_payload": payload}

    assert evaluator.validate_existing_manifest(
        manifest,
        expected_fingerprint=fingerprint,
    ) is manifest

    tampered = {
        **manifest,
        "fingerprint_payload": {**payload, "rows": [4]},
    }
    with pytest.raises(ValueError, match="does not hash"):
        evaluator.validate_existing_manifest(
            tampered,
            expected_fingerprint=fingerprint,
        )


def test_existing_manifest_must_match_current_run_fingerprint() -> None:
    payload = {"task": evaluator.TASK_NAME, "split": "val", "rows": [3]}
    fingerprint = evaluator.fingerprint_payload_sha256(payload)

    with pytest.raises(ValueError, match="differs from this run"):
        evaluator.validate_existing_manifest(
            {"fingerprint": fingerprint, "fingerprint_payload": payload},
            expected_fingerprint="0" * 64,
        )


def test_hard32_receipt_is_atomic_self_hashing_and_checkpoint_bound(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output_dir = tmp_path / "eval"
    output_dir.mkdir()
    for name in ("manifest.json", "summary.json"):
        (output_dir / name).write_text("{}\n", encoding="utf-8")
    for condition in evaluator.CONDITIONS:
        (output_dir / f"{condition}.jsonl").write_text("{}\n", encoding="utf-8")
    memory_dir = tmp_path / "checkpoint-32"
    memory_dir.mkdir()
    (memory_dir / "delta_mem_adapter.pt").write_bytes(b"adapter")
    (memory_dir / "delta_mem_config.json").write_text("{}\n", encoding="utf-8")
    dataset = tmp_path / "val.jsonl"
    dataset.write_text("official\n", encoding="utf-8")
    selection = tmp_path / "holdout_source_indices.json"
    selection.write_text("selection\n", encoding="utf-8")
    monkeypatch.setattr(
        evaluator,
        "OFFICIAL_SCENE_V4_VAL_SHA256",
        evaluator.sha256_file(dataset),
    )
    monkeypatch.setattr(
        evaluator,
        "HARD32_SELECTION_SHA256",
        evaluator.sha256_file(selection),
    )
    lineage = {"lineage_kind": "identity_checkpoint_receipt", "checkpoint_step": 32}
    monkeypatch.setattr(evaluator, "scene_v6_training_lineage", lambda path: lineage)
    empty_mapping_sha256 = evaluator.sha256_text("[]")
    monkeypatch.setattr(
        evaluator,
        "HARD32_FROZEN_DONOR_MAPPING_ROWS_SHA256",
        empty_mapping_sha256,
    )
    contract = {
        "name": "scene_v6_identity_hard32",
        "rows": 32,
        "conditions": list(evaluator.CONDITIONS),
    }
    gate = {
        "status": "pass",
        "all_gates_passed": True,
        "full170_authorized_for_bound_checkpoint": True,
    }
    receipt = evaluator.build_hard32_receipt(
        output_dir=output_dir,
        fingerprint="f" * 64,
        contract=contract,
        candidate_lineage=lineage,
        code_fingerprint={"evaluator_sha256": evaluator.sha256_file(Path(evaluator.__file__))},
        dataset_file=dataset,
        selection_file=selection,
        donor_mapping=[],
        gate=gate,
        semantic_evidence={"rows": []},
        base_outcome_evidence={"rows": []},
        memory_dir=memory_dir,
        conditions=list(evaluator.CONDITIONS),
    )
    receipt_path = output_dir / "hard32_receipt.json"
    evaluator.write_json_atomic(receipt_path, receipt)

    authorization = evaluator.validate_hard32_pass_receipt(
        receipt_path,
        memory_dir=memory_dir,
    )

    assert receipt["schema"] == "scene_v6_identity_hard32_receipt.v2"
    assert authorization["payload_sha256"] == receipt["receipt_sha256"]
    assert authorization["checkpoint_adapter_sha256"] == evaluator.sha256_file(
        memory_dir / "delta_mem_adapter.pt"
    )

    stale_v1 = {**receipt, "schema": "scene_v6_identity_hard32_receipt.v1"}
    stale_v1_unsigned = dict(stale_v1)
    stale_v1_unsigned.pop("receipt_sha256")
    stale_v1["receipt_sha256"] = evaluator.fingerprint_payload_sha256(
        stale_v1_unsigned
    )
    receipt_path.write_text(json.dumps(stale_v1), encoding="utf-8")
    with pytest.raises(ValueError, match="schema differs"):
        evaluator.validate_hard32_pass_receipt(receipt_path, memory_dir=memory_dir)

    tampered = json.loads(receipt_path.read_text(encoding="utf-8"))
    tampered["status"] = "fail"
    receipt_path.write_text(json.dumps(tampered), encoding="utf-8")
    with pytest.raises(ValueError, match="checksum differs"):
        evaluator.validate_hard32_pass_receipt(receipt_path, memory_dir=memory_dir)


def test_historical_v6_receipt_self_hashes_and_binds_three_outputs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output_dir = tmp_path / "eval"
    output_dir.mkdir()
    for name in ("manifest.json", "summary.json", "progress.json"):
        (output_dir / name).write_text("{}\n", encoding="utf-8")
    for condition in evaluator.HISTORICAL_V6_HARD32_CONDITIONS:
        (output_dir / f"{condition}.jsonl").write_text("{}\n", encoding="utf-8")
    lineage = {
        "lineage_kind": "historical_artifact_identity_without_source_receipt",
        "lineage_limitation": evaluator.HISTORICAL_V6_LINEAGE_LIMITATION,
    }
    hard32 = {"holdout": {"sha256": evaluator.HARD32_HOLDOUT_SHA256}}
    revision = {"commit": "a" * 40, "dirty": False}
    monkeypatch.setattr(
        evaluator, "validate_historical_v6_checkpoint", lambda path: lineage
    )
    monkeypatch.setattr(
        evaluator,
        "validate_historical_v6_hard32_artifacts",
        lambda **kwargs: hard32,
    )
    monkeypatch.setattr(evaluator, "git_revision", lambda path: revision)
    base_model_binding = {"path": "/model", "weights": {}, "prompt_artifacts": {}}
    monkeypatch.setattr(
        evaluator,
        "historical_base_model_binding",
        lambda path: base_model_binding,
    )
    monkeypatch.setattr(
        evaluator,
        "scene_state_code_fingerprint",
        lambda path: code,
    )
    monkeypatch.setattr(
        evaluator,
        "require_historical_tracked_worktree_clean",
        lambda: {
            "repository": str(evaluator.PROJECT_ROOT),
            "tracked_worktree_clean": True,
            "untracked_files_ignored": True,
        },
    )
    code = {"evaluator_sha256": evaluator.sha256_file(Path(evaluator.__file__))}
    contract = {
        "name": evaluator.HISTORICAL_V6_HARD32_CONTRACT,
        "rows": 32,
        "conditions": list(evaluator.HISTORICAL_V6_HARD32_CONDITIONS),
        "full170_authorized": False,
        "test_authorized": False,
        "checkpoint_selection_authorized": False,
    }
    evidence = {
        "schema": "rwkv_ms_scene_historical_v6_hard32_evidence.v1",
        "full170_authorized": False,
        "test_authorized": False,
    }
    fingerprint = "f" * 64
    receipt = evaluator.build_historical_v6_hard32_receipt(
        output_dir=output_dir,
        fingerprint=fingerprint,
        contract=contract,
        candidate_lineage=lineage,
        code_fingerprint=code,
        repository_revision=revision,
        base_model=tmp_path / "model",
        base_model_binding=base_model_binding,
        dataset_file=tmp_path / "val.jsonl",
        selection_file=tmp_path / "selection.json",
        memory_dir=tmp_path / "checkpoint-672",
        conditions=list(evaluator.HISTORICAL_V6_HARD32_CONDITIONS),
        evidence=evidence,
    )
    unsigned = dict(receipt)
    recorded_sha256 = unsigned.pop("receipt_sha256")
    assert recorded_sha256 == evaluator.fingerprint_payload_sha256(unsigned)
    assert receipt["full170_authorized"] is False
    receipt_path = output_dir / "historical_v6_hard32_receipt.json"
    evaluator.write_json_atomic(receipt_path, receipt)

    validated = evaluator.validate_historical_v6_hard32_receipt(
        receipt_path,
        fingerprint=fingerprint,
        memory_dir=tmp_path / "checkpoint-672",
        base_model=tmp_path / "model",
        base_model_binding=base_model_binding,
        dataset_file=tmp_path / "val.jsonl",
        selection_file=tmp_path / "selection.json",
        code_fingerprint=code,
        repository_revision=revision,
        evidence=evidence,
    )
    assert validated["payload_sha256"] == recorded_sha256
    assert validated["full170_authorized"] is False

    monkeypatch.setattr(
        evaluator,
        "historical_base_model_binding",
        lambda path: {**base_model_binding, "weights": {"changed": True}},
    )
    with pytest.raises(ValueError, match="base-model binding differs"):
        evaluator.validate_historical_v6_hard32_receipt(
            receipt_path,
            fingerprint=fingerprint,
            memory_dir=tmp_path / "checkpoint-672",
            base_model=tmp_path / "model",
            base_model_binding=base_model_binding,
            dataset_file=tmp_path / "val.jsonl",
            selection_file=tmp_path / "selection.json",
            code_fingerprint=code,
            repository_revision=revision,
            evidence=evidence,
        )
    monkeypatch.setattr(
        evaluator,
        "historical_base_model_binding",
        lambda path: base_model_binding,
    )
    monkeypatch.setattr(
        evaluator,
        "scene_state_code_fingerprint",
        lambda path: {**code, "common_sha256": "0" * 64},
    )
    with pytest.raises(ValueError, match="code binding differs"):
        evaluator.validate_historical_v6_hard32_receipt(
            receipt_path,
            fingerprint=fingerprint,
            memory_dir=tmp_path / "checkpoint-672",
            base_model=tmp_path / "model",
            base_model_binding=base_model_binding,
            dataset_file=tmp_path / "val.jsonl",
            selection_file=tmp_path / "selection.json",
            code_fingerprint=code,
            repository_revision=revision,
            evidence=evidence,
        )
    monkeypatch.setattr(
        evaluator,
        "scene_state_code_fingerprint",
        lambda path: code,
    )

    (output_dir / "normal_full.jsonl").write_text("tampered\n", encoding="utf-8")
    with pytest.raises(ValueError, match="normal_full output artifact differs"):
        evaluator.validate_historical_v6_hard32_receipt(
            receipt_path,
            fingerprint=fingerprint,
            memory_dir=tmp_path / "checkpoint-672",
            base_model=tmp_path / "model",
            base_model_binding=base_model_binding,
            dataset_file=tmp_path / "val.jsonl",
            selection_file=tmp_path / "selection.json",
            code_fingerprint=code,
            repository_revision=revision,
            evidence=evidence,
        )


def test_read_resume_records_repairs_only_partial_final_line(tmp_path: Path) -> None:
    output = tmp_path / "state_only.jsonl"
    valid = {"source_index": 3, "condition": "state_only"}
    output.write_bytes((json.dumps(valid) + "\n{\"source_index\":").encode("utf-8"))

    assert evaluator.read_resume_records(output) == [valid]
    assert output.read_text(encoding="utf-8") == json.dumps(valid) + "\n"

    output.write_text(json.dumps(valid) + "\nnot-json\n", encoding="utf-8")
    with pytest.raises(json.JSONDecodeError):
        evaluator.read_resume_records(output)


def test_read_resume_records_terminates_valid_final_record(tmp_path: Path) -> None:
    output = tmp_path / "state_only.jsonl"
    valid = {"source_index": 3, "condition": "state_only"}
    output.write_bytes(json.dumps(valid).encode("utf-8"))

    assert evaluator.read_resume_records(output) == [valid]
    assert output.read_bytes() == json.dumps(valid).encode("utf-8") + b"\n"


def test_condition_protocols_explicitly_discard_prime_kv_cache() -> None:
    assert evaluator.CONDITION_PROTOCOLS["state_only"]["kv_cache_carried_from_prime"] is False
    assert evaluator.CONDITION_PROTOCOLS["state_only"]["rwkv_writes_during_prime"] is True
    assert (
        evaluator.CONDITION_PROTOCOLS["state_only_no_write"]["rwkv_writes_during_prime"]
        is False
    )
    assert evaluator.CONDITION_PROTOCOLS["state_only"]["attention_context"] == "system_only"
    donor = evaluator.CONDITION_PROTOCOLS["state_only_donor"]
    assert donor["kv_cache_carried_from_prime"] is False
    assert donor["rwkv_writes_during_prime"] is True
    assert donor["rwkv_writes_during_generation"] is False
    assert donor["donor_pool"] == "selected_validation_rows_only"
    assert donor["donor_rule"] == "next_selected_row_cyclic"
