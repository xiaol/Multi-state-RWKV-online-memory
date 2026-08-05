from __future__ import annotations

import copy
import json
import os
from pathlib import Path

import pytest

from experiments.rethinking_rwkv_ms_gemma import (
    prepare_synthetic_compositional_associative_retrieval_canary_v3 as v3,
)


@pytest.fixture(scope="module")
def tokenizer():
    os.environ["HF_ENDPOINT"] = v3.HF_MIRROR_ENDPOINT
    assert v3.DEFAULT_MODEL_PATH.is_dir()
    return v3.load_local_tokenizer(v3.DEFAULT_MODEL_PATH)


@pytest.fixture(scope="module")
def partitions(tokenizer):
    return v3.build_partitions(tokenizer)


def _fake_model_identity(root: Path) -> dict[str, object]:
    root.mkdir()
    for ordinal, name in enumerate(v3.MODEL_ARTIFACT_NAMES):
        (root / name).write_bytes(f"v3-fixture-{ordinal}-{name}\n".encode("ascii"))
    return v3.bind_model_artifacts(root)


def test_spec_locks_compositional_heldout_contract() -> None:
    spec = v3.canary_spec()

    assert len(v3.KEY_LABELS) == len(set(v3.KEY_LABELS)) == 24
    assert len(v3.VALUE_LABELS) == len(set(v3.VALUE_LABELS)) == 24
    assert not set(v3.KEY_LABELS) & set(v3.VALUE_LABELS)
    assert v3.RECORDS_PER_EPISODE == v3.RWKV_MS_NUM_STATES == 4
    assert set(v3.TRAIN_OFFSETS).isdisjoint(v3.HELDOUT_OFFSETS)
    assert set(v3.TRAIN_OFFSETS) | set(v3.HELDOUT_OFFSETS) == set(range(24))
    assert spec["episode"]["query_counterfactuals_per_byte_identical_state"] == 4
    assert spec["split_guarantees"] == {
        "keys_seen_in_both_splits": True,
        "values_seen_in_both_splits": True,
        "key_value_pair_intersection": 0,
        "mapping_intersection": 0,
        "mapping_query_intersection": 0,
        "record_order_pattern_intersection": 0,
    }
    assert spec["acceptance_gate"] == {
        "training_seeds": [42, 43, 44],
        "required_seed_passes": 3,
        "heldout_answer_accuracy_min": 0.95,
        "heldout_semantic_route_accuracy_min": 0.95,
        "heldout_query_counterfactual_route_accuracy_min": 0.95,
        "heldout_donor_expected_answer_accuracy_min": 0.95,
        "heldout_value_swap_expected_answer_accuracy_min": 0.95,
        "heldout_no_write_answer_accuracy_max": 0.35,
        "heldout_no_write_route_absent_fraction_min": 1.0,
    }
    assert v3.SOURCE_CONTRACT["requires_record_local_collator"] is True
    assert v3.SOURCE_CONTRACT["compatible_with_flat_episode_write_collator"] is False
    assert v3.SOURCE_CONTRACT["hard32_accessed"] is False
    assert v3.SOURCE_CONTRACT["protected_evaluation_included"] is False


def test_partitions_are_deterministic_for_local_tokenizer(
    tokenizer,
    partitions,
) -> None:
    rebuilt = {
        split: v3.build_partition_rows(tokenizer, split)
        for split in v3.PARTITION_ORDER
    }

    assert v3.canonical_sha256(rebuilt) == v3.canonical_sha256(partitions)
    assert len(partitions["train"]) == 384
    assert len(partitions["heldout"]) == 192


def test_split_manifest_proves_unseen_pairs_mappings_and_queries(
    partitions,
) -> None:
    split_manifest = v3.build_split_manifest(partitions)
    audit = split_manifest["audit"]

    assert audit["passed"] is True
    assert audit["train_key_coverage"] == audit["heldout_key_coverage"] == 24
    assert audit["train_value_coverage"] == audit["heldout_value_coverage"] == 24
    assert audit["shared_key_count"] == audit["shared_value_count"] == 24
    assert audit["key_value_pair_intersection"] == []
    assert audit["mapping_intersection_sha256"] == []
    assert audit["mapping_query_intersection_sha256"] == []
    assert audit["row_id_intersection"] == []
    assert audit["record_order_pattern_intersection"] == []
    assert audit["record_order_pattern_intersection_count"] == 0
    assert audit["train_record_order_pattern_count"] == 18
    assert audit["heldout_record_order_pattern_count"] == 6
    assert audit["query_target_slots_balanced"] is True
    assert audit["record_orders_randomized"] is True
    assert len(audit["partitions"]["train"]["key_value_pairs"]) == 384
    assert len(audit["partitions"]["heldout"]["key_value_pairs"]) == 192
    assert len(audit["partitions"]["train"]["mapping_signatures_sha256"]) == 96
    assert len(audit["partitions"]["heldout"]["mapping_signatures_sha256"]) == 48
    assert audit["partitions"]["train"]["query_target_slot_counts"] == {
        "0": 96,
        "1": 96,
        "2": 96,
        "3": 96,
    }
    assert audit["partitions"]["heldout"]["query_target_slot_counts"] == {
        "0": 48,
        "1": 48,
        "2": 48,
        "3": 48,
    }
    unsigned = dict(split_manifest)
    declared = unsigned.pop("manifest_sha256")
    assert declared == v3.canonical_sha256(unsigned)


@pytest.mark.parametrize("split", v3.PARTITION_ORDER)
def test_rows_expose_exact_record_major_masks_and_query_route(
    partitions,
    split: str,
) -> None:
    for row in partitions[split]:
        records = row["record_local_writes"]
        target_slot = row["query_route_target_slot"]

        assert len(records) == 4
        assert row["write_record_slot_indices"] == [0, 1, 2, 3]
        assert row["write_record_input_ids"] == [
            record["tokenization"]["input_ids"] for record in records
        ]
        assert row["write_record_attention_mask"] == [
            record["tokenization"]["attention_mask"] for record in records
        ]
        assert row["write_record_key_mask"] == [
            record["tokenization"]["key_token_mask"] for record in records
        ]
        assert row["write_record_value_mask"] == [
            record["tokenization"]["value_token_mask"] for record in records
        ]
        for record_index, record in enumerate(records):
            key_mask = row["write_record_key_mask"][record_index]
            value_mask = row["write_record_value_mask"][record_index]
            assert len(key_mask) == len(value_mask) == len(
                row["write_record_input_ids"][record_index]
            )
            assert any(key_mask) and any(value_mask)
            assert not any(
                key_selected and value_selected
                for key_selected, value_selected in zip(
                    key_mask, value_mask, strict=True
                )
            )
        assert row["query"]["target_slot"] == target_slot
        assert row["query"]["key"] == records[target_slot]["key"]
        assert row["query"]["target_value"] == records[target_slot]["value"]
        assert row["read_route_input_ids"] == row["read_route"]["input_ids"]
        assert row["read_route_target_mask"] == row["read_route"][
            "query_key_token_mask"
        ]
        assert len(row["read_route_target_mask"]) == len(
            row["read_route_input_ids"]
        )
        assert any(row["read_route_target_mask"])
        assert len(row["messages"]) == 3
        assert all(
            record["content"] not in json.dumps(row["messages"])
            for record in records
        )


@pytest.mark.parametrize("split", v3.PARTITION_ORDER)
def test_each_state_has_four_byte_identical_query_counterfactuals(
    partitions,
    split: str,
) -> None:
    families: dict[str, list[dict[str, object]]] = {}
    for row in partitions[split]:
        families.setdefault(row["memory_state_id"], []).append(row)

    expected_families = 96 if split == "train" else 48
    assert len(families) == expected_families
    for state_id, family in families.items():
        family.sort(key=lambda row: row["query_route_target_slot"])
        assert [row["query_route_target_slot"] for row in family] == [0, 1, 2, 3]
        assert len({row["memory_state_sha256"] for row in family}) == 1
        assert len(
            {
                v3.canonical_sha256(row["write_record_input_ids"])
                for row in family
            }
        ) == 1
        expected_ids = [row["row_id"] for row in family]
        for row in family:
            assert row["query_counterfactuals"] == {
                "memory_state_id": state_id,
                "byte_identical_record_writes_required": True,
                "row_id_by_target_slot": expected_ids,
            }


@pytest.mark.parametrize("split", v3.PARTITION_ORDER)
def test_donor_and_value_swap_interventions_are_exact(
    partitions,
    split: str,
) -> None:
    rows = partitions[split]
    for ordinal, row in enumerate(rows):
        donor = rows[row["donor"]["row_ordinal"]]
        target_slot = row["query"]["target_slot"]
        source_slots = row["value_swap"]["source_slot_by_destination_slot"]
        swapped_source_slot = source_slots[target_slot]
        swapped_record = row["record_local_writes"][swapped_source_slot]

        assert donor["donor"]["row_ordinal"] == ordinal
        assert donor["query"]["key"] == row["query"]["key"]
        assert donor["query"]["target_slot"] == target_slot
        assert [record["key"] for record in donor["record_local_writes"]] == [
            record["key"] for record in row["record_local_writes"]
        ]
        assert all(
            donor_record["value"] != source_record["value"]
            for donor_record, source_record in zip(
                donor["record_local_writes"],
                row["record_local_writes"],
                strict=True,
            )
        )
        assert row["donor"]["expected_target_value"] == donor["query"][
            "target_value"
        ]
        assert sorted(source_slots) == [0, 1, 2, 3]
        assert all(source != destination for destination, source in enumerate(source_slots))
        assert row["value_swap"]["expected_target_value"] == swapped_record["value"]
        assert row["value_swap"]["expected_target_value"] != row["query"][
            "target_value"
        ]


def test_write_and_load_bundle_recompute_all_hashes(
    tmp_path: Path,
    tokenizer,
    partitions,
) -> None:
    model = _fake_model_identity(tmp_path / "model")
    output_dir = tmp_path / "bundle"
    result = v3.write_bundle(
        output_dir,
        model=model,
        tokenizer=tokenizer,
        partitions=partitions,
    )
    loaded = v3.load_source_bundle(Path(result["source_manifest"]))

    assert result["train_rows"] == 384
    assert result["heldout_rows"] == 192
    assert loaded["manifest_sha256"] == result["source_manifest_sha256"]
    assert loaded["split_manifest"]["audit"]["passed"] is True
    assert loaded["partitions"] == partitions
    manifest = loaded["manifest"]
    assert all(
        not Path(manifest["partitions"][split][artifact]["path"]).is_absolute()
        for split in v3.PARTITION_ORDER
        for artifact in ("data", "row_manifest")
    )
    assert manifest["contract"]["external_dataset_access"] is False
    assert manifest["contract"]["hard32_accessed"] is False
    assert manifest["contract"]["protected_evaluation_included"] is False

    heldout_path = Path(result["heldout_file"])
    heldout_path.write_bytes(heldout_path.read_bytes() + b"\n")
    with pytest.raises(ValueError, match="heldout data binding differs"):
        v3.load_source_bundle(Path(result["source_manifest"]))


def test_hf_endpoint_is_pinned_to_mirror(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("HF_ENDPOINT", raising=False)
    assert v3.configure_hf_mirror() == v3.HF_MIRROR_ENDPOINT
    assert os.environ["HF_ENDPOINT"] == v3.HF_MIRROR_ENDPOINT

    monkeypatch.setenv("HF_ENDPOINT", "https://huggingface.co")
    with pytest.raises(ValueError, match="HF_ENDPOINT must be"):
        v3.configure_hf_mirror()


def test_validation_rejects_a_declared_train_pair_in_heldout(partitions) -> None:
    corrupted = copy.deepcopy(partitions["heldout"])
    source = partitions["train"][0]["record_local_writes"][0]
    corrupted[0]["record_local_writes"][0]["key_index"] = source["key_index"]
    corrupted[0]["record_local_writes"][0]["key"] = source["key"]
    corrupted[0]["record_local_writes"][0]["value_index"] = source["value_index"]
    corrupted[0]["record_local_writes"][0]["value"] = source["value"]

    with pytest.raises(ValueError, match="record mapping differs"):
        v3.audit_split_leakage(
            {"train": partitions["train"], "heldout": corrupted}
        )
