from __future__ import annotations

from dataclasses import replace

import pytest

from experiments.rethinking_rwkv_ms_gemma import run_natural_memory_gate as runner


def _condition_family() -> list[runner.NaturalMemoryExample]:
    examples: list[runner.NaturalMemoryExample] = []
    for index, condition in enumerate(runner.DEFAULT_TRAINING_CONDITIONS):
        examples.append(
            runner.NaturalMemoryExample(
                row_id=f"query-0::training-condition={condition}",
                memory_state_id=f"episode-0:{condition}",
                source_split="train",
                source_mapping_offset=0,
                condition=condition,
                write_records=({"input_ids": [index + 1]},),
                write_slots=(0,),
                read_input_ids=(10, 11, 12 + index),
                read_attention_mask=(1, 1, 1),
                query_mask=(False, True, False),
                answer_mask=(False, False, True),
                labels=(-100, -100, 12 + index),
                target_slot=0,
                expected_answer_token_ids=(12 + index,),
                expected_value=f'"value-{index}"',
                target_slot_rewrite_selection=(
                    {"semantic_target_slot": 0}
                    if condition == "target_slot_rewrite"
                    else None
                ),
                episode_id="episode-0",
                task="attribution",
                semantic_target_slot=0,
                write_record_ids=(f"record-{condition}",),
                write_semantic_slots=(0,),
                write_value_jsons=(f'"value-{index}"',),
                record_payload_sha256=f"{index + 1:064x}",
                binding_absent_from_training=True,
                query_prefix_length=2,
            )
        )
    return examples


def test_training_dataset_audit_rejects_missing_and_duplicate_condition_families(
) -> None:
    family = _condition_family()
    assert runner.audit_training_dataset(family)["passed"] is True

    incomplete = runner.audit_training_dataset(family[:-1])
    assert incomplete["condition_set_exact"] is False
    assert incomplete["paired_condition_coverage"] is False
    assert incomplete["passed"] is False

    duplicated = runner.audit_training_dataset([*family, family[0]])
    assert duplicated["unique_row_ids"] is False
    assert duplicated["paired_condition_coverage"] is False
    assert duplicated["passed"] is False


@pytest.mark.parametrize(
    "replacement",
    [
        {"episode_id": "episode-other"},
        {
            "read_input_ids": (99, 11, 13),
            "read_attention_mask": (1, 1, 1),
            "query_mask": (False, True, False),
        },
    ],
)
def test_training_dataset_audit_rejects_cross_condition_family_metadata_drift(
    replacement: dict,
) -> None:
    family = _condition_family()
    family[1] = replace(family[1], **replacement)

    audit = runner.audit_training_dataset(family)

    assert audit["paired_condition_coverage"] is True
    assert audit["family_invariants_passed"] is False
    assert audit["family_invariant_failure_count"] == 1
    assert audit["passed"] is False


def test_training_dataset_payload_digest_binds_encoded_objective_fields() -> None:
    family = _condition_family()
    original = runner.audit_training_dataset(family)
    family[1] = replace(family[1], labels=(-100, -100, 999))

    changed = runner.audit_training_dataset(family)

    assert changed["training_row_id_set_sha256"] == original[
        "training_row_id_set_sha256"
    ]
    assert changed["ordered_training_examples_sha256"] != original[
        "ordered_training_examples_sha256"
    ]
