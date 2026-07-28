from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from experiments.rethinking_rwkv_ms_gemma import run_scene_state_eval as state_eval
from experiments.rethinking_rwkv_ms_gemma import run_scene_train32_eval as evaluator


def _self_hashed(payload: dict[str, Any], field: str) -> dict[str, Any]:
    result = dict(payload)
    result[field] = evaluator.canonical_sha256(result)
    return result


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _write_jsonl(path: Path, payloads: list[dict[str, Any]]) -> list[str]:
    lines = [
        json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        for payload in payloads
    ]
    path.write_text("".join(f"{line}\n" for line in lines), encoding="utf-8")
    return lines


def _tiny_bundle(root: Path) -> dict[str, Any]:
    root.mkdir()
    data_path = root / "tiny2.jsonl"
    rows_path = root / "tiny2_rows.jsonl"
    pair_path = root / "tiny2_pair_manifest.json"
    source_path = root / "tiny2_source_manifest.json"
    source_lock_path = root / "source_lock.json"
    golds = ([2, 4], [1, 3])
    official_indices = (383, 619)
    write_counts = (100, 121)
    token_ids = (101, 202)
    data_rows = [
        {
            "messages": [
                {"role": "system", "content": "scene system"},
                {"role": "user", "content": f"[P1] row {ordinal}"},
                {
                    "role": "assistant",
                    "content": json.dumps({"boundaries": list(gold)}),
                },
            ]
        }
        for ordinal, gold in enumerate(golds)
    ]
    raw_rows = _write_jsonl(data_path, data_rows)
    row_payloads = []
    for ordinal, (gold, source_index, write_count, target_id) in enumerate(
        zip(golds, official_indices, write_counts, token_ids, strict=True)
    ):
        baseline_score = {
            "schema_valid": False,
            "tp": 0,
            "fp": 0,
            "fn": 2,
        }
        row_payloads.append(
            _self_hashed(
                {
                    "schema": evaluator.ROW_SCHEMA,
                    "train_row_ordinal": ordinal,
                    "source_split": "train",
                    "official_source_index": source_index,
                    "row_sha256": evaluator.sha256_text(raw_rows[ordinal]),
                    "gold_boundaries": list(gold),
                    "gold_boundary_count": 2,
                    "label_sha256": evaluator.canonical_sha256(list(gold)),
                    "base_record_sha256": str(ordinal + 1) * 64,
                    "strict_score": baseline_score,
                    "strict_failure_stratum": "invalid_schema",
                    "token_metadata": {
                        "write_rendered_sha256": evaluator.sha256_text(
                            f"write-{ordinal}"
                        ),
                        "write_input_ids_sha256": evaluator.canonical_sha256(
                            [ordinal, write_count]
                        ),
                        "write_token_count": write_count,
                        "generation_prefix_rendered_sha256": evaluator.sha256_text(
                            "generation-prefix"
                        ),
                        "generation_prefix_input_ids_sha256": evaluator.canonical_sha256(
                            [7, 8, 9]
                        ),
                        "generation_prefix_token_count": 3,
                        "semantic_target_positions": [10],
                        "semantic_target_token_ids": [target_id],
                    },
                },
                "record_sha256",
            )
        )
    _write_jsonl(rows_path, row_payloads)
    entries = []
    for ordinal in range(2):
        donor = 1 - ordinal
        source_row = row_payloads[ordinal]
        donor_row = row_payloads[donor]
        entries.append(
            _self_hashed(
                {
                    "train_row_ordinal": ordinal,
                    "donor_train_row_ordinal": donor,
                    "official_source_index": source_row["official_source_index"],
                    "donor_official_source_index": donor_row[
                        "official_source_index"
                    ],
                    "source_row_sha256": source_row["row_sha256"],
                    "donor_row_sha256": donor_row["row_sha256"],
                    "source_label_sha256": source_row["label_sha256"],
                    "donor_label_sha256": donor_row["label_sha256"],
                    "source_boundary_count": 2,
                    "donor_boundary_count": 2,
                    "source_write_sha256": source_row["token_metadata"][
                        "write_input_ids_sha256"
                    ],
                    "donor_write_sha256": donor_row["token_metadata"][
                        "write_input_ids_sha256"
                    ],
                    "source_write_token_count": source_row["token_metadata"][
                        "write_token_count"
                    ],
                    "donor_write_token_count": donor_row["token_metadata"][
                        "write_token_count"
                    ],
                    "source_generation_prefix_sha256": source_row[
                        "token_metadata"
                    ]["generation_prefix_input_ids_sha256"],
                    "donor_generation_prefix_sha256": donor_row[
                        "token_metadata"
                    ]["generation_prefix_input_ids_sha256"],
                    "source_base_record_sha256": source_row[
                        "base_record_sha256"
                    ],
                    "donor_base_record_sha256": donor_row[
                        "base_record_sha256"
                    ],
                    "source_strict_failure_stratum": source_row[
                        "strict_failure_stratum"
                    ],
                    "donor_strict_failure_stratum": donor_row[
                        "strict_failure_stratum"
                    ],
                    "source_strict_score_sha256": evaluator.canonical_sha256(
                        source_row["strict_score"]
                    ),
                    "donor_strict_score_sha256": evaluator.canonical_sha256(
                        donor_row["strict_score"]
                    ),
                    "write_token_count_delta": 21,
                    "selected_target_positions": [10],
                    "selected_target_predictor_positions": [9],
                    "selected_target_token_ids": [token_ids[ordinal]],
                    "donor_target_token_ids": [token_ids[donor]],
                    "first_differing_semantic_ordinal": 0,
                    "causal_prefix_sha256": "c" * 64,
                    "target_stratum": "same_cardinality_value",
                },
                "entry_sha256",
            )
        )
    pair_payload = _self_hashed(
        {
            "schema": evaluator.PAIR_SCHEMA,
            "dataset": {
                "path": str(data_path.resolve()),
                "sha256": evaluator.sha256_file(data_path),
                "rows": 2,
                "ordered_row_sha256": evaluator.canonical_sha256(
                    [row["row_sha256"] for row in row_payloads]
                ),
            },
            "quotas": evaluator.CONTRACT_SPECS["scene_v7_tiny_overfit"][
                "quotas"
            ],
            "directed_pairs": entries,
            "entries_sha256": evaluator.canonical_sha256(entries),
        },
        "manifest_sha256",
    )
    _write_json(pair_path, pair_payload)
    source_payload = _self_hashed(
        {
            "schema": evaluator.SOURCE_SCHEMA,
            "task": evaluator.TASK_NAME,
            "contract": {
                "source_split": "train",
                "val_rows": 0,
                "test_rows": 0,
                "episode_contract": evaluator.EPISODE_CONTRACT,
            },
            "partitions": {
                "train": {
                    "source_split": "train",
                    "rows": 2,
                    "data": {
                        "path": str(data_path.resolve()),
                        "sha256": evaluator.sha256_file(data_path),
                    },
                    "row_manifest": {
                        "path": str(rows_path.resolve()),
                        "sha256": evaluator.sha256_file(rows_path),
                    },
                }
            },
            "v7_pairing": {
                "schema": evaluator.PAIR_BINDING_SCHEMA,
                "dataset_sha256": evaluator.sha256_file(data_path),
                "directed_entry_count": 2,
                "quotas": evaluator.CONTRACT_SPECS["scene_v7_tiny_overfit"][
                    "quotas"
                ],
                "entries_sha256": pair_payload["entries_sha256"],
                "pair_manifest": {
                    "path": str(pair_path.resolve()),
                    "sha256": evaluator.sha256_file(pair_path),
                    "manifest_sha256": pair_payload["manifest_sha256"],
                },
            },
            "parent_train32_sha256": "d" * 64,
        },
        "manifest_sha256",
    )
    _write_json(source_path, source_payload)
    artifact_paths = {
        "tiny2": data_path,
        "tiny2_rows": rows_path,
        "tiny2_pair_manifest": pair_path,
        "tiny2_source_manifest": source_path,
    }
    lock_payload = _self_hashed(
        {
            "schema": "rwkv_ms_scene_memory_v7_source_lock.v1",
            "artifacts": {
                name: {
                    "path": str(path.resolve()),
                    "sha256": evaluator.sha256_file(path),
                }
                for name, path in artifact_paths.items()
            },
        },
        "lock_sha256",
    )
    _write_json(source_lock_path, lock_payload)
    return {
        "dataset": data_path,
        "rows": rows_path,
        "pair": pair_path,
        "source": source_path,
        "source_lock": source_lock_path,
        "row_payloads": row_payloads,
        "pair_payload": pair_payload,
    }


def _validate_tiny(bundle: dict[str, Any]) -> dict[str, Any]:
    return evaluator.validate_v7_contract(
        contract="scene_v7_tiny_overfit",
        dataset_file=bundle["dataset"],
        row_manifest_file=bundle["rows"],
        pair_manifest_file=bundle["pair"],
        source_manifest_file=bundle["source"],
        expected_dataset_sha256=evaluator.sha256_file(bundle["dataset"]),
        expected_row_manifest_sha256=evaluator.sha256_file(bundle["rows"]),
        expected_pair_manifest_sha256=evaluator.sha256_file(bundle["pair"]),
        expected_source_manifest_sha256=evaluator.sha256_file(bundle["source"]),
        source_lock_file=bundle["source_lock"],
    )


def _generation_record(
    *,
    condition: str,
    ordinal: int,
    gold: list[int],
    prediction: Any,
) -> dict[str, Any]:
    raw = json.dumps(prediction, separators=(",", ":"))
    parsed = json.loads(raw)
    gold_payload = {"boundaries": gold}
    return {
        "condition": condition,
        "train_row_ordinal": ordinal,
        "gold": gold_payload,
        "raw_generation": raw,
        "parsed_json": parsed,
        "score_strict": evaluator.score_prediction(
            "scene", parsed, gold_payload
        ),
        "score_recovered": evaluator.recovered_scene_score(parsed, gold_payload),
    }


def _passing_tiny_records() -> tuple[dict[str, list[dict[str, Any]]], dict[str, Any]]:
    golds = ([2, 4], [1, 3])
    correct = [
        _generation_record(
            condition="state_only",
            ordinal=index,
            gold=list(gold),
            prediction={"boundaries": list(gold)},
        )
        for index, gold in enumerate(golds)
    ]
    donor = [
        _generation_record(
            condition="state_only_donor",
            ordinal=index,
            gold=list(golds[index]),
            prediction={"boundaries": list(golds[1 - index])},
        )
        for index in range(2)
    ]
    zero_prediction = {"boundaries": list(golds[0])}
    zero = [
        _generation_record(
            condition="state_only_no_write",
            ordinal=index,
            gold=list(golds[index]),
            prediction=zero_prediction,
        )
        for index in range(2)
    ]
    pairing = {
        "directed_pairs": [
            {
                "train_row_ordinal": 0,
                "donor_train_row_ordinal": 1,
                "target_stratum": "same_cardinality_value",
            },
            {
                "train_row_ordinal": 1,
                "donor_train_row_ordinal": 0,
                "target_stratum": "same_cardinality_value",
            },
        ]
    }
    return {
        "state_only": correct,
        "state_only_donor": donor,
        "state_only_no_write": zero,
    }, pairing


def test_runtime_generation_and_scoring_helpers_are_reused() -> None:
    assert evaluator.evaluate_condition is state_eval.evaluate_condition
    assert evaluator.generate_messages is state_eval.generate_messages
    assert evaluator.prime_online_state is state_eval.prime_online_state
    assert evaluator.score_prediction is state_eval.score_prediction
    assert evaluator.is_canonical_scene_prediction is (
        state_eval.is_canonical_scene_prediction
    )


def test_tiny_train_source_and_pair_contract_passes(tmp_path: Path) -> None:
    bundle = _tiny_bundle(tmp_path / "bundle")

    validated = _validate_tiny(bundle)

    assert validated["expected_rows"] == 2
    assert [row["source_index"] for row in validated["rows"]] == [383, 619]
    assert validated["pairing"]["donor_by_ordinal"] == {0: 1, 1: 0}


def test_validation_test_source_and_train32_row_count_are_rejected(
    tmp_path: Path,
) -> None:
    bundle = _tiny_bundle(tmp_path / "bundle")
    source = json.loads(bundle["source"].read_text(encoding="utf-8"))
    source["contract"]["source_split"] = "val"
    source["manifest_sha256"] = evaluator.self_hash_payload(
        source, hash_field="manifest_sha256"
    )
    _write_json(bundle["source"], source)

    with pytest.raises(evaluator.V7EvaluationContractError, match="not train"):
        evaluator.validate_v7_contract(
            contract="scene_v7_tiny_overfit",
            dataset_file=bundle["dataset"],
            row_manifest_file=bundle["rows"],
            pair_manifest_file=bundle["pair"],
            source_manifest_file=bundle["source"],
        )

    with pytest.raises(evaluator.V7EvaluationContractError, match="exactly 32"):
        evaluator.load_v7_rows(
            bundle["dataset"],
            bundle["rows"],
            contract="scene_v7_train32_overfit",
        )

    val_path = tmp_path / "val.jsonl"
    val_path.write_text(bundle["dataset"].read_text(encoding="utf-8"), encoding="utf-8")
    with pytest.raises(evaluator.V7EvaluationContractError, match="validation/test"):
        evaluator.load_v7_rows(
            val_path,
            bundle["rows"],
            contract="scene_v7_tiny_overfit",
        )


def test_tampered_pair_source_hash_fails_closed(tmp_path: Path) -> None:
    bundle = _tiny_bundle(tmp_path / "bundle")
    rows = evaluator.load_v7_rows(
        bundle["dataset"],
        bundle["rows"],
        contract="scene_v7_tiny_overfit",
    )
    pair = json.loads(bundle["pair"].read_text(encoding="utf-8"))
    pair["directed_pairs"][0]["source_row_sha256"] = "0" * 64
    pair["directed_pairs"][0]["entry_sha256"] = evaluator.self_hash_payload(
        pair["directed_pairs"][0], hash_field="entry_sha256"
    )
    pair["entries_sha256"] = evaluator.canonical_sha256(pair["directed_pairs"])
    pair["manifest_sha256"] = evaluator.self_hash_payload(
        pair, hash_field="manifest_sha256"
    )
    _write_json(bundle["pair"], pair)

    with pytest.raises(evaluator.V7EvaluationContractError, match="source_row_sha256"):
        evaluator.load_v7_pairing(
            bundle["pair"],
            dataset_file=bundle["dataset"],
            rows=rows,
            contract="scene_v7_tiny_overfit",
        )


def test_tiny_gate_uses_canonical_strict_generation_only() -> None:
    records, pairing = _passing_tiny_records()
    gate = evaluator.build_gate(
        contract="scene_v7_tiny_overfit",
        records_by_condition=records,
        pairing=pairing,
    )
    assert gate["status"] == "pass"
    assert gate["metrics"]["correct_vs_zero"]["zero_unique_raw_outputs"] == 1
    assert gate["metrics"]["donor_identity_recovery"]["strict_exact_rows"] == 2

    wrapped_records, pairing = _passing_tiny_records()
    wrapped = [{"boundaries": [2, 4]}]
    wrapped_records["state_only"][0] = _generation_record(
        condition="state_only",
        ordinal=0,
        gold=[2, 4],
        prediction=wrapped,
    )
    wrapped_gate = evaluator.build_gate(
        contract="scene_v7_tiny_overfit",
        records_by_condition=wrapped_records,
        pairing=pairing,
    )
    assert wrapped_gate["status"] == "fail"
    assert not wrapped_gate["gates"]["all_correct_state_outputs_canonical"]
    assert wrapped_gate["format_recovery_diagnostic_only"][
        "can_satisfy_gate"
    ] is False


def test_resume_fingerprint_mismatch_fails_closed(tmp_path: Path) -> None:
    bundle = _tiny_bundle(tmp_path / "bundle")
    contract = _validate_tiny(bundle)
    sample = contract["rows"][0]
    parsed = {"boundaries": [2, 4]}
    record = evaluator._record_with_self_hash(
        {
            "schema": evaluator.RECORD_SCHEMA,
            "status": "ok",
            "fingerprint": "f" * 64,
            "condition": "state_only",
            "task": evaluator.TASK_NAME,
            "task_kind": "scene",
            "split": "train",
            "train_row_ordinal": 0,
            "source_index": sample["source_index"],
            "row_sha256": sample["row_sha256"],
            "gold": sample["gold"],
            "donor_train_row_ordinal": None,
            "donor_source_index": None,
            "donor_row_sha256": None,
            "raw_generation": json.dumps(parsed),
            "parsed_json": parsed,
            "score_strict": evaluator.score_prediction(
                "scene", parsed, sample["gold"]
            ),
            "score_recovered": evaluator.recovered_scene_score(
                parsed, sample["gold"]
            ),
            "input_rendered_sha256": sample["token_metadata"][
                "generation_prefix_rendered_sha256"
            ],
            "prime": {
                "rendered_sha256": sample["token_metadata"][
                    "write_rendered_sha256"
                ]
            },
            "output_tokens": 5,
            "hit_max_new_tokens": False,
        }
    )

    validated = evaluator.validate_v7_resume_records(
        [record],
        condition="state_only",
        fingerprint="f" * 64,
        rows=contract["rows"],
        donor_by_ordinal={0: 1, 1: 0},
    )
    assert set(validated) == {0}

    tampered = evaluator._record_with_self_hash(
        {**record, "fingerprint": "e" * 64}
    )
    with pytest.raises(evaluator.V7EvaluationContractError, match="fingerprint"):
        evaluator.validate_v7_resume_records(
            [tampered],
            condition="state_only",
            fingerprint="f" * 64,
            rows=contract["rows"],
            donor_by_ordinal={0: 1, 1: 0},
        )


def test_tiny_receipt_self_hashes_and_never_authorizes_hard32(
    tmp_path: Path,
) -> None:
    bundle = _tiny_bundle(tmp_path / "bundle")
    contract = _validate_tiny(bundle)
    output = tmp_path / "output"
    output.mkdir()
    for filename in (
        "manifest.json",
        "state_only.jsonl",
        "state_only_donor.jsonl",
        "state_only_no_write.jsonl",
        "summary.json",
    ):
        (output / filename).write_text("{}\n", encoding="utf-8")
    records, pairing = _passing_tiny_records()
    gate = evaluator.build_gate(
        contract="scene_v7_tiny_overfit",
        records_by_condition=records,
        pairing=pairing,
    )
    checkpoint = {
        "memory_dir": "/checkpoint-32",
        "adapter_sha256": "a" * 64,
        "config_sha256": "b" * 64,
    }
    receipt = evaluator.build_receipt(
        contract="scene_v7_tiny_overfit",
        fingerprint="f" * 64,
        output_dir=output,
        input_contract=contract,
        checkpoint=checkpoint,
        gate=gate,
    )
    receipt_path = output / "gate_receipt.json"
    _write_json(receipt_path, receipt)

    validated = evaluator.validate_receipt(
        receipt_path,
        expected_contract="scene_v7_tiny_overfit",
        expected_checkpoint=checkpoint,
        require_pass=True,
    )
    stored_receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    assert validated["receipt_sha256"] == evaluator.self_hash_payload(stored_receipt)
    assert validated["hard32_authorization"]["authorized"] is False
    with pytest.raises(evaluator.V7EvaluationContractError, match="contract"):
        evaluator.validate_fixed_hard32_authorization(
            receipt_path,
            expected_checkpoint=checkpoint,
        )

    tampered = dict(receipt)
    tampered["input_artifacts"] = {
        **tampered["input_artifacts"],
        "dataset": {
            **tampered["input_artifacts"]["dataset"],
            "expected_sha256": "0" * 64,
        },
    }
    tampered["receipt_sha256"] = evaluator.self_hash_payload(tampered)
    with pytest.raises(evaluator.V7EvaluationContractError, match="expected SHA"):
        evaluator.validate_receipt(tampered)


def test_checkpoint_protocol_and_pairing_are_required(tmp_path: Path) -> None:
    bundle = _tiny_bundle(tmp_path / "bundle")
    contract = _validate_tiny(bundle)
    checkpoint = tmp_path / "checkpoint-32"
    checkpoint.mkdir()
    _write_json(
        checkpoint / "delta_mem_config.json",
        {
            "target_layers": list(range(42)),
            "delta_heads": ["q", "o"],
            "rank": 4,
            "rwkv_ms_semantics_version": 2,
            "memory_backend": "rwkv_ms",
        },
    )
    for filename in (
        "delta_mem_adapter.pt",
        "optimizer.pt",
        "scheduler.pt",
        "rng_state.pth",
    ):
        (checkpoint / filename).write_bytes(b"fixture")
    train_pair = _self_hashed(
        {
            "sample_count": 2,
            "source_pair_manifest_path": contract["pair_manifest_file"],
            "source_pair_manifest_file_sha256": contract["artifact_bindings"][
                "pair_manifest"
            ]["actual_sha256"],
            "source_pair_manifest_sha256": contract["pair_manifest_sha256"],
            "source_entries_sha256": contract["pairing"]["entries_sha256"],
            "target_stratum_row_counts": contract["expected_quotas"],
        },
        "manifest_sha256",
    )
    materialized = _self_hashed(
        {
            "schema_version": 2,
            "objective_version": "scene_state_generation_ce_v1",
            "target_stratum_row_counts": contract["expected_quotas"],
            "splits": {"train": train_pair},
        },
        "manifest_sha256",
    )
    _write_json(
        checkpoint / "scene_state_identity_pairing_manifest.json",
        materialized,
    )
    protocol = {
        "schema_version": 10,
        "memory_objective_version": "scene_state_generation_ce_v1",
        "memory_loss_mode": "scene_state_generation_ce",
        "training_mode": "episode",
        "assistant_loss_mode": "final_assistant_only",
        "episode_recent_messages": 0,
        "episode_read_write_enabled": False,
        "max_write_length": 2048,
        "per_device_train_batch_size": 1,
        "gradient_accumulation_steps": 1,
        "frozen_mlp_activation_checkpointing": True,
        "validation_split_ratio": 0.0,
        "eval_samples": 0,
        "scene_boundary_payload_ce_weight": 0.0,
        "memory_dropout_no_memory_prob": 0.0,
        "memory_dropout_state_only_prob": 0.0,
        "memory_base_kl_weight": 0.0,
        "memory_kl_weight": 0.0,
        "memory_representation_weight": 0.0,
        "write_sparsity_weight": 0.0,
        "scene_generation_read_protocol": (
            "exact_system_only_generation_prefix_same_read_correct_donor_zero_v1"
        ),
        "scene_generation_zero_protocol": (
            "adapter_active_reset_state_writes_disabled_detached_reference_v1"
        ),
        "scene_generation_objective_formula": (
            "weighted_generation_ce(schema=2,decision=4,termination=1) + "
            "first_wrong_gold_prefix_top1_hinge(0.2) + "
            "correct_source_vs_donor_two_token_ce + "
            "donor_donor_vs_source_two_token_ce + "
            "correct_vs_detached_zero_decision_margin_hinge(0.2)"
        ),
        "train_file": contract["dataset_file"],
        "train_samples": 2,
        "scene_state_source_manifest": {
            "path": contract["source_manifest_file"],
            "file_sha256": contract["artifact_bindings"]["source_manifest"][
                "actual_sha256"
            ],
            "schema": evaluator.SOURCE_SCHEMA,
            "train_file": contract["dataset_file"],
            "train_file_sha256": contract["artifact_bindings"]["dataset"][
                "actual_sha256"
            ],
            "train_rows": 2,
            "train_source_split": "train",
            "episode_contract": evaluator.EPISODE_CONTRACT,
        },
        "scene_state_identity_pairing": {
            "manifest_sha256": materialized["manifest_sha256"],
            "target_stratum_row_counts": contract["expected_quotas"],
        },
        "max_steps": 32,
    }
    _write_json(checkpoint / "training_protocol.json", protocol)
    _write_json(
        checkpoint / "trainer_state.json",
        {"global_step": 32, "max_steps": 32},
    )

    identity = evaluator.validate_v7_checkpoint(
        checkpoint,
        input_contract=contract,
    )
    assert identity["global_step"] == 32
    assert identity["objective_version"] == "scene_state_generation_ce_v1"

    protocol["memory_objective_version"] = "wrong"
    _write_json(checkpoint / "training_protocol.json", protocol)
    with pytest.raises(evaluator.V7EvaluationContractError, match="objective"):
        evaluator.validate_v7_checkpoint(
            checkpoint,
            input_contract=contract,
        )
