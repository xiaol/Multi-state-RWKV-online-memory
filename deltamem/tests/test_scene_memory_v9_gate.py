from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from experiments.rethinking_rwkv_ms_gemma import run_scene_memory_v9_gate as gate
from experiments.rethinking_rwkv_ms_gemma import run_scene_state_eval as state_eval


def _pairing() -> tuple[dict[str, Any], dict[int, int]]:
    canonical = (
        (1, 14),
        (22, 26),
        (3, 24),
        (5, 9),
        (10, 23),
        (19, 28),
        (20, 31),
    )
    value = set(gate.VALUE14_ORDINALS)
    presence = [ordinal for ordinal in range(32) if ordinal not in value]
    pairs = [*canonical, *zip(presence[::2], presence[1::2], strict=True)]
    donor: dict[int, int] = {}
    entries: list[dict[str, Any]] = []
    cross = {1, 14, 22, 26}
    for left, right in pairs:
        donor[left] = right
        donor[right] = left
    for ordinal in range(32):
        stratum = (
            "presence"
            if ordinal not in value
            else (
                "cross_cardinality_value"
                if ordinal in cross
                else "same_cardinality_value"
            )
        )
        entries.append(
            {
                "train_row_ordinal": ordinal,
                "donor_train_row_ordinal": donor[ordinal],
                "target_stratum": stratum,
            }
        )
    return {
        "directed_pairs": entries,
        "donor_by_ordinal": donor,
        "manifest_sha256": "a" * 64,
        "entries_sha256": gate.canonical_sha256(entries),
    }, donor


def _semantic(
    *,
    ordinal: int,
    donor_ordinal: int,
    condition: str,
    value_position: int,
    source_preferences: int,
    donor_preferences: int,
    switches: int,
    causal_rows: int,
    source_separation_rows: int,
) -> dict[str, Any]:
    correct_source_nll = 0.1 if value_position < source_preferences else 1.1
    if condition == "state_only":
        selected = correct_source_nll
        alternative = 1.1 if value_position < source_preferences else 0.1
    elif condition == "state_only_donor":
        donor_preferred = (
            value_position < switches
            or value_position >= 14 - (donor_preferences - switches)
        )
        selected = (
            correct_source_nll + 1.0
            if value_position < source_separation_rows
            else max(correct_source_nll - 0.05, 0.0)
        )
        alternative = max(selected - 0.05, 0.0) if donor_preferred else selected + 1.0
    else:
        selected = (
            correct_source_nll + 1.0
            if value_position < causal_rows
            else max(correct_source_nll - 0.05, 0.0)
        )
        alternative = 1.1
    return {
        "pair_target": {
            "mask_mode": gate.v8.PAIR_TARGET_DECISION_MASK_MODE,
            "normalization": gate.v8.PAIR_TARGET_DECISION_NLL_NORMALIZATION,
            "token_count": 1,
            "mean_nll": selected,
            "alternative_target_mean_nll": alternative,
            "selected_over_alternative_logprob_margin": alternative - selected,
            "selected_target_positions": [7],
            "selected_target_token_ids": [1000 + ordinal],
            "donor_target_token_ids": [1000 + donor_ordinal],
            "first_differing_semantic_ordinal": 0,
            "causal_prefix_sha256": f"causal-{ordinal}",
            "donor_source_index": donor_ordinal,
            "donor_row_sha256": f"row-{donor_ordinal}",
            "read_rendered_sha256": f"read-{ordinal}",
        }
    }


def _records(
    *,
    correct_exact: int,
    donor_exact: int,
    switches: int,
    source_preferences: int = 10,
    donor_preferences: int = 10,
    causal_rows: int = 11,
    source_separation_rows: int = 13,
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, Any]]:
    assert switches <= source_preferences
    assert switches <= donor_preferences
    assert donor_preferences - switches <= 14 - source_preferences
    pairing, donor = _pairing()
    value_position = {
        ordinal: position for position, ordinal in enumerate(gate.VALUE14_ORDINALS)
    }
    gold = {ordinal: {"boundaries": [ordinal + 1]} for ordinal in range(32)}
    records: dict[str, list[dict[str, Any]]] = {
        condition: [] for condition in gate.CONDITIONS
    }
    for condition in gate.CONDITIONS:
        for ordinal in range(32):
            position = value_position.get(ordinal)
            if condition == "state_only":
                parsed = (
                    gold[ordinal]
                    if position is None or position < correct_exact
                    else {"boundaries": [900 + ordinal]}
                )
                raw = f"correct-{ordinal}"
            elif condition == "state_only_donor":
                parsed = (
                    gold[donor[ordinal]]
                    if position is None or position < donor_exact
                    else {"boundaries": [800 + ordinal]}
                )
                raw = f"donor-{ordinal}"
            else:
                parsed = {"boundaries": []}
                raw = "zero-invariant"
            record: dict[str, Any] = {
                "condition": condition,
                "train_row_ordinal": ordinal,
                "source_index": ordinal,
                "row_sha256": f"row-{ordinal}",
                "gold": gold[ordinal],
                "parsed_json": parsed,
                "raw_generation": raw,
                "score_strict": gate.v8.score_prediction("scene", parsed, gold[ordinal]),
                "score_recovered": gate.v8.recovered_scene_score(parsed, gold[ordinal]),
                "hit_max_new_tokens": False,
                "input_tokens": 8,
                "output_tokens": 4,
                "elapsed_seconds": 0.01,
            }
            if position is not None:
                record["semantic_decision_nll"] = _semantic(
                    ordinal=ordinal,
                    donor_ordinal=donor[ordinal],
                    condition=condition,
                    value_position=position,
                    source_preferences=source_preferences,
                    donor_preferences=donor_preferences,
                    switches=switches,
                    causal_rows=causal_rows,
                    source_separation_rows=source_separation_rows,
                )
            records[condition].append(record)
    return records, pairing


def test_v9_gate_requires_generation_and_bidirectional_progress() -> None:
    improved, pairing = _records(correct_exact=4, donor_exact=4, switches=7)

    result = gate.build_v9_gate(
        records_by_condition=improved,
        pairing=pairing,
        checkpoint_step=7,
    )

    assert result["status"] == "pass"
    assert result["training_continuation_authorized"] is True
    assert result["next_checkpoint_step"] == 14
    assert result["hard32_authorized"] is False
    assert result["hard32_access"] == gate.HARD32_ACCESS_POLICY


def test_v9_gate_rejects_unchanged_v8_step56_behavior() -> None:
    unchanged, pairing = _records(correct_exact=3, donor_exact=3, switches=6)

    result = gate.build_v9_gate(
        records_by_condition=unchanged,
        pairing=pairing,
        checkpoint_step=7,
    )

    assert result["status"] == "fail"
    assert result["training_continuation_authorized"] is False
    assert result["gates"][
        "value14_correct_generation_strictly_improves"
    ] is False
    assert result["gates"][
        "value14_bidirectional_switch_strictly_improves"
    ] is False


def test_v9_later_gate_requires_strict_block_over_block_progress() -> None:
    step7_records, pairing = _records(correct_exact=4, donor_exact=4, switches=7)
    step7 = gate.build_v9_gate(
        records_by_condition=step7_records,
        pairing=pairing,
        checkpoint_step=7,
    )
    unchanged_records, _ = _records(correct_exact=4, donor_exact=4, switches=7)
    unchanged = gate.build_v9_gate(
        records_by_condition=unchanged_records,
        pairing=pairing,
        checkpoint_step=14,
        previous_gate=step7,
    )
    improved_records, _ = _records(correct_exact=5, donor_exact=5, switches=8)
    improved = gate.build_v9_gate(
        records_by_condition=improved_records,
        pairing=pairing,
        checkpoint_step=14,
        previous_gate=step7,
    )

    assert unchanged["status"] == "fail"
    assert improved["status"] == "pass"
    assert improved["comparison"]["checkpoint_step"] == 7
    assert improved["next_checkpoint_step"] == 21


@pytest.mark.parametrize(
    ("regression", "gate_name"),
    (
        (
            {"source_preferences": 9},
            "value14_source_token_preference_does_not_regress",
        ),
        (
            {"donor_preferences": 9},
            "value14_donor_token_preference_does_not_regress",
        ),
        (
            {"source_separation_rows": 12},
            "value14_source_state_separation_does_not_regress",
        ),
        (
            {"causal_rows": 10},
            "value14_causal_zero_control_does_not_regress",
        ),
    ),
)
def test_v9_gate_rejects_selected_or_causal_evidence_regression(
    regression: dict[str, int],
    gate_name: str,
) -> None:
    records, pairing = _records(
        correct_exact=4,
        donor_exact=4,
        switches=7,
        **regression,
    )

    result = gate.build_v9_gate(
        records_by_condition=records,
        pairing=pairing,
        checkpoint_step=7,
    )

    assert result["status"] == "fail"
    assert result["gates"][gate_name] is False
    assert result["training_continuation_authorized"] is False


def test_v9_later_gate_rejects_missing_or_nonpassing_predecessor() -> None:
    records, pairing = _records(correct_exact=5, donor_exact=5, switches=8)

    with pytest.raises(
        gate.V9EvaluationContractError,
        match="requires its previous gate",
    ):
        gate.build_v9_gate(
            records_by_condition=records,
            pairing=pairing,
            checkpoint_step=14,
        )

    failed_records, _ = _records(correct_exact=3, donor_exact=3, switches=6)
    failed = gate.build_v9_gate(
        records_by_condition=failed_records,
        pairing=pairing,
        checkpoint_step=7,
    )
    with pytest.raises(
        gate.V9EvaluationContractError,
        match="not a passing immediate predecessor",
    ):
        gate.build_v9_gate(
            records_by_condition=records,
            pairing=pairing,
            checkpoint_step=14,
            previous_gate=failed,
        )


def test_v9_final_step_is_progression_pass_but_not_benchmark_authorization() -> None:
    pairing: dict[str, Any] | None = None
    previous: dict[str, Any] | None = None
    for step, exact, switches in (
        (7, 4, 7),
        (14, 5, 8),
        (21, 6, 9),
        (28, 7, 10),
    ):
        records, current_pairing = _records(
            correct_exact=exact,
            donor_exact=exact,
            switches=switches,
        )
        pairing = current_pairing if pairing is None else pairing
        previous = gate.build_v9_gate(
            records_by_condition=records,
            pairing=pairing,
            checkpoint_step=step,
            previous_gate=previous,
        )

    assert previous is not None
    assert previous["status"] == "pass"
    assert previous["final_checkpoint_reached"] is True
    assert previous["training_continuation_authorized"] is False
    assert previous["next_checkpoint_step"] is None
    assert previous["final_benchmark_candidate"] is False
    assert previous["hard32_authorized"] is False


def test_v9_previous_receipt_is_bound_to_lineage_source_checkpoint(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    previous_checkpoint = tmp_path / "checkpoint-7"
    previous_checkpoint.mkdir()
    lineage_path = tmp_path / "checkpoint-14" / "continuation_manifest.json"
    lineage_path.parent.mkdir()
    lineage_path.write_text(
        json.dumps(
            {
                "source_checkpoint": str(previous_checkpoint.resolve()),
                "source_global_step": 7,
                "source_training_protocol_sha256": "protocol-7",
                "source_lineage_filename": "warm_start_lineage_manifest.json",
                "source_lineage_file_sha256": "lineage-7",
                "root_warm_start_receipt_sha256": "warm-root",
            }
        ),
        encoding="utf-8",
    )
    previous_receipt = tmp_path / "step7-gate-receipt.json"
    previous_receipt.write_text("{}\n", encoding="utf-8")
    checkpoint = {
        "global_step": 14,
        "lineage_artifact": gate._artifact_binding(
            lineage_path,
            description="test lineage",
        ),
    }
    calls: list[Path] = []

    def validated(*_args, memory_dir: Path, **_kwargs):
        calls.append(Path(memory_dir))
        return {
            "checkpoint": {
                "memory_dir": str(previous_checkpoint.resolve()),
                "global_step": 7,
                "training_protocol_canonical_sha256": "protocol-7",
                "lineage": {
                    "lineage_filename": "warm_start_lineage_manifest.json",
                    "lineage_file_sha256": "lineage-7",
                    "root_warm_start_receipt_sha256": "warm-root",
                },
            },
            "gate": {"next_checkpoint_step": 14},
        }

    monkeypatch.setattr(gate, "validate_gate_receipt_for_checkpoint", validated)

    result = gate.validate_previous_gate_receipt(
        previous_receipt,
        checkpoint=checkpoint,
        input_contract={},
        warm_contract={},
        ssd_root=tmp_path,
    )

    assert result is not None
    assert calls == [previous_checkpoint.resolve()]


def test_v9_previous_receipt_rejects_lineage_hash_mismatch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    previous_checkpoint = tmp_path / "checkpoint-7"
    previous_checkpoint.mkdir()
    lineage_path = tmp_path / "checkpoint-14" / "continuation_manifest.json"
    lineage_path.parent.mkdir()
    lineage_path.write_text(
        json.dumps(
            {
                "source_checkpoint": str(previous_checkpoint.resolve()),
                "source_global_step": 7,
                "source_training_protocol_sha256": "protocol-7",
                "source_lineage_filename": "warm_start_lineage_manifest.json",
                "source_lineage_file_sha256": "expected-lineage-7",
                "root_warm_start_receipt_sha256": "warm-root",
            }
        ),
        encoding="utf-8",
    )
    receipt = tmp_path / "step7-gate-receipt.json"
    receipt.write_text("{}\n", encoding="utf-8")
    checkpoint = {
        "global_step": 14,
        "lineage_artifact": gate._artifact_binding(
            lineage_path,
            description="test lineage",
        ),
    }
    monkeypatch.setattr(
        gate,
        "validate_gate_receipt_for_checkpoint",
        lambda *_args, **_kwargs: {
            "checkpoint": {
                "memory_dir": str(previous_checkpoint.resolve()),
                "global_step": 7,
                "training_protocol_canonical_sha256": "protocol-7",
                "lineage": {
                    "lineage_filename": "warm_start_lineage_manifest.json",
                    "lineage_file_sha256": "tampered-lineage-7",
                    "root_warm_start_receipt_sha256": "warm-root",
                },
            },
            "gate": {"next_checkpoint_step": 14},
        },
    )

    with pytest.raises(
        gate.V9EvaluationContractError,
        match="exact immediate lineage",
    ):
        gate.validate_previous_gate_receipt(
            receipt,
            checkpoint=checkpoint,
            input_contract={},
            warm_contract={},
            ssd_root=tmp_path,
        )


def test_v9_continuation_authorization_binds_exact_next_target(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    receipt = tmp_path / "gate-receipt.json"
    receipt.write_text("{}\n", encoding="utf-8")
    checkpoint = {
        "memory_dir": str((tmp_path / "checkpoint-7").resolve()),
        "global_step": 7,
    }
    validated = {
        "receipt_sha256": "a" * 64,
        "checkpoint": checkpoint,
        "gate": {
            "training_continuation_authorized": True,
            "next_checkpoint_step": 14,
        },
        "training_authorization": {
            "authorization_kind": gate.CONTINUATION_AUTHORIZATION_KIND,
            "authorized": True,
            "next_checkpoint_step": 14,
            "checkpoint_binding": checkpoint,
            "hard32_authorized": False,
        },
    }
    monkeypatch.setattr(
        gate,
        "validate_gate_receipt_for_checkpoint",
        lambda *_args, **_kwargs: validated,
    )

    result = gate.validate_continuation_authorization(
        receipt,
        source_checkpoint=checkpoint["memory_dir"],
        target_step=14,
        input_contract={},
        warm_contract={},
        ssd_root=tmp_path,
    )

    assert result["source_step"] == 7
    assert result["target_step"] == 14
    assert result["gate_receipt_sha256"] == "a" * 64
    assert result["hard32_authorized"] is False


def test_v9_continuation_authorization_rejects_stale_or_failing_receipt(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    receipt = tmp_path / "gate-receipt.json"
    receipt.write_text("{}\n", encoding="utf-8")
    checkpoint = {
        "memory_dir": str((tmp_path / "checkpoint-7").resolve()),
        "global_step": 7,
    }
    validated = {
        "receipt_sha256": "a" * 64,
        "checkpoint": checkpoint,
        "gate": {
            "training_continuation_authorized": True,
            "next_checkpoint_step": 14,
        },
        "training_authorization": {
            "authorization_kind": gate.CONTINUATION_AUTHORIZATION_KIND,
            "authorized": True,
            "next_checkpoint_step": 14,
            "checkpoint_binding": checkpoint,
            "hard32_authorized": False,
        },
    }
    monkeypatch.setattr(
        gate,
        "validate_gate_receipt_for_checkpoint",
        lambda *_args, **_kwargs: validated,
    )

    with pytest.raises(
        gate.V9EvaluationContractError,
        match="immediate target",
    ):
        gate.validate_continuation_authorization(
            receipt,
            source_checkpoint=checkpoint["memory_dir"],
            target_step=21,
            input_contract={},
            warm_contract={},
            ssd_root=tmp_path,
        )

    validated["training_authorization"]["authorized"] = False
    with pytest.raises(
        gate.V9EvaluationContractError,
        match="does not authorize",
    ):
        gate.validate_continuation_authorization(
            receipt,
            source_checkpoint=checkpoint["memory_dir"],
            target_step=14,
            input_contract={},
            warm_contract={},
            ssd_root=tmp_path,
        )


def _input_contract(root: Path) -> dict[str, Any]:
    pairing, _ = _pairing()
    artifacts: dict[str, dict[str, Any]] = {}
    for name in ("train32", "train32_pair_manifest", "v9_source_manifest"):
        path = root / f"{name}.json"
        path.write_text(f"{name}\n", encoding="utf-8")
        artifacts[name] = gate._artifact_binding(path, description=name)
    return {
        "artifacts": artifacts,
        "pairing": pairing,
        "launch_data": {},
    }


def _protocol(input_contract: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": gate.launch.OBJECTIVE_SCHEMA_VERSION,
        "memory_objective_version": gate.launch.OBJECTIVE_VERSION,
        "train_sampler_mode": gate.launch.FIXED_SAMPLER_MODE,
        "scene_generation_objective_formula": gate.V9_OBJECTIVE_FORMULA,
        "scene_generation_backward_mode": (
            "sequential_pair_zero_probe_teacher_then_aligned_replay_v3"
        ),
        "scene_generation_zero_protocol": (
            "shared_exact_selected_causal_prefix_adapter_active_reset_state_detached_v2"
        ),
        "scene_generation_generated_prefix_correction_weight": 0.5,
        "scene_generation_generated_prefix_correction_mode": (
            "levenshtein_raw_generated_prefix_gold_ce_wrong_unlikelihood_v3"
        ),
        "scene_generation_generated_prefix_max_correction_events": 4,
        "scene_generation_pair_unit": (
            "canonical_low_with_reciprocal_full_payload_v1"
        ),
        "scene_generation_pair_physical_batch_size": 1,
        "scene_generation_pair_directional_exposures": 2,
        "scene_state_source_manifest": {
            "path": input_contract["artifacts"]["v9_source_manifest"]["path"],
            "file_sha256": input_contract["artifacts"]["v9_source_manifest"][
                "sha256"
            ],
            "schema": gate.launch.SOURCE_SCHEMA,
            "train_file": input_contract["artifacts"]["train32"]["path"],
            "train_file_sha256": input_contract["artifacts"]["train32"][
                "sha256"
            ],
            "train_rows": 32,
            "train_source_split": "train",
            "episode_contract": gate.v8.v7.EPISODE_CONTRACT,
        },
    }


def _checkpoint_pairing(input_contract: dict[str, Any]) -> dict[str, Any]:
    counts = {
        "presence": 18,
        "same_cardinality_value": 10,
        "cross_cardinality_value": 4,
    }
    train = {
        "sample_count": 32,
        "source_pair_manifest_path": input_contract["artifacts"][
            "train32_pair_manifest"
        ]["path"],
        "source_pair_manifest_file_sha256": input_contract["artifacts"][
            "train32_pair_manifest"
        ]["sha256"],
        "source_pair_manifest_sha256": input_contract["pairing"]["manifest_sha256"],
        "source_entries_sha256": input_contract["pairing"]["entries_sha256"],
        "target_stratum_row_counts": counts,
        "symmetric_full_pair_materialized": True,
    }
    train["manifest_sha256"] = gate.self_hash_payload(
        train,
        hash_field="manifest_sha256",
    )
    pairing = {
        "schema_version": 2,
        "objective_version": gate.PAIRING_OBJECTIVE_VERSION,
        "splits": {"train": train},
    }
    pairing["manifest_sha256"] = gate.self_hash_payload(
        pairing,
        hash_field="manifest_sha256",
    )
    return pairing


def _write_checkpoint(root: Path, input_contract: dict[str, Any]) -> Path:
    checkpoint = root / "checkpoint-7"
    checkpoint.mkdir()
    protocol = _protocol(input_contract)
    pairing = _checkpoint_pairing(input_contract)
    protocol["scene_state_identity_pairing"] = {
        "manifest_sha256": pairing["manifest_sha256"],
        "target_stratum_row_counts": pairing["splits"]["train"][
            "target_stratum_row_counts"
        ],
    }
    payloads = {
        "training_protocol.json": protocol,
        "scene_state_identity_pairing_manifest.json": pairing,
        "delta_mem_config.json": {},
        "trainer_state.json": {"global_step": 7, "max_steps": 7},
        "warm_start_lineage_manifest.json": {},
    }
    for filename, payload in payloads.items():
        (checkpoint / filename).write_text(
            json.dumps(payload, sort_keys=True),
            encoding="utf-8",
        )
    for filename in ("delta_mem_adapter.pt", "optimizer.pt", "scheduler.pt", "rng_state.pth"):
        (checkpoint / filename).write_bytes(filename.encode())
    return checkpoint


def test_v9_checkpoint_gate_rejects_objective_formula_drift(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    input_contract = _input_contract(tmp_path)
    checkpoint = _write_checkpoint(tmp_path, input_contract)
    protocol_path = checkpoint / "training_protocol.json"
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    protocol["scene_generation_objective_formula"] = "pair_token_ce_only"
    protocol_path.write_text(json.dumps(protocol), encoding="utf-8")
    monkeypatch.setattr(
        gate.launch,
        "validate_checkpoint_contract",
        lambda **_kwargs: {
            "checkpoint": str(checkpoint),
            "checkpoint_step": 7,
            "lineage_filename": "warm_start_lineage_manifest.json",
            "training_protocol_sha256": gate.canonical_sha256(protocol),
        },
    )

    with pytest.raises(
        gate.V9EvaluationContractError,
        match="symmetric objective protocol differs",
    ):
        gate.validate_v9_checkpoint(
            checkpoint,
            input_contract=input_contract,
            warm_contract={},
            ssd_root=tmp_path,
        )


def test_v9_checkpoint_validation_reaches_pairing_and_artifact_binding(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    input_contract = _input_contract(tmp_path)
    checkpoint = _write_checkpoint(tmp_path, input_contract)
    protocol = json.loads(
        (checkpoint / "training_protocol.json").read_text(encoding="utf-8")
    )
    monkeypatch.setattr(
        gate.launch,
        "validate_checkpoint_contract",
        lambda **_kwargs: {
            "checkpoint": str(checkpoint),
            "checkpoint_step": 7,
            "lineage_filename": "warm_start_lineage_manifest.json",
            "training_protocol_sha256": gate.canonical_sha256(protocol),
        },
    )
    monkeypatch.setattr(
        gate,
        "memory_architecture_contract",
        lambda _checkpoint: {
            "target_layers": list(range(42)),
            "delta_heads": ["q", "o"],
            "rank": 4,
            "rwkv_ms_semantics_version": 2,
            "memory_backend": "rwkv_ms",
        },
    )

    result = gate.validate_v9_checkpoint(
        checkpoint,
        input_contract=input_contract,
        warm_contract={},
        ssd_root=tmp_path,
    )

    assert result["global_step"] == 7
    assert result["pairing_manifest_sha256"]
    assert result["lineage"]["checkpoint_step"] == 7


def test_v9_pairing_requires_reciprocal_full_payload_marker(
    tmp_path: Path,
) -> None:
    input_contract = _input_contract(tmp_path)
    pairing = _checkpoint_pairing(input_contract)
    pairing["splits"]["train"].pop("symmetric_full_pair_materialized")
    pairing["splits"]["train"]["manifest_sha256"] = gate.self_hash_payload(
        pairing["splits"]["train"],
        hash_field="manifest_sha256",
    )
    pairing["manifest_sha256"] = gate.self_hash_payload(
        pairing,
        hash_field="manifest_sha256",
    )
    protocol = _protocol(input_contract)
    protocol["scene_state_identity_pairing"] = {
        "manifest_sha256": pairing["manifest_sha256"],
        "target_stratum_row_counts": pairing["splits"]["train"][
            "target_stratum_row_counts"
        ],
    }

    with pytest.raises(
        gate.V9EvaluationContractError,
        match="symmetric_full_pair_materialized",
    ):
        gate._validate_v9_pairing(
            pairing,
            protocol=protocol,
            input_contract=input_contract,
        )


def test_v9_train_preflight_never_touches_hard32(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    protected = Path(state_eval.HISTORICAL_V6_HARD32_HOLDOUT)
    original_open = Path.open
    original_resolve = Path.resolve
    original_hash = gate.launch.sha256_file

    def guarded_open(path: Path, *args, **kwargs):
        if Path(path).absolute() == protected.absolute():
            raise AssertionError("V9 gate attempted to open Hard32")
        return original_open(path, *args, **kwargs)

    def guarded_resolve(path: Path, *args, **kwargs):
        if Path(path).absolute() == protected.absolute():
            raise AssertionError("V9 gate attempted to resolve Hard32")
        return original_resolve(path, *args, **kwargs)

    def guarded_hash(path: Path) -> str:
        if Path(path).absolute() == protected.absolute():
            raise AssertionError("V9 gate attempted to hash Hard32")
        return original_hash(path)

    monkeypatch.setattr(Path, "open", guarded_open)
    monkeypatch.setattr(Path, "resolve", guarded_resolve)
    monkeypatch.setattr(gate.launch, "sha256_file", guarded_hash)

    result = gate.validate_v9_train_inputs()

    assert result["hard32_access"] == gate.HARD32_ACCESS_POLICY
    assert result["value14_ordinals"] == list(gate.VALUE14_ORDINALS)


def test_v9_gate_paths_must_stay_on_ssd_outside_protected_names(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    allowed = tmp_path / "gates" / "step7"
    assert gate._ssd_path(
        allowed,
        description="test gate output",
        ssd_root=tmp_path,
    ) == allowed.resolve()

    with pytest.raises(gate.V9EvaluationContractError, match="must stay on the SSD"):
        gate._ssd_path(
            tmp_path.parent / "outside" / "step7",
            description="test gate output",
            ssd_root=tmp_path,
        )
    with pytest.raises(gate.V9EvaluationContractError, match="protected paths"):
        gate._ssd_path(
            tmp_path / "evaluation" / "step7",
            description="test gate output",
            ssd_root=tmp_path,
        )

    protected = tmp_path / "hard32" / "artifact.json"
    protected.parent.mkdir(parents=True)
    protected.write_text("{}\n", encoding="utf-8")
    original_open = Path.open

    def guarded_open(path: Path, *args, **kwargs):
        if path.resolve() == protected.resolve():
            raise AssertionError("protected artifact was opened before rejection")
        return original_open(path, *args, **kwargs)

    monkeypatch.setattr(Path, "open", guarded_open)
    with pytest.raises(gate.V9EvaluationContractError, match="protected paths"):
        gate._verify_artifact_binding(
            {
                "path": str(protected),
                "bytes": protected.stat().st_size,
                "sha256": "0" * 64,
            },
            description="protected receipt artifact",
            ssd_root=tmp_path,
        )

    receipt = tmp_path / "gates" / "step7" / "gate_receipt.json"
    receipt.parent.mkdir(parents=True)
    receipt.write_text("{}\n", encoding="utf-8")
    assert gate._regular_file(
        receipt,
        description="test gate receipt",
        ssd_root=tmp_path,
    ) == receipt.resolve()

    symlink = tmp_path / "gate-receipt-link.json"
    symlink.symlink_to(receipt)
    with pytest.raises(gate.V9EvaluationContractError, match="symlink"):
        gate._regular_file(
            symlink,
            description="test gate receipt",
            ssd_root=tmp_path,
        )
