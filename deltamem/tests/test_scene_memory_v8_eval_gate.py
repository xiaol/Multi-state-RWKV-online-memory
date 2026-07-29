from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from experiments.rethinking_rwkv_ms_gemma import run_scene_memory_v8_gate as gate
from experiments.rethinking_rwkv_ms_gemma import run_scene_state_eval as state_eval


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _self_hashed(payload: dict[str, Any], field: str) -> dict[str, Any]:
    result = dict(payload)
    result[field] = gate.self_hash_payload(result, hash_field=field)
    return result


def _donor_mapping() -> tuple[dict[int, int], dict[int, str]]:
    same = (1, 3, 5, 9, 10, 14, 19, 20, 22, 23)
    cross = (24, 26, 28, 31)
    value = set((*same, *cross))
    presence = tuple(index for index in range(32) if index not in value)
    donor: dict[int, int] = {}
    strata: dict[int, str] = {}
    for ordinals, stratum in (
        (same, "same_cardinality_value"),
        (cross, "cross_cardinality_value"),
        (presence, "presence"),
    ):
        assert len(ordinals) % 2 == 0
        for offset in range(0, len(ordinals), 2):
            left, right = ordinals[offset : offset + 2]
            donor[left] = right
            donor[right] = left
            strata[left] = stratum
            strata[right] = stratum
    return donor, strata


def _pairing() -> dict[str, Any]:
    donor, strata = _donor_mapping()
    entries = [
        {
            "train_row_ordinal": ordinal,
            "donor_train_row_ordinal": donor[ordinal],
            "target_stratum": strata[ordinal],
        }
        for ordinal in range(32)
    ]
    return {
        "directed_pairs": entries,
        "donor_by_ordinal": donor,
        "manifest_sha256": "a" * 64,
        "entries_sha256": gate.canonical_sha256(entries),
    }


def _semantic_pair(
    *,
    ordinal: int,
    donor_ordinal: int,
    condition: str,
) -> dict[str, Any]:
    if condition == "state_only":
        selected_nll, alternative_nll = 0.1, 1.1
    elif condition == "state_only_donor":
        selected_nll, alternative_nll = 1.1, 0.1
    else:
        selected_nll, alternative_nll = 1.1, 1.1
    return {
        "all_semantic": {
            "mask_mode": state_eval.SEMANTIC_DECISION_MASK_MODE,
            "normalization": state_eval.SEMANTIC_DECISION_NLL_NORMALIZATION,
            "selected_target_positions": [10],
            "selected_target_token_ids": [1000 + ordinal],
            "token_count": 1,
            "nll_sum": selected_nll,
            "mean_nll": selected_nll,
            "read_rendered_sha256": "b" * 64,
        },
        "pair_target": {
            "mask_mode": gate.PAIR_TARGET_DECISION_MASK_MODE,
            "normalization": gate.PAIR_TARGET_DECISION_NLL_NORMALIZATION,
            "target_mode": gate.PAIR_TARGET_DECISION_MASK_MODE,
            "selected_target_positions": [10],
            "selected_target_token_ids": [1000 + ordinal],
            "donor_target_token_ids": [1000 + donor_ordinal],
            "alternative_target_token_ids": [1000 + donor_ordinal],
            "first_differing_semantic_ordinal": 0,
            "causal_prefix_sha256": "c" * 64,
            "donor_source_index": donor_ordinal,
            "donor_row_sha256": f"{donor_ordinal:064x}",
            "token_count": 1,
            "nll_sum": selected_nll,
            "mean_nll": selected_nll,
            "alternative_target_nll_sum": alternative_nll,
            "alternative_target_mean_nll": alternative_nll,
            "selected_over_alternative_logprob_margin": (
                alternative_nll - selected_nll
            ),
            "read_rendered_sha256": "b" * 64,
        },
    }


def _records(*, root_like: bool = False, fingerprint: str | None = None):
    donor, _ = _donor_mapping()
    records = {condition: [] for condition in gate.CONDITIONS}
    root_correct = set(gate.VALUE14_ORDINALS[:2])
    for condition in gate.CONDITIONS:
        for ordinal in range(32):
            gold = {"boundaries": [ordinal + 1]}
            if condition == "state_only":
                prediction = (
                    gold
                    if not root_like or ordinal not in gate.VALUE14_SET or ordinal in root_correct
                    else {"boundaries": [900 + ordinal]}
                )
            elif condition == "state_only_donor":
                prediction = {"boundaries": [donor[ordinal] + 1]}
            else:
                prediction = {"boundaries": [999]}
            raw = json.dumps(prediction, separators=(",", ":"))
            record: dict[str, Any] = {
                "schema": gate.GATE_RECORD_SCHEMA,
                "status": "ok",
                "completed_at": "2026-01-01T00:00:00Z",
                "fingerprint": fingerprint or "f" * 64,
                "condition": condition,
                "task": gate.TASK_NAME,
                "task_kind": "scene",
                "split": "train",
                "train_row_ordinal": ordinal,
                "source_index": ordinal,
                "row_sha256": f"{ordinal:064x}",
                "gold": gold,
                "donor_train_row_ordinal": donor[ordinal],
                "donor_source_index": donor[ordinal],
                "donor_row_sha256": f"{donor[ordinal]:064x}",
                "raw_generation": raw,
                "parsed_json": prediction,
                "score_strict": gate.score_prediction("scene", prediction, gold),
                "score_recovered": gate.recovered_scene_score(prediction, gold),
                "semantic_decision_nll": (
                    _semantic_pair(
                        ordinal=ordinal,
                        donor_ordinal=donor[ordinal],
                        condition=condition,
                    )
                    if ordinal in gate.VALUE14_SET
                    else None
                ),
                "input_tokens": 1,
                "output_tokens": 1,
                "hit_max_new_tokens": False,
                "elapsed_seconds": 0.0,
            }
            records[condition].append(gate._record_with_self_hash(record))
    return records


def test_value14_gate_passes_only_with_correct_and_donor_identity_switches() -> None:
    result = gate.build_v8_gate(records_by_condition=_records(), pairing=_pairing())

    assert result["status"] == "pass"
    assert result["full_answer_ce_used_for_gate"] is False
    assert result["metrics"]["value14_generation"]["correct_strict_exact_rows"] == 14
    assert result["metrics"]["value14_generation"]["donor_identity_strict_exact_rows"] == 14
    selected = result["metrics"]["value14_selected_token_identity"]["overall"]
    assert selected["bidirectional_identity_switch_rows"] == 14
    assert selected["correct_state_beats_zero_on_source_token_rows"] == 14


def test_v7_root_like_two_of_fourteen_cannot_pass_v8_gate() -> None:
    result = gate.build_v8_gate(
        records_by_condition=_records(root_like=True),
        pairing=_pairing(),
    )

    assert result["status"] == "fail"
    assert result["metrics"]["value14_generation"]["correct_strict_exact_rows"] == 2
    assert result["gates"]["value14_material_gain_over_v7_root"] is False


def test_value14_gate_rejects_selected_token_identity_drift() -> None:
    records = _records()
    drifted = records["state_only_donor"][gate.VALUE14_ORDINALS[0]]
    drifted["semantic_decision_nll"]["pair_target"]["causal_prefix_sha256"] = "d" * 64

    with pytest.raises(gate.V8EvaluationContractError, match="causal_prefix_sha256"):
        gate.build_v8_gate(records_by_condition=records, pairing=_pairing())


def _input_contract(root: Path) -> dict[str, Any]:
    artifacts: dict[str, dict[str, Any]] = {}
    for name in (
        "train32",
        "train32_rows",
        "train32_pair_manifest",
        "train32_source_manifest",
        "v7_source_lock",
        "v8_source_lock",
        "v8_bundle_manifest",
        "v8_source_manifest",
        "v8_schedule",
        "v8_schedule_manifest",
    ):
        path = root / "sources" / f"{name}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"{name}\n", encoding="utf-8")
        artifacts[name] = gate._artifact_binding(path, description=name)
    pairing = _pairing()
    rows = [
        {
            "train_row_ordinal": ordinal,
            "source_index": ordinal,
            "row_sha256": f"{ordinal:064x}",
            "gold": {"boundaries": [ordinal + 1]},
        }
        for ordinal in range(32)
    ]
    return {
        "rows": rows,
        "pairing": pairing,
        "artifacts": artifacts,
        "benchmark_lock": gate.validate_benchmark_lock(),
        "v8_source_manifest_sha256": "e" * 64,
        "v8_schedule_entries_sha256": gate.launch.SCHEDULE_ENTRIES_SHA256,
    }


def _config() -> dict[str, Any]:
    return {
        "rank": 4,
        "alpha": 8.0,
        "memory_backend": "rwkv_ms",
        "target_layers": list(range(42)),
        "delta_heads": ["q", "o"],
        "memory_fusion_mode": "add",
        "memory_fusion_placement": "attention_output",
        "memory_fusion_residual_scale": 1.0,
        "memory_fusion_residual_scale_max": 1.0,
        "trainable_delta_scale": True,
        "delta_scale_init": 0.1,
        "delta_scale_max": 0.5,
        "delta_scale_granularity": "head",
        "delta_scale_parameterization": "alpha_over_rank",
        "output_init": "base_slice_fixed",
        "base_slice_ref_width": 8,
        "online_gain": 0.2,
        "rwkv_ms_num_states": 4,
        "rwkv_ms_chunk_size": 128,
        "rwkv_ms_semantics_version": 2,
    }


def _checkpoint_pairing(input_contract: dict[str, Any]) -> dict[str, Any]:
    counts = {
        "presence": 18,
        "same_cardinality_value": 10,
        "cross_cardinality_value": 4,
    }
    train = _self_hashed(
        {
            "sample_count": 32,
            "source_pair_manifest_path": input_contract["artifacts"][
                "train32_pair_manifest"
            ]["path"],
            "source_pair_manifest_file_sha256": input_contract["artifacts"][
                "train32_pair_manifest"
            ]["sha256"],
            "source_pair_manifest_sha256": input_contract["pairing"][
                "manifest_sha256"
            ],
            "source_entries_sha256": input_contract["pairing"]["entries_sha256"],
            "target_stratum_row_counts": counts,
        },
        "manifest_sha256",
    )
    return _self_hashed(
        {
            "schema_version": 2,
            "objective_version": gate.V8_OBJECTIVE["training_objective_version"],
            "target_stratum_row_counts": counts,
            "splits": {"train": train},
        },
        "manifest_sha256",
    )


def _protocol(
    step: int,
    input_contract: dict[str, Any],
    pairing: dict[str, Any],
) -> dict[str, Any]:
    artifacts = input_contract["artifacts"]
    return {
        "schema_version": 11,
        "memory_objective_version": gate.V8_OBJECTIVE["training_objective_version"],
        "memory_loss_mode": "scene_state_generation_ce",
        "train_file": artifacts["train32"]["path"],
        "train_samples": 32,
        "eval_samples": 0,
        "training_mode": "episode",
        "assistant_loss_mode": "final_assistant_only",
        "episode_recent_messages": 0,
        "episode_read_write_enabled": False,
        "per_device_train_batch_size": 1,
        "gradient_accumulation_steps": 1,
        "learning_rate": 2e-4,
        "lr_scheduler_type": "constant_with_warmup",
        "warmup_ratio": 0.0,
        "warmup_steps": 4,
        "save_steps": 14,
        "num_train_epochs": 1.0,
        "train_sampler_seed": None,
        "train_sampler_mode": "explicit_ordered_train_row_ordinal_v1",
        "max_steps": step,
        "memory_kl_weight": 0.0,
        "memory_base_kl_weight": 0.0,
        "memory_representation_weight": 0.0,
        "write_sparsity_weight": 0.0,
        "scene_generation_generated_unlikelihood_weight": 0.5,
        "scene_generation_generated_unlikelihood_mode": (
            "greedy_correct_state_edit_aligned_wrong_tokens_v2"
        ),
        "scene_generation_generated_unlikelihood_scope": (
            "same_and_cross_cardinality_value_rows_v1"
        ),
        "scene_generation_generated_unlikelihood_max_wrong_tokens": 4,
        "scene_generation_generated_rollout_extra_tokens": 4,
        "scene_generation_generated_rollout_max_tokens": 24,
        "scene_generation_generated_rollout_decoding": (
            "greedy_use_cache_true_exact_system_only_prompt_v1"
        ),
        "scene_generation_generated_replay_state_gradient": True,
        "scene_generation_generated_replay_read_path_gradient": True,
        "scene_generation_objective_formula": (
            "weighted_generation_ce(schema=2,decision=4,termination=1) + "
            "first_wrong_gold_prefix_top1_hinge(0.2) + "
            "correct_source_vs_donor_two_token_ce + "
            "donor_donor_vs_source_two_token_ce + "
            "correct_vs_detached_zero_decision_margin_hinge(0.2) + "
            "0.5 * correct_state_generated_prefix_unlikelihood"
        ),
        "scene_state_source_manifest": {
            "path": artifacts["v8_source_manifest"]["path"],
            "file_sha256": artifacts["v8_source_manifest"]["sha256"],
            "schema": gate.launch.SOURCE_SCHEMA,
            "train_file": artifacts["train32"]["path"],
            "train_file_sha256": artifacts["train32"]["sha256"],
            "train_rows": 32,
            "train_source_split": "train",
            "episode_contract": gate.v7.EPISODE_CONTRACT,
        },
        "train_schedule": {
            "schema": gate.launch.CURRICULUM_SCHEMA,
            "schedule_path": artifacts["v8_schedule"]["path"],
            "schedule_file_sha256": artifacts["v8_schedule"]["sha256"],
            "schedule_entries_sha256": gate.launch.SCHEDULE_ENTRIES_SHA256,
            "schedule_manifest_path": artifacts["v8_schedule_manifest"]["path"],
            "schedule_manifest_file_sha256": artifacts["v8_schedule_manifest"][
                "sha256"
            ],
            "schedule_manifest_sha256": gate.launch.SCHEDULE_MANIFEST_CANONICAL_SHA256,
            "ordered_train_row_ordinals_sha256": gate.launch.SCHEDULE_ORDINALS_SHA256,
            "total_steps": 152,
            "checkpoint_steps": list(gate.LOCKED_CHECKPOINT_STEPS),
            "value14_ordinals": list(gate.VALUE14_ORDINALS),
        },
        "scene_state_identity_pairing": {
            "manifest_sha256": pairing["manifest_sha256"],
            "target_stratum_row_counts": pairing["target_stratum_row_counts"],
        },
    }


def _build_checkpoint_chain(
    root: Path,
    input_contract: dict[str, Any],
    monkeypatch: pytest.MonkeyPatch,
) -> dict[int, Path]:
    source_root = root / "pinned-v7" / "checkpoint-256"
    source_root.mkdir(parents=True)
    source_artifact = source_root / "delta_mem_adapter.pt"
    source_artifact.write_bytes(b"pinned")
    warm_lock = {
        "lock_sha256": "1" * 64,
        "source_checkpoint": str(source_root.resolve()),
        "artifacts": {
            "delta_mem_adapter.pt": gate._artifact_binding(
                source_artifact,
                description="test pinned adapter",
            )
        },
    }
    monkeypatch.setattr(gate, "load_v8_warm_start_lock", lambda _path: warm_lock)
    checkpoints: dict[int, Path] = {}
    root_receipt_sha: str | None = None
    for step in gate.LOCKED_CHECKPOINT_STEPS[:4]:
        checkpoint = root / f"run-{step}" / "trainer" / f"checkpoint-{step}"
        checkpoint.mkdir(parents=True)
        pairing = _checkpoint_pairing(input_contract)
        protocol = _protocol(step, input_contract, pairing)
        _write_json(checkpoint / "delta_mem_config.json", _config())
        _write_json(checkpoint / "trainer_state.json", {"global_step": step, "max_steps": step})
        _write_json(checkpoint / "training_protocol.json", protocol)
        _write_json(checkpoint / "scene_state_identity_pairing_manifest.json", pairing)
        (checkpoint / "delta_mem_adapter.pt").write_bytes(f"adapter-{step}".encode())
        (checkpoint / "optimizer.pt").write_bytes(b"optimizer")
        (checkpoint / "scheduler.pt").write_bytes(b"scheduler")
        (checkpoint / "rng_state.pth").write_bytes(b"rng")
        protocol_sha = gate.canonical_sha256(protocol)
        if step == 14:
            lineage = {
                "schema": gate.WARM_START_RECEIPT_SCHEMA,
                "schema_version": 1,
                "mode": gate.WARM_START_MODE,
                "source_checkpoint": str(source_root.resolve()),
                "source_lock": {
                    "path": str(gate.WARM_START_LOCK.resolve()),
                    "lock_sha256": warm_lock["lock_sha256"],
                },
                "source_artifacts": warm_lock["artifacts"],
                "source_global_step": 256,
                "trainer_resume_from_checkpoint": None,
                "target_initial_global_step": 0,
                "pre_train_global_step": 0,
                "fresh_optimizer_created": True,
                "fresh_optimizer_state_entries_before_train": 0,
                "fresh_scheduler_created_before_train": False,
                "target_training_protocol_sha256": protocol_sha,
                "target_fresh_start": {
                    "initial_global_step": 0,
                    "optimizer_state": "fresh",
                    "scheduler_state": "fresh",
                    "trainer_state": "fresh",
                },
            }
            lineage = _self_hashed(lineage, "receipt_sha256")
            root_receipt_sha = lineage["receipt_sha256"]
            _write_json(checkpoint / "warm_start_lineage_manifest.json", lineage)
        else:
            source_step = gate.LOCKED_CHECKPOINT_STEPS[
                gate.LOCKED_CHECKPOINT_STEPS.index(step) - 1
            ]
            source = checkpoints[source_step]
            source_lineage_name = (
                "warm_start_lineage_manifest.json"
                if source_step == 14
                else "continuation_manifest.json"
            )
            source_protocol = json.loads(
                (source / "training_protocol.json").read_text(encoding="utf-8")
            )
            lineage = {
                "schema_version": 1,
                "mode": "extend",
                "source_checkpoint": str(source.resolve()),
                "source_global_step": source_step,
                "source_effective_max_steps": source_step,
                "source_max_steps": source_step,
                "target_max_steps": step,
                "source_lineage_filename": source_lineage_name,
                "source_lineage_file_sha256": gate.sha256_file(
                    source / source_lineage_name
                ),
                "source_training_protocol_sha256": gate.canonical_sha256(
                    source_protocol
                ),
                "target_training_protocol_sha256": protocol_sha,
                "root_warm_start_receipt_sha256": root_receipt_sha,
            }
            lineage = _self_hashed(lineage, "manifest_sha256")
            _write_json(checkpoint / "continuation_manifest.json", lineage)
        checkpoints[step] = checkpoint
    return checkpoints


def test_checkpoint_gate_rejects_pre56_missing_and_drifted_lineage(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    inputs = _input_contract(tmp_path)
    checkpoints = _build_checkpoint_chain(tmp_path, inputs, monkeypatch)

    validated = gate.validate_v8_checkpoint(checkpoints[56], input_contract=inputs)
    assert validated["global_step"] == 56
    assert [row["step"] for row in validated["lineage"]["chain"]] == [14, 28, 42, 56]
    with pytest.raises(gate.V8EvaluationContractError, match="unavailable before step56"):
        gate.validate_v8_checkpoint(checkpoints[42], input_contract=inputs)

    lineage_path = checkpoints[56] / "continuation_manifest.json"
    original = lineage_path.read_text(encoding="utf-8")
    lineage_path.unlink()
    with pytest.raises(gate.V8EvaluationContractError, match="lineage manifest"):
        gate.validate_v8_checkpoint(checkpoints[56], input_contract=inputs)
    lineage_path.write_text(original, encoding="utf-8")
    payload = json.loads(original)
    payload["source_training_protocol_sha256"] = "9" * 64
    _write_json(lineage_path, payload)
    with pytest.raises(gate.V8EvaluationContractError, match="manifest_sha256 differs"):
        gate.validate_v8_checkpoint(checkpoints[56], input_contract=inputs)


def _write_gate_outputs(
    output: Path,
    *,
    fingerprint: str,
    records: dict[str, list[dict[str, Any]]],
    result: dict[str, Any],
) -> None:
    output.mkdir(parents=True)
    manifest = {
        "schema": gate.GATE_MANIFEST_SCHEMA,
        "fingerprint": fingerprint,
        "fingerprint_payload": {"fixture": "v8"},
    }
    _write_json(output / "manifest.json", manifest)
    for condition, condition_records in records.items():
        (output / f"{condition}.jsonl").write_text(
            "".join(json.dumps(row, sort_keys=True) + "\n" for row in condition_records),
            encoding="utf-8",
        )
    summary = _self_hashed(
        {
            "schema": gate.GATE_SUMMARY_SCHEMA,
            "fingerprint": fingerprint,
            "gate": result,
        },
        "summary_sha256",
    )
    _write_json(output / "summary.json", summary)


def test_receipt_passes_exact_checkpoint_and_rejects_wrong_checkpoint(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    inputs = _input_contract(tmp_path)
    checkpoints = _build_checkpoint_chain(tmp_path, inputs, monkeypatch)
    checkpoint = gate.validate_v8_checkpoint(checkpoints[56], input_contract=inputs)
    fingerprint_payload = {"fixture": "v8"}
    fingerprint = gate.fingerprint_payload_sha256(fingerprint_payload)
    records = _records(fingerprint=fingerprint)
    result = gate.build_v8_gate(records_by_condition=records, pairing=inputs["pairing"])
    output = tmp_path / "eval"
    _write_gate_outputs(
        output,
        fingerprint=fingerprint,
        records=records,
        result=result,
    )
    receipt = gate.build_gate_receipt(
        output_dir=output,
        fingerprint=fingerprint,
        input_contract=inputs,
        checkpoint=checkpoint,
        gate=result,
    )
    receipt_path = output / "gate_receipt.json"
    gate.atomic_write_canonical_json(receipt_path, receipt)

    validated = gate.validate_gate_receipt_for_checkpoint(
        receipt_path,
        memory_dir=checkpoints[56],
        input_contract=inputs,
    )
    assert validated["hard32_authorization"]["scope"] == gate.HARD32_AUTHORIZATION_SCOPE
    assert validated["hard32_authorization"]["full170_authorized"] is False
    (checkpoints[56] / "delta_mem_adapter.pt").write_bytes(b"drifted-adapter")
    with pytest.raises(gate.V8EvaluationContractError, match="checkpoint binding differs"):
        gate.validate_gate_receipt_for_checkpoint(
            receipt_path,
            memory_dir=checkpoints[56],
            input_contract=inputs,
        )


def test_v7_or_failed_receipt_cannot_reach_hard32_validation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    receipt_path = tmp_path / "v7_receipt.json"
    _write_json(receipt_path, {"schema": gate.v7.RECEIPT_SCHEMA})
    calls = 0

    def forbidden_hard32(**_kwargs):
        nonlocal calls
        calls += 1
        raise AssertionError("Hard32 must not be accessed before V8 receipt validation")

    monkeypatch.setattr(
        state_eval,
        "validate_historical_v6_hard32_artifacts",
        forbidden_hard32,
    )
    with pytest.raises(gate.V8EvaluationContractError, match="receipt schema differs"):
        state_eval.validate_scene_v8_train32_hard32_authorization(
            receipt_path,
            memory_dir=tmp_path / "checkpoint-56",
            dataset_file=Path(state_eval.HISTORICAL_V6_OFFICIAL_VAL),
            selection_file=Path(state_eval.HISTORICAL_V6_HARD32_SELECTION),
        )
    assert calls == 0


def test_train_input_preflight_never_resolves_opens_or_hashes_hard32(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    protected = Path(state_eval.HISTORICAL_V6_HARD32_HOLDOUT)
    original_open = Path.open
    original_resolve = Path.resolve
    original_hash = gate.launch.sha256_file

    def guarded_open(path: Path, *args, **kwargs):
        if Path(path).absolute() == protected.absolute():
            raise AssertionError("pre-gate attempted to open Hard32")
        return original_open(path, *args, **kwargs)

    def guarded_resolve(path: Path, *args, **kwargs):
        if Path(path).absolute() == protected.absolute():
            raise AssertionError("pre-gate attempted to resolve Hard32")
        return original_resolve(path, *args, **kwargs)

    def guarded_hash(path: Path) -> str:
        if Path(path).absolute() == protected.absolute():
            raise AssertionError("pre-gate attempted to hash Hard32")
        return original_hash(path)

    monkeypatch.setattr(Path, "open", guarded_open)
    monkeypatch.setattr(Path, "resolve", guarded_resolve)
    monkeypatch.setattr(gate.launch, "sha256_file", guarded_hash)

    result = gate.validate_v8_train_inputs()
    assert result["value14_ordinals"] == list(gate.VALUE14_ORDINALS)

