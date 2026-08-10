from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
EXPERIMENTS = ROOT / "experiments" / "rethinking_rwkv_ms_gemma"


def canonical_sha256(payload: object) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def read_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def test_v5_preserves_scientific_contract_and_binds_accumulation() -> None:
    v4 = read_json(EXPERIMENTS / "natural_memory_replication_protocol_v4.json")
    v5 = read_json(EXPERIMENTS / "natural_memory_replication_protocol_v5.json")
    receipt = v5.pop("protocol_receipt")
    assert receipt == {
        "algorithm": "sha256",
        "payload_scope": "canonical_protocol_without_receipt",
        "payload_sha256": canonical_sha256(v5),
    }

    for field in (
        "baseline_proof",
        "immutable_benchmark_contract",
        "immutable_training_contract",
        "policy",
    ):
        assert v5[field] == v4[field]

    execution = v5["execution_contract"]
    for field, value in v4["execution_contract"].items():
        assert execution[field] == value
    assert execution["training_local_microbatch_size"] == 2
    assert execution["training_gradient_accumulation_steps"] == 2
    assert execution["training_local_batch_size"] == 4
    assert execution["training_global_batch_size"] == 16

    assert [
        (replication["id"], replication["split_seed"], replication["training_seed"])
        for replication in v5["replications"]
    ] == [("r10", 20260822, 51), ("r11", 20260823, 52)]


def test_v5_seed_selection_is_training_only_and_next_feasible() -> None:
    v5 = read_json(EXPERIMENTS / "natural_memory_replication_protocol_v5.json")
    screen_binding = v5["seed_feasibility_screen"]
    screen_path = EXPERIMENTS / screen_binding["file"]
    assert hashlib.sha256(screen_path.read_bytes()).hexdigest() == screen_binding[
        "file_sha256"
    ]
    screen = read_json(screen_path)
    receipt = screen.pop("receipt")
    assert receipt["payload_sha256"] == canonical_sha256(screen)
    assert receipt["payload_sha256"] == screen_binding["payload_sha256"]
    assert screen["screening_scope"] == {
        "allowed_inputs": "publisher_training_splits_only",
        "balance_limit": 0.03,
        "classification": "generator_feasibility_only",
        "development_rows_generated": 0,
        "episodes_generated": 0,
        "hard32_opened": False,
        "model_binding_opened": False,
        "native_validation_opened": False,
        "optimizer_updates": 0,
        "sealed_rows_generated": 0,
        "test_opened": False,
    }

    feasible = [row for row in screen["screened"] if row["feasible"]]
    selected = [
        row
        for row in feasible
        if row["seed"] not in screen_binding["previously_preregistered_seeds"]
    ][:2]
    assert [row["seed"] for row in selected] == screen_binding["selected_seeds"]
    assert {
        str(row["seed"]): row["component_assignment_sha256"] for row in selected
    } == screen_binding["selected_component_assignment_sha256"]


def test_v5_preselects_native_candidate_and_keeps_protected_splits_closed() -> None:
    v5 = read_json(EXPERIMENTS / "natural_memory_replication_protocol_v5.json")
    native = v5["native_validation_contract"]
    assert native["memory_candidate"] == "r10.adapter_files_sha256"
    assert native["selection_rule"] == (
        "r10_preselected_before_materialization_and_independent_of_development_or_"
        "sealed_metrics"
    )
    assert native["protected_splits_forbidden"] == ["test", "Hard32"]
    assert native["success_rule"] == (
        "memory_metric_gte_base_metric_on_all_three_tasks_and_memory_metric_gt_"
        "base_metric_on_at_least_two_tasks"
    )


def test_v5_capacity_gate_binds_optimizer_batch_and_microbatches() -> None:
    v5 = read_json(EXPERIMENTS / "natural_memory_replication_protocol_v5.json")
    capacity = v5["capacity_preflight"]
    assert capacity["required_local_batch_size"] == 4
    assert capacity["required_local_microbatch_size"] == 2
    assert capacity["required_gradient_accumulation_steps"] == 2
    assert capacity["failure_is_terminal_for_replication"] is True
    assert capacity["seed_substitution_forbidden"] is True
    assert capacity["predecessor_failure"] == {
        "completed_optimizer_steps": 3,
        "environment_adjusted_reserved_headroom_bytes": 110493696,
        "local_optimizer_batch_size": 4,
        "minimum_observed_free_after_phase_bytes": 513146880,
        "profile_receipt_payload_sha256": (
            "7f903e49de80833d42dc79a2fa961d1a3a054288dcea81e93a3ba799ad4bf2dd"
        ),
        "replication_id": "r8",
    }


def test_v5_source_hashes_bind_the_pre_authorization_runner() -> None:
    v5 = read_json(EXPERIMENTS / "natural_memory_replication_protocol_v5.json")
    assert v5["source_code_sha256"] == {
        "distributed": (
            "f1dffce7cebf47d3c3e864efca74a641fe221e1f903f8caea2e7badac0b6b9eb"
        ),
        "prepare": (
            "a737e9f5db35777c84227bfec30e73f24ae001b4a1c6173e020682b65b8892a9"
        ),
        "profile": (
            "50ef67e3e13ce6f726f0f1ec13e05a5e4755a192fad49c3f015f113070525563"
        ),
        "runner": (
            "85258bf99a8d51c5cae6a922eebd7099681b10e31b7cadcc7cb1d4265e88a54f"
        ),
    }


def test_v5_amendment_authorizes_only_the_current_runner() -> None:
    protocol = read_json(
        EXPERIMENTS / "natural_memory_replication_protocol_v5.json"
    )
    amendment = read_json(
        EXPERIMENTS
        / "natural_memory_replication_protocol_v5_amendment_runner_authorization.json"
    )
    receipt = amendment.pop("amendment_receipt")
    assert receipt == {
        "algorithm": "sha256",
        "payload_scope": "canonical_amendment_without_receipt",
        "payload_sha256": canonical_sha256(amendment),
    }
    runner_path = EXPERIMENTS / "run_natural_memory_gate.py"
    assert amendment["runner_change"] == {
        "new_sha256": hashlib.sha256(runner_path.read_bytes()).hexdigest(),
        "old_sha256": protocol["source_code_sha256"]["runner"],
    }
    assert amendment["authorized_replication_ids"] == [
        replication["id"] for replication in protocol["replications"]
    ]
    assert amendment["scope"] == {
        "classification": "infrastructure_only",
        "data_changed": False,
        "gate_changed": False,
        "hyperparameters_changed": False,
        "training_math_changed": False,
    }
