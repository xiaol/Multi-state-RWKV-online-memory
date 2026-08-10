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


def test_v4_preserves_frozen_contract_and_binds_next_untouched_seeds() -> None:
    v3 = read_json(EXPERIMENTS / "natural_memory_replication_protocol_v3.json")
    v4 = read_json(EXPERIMENTS / "natural_memory_replication_protocol_v4.json")
    receipt = v4.pop("protocol_receipt")
    assert receipt == {
        "algorithm": "sha256",
        "payload_scope": "canonical_protocol_without_receipt",
        "payload_sha256": canonical_sha256(v4),
    }

    for field in (
        "baseline_proof",
        "execution_contract",
        "immutable_benchmark_contract",
        "immutable_training_contract",
        "policy",
    ):
        assert v4[field] == v3[field]

    assert [
        (replication["id"], replication["split_seed"], replication["training_seed"])
        for replication in v4["replications"]
    ] == [("r8", 20260817, 49), ("r9", 20260818, 50)]
    assert v4["policy"]["sealed_attempts_per_replication"] == 1
    assert v4["policy"]["sealed_gate"] == "all_52_named_checks_true"


def test_v4_seed_selection_is_derived_only_from_the_preexisting_screen() -> None:
    v4 = read_json(EXPERIMENTS / "natural_memory_replication_protocol_v4.json")
    screen_binding = v4["seed_feasibility_screen"]
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
    unused = [
        row
        for row in feasible
        if row["seed"] not in screen_binding["previously_preregistered_seeds"]
    ]
    selected = unused[:2]
    assert [row["seed"] for row in selected] == screen_binding["selected_seeds"]
    assert {
        str(row["seed"]): row["component_assignment_sha256"] for row in selected
    } == screen_binding["selected_component_assignment_sha256"]


def test_v4_preselects_native_candidate_and_keeps_protected_splits_closed() -> None:
    v4 = read_json(EXPERIMENTS / "natural_memory_replication_protocol_v4.json")
    native = v4["native_validation_contract"]
    assert native["memory_candidate"] == "r8.adapter_files_sha256"
    assert native["selection_rule"] == (
        "r8_preselected_before_materialization_and_independent_of_development_or_"
        "sealed_metrics"
    )
    assert native["protected_splits_forbidden"] == ["test", "Hard32"]
    assert native["success_rule"] == (
        "memory_metric_gte_base_metric_on_all_three_tasks_and_memory_metric_gt_"
        "base_metric_on_at_least_two_tasks"
    )


def test_v4_keeps_required_local_batch_four_as_a_terminal_preflight() -> None:
    v4 = read_json(EXPERIMENTS / "natural_memory_replication_protocol_v4.json")
    capacity = v4["capacity_preflight"]
    assert capacity["required_local_batch_size"] == 4
    assert capacity["failure_is_terminal_for_replication"] is True
    assert capacity["seed_substitution_forbidden"] is True
    assert capacity["predecessor_failure"] == {
        "completed_optimizer_steps": 3,
        "environment_adjusted_reserved_headroom_bytes": 135659520,
        "local_batch_size": 4,
        "minimum_observed_free_after_phase_bytes": 34996224,
        "profile_receipt_payload_sha256": (
            "4ba56eb1013978fc725f59edae818f81272a36e66e248cf75a7160f7d16d38ad"
        ),
        "replication_id": "r6",
    }


def test_v4_amendment_authorizes_only_the_current_runner() -> None:
    protocol = read_json(
        EXPERIMENTS / "natural_memory_replication_protocol_v4.json"
    )
    amendment = read_json(
        EXPERIMENTS
        / "natural_memory_replication_protocol_v4_amendment_runner_authorization.json"
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
