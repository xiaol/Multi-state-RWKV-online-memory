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


def test_v6_receipt_and_family_balanced_contract() -> None:
    v5 = read_json(EXPERIMENTS / "natural_memory_replication_protocol_v5.json")
    v6 = read_json(EXPERIMENTS / "natural_memory_replication_protocol_v6.json")
    receipt = v6.pop("protocol_receipt")
    assert receipt == {
        "algorithm": "sha256",
        "payload_scope": "canonical_protocol_without_receipt",
        "payload_sha256": canonical_sha256(v6),
    }

    assert v6["baseline_proof"] == v5["baseline_proof"]
    assert v6["immutable_benchmark_contract"] == v5["immutable_benchmark_contract"]
    for field in (
        "answer_exact_min",
        "answer_weight",
        "attn_implementation",
        "batch_size",
        "dtype",
        "epochs",
        "eval_batch_size",
        "greedy_answer_evaluation",
        "hard_negative_margin",
        "hard_negative_weight",
        "key_dim",
        "learning_rate",
        "max_grad_norm",
        "max_steps",
        "rank",
        "rewrite_output_change_min",
        "route_accuracy_min",
        "route_weight",
        "target_layers",
        "temperature",
        "training_conditions",
    ):
        assert v6["immutable_training_contract"][field] == v5[
            "immutable_training_contract"
        ][field]
    assert {
        field: v6["immutable_training_contract"][field]
        for field in (
            "family_size",
            "global_families_per_update",
            "local_family_per_rank",
        )
    } == {
        "family_size": 4,
        "global_families_per_update": 4,
        "local_family_per_rank": 1,
    }
    assert v6["development_intervention"]["preserved_fields"] == [
        "architecture",
        "five_state_ce_objective",
        "optimizer",
        "optimizer_batch",
        "epochs",
        "updates",
        "thresholds",
        "benchmark_generator",
    ]


def test_v6_binds_passing_opened_development_ablation() -> None:
    v6 = read_json(EXPERIMENTS / "natural_memory_replication_protocol_v6.json")
    intervention = v6["development_intervention"]
    plan_path = EXPERIMENTS / intervention["plan_file"]
    assert hashlib.sha256(plan_path.read_bytes()).hexdigest() == intervention[
        "plan_file_sha256"
    ]
    plan = read_json(plan_path)
    receipt = plan.pop("plan_receipt")
    assert receipt["payload_sha256"] == canonical_sha256(plan)
    assert receipt["payload_sha256"] == intervention["plan_payload_sha256"]
    assert intervention["plan_git_commit"] == (
        "862eb7d8d5ae16b72a5360fec73a731a674b0b23"
    )
    assert intervention["ablation_gate_passed"] is True
    assert intervention["family_schedule"] == {
        "global_families_per_update": 4,
        "global_query_slot_counts_per_update": [4, 4, 4, 4],
        "local_family_per_rank": 1,
        "local_query_slots_per_update": [0, 1, 2, 3],
    }


def test_v6_seed_selection_is_training_only_and_next_feasible() -> None:
    v6 = read_json(EXPERIMENTS / "natural_memory_replication_protocol_v6.json")
    screen_binding = v6["seed_feasibility_screen"]
    screen_path = EXPERIMENTS / screen_binding["file"]
    assert hashlib.sha256(screen_path.read_bytes()).hexdigest() == screen_binding[
        "file_sha256"
    ]
    screen = read_json(screen_path)
    receipt = screen.pop("receipt")
    assert receipt["payload_sha256"] == canonical_sha256(screen)
    assert receipt["payload_sha256"] == screen_binding["payload_sha256"]
    assert screen["screening_scope"]["allowed_inputs"] == (
        "publisher_training_splits_only"
    )
    assert screen["screening_scope"]["native_validation_opened"] is False
    assert screen["screening_scope"]["sealed_rows_generated"] == 0
    assert screen["screening_scope"]["test_opened"] is False

    feasible = [row for row in screen["screened"] if row["feasible"]]
    selected = [
        row
        for row in feasible
        if row["seed"] not in screen_binding["previously_preregistered_seeds"]
    ][:2]
    assert [row["seed"] for row in selected] == [20260825, 20260826]
    assert {
        str(row["seed"]): row["component_assignment_sha256"] for row in selected
    } == screen_binding["selected_component_assignment_sha256"]
    assert [
        (replication["id"], replication["split_seed"], replication["training_seed"])
        for replication in v6["replications"]
    ] == [("r12", 20260825, 53), ("r13", 20260826, 54)]


def test_v6_preselects_native_candidate_and_keeps_protected_splits_closed() -> None:
    v6 = read_json(EXPERIMENTS / "natural_memory_replication_protocol_v6.json")
    native = v6["native_validation_contract"]
    assert native["memory_candidate"] == "r12.adapter_files_sha256"
    assert native["selection_rule"] == (
        "r12_preselected_before_materialization_and_independent_of_development_or_"
        "sealed_metrics"
    )
    assert native["protected_splits_forbidden"] == ["test", "Hard32"]
    assert native["success_rule"] == (
        "memory_metric_gte_base_metric_on_all_three_tasks_and_memory_metric_gt_"
        "base_metric_on_at_least_two_tasks"
    )


def test_v6_source_hashes_bind_pre_authorization_sources() -> None:
    v6 = read_json(EXPERIMENTS / "natural_memory_replication_protocol_v6.json")
    assert v6["source_code_sha256"] == {
        "distributed": (
            "09fd08b4750469c1364c28a935b443895f988f17ccc8102ede3ecce6bed6f44d"
        ),
        "prepare": (
            "a737e9f5db35777c84227bfec30e73f24ae001b4a1c6173e020682b65b8892a9"
        ),
        "profile": (
            "50ef67e3e13ce6f726f0f1ec13e05a5e4755a192fad49c3f015f113070525563"
        ),
        "runner": (
            "5de6b363a6621c9ad960e491ae9575192bddb2165892b1705340e226b8d73fbf"
        ),
    }


def test_v6_predecessor_is_terminal_v5_without_sealed_access() -> None:
    v6 = read_json(EXPERIMENTS / "natural_memory_replication_protocol_v6.json")
    predecessor = v6["predecessor_protocol"]
    outcome_path = EXPERIMENTS / predecessor["outcome_file"]
    assert hashlib.sha256(outcome_path.read_bytes()).hexdigest() == predecessor[
        "outcome_file_sha256"
    ]
    outcome = read_json(outcome_path)
    receipt = outcome.pop("outcome_receipt")
    assert receipt["payload_sha256"] == canonical_sha256(outcome)
    assert receipt["payload_sha256"] == predecessor["outcome_payload_sha256"]
    assert predecessor["outcome"] == "terminal_protocol_failure"
    assert outcome["protected_evaluation"] == {
        "hard32_opened": False,
        "sealed_r10_opened": False,
        "sealed_r11_opened": False,
        "test_opened": False,
    }
