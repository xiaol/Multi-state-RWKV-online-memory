from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
EXPERIMENTS = ROOT / "experiments" / "rethinking_rwkv_ms_gemma"
AUTHORIZATION = EXPERIMENTS / "natural_memory_native_validation_v6_authorization.json"


def canonical_sha256(payload: object) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def read_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def test_native_v6_authorization_receipt_and_preselected_candidate() -> None:
    authorization = read_json(AUTHORIZATION)
    receipt = authorization.pop("receipt")
    assert receipt == {
        "algorithm": "sha256",
        "payload_scope": "canonical_authorization_without_receipt",
        "payload_sha256": canonical_sha256(authorization),
    }
    protocol = read_json(EXPERIMENTS / authorization["protocol"]["file"])
    assert authorization["candidate"]["id"] == "r12"
    assert protocol["native_validation_contract"]["memory_candidate"] == (
        "r12.adapter_files_sha256"
    )
    development_receipt = read_json(
        EXPERIMENTS
        / "local_artifacts/natural_memory_gate_replication_r12_development_run_split20260825_seed53/run_receipt.json"
    )
    assert authorization["candidate"]["adapter_files_aggregate_sha256"] == (
        development_receipt["adapter_files_sha256"]
    )
    adapter_dir = EXPERIMENTS / authorization["candidate"]["adapter_path"]
    assert hashlib.sha256((adapter_dir / "delta_mem_adapter.pt").read_bytes()).hexdigest() == (
        authorization["candidate"]["adapter_file_sha256"]
    )
    assert hashlib.sha256((adapter_dir / "delta_mem_config.json").read_bytes()).hexdigest() == (
        authorization["candidate"]["config_sha256"]
    )


def test_native_v6_authorization_requires_two_sealed_passes() -> None:
    authorization = read_json(AUTHORIZATION)
    paths = {
        "r12": {
            "development": EXPERIMENTS
            / "local_artifacts/natural_memory_gate_replication_r12_development_run_split20260825_seed53/run_receipt.json",
            "sealed": EXPERIMENTS
            / "local_artifacts/natural_memory_gate_replication_r12_sealed_run_split20260825_seed53/run_receipt.json",
        },
        "r13": {
            "development": EXPERIMENTS
            / "local_artifacts/natural_memory_gate_replication_r13_development_run_split20260826_seed54/run_receipt.json",
            "sealed": EXPERIMENTS
            / "local_artifacts/natural_memory_gate_replication_r13_sealed_run_split20260826_seed54/run_receipt.json",
        },
    }
    for replication_id, receipts in paths.items():
        binding = authorization["replication_unlock"][replication_id]
        for profile, expected_checks in (("development", 54), ("sealed", 52)):
            path = receipts[profile]
            run_receipt = read_json(path)
            assert hashlib.sha256(path.read_bytes()).hexdigest() == binding[
                f"{profile}_run_receipt_file_sha256"
            ]
            assert run_receipt["run_receipt_sha256"] == binding[
                f"{profile}_run_receipt_sha256"
            ]
            assert run_receipt["gate_passed"] is True
            assert len(run_receipt["gate"]["checks"]) == expected_checks
            assert run_receipt["gate"]["failed_checks"] == []
        assert binding["sealed_attempts"] == 1


def test_native_v6_authorization_binds_validation_data_code_and_policy() -> None:
    authorization = read_json(AUTHORIZATION)
    source_files = {
        "runner_sha256": EXPERIMENTS / "run_novel_agent_eval.py",
        "analyzer_sha256": EXPERIMENTS / "analyze_novel_agent_eval.py",
        "sharder_sha256": EXPERIMENTS / "shard_novel_agent_eval.py",
    }
    for field, path in source_files.items():
        assert hashlib.sha256(path.read_bytes()).hexdigest() == authorization[
            "code_bindings"
        ][field]
    for task in authorization["evaluation"]["tasks"].values():
        path = Path(task["dataset_path"])
        assert hashlib.sha256(path.read_bytes()).hexdigest() == task["dataset_sha256"]
    assert authorization["evaluation"]["publisher_split_alias"] == "val"
    assert authorization["protected_evaluation"] == {
        "hard32_forbidden": True,
        "hard32_opened": False,
        "test_forbidden": True,
        "test_opened": False,
    }
    assert authorization["success_rule"] == {
        "definition": (
            "normal_gte_base_on_all_three_tasks_and_normal_gt_base_on_at_least_"
            "two_tasks"
        ),
        "minimum_strictly_better_tasks": 2,
        "no_regression_required_tasks": [
            "attribution-v3.2",
            "narrative-v3.2",
            "scene-v4-current",
        ],
    }
