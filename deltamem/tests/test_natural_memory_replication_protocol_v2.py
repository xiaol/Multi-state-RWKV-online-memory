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


def test_v2_preserves_frozen_contract_and_binds_feasible_seeds() -> None:
    v1 = json.loads(
        (EXPERIMENTS / "natural_memory_replication_protocol_v1.json").read_text(
            encoding="utf-8"
        )
    )
    v2 = json.loads(
        (EXPERIMENTS / "natural_memory_replication_protocol_v2.json").read_text(
            encoding="utf-8"
        )
    )
    receipt = v2.pop("protocol_receipt")
    assert receipt == {
        "algorithm": "sha256",
        "payload_scope": "canonical_protocol_without_receipt",
        "payload_sha256": canonical_sha256(v2),
    }
    for field in (
        "baseline_proof",
        "execution_contract",
        "immutable_benchmark_contract",
        "immutable_training_contract",
    ):
        assert v2[field] == v1[field]
    assert [
        (replication["id"], replication["split_seed"], replication["training_seed"])
        for replication in v2["replications"]
    ] == [("r4", 20260813, 45), ("r5", 20260814, 46)]
    assert v2["seed_feasibility_screen"]["selected_seeds"] == [
        replication["split_seed"] for replication in v2["replications"]
    ]
    assert (
        v2["native_validation_contract"]["selection_rule"]
        == "r4_preselected_before_materialization_and_independent_of_development_or_sealed_metrics"
    )


def test_v2_amendment_authorizes_only_the_current_runner() -> None:
    protocol = json.loads(
        (EXPERIMENTS / "natural_memory_replication_protocol_v2.json").read_text(
            encoding="utf-8"
        )
    )
    amendment = json.loads(
        (
            EXPERIMENTS
            / "natural_memory_replication_protocol_v2_amendment_runner_authorization.json"
        ).read_text(encoding="utf-8")
    )
    receipt = amendment.pop("amendment_receipt")
    assert receipt["payload_sha256"] == canonical_sha256(amendment)
    runner_path = EXPERIMENTS / "run_natural_memory_gate.py"
    assert amendment["runner_change"]["new_sha256"] == hashlib.sha256(
        runner_path.read_bytes()
    ).hexdigest()
    assert amendment["authorized_replication_ids"] == [
        replication["id"] for replication in protocol["replications"]
    ]
