from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
from typing import Any

import pytest

from experiments.rethinking_rwkv_ms_gemma import run_natural_memory_gate as runner


def _write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _production_training_dataset_audit() -> dict[str, Any]:
    conditions = list(runner.SUPERVISED_COMPOSITIONAL_TRAINING_CONDITIONS)
    tasks = list(runner.PRODUCTION_TASKS)
    rows_per_condition_task = {
        condition: {
            task: runner.PRODUCTION_ROWS_PER_CONDITION_TASK for task in tasks
        }
        for condition in conditions
    }
    audit: dict[str, Any] = {
        "schema": runner.TRAINING_DATASET_AUDIT_SCHEMA,
        "training_conditions": conditions,
        "tasks": tasks,
        "rows": runner.PRODUCTION_TRAINING_ROWS,
        "unique_row_ids": True,
        "row_id_policy": runner.TRAINING_ROW_ID_POLICY,
        "row_id_policy_passed": True,
        "sampling_policy": runner.TRAINING_SAMPLING_POLICY,
        "payload_digest_policy": runner.TRAINING_PAYLOAD_DIGEST_POLICY,
        "family_invariant_policy": runner.TRAINING_FAMILY_INVARIANT_POLICY,
        "condition_set_exact": True,
        "condition_task_strata_exact": True,
        "condition_task_strata_balanced": True,
        "rows_per_condition_task": rows_per_condition_task,
        "answer_tokens_per_condition_task": rows_per_condition_task,
        "rows_by_condition": {
            condition: runner.PRODUCTION_ROWS_PER_CONDITION_TASK * len(tasks)
            for condition in conditions
        },
        "rows_by_task": {
            task: runner.PRODUCTION_ROWS_PER_CONDITION_TASK * len(conditions)
            for task in tasks
        },
        "source_query_condition_families": (
            runner.PRODUCTION_ROWS_PER_CONDITION_TASK * len(tasks)
        ),
        "complete_source_query_condition_families": (
            runner.PRODUCTION_ROWS_PER_CONDITION_TASK * len(tasks)
        ),
        "paired_condition_coverage": True,
        "family_invariants_passed": True,
        "family_invariant_failure_count": 0,
        "training_row_id_set_sha256": "a" * 64,
        "ordered_training_examples_sha256": "b" * 64,
        "passed": True,
    }
    return runner.bind_production_training_contract(
        audit,
        epochs=runner.PRODUCTION_EPOCHS,
        global_batch_size=runner.distributed.REQUIRED_GLOBAL_BATCH_SIZE,
        requested_max_steps=runner.PRODUCTION_UPDATES,
        schedule_mode="complete",
    )


def _literal_legacy_training_dataset_audit() -> dict[str, Any]:
    audit = _production_training_dataset_audit()
    audit["schedule_contract"] = {
        "epochs": 8,
        "global_batch_size": 4,
        "rows_divide_global_batch": True,
        "complete_epoch_updates": 3840,
        "requested_max_steps": 3840,
        "complete_epoch_schedule_requested": True,
    }
    audit["schedule_mode"] = "complete"
    audit["schedule_contract_checks"] = {
        "schedule_mode": True,
        "epochs_exact": True,
        "global_batch_exact": True,
        "rows_divide_global_batch": True,
        "complete_epoch_updates_exact": True,
        "requested_max_steps_exact": True,
        "complete_epoch_schedule_requested": True,
    }
    audit["schedule_contract_passed"] = True
    audit["production_contract_checks"] = {
        "dataset": True,
        "schedule": True,
    }
    audit["production_contract_passed"] = True
    return audit


def _passing_gate() -> dict[str, Any]:
    return {
        "schema": runner.ACCEPTANCE_SCHEMA,
        "thresholds": {
            "answer_exact_min": 0.8,
            "route_accuracy_min": 0.95,
            "rewrite_output_change_min": 0.8,
        },
        "checks": {"retained_evaluation": True},
        "failed_checks": [],
        "passed": True,
    }


def _write_chain(tmp_path: Path) -> dict[str, Any]:
    run_dir = tmp_path / "development-run"
    adapter_dir = tmp_path / "adapter"
    run_dir.mkdir()
    adapter_dir.mkdir()
    (adapter_dir / "delta_mem_adapter.pt").write_bytes(b"sealed-adapter")

    audit = _production_training_dataset_audit()
    gate = _passing_gate()
    protocol = {
        "schema": runner.PROTOCOL_SCHEMA,
        "runner_schema": runner.RUN_SCHEMA,
        "profile": "development",
        "training_dataset_audit": deepcopy(audit),
    }
    training = {
        "schema": runner.TRAINING_CONFIGURATION_SCHEMA,
        "profile": "development",
        "training_dataset_audit": deepcopy(audit),
    }
    evaluation = {
        "schema": runner.EVALUATION_SCHEMA,
        "profile": "development",
        "training": {"training_dataset_audit": deepcopy(audit)},
        "gate": deepcopy(gate),
    }
    _write_json(run_dir / "protocol.json", protocol)
    _write_json(run_dir / "training_configuration.json", training)
    _write_json(run_dir / "evaluation.json", evaluation)

    adapter_files = runner.snapshot_directory_files(adapter_dir)
    manifest_hash = "c" * 64
    benchmark_hash = "d" * 64
    receipt = {
        "schema": runner.RUN_SCHEMA,
        "profile": "development",
        "gate_passed": True,
        "source_manifest_payload_sha256": manifest_hash,
        "protocol_sha256": runner._sha256_json(protocol),
        "training_configuration_sha256": runner._sha256_json(training),
        "training_dataset_audit_sha256": runner._sha256_json(audit),
        "evaluation_sha256": runner._sha256_json(evaluation),
        "adapter_files": adapter_files,
        "adapter_files_sha256": runner._sha256_json(adapter_files),
        "gate": deepcopy(gate),
    }
    receipt = runner._signed_payload(receipt, "run_receipt_sha256")
    _write_json(run_dir / "run_receipt.json", receipt)

    lock = {
        "schema": runner.source.SEALED_LOCK_SCHEMA,
        "configuration_frozen": True,
        "development_gate_passed": True,
        "benchmark_contract_sha256": benchmark_hash,
        "development_manifest_payload_sha256": manifest_hash,
        "runner_protocol_sha256": receipt["protocol_sha256"],
        "training_configuration_sha256": receipt[
            "training_configuration_sha256"
        ],
        "training_dataset_audit_sha256": receipt[
            "training_dataset_audit_sha256"
        ],
        "evaluation_sha256": receipt["evaluation_sha256"],
        "development_run_receipt_sha256": receipt["run_receipt_sha256"],
        "adapter_files_sha256": receipt["adapter_files_sha256"],
    }
    sealed_manifest = {
        "benchmark_contract_sha256": benchmark_hash,
        "sealed_lock": {
            "receipt": lock,
            "receipt_sha256": runner._sha256_json(lock),
        },
    }
    return {
        "run_dir": run_dir,
        "adapter_dir": adapter_dir,
        "sealed_manifest": sealed_manifest,
    }


def _rebind_chain(chain: dict[str, Any], *, bind_adapter: bool = True) -> None:
    run_dir = chain["run_dir"]
    protocol = _read_json(run_dir / "protocol.json")
    training = _read_json(run_dir / "training_configuration.json")
    evaluation = _read_json(run_dir / "evaluation.json")
    audit = training["training_dataset_audit"]
    receipt_path = run_dir / "run_receipt.json"
    receipt = _read_json(receipt_path)
    receipt.pop("run_receipt_sha256", None)
    receipt["protocol_sha256"] = runner._sha256_json(protocol)
    receipt["training_configuration_sha256"] = runner._sha256_json(training)
    receipt["training_dataset_audit_sha256"] = runner._sha256_json(audit)
    receipt["evaluation_sha256"] = runner._sha256_json(evaluation)
    if bind_adapter:
        adapter_files = runner.snapshot_directory_files(chain["adapter_dir"])
        receipt["adapter_files"] = adapter_files
        receipt["adapter_files_sha256"] = runner._sha256_json(adapter_files)
    receipt = runner._signed_payload(receipt, "run_receipt_sha256")
    _write_json(receipt_path, receipt)

    sealed_lock = chain["sealed_manifest"]["sealed_lock"]
    lock = sealed_lock["receipt"]
    lock["runner_protocol_sha256"] = receipt["protocol_sha256"]
    lock["training_configuration_sha256"] = receipt[
        "training_configuration_sha256"
    ]
    lock["training_dataset_audit_sha256"] = receipt[
        "training_dataset_audit_sha256"
    ]
    lock["evaluation_sha256"] = receipt["evaluation_sha256"]
    lock["development_run_receipt_sha256"] = receipt["run_receipt_sha256"]
    if bind_adapter:
        lock["adapter_files_sha256"] = receipt["adapter_files_sha256"]
    sealed_lock["receipt_sha256"] = runner._sha256_json(lock)


def _validate(chain: dict[str, Any]) -> dict[str, Any]:
    return runner.validate_sealed_lock_chain(
        chain["sealed_manifest"],
        chain["run_dir"],
        adapter_path=chain["adapter_dir"],
    )


def test_sealed_lock_chain_accepts_complete_retained_evidence(tmp_path: Path) -> None:
    chain = _write_chain(tmp_path)

    result = _validate(chain)

    assert result["passed"] is True
    assert result["evaluation_sha256"] == runner._sha256_json(
        _read_json(chain["run_dir"] / "evaluation.json")
    )
    assert result["adapter_files_sha256"] == runner._sha256_json(
        runner.snapshot_directory_files(chain["adapter_dir"])
    )


def test_sealed_lock_chain_accepts_literal_legacy_batch_four_schedule(
    tmp_path: Path,
) -> None:
    chain = _write_chain(tmp_path)
    legacy_audit = _literal_legacy_training_dataset_audit()
    protocol_path = chain["run_dir"] / "protocol.json"
    protocol = _read_json(protocol_path)
    protocol["training_dataset_audit"] = deepcopy(legacy_audit)
    _write_json(protocol_path, protocol)
    training_path = chain["run_dir"] / "training_configuration.json"
    training = _read_json(training_path)
    training["training_dataset_audit"] = deepcopy(legacy_audit)
    _write_json(training_path, training)
    evaluation_path = chain["run_dir"] / "evaluation.json"
    evaluation = _read_json(evaluation_path)
    evaluation["training"]["training_dataset_audit"] = deepcopy(legacy_audit)
    _write_json(evaluation_path, evaluation)
    _rebind_chain(chain)

    assert runner.validate_production_training_contract(
        legacy_audit,
        schedule_mode="complete",
    ) is False
    assert runner.validate_retained_production_training_contract(
        legacy_audit,
        schedule_mode="complete",
    ) is True
    assert _validate(chain)["passed"] is True


def test_retained_schedule_allowlist_rejects_unrecognized_batch_geometry() -> None:
    unsupported = _literal_legacy_training_dataset_audit()
    unsupported["schedule_contract"].update(
        {
            "global_batch_size": 8,
            "complete_epoch_updates": 1920,
            "requested_max_steps": 1920,
        }
    )

    assert runner.validate_retained_production_training_contract(
        unsupported,
        schedule_mode="complete",
    ) is False


def test_sealed_lock_chain_requires_retained_evaluation(tmp_path: Path) -> None:
    chain = _write_chain(tmp_path)
    (chain["run_dir"] / "evaluation.json").unlink()

    with pytest.raises(ValueError, match="artifact is missing.*evaluation.json"):
        _validate(chain)


def test_sealed_lock_chain_rejects_incompatible_evaluation_schema(
    tmp_path: Path,
) -> None:
    chain = _write_chain(tmp_path)
    path = chain["run_dir"] / "evaluation.json"
    evaluation = _read_json(path)
    evaluation["schema"] = "incompatible"
    _write_json(path, evaluation)
    _rebind_chain(chain)

    with pytest.raises(ValueError, match="incompatible protocol schema"):
        _validate(chain)


def test_sealed_lock_chain_rejects_non_development_evaluation(tmp_path: Path) -> None:
    chain = _write_chain(tmp_path)
    path = chain["run_dir"] / "evaluation.json"
    evaluation = _read_json(path)
    evaluation["profile"] = "sealed_validation"
    _write_json(path, evaluation)
    _rebind_chain(chain)

    with pytest.raises(ValueError, match="do not share the development profile"):
        _validate(chain)


def test_sealed_lock_chain_rejects_mismatched_audit_copies(tmp_path: Path) -> None:
    chain = _write_chain(tmp_path)
    path = chain["run_dir"] / "evaluation.json"
    evaluation = _read_json(path)
    evaluation["training"]["training_dataset_audit"]["rows"] += 1
    _write_json(path, evaluation)
    _rebind_chain(chain)

    with pytest.raises(ValueError, match="different training dataset audits"):
        _validate(chain)


def test_sealed_lock_chain_rejects_consistently_rehashed_invalid_audit(
    tmp_path: Path,
) -> None:
    chain = _write_chain(tmp_path)
    for name, location in (
        ("protocol.json", ("training_dataset_audit",)),
        ("training_configuration.json", ("training_dataset_audit",)),
        ("evaluation.json", ("training", "training_dataset_audit")),
    ):
        path = chain["run_dir"] / name
        artifact = _read_json(path)
        audit = artifact
        for key in location:
            audit = audit[key]
        audit["production_contract_passed"] = False
        _write_json(path, artifact)
    _rebind_chain(chain)

    with pytest.raises(ValueError, match="audit is not production-valid"):
        _validate(chain)


def test_sealed_lock_chain_rejects_gate_copy_mismatch(tmp_path: Path) -> None:
    chain = _write_chain(tmp_path)
    path = chain["run_dir"] / "evaluation.json"
    evaluation = _read_json(path)
    evaluation["gate"]["checks"]["extra"] = True
    _write_json(path, evaluation)
    _rebind_chain(chain)

    with pytest.raises(ValueError, match="receipt gate differs"):
        _validate(chain)


@pytest.mark.parametrize(
    "gate_update",
    [
        {"passed": False, "failed_checks": ["retained_evaluation"]},
        {"checks": {}},
        {"checks": {"retained_evaluation": 1}},
        {"schema": "incompatible"},
    ],
)
def test_sealed_lock_chain_rejects_structurally_false_gate(
    tmp_path: Path,
    gate_update: dict[str, Any],
) -> None:
    chain = _write_chain(tmp_path)
    evaluation_path = chain["run_dir"] / "evaluation.json"
    receipt_path = chain["run_dir"] / "run_receipt.json"
    evaluation = _read_json(evaluation_path)
    receipt = _read_json(receipt_path)
    gate = deepcopy(evaluation["gate"])
    gate.update(gate_update)
    evaluation["gate"] = gate
    receipt["gate"] = deepcopy(gate)
    _write_json(evaluation_path, evaluation)
    _write_json(receipt_path, receipt)
    _rebind_chain(chain)

    with pytest.raises(ValueError, match="does not prove a passing gate"):
        _validate(chain)


def test_sealed_lock_chain_rejects_unbound_evaluation_hash(tmp_path: Path) -> None:
    chain = _write_chain(tmp_path)
    path = chain["run_dir"] / "evaluation.json"
    evaluation = _read_json(path)
    evaluation["retained_note"] = "tampered"
    _write_json(path, evaluation)

    with pytest.raises(ValueError, match="evaluation_sha256"):
        _validate(chain)


def test_sealed_lock_chain_rejects_receipt_evaluation_hash_mismatch(
    tmp_path: Path,
) -> None:
    chain = _write_chain(tmp_path)
    path = chain["run_dir"] / "evaluation.json"
    evaluation = _read_json(path)
    evaluation["retained_note"] = "tampered"
    _write_json(path, evaluation)
    sealed_lock = chain["sealed_manifest"]["sealed_lock"]
    sealed_lock["receipt"]["evaluation_sha256"] = runner._sha256_json(evaluation)
    sealed_lock["receipt_sha256"] = runner._sha256_json(sealed_lock["receipt"])

    with pytest.raises(ValueError, match="receipt does not bind its protocol files"):
        _validate(chain)


def test_sealed_lock_chain_rejects_bad_receipt_signature(tmp_path: Path) -> None:
    chain = _write_chain(tmp_path)
    path = chain["run_dir"] / "run_receipt.json"
    receipt = _read_json(path)
    receipt["unbound_note"] = "tampered"
    _write_json(path, receipt)

    with pytest.raises(ValueError, match="receipt signature is invalid"):
        _validate(chain)


def test_sealed_lock_chain_rejects_unbound_lock_receipt(tmp_path: Path) -> None:
    chain = _write_chain(tmp_path)
    chain["sealed_manifest"]["sealed_lock"]["receipt"][
        "configuration_frozen"
    ] = False

    with pytest.raises(ValueError, match="lock receipt is invalid or not frozen"):
        _validate(chain)


def test_sealed_lock_chain_rejects_receipt_adapter_aggregate_mismatch(
    tmp_path: Path,
) -> None:
    chain = _write_chain(tmp_path)
    receipt_path = chain["run_dir"] / "run_receipt.json"
    receipt = _read_json(receipt_path)
    receipt["adapter_files_sha256"] = "e" * 64
    _write_json(receipt_path, receipt)
    _rebind_chain(chain, bind_adapter=False)

    with pytest.raises(ValueError, match="exact adapter artifacts"):
        _validate(chain)
