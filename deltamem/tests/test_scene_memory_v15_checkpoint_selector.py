from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

import pytest

from experiments.rethinking_rwkv_ms_gemma import (
    scene_memory_v15_data_contract as data_contract,
)
from experiments.rethinking_rwkv_ms_gemma import (
    scene_memory_v15_launch_contract as launch,
)
from experiments.rethinking_rwkv_ms_gemma import (
    select_scene_memory_v15_checkpoint as selector,
)


def _summary(step: int, values: tuple[int, int, int, float, float, float]) -> dict[str, Any]:
    exact, own, satisfied, margin, hinge, nll = values
    summary: dict[str, Any] = {
        "schema": selector.CHECKPOINT_SUMMARY_SCHEMA,
        "checkpoint_step": step,
        "correct_state_parsed_boundary_exact_rows": exact,
        "identity_own_beats_paired_rows": own,
        "identity_margin_satisfied_rows": satisfied,
        "mean_identity_logit_margin": margin,
        "mean_identity_hinge": hinge,
        "mean_correct_state_semantic_nll": nll,
    }
    summary["summary_sha256"] = selector.self_hash_payload(
        summary,
        field="summary_sha256",
    )
    return summary


def _rows() -> tuple[list[dict[str, Any]], dict[int, int]]:
    rows = [
        {
            "train_row_ordinal": ordinal,
            "official_source_index": 1000 + ordinal,
            "row_sha256": f"{ordinal + 1:064x}",
            "label_sha256": f"{ordinal + 101:064x}",
            "gold_content": f'{{"boundaries":[{ordinal}]}}',
        }
        for ordinal in range(selector.TRAIN32_ROWS)
    ]
    donors = {ordinal: ordinal ^ 1 for ordinal in range(selector.TRAIN32_ROWS)}
    return rows, donors


def _pair_entries(donors: dict[int, int]) -> dict[int, dict[str, Any]]:
    return {
        ordinal: {
            "train_row_ordinal": ordinal,
            "donor_train_row_ordinal": donor,
            "first_differing_semantic_ordinal": 0,
            "selected_target_token_ids": [1],
            "donor_target_token_ids": [2],
            "causal_prefix_sha256": "a" * 64,
        }
        for ordinal, donor in donors.items()
    }


def _result(*, exact: bool, margin: float, nll: float, gold: str) -> dict[str, Any]:
    return {
        "score_strict": {
            "schema_valid": True,
            "tp": 1,
            "fp": 0 if exact else 1,
            "fn": 0,
        },
        "semantic_decision_nll": {
            "all_semantic": {"mean_nll": nll},
            "pair_target": {
                "selected_over_alternative_logprob_margin": margin,
                "target_mode": "first_pair_distinguishing_boundaries_semantic_token_v1",
                "first_differing_semantic_ordinal": 0,
                "selected_target_token_ids": [1],
                "donor_target_token_ids": [2],
                "causal_prefix_sha256": "a" * 64,
            },
        },
        "raw_generation": gold if exact else f"prefix {gold}",
        "parsed_json": {"boundaries": [1]},
        "output_tokens": 4,
        "hit_max_new_tokens": False,
        "input_rendered_sha256": "b" * 64,
    }


def _artifact(path: Path, content: bytes) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)
    return selector.artifact_binding(path)


def _selection_fixture(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> dict[str, Any]:
    rows, donors = _rows()
    pair_entries = _pair_entries(donors)
    monkeypatch.setattr(
        selector,
        "_selector_train_contract",
        lambda: {
            "rows": rows,
            "pairing": {
                "donor_by_ordinal": donors,
                "by_ordinal": pair_entries,
            },
        },
    )
    run_root = tmp_path / "scene_memory_v15"
    selection_root = run_root / "selection"
    hard32_root = run_root / "hard32"
    run_name = "scene_memory_v15_production_unit_step4"
    training_run = run_root / run_name
    selection_run = selection_root / run_name
    monkeypatch.setattr(selector, "RUN_ROOT", run_root)
    monkeypatch.setattr(selector, "SELECTION_ROOT", selection_root)
    monkeypatch.setattr(selector, "HARD32_OUTPUT_ROOT", hard32_root)
    checkpoints: list[dict[str, Any]] = []
    for step in selector.CHECKPOINT_STEPS:
        checkpoint_dir = training_run / "trainer" / f"checkpoint-{step}"
        checkpoints.append(
            {
                "path": str(checkpoint_dir.absolute()),
                "checkpoint_step": step,
                "consumed_pair_presentations": step * 16,
                "artifacts": {
                    name: _artifact(
                        checkpoint_dir / name,
                        f"{name}-{step}".encode(),
                    )
                    for name in launch.REQUIRED_CHECKPOINT_ARTIFACTS
                },
                "rng_state_artifacts": {},
            }
        )
    launch_binding = _artifact(run_root / "logs" / f"{run_name}.launch.json", b"launch")
    completion_binding = _artifact(
        run_root / "logs" / f"{run_name}.completion.json",
        b"completion",
    )
    runtime = selector.frozen_selector_runtime()
    source_lock = {"sha256": "c" * 64}
    base_model = {"path": str((run_root / "base-model").absolute()), "identity": True}
    fingerprint_payload = selector.build_selector_fingerprint_payload(
        checkpoints=checkpoints,
        launch_receipt=launch_binding,
        completion_receipt=completion_binding,
        base_model=base_model,
        source_lock=source_lock,
        runtime=runtime,
    )
    fingerprint = selector.canonical_sha256(fingerprint_payload)
    manifest_path = selection_run / selector.SELECTOR_MANIFEST_FILENAME
    selector.atomic_write_json(
        manifest_path,
        selector.build_selector_manifest(
            fingerprint_payload=fingerprint_payload,
            created_at="2026-08-01T00:00:00Z",
        ),
    )
    manifest_binding = selector.artifact_binding(manifest_path)
    summaries: list[dict[str, Any]] = []
    audit_paths: list[Path] = []
    for checkpoint in checkpoints:
        step = int(checkpoint["checkpoint_step"])
        records = [
            selector.build_post_save_row_record(
                checkpoint_step=step,
                sample=row,
                donor_sample=rows[donors[ordinal]],
                result=_result(
                    exact=ordinal < 4 + step,
                    margin=float(step) / 4.0,
                    nll=2.0 - float(step) / 10.0,
                    gold=row["gold_content"],
                ),
                fingerprint=fingerprint,
            )
            for ordinal, row in enumerate(rows)
        ]
        audit_path = selection_run / f"checkpoint-{step}.train32.jsonl"
        selector.atomic_write_jsonl(audit_path, records)
        audit_paths.append(audit_path)
        summaries.append(
            selector.build_checkpoint_summary(
                checkpoint=checkpoint,
                records=records,
                records_binding=selector.artifact_binding(audit_path),
            )
        )
    receipt = selector.build_selection_receipt(
        fingerprint=fingerprint,
        train_inputs={
            "source_split": "train",
            "rows": selector.TRAIN32_ROWS,
            "train32_sha256": data_contract.TRAIN32_SHA256,
            "train32_rows_sha256": data_contract.TRAIN32_ROWS_SHA256,
            "pair_manifest_sha256": data_contract.PAIR_MANIFEST_FILE_SHA256,
            "source_manifest_sha256": data_contract.SOURCE_MANIFEST_FILE_SHA256,
            "source_lock": source_lock,
            "hard32_rows": 0,
        },
        launch_receipt=launch_binding,
        completion_receipt=completion_binding,
        selector_manifest=manifest_binding,
        base_model=base_model,
        summaries=summaries,
        runtime=runtime,
        created_at="2026-08-01T00:00:00Z",
    )
    receipt_path = selection_run / selector.SELECTION_RECEIPT_FILENAME
    selector.atomic_write_json(receipt_path, receipt)
    lock = selector.build_candidate_lock(
        selection_receipt=receipt,
        selection_receipt_binding=selector.artifact_binding(receipt_path),
        hard32_output_dir=hard32_root / f"{run_name}_checkpoint-4",
    )
    lock_path = selection_run / selector.CANDIDATE_LOCK_FILENAME
    selector.atomic_write_json(lock_path, lock)
    return {
        "rows": rows,
        "donors": donors,
        "pair_entries": pair_entries,
        "summaries": summaries,
        "checkpoints": checkpoints,
        "audit_paths": audit_paths,
        "fingerprint": fingerprint,
        "fingerprint_payload": fingerprint_payload,
        "manifest_path": manifest_path,
        "manifest_binding": manifest_binding,
        "base_model": base_model,
        "runtime": runtime,
        "receipt": receipt,
        "receipt_path": receipt_path,
        "lock": lock,
        "lock_path": lock_path,
        "run_root": run_root,
        "selection_root": selection_root,
        "run_name": run_name,
    }


def _write_forged_manifest_receipt(
    fixture: dict[str, Any],
    manifest: dict[str, Any],
    *,
    receipt_updates: dict[str, Any] | None = None,
) -> dict[str, Any]:
    selector.atomic_write_json(fixture["manifest_path"], manifest)
    forged = copy.deepcopy(fixture["receipt"])
    forged["selector_manifest"] = selector.artifact_binding(fixture["manifest_path"])
    if receipt_updates is not None:
        forged.update(receipt_updates)
    forged["receipt_sha256"] = selector.self_hash_payload(
        forged,
        field="receipt_sha256",
    )
    return forged


def test_rank_key_is_the_frozen_lexicographic_order() -> None:
    summary = _summary(3, (17, 16, 9, 1.25, 0.2, 0.75))

    assert selector.checkpoint_rank_key(summary) == (
        -17,
        -16,
        -9,
        -1.25,
        0.2,
        0.75,
        3,
    )


def test_parsed_exact_is_not_raw_token_exact() -> None:
    rows, donors = _rows()
    result = _result(exact=True, margin=1.0, nll=0.5, gold=rows[0]["gold_content"])
    result["raw_generation"] = "  " + rows[0]["gold_content"] + "  "

    record = selector.build_post_save_row_record(
        checkpoint_step=1,
        sample=rows[0],
        donor_sample=rows[donors[0]],
        result=result,
        fingerprint="f" * 64,
    )

    assert record["parsed_boundary_exact"] is True
    assert record["raw_token_exact_telemetry"] is True
    result["raw_generation"] = "answer: " + rows[0]["gold_content"]
    record = selector.build_post_save_row_record(
        checkpoint_step=1,
        sample=rows[0],
        donor_sample=rows[donors[0]],
        result=result,
        fingerprint="f" * 64,
    )
    assert record["parsed_boundary_exact"] is True
    assert record["raw_token_exact_telemetry"] is False


@pytest.mark.parametrize(
    ("field", "forged_value"),
    (
        ("parsed_boundary_exact", False),
        ("raw_token_exact_telemetry", False),
    ),
)
def test_exact_generation_claims_reproduce_from_embedded_evidence(
    field: str,
    forged_value: bool,
) -> None:
    rows, donors = _rows()
    record = selector.build_post_save_row_record(
        checkpoint_step=1,
        sample=rows[0],
        donor_sample=rows[donors[0]],
        result=_result(
            exact=True,
            margin=1.0,
            nll=0.5,
            gold=rows[0]["gold_content"],
        ),
        fingerprint="f" * 64,
    )
    record[field] = forged_value
    record["record_sha256"] = selector.self_hash_payload(
        record,
        field="record_sha256",
    )

    with pytest.raises(
        selector.V15SelectionError,
        match="exact-generation claims do not reproduce",
    ):
        selector.validate_post_save_row_records(
            [record],
            checkpoint_step=1,
            fingerprint="f" * 64,
            rows=rows,
            donor_by_ordinal=donors,
            pair_by_ordinal=_pair_entries(donors),
            require_complete=False,
        )


@pytest.mark.parametrize(
    ("field", "forged_value"),
    (
        ("target_mode", "non_causal_target"),
        ("first_differing_semantic_ordinal", 1),
        ("selected_target_token_ids", [7]),
        ("donor_target_token_ids", [8]),
        ("causal_prefix_sha256", "d" * 64),
    ),
)
def test_causal_pair_target_must_match_locked_train32_pairing(
    field: str,
    forged_value: Any,
) -> None:
    rows, donors = _rows()
    record = selector.build_post_save_row_record(
        checkpoint_step=1,
        sample=rows[0],
        donor_sample=rows[donors[0]],
        result=_result(
            exact=True,
            margin=1.0,
            nll=0.5,
            gold=rows[0]["gold_content"],
        ),
        fingerprint="f" * 64,
    )
    record["pair_target"][field] = forged_value
    record["record_sha256"] = selector.self_hash_payload(
        record,
        field="record_sha256",
    )

    with pytest.raises(
        selector.V15SelectionError,
        match="causal pair target differs from locked Train32 pairing",
    ):
        selector.validate_post_save_row_records(
            [record],
            checkpoint_step=1,
            fingerprint="f" * 64,
            rows=rows,
            donor_by_ordinal=donors,
            pair_by_ordinal=_pair_entries(donors),
            require_complete=False,
        )


def test_complete_receipt_and_candidate_lock_reproduce(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _selection_fixture(tmp_path, monkeypatch)

    receipt = selector.validate_selection_receipt(fixture["receipt_path"])
    lock = selector.validate_candidate_lock(
        fixture["lock_path"],
        selection_receipt=fixture["receipt_path"],
    )

    assert receipt["selected_checkpoint_step"] == 4
    assert receipt["ranked_checkpoint_steps"] == [4, 3, 2, 1]
    assert lock["candidate_count"] == 1
    assert lock["rejected_checkpoint_steps"] == [1, 2, 3]


def test_incomplete_claimed_audit_cannot_authorize(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _selection_fixture(tmp_path, monkeypatch)
    audit_path = fixture["audit_paths"][0]
    lines = audit_path.read_text(encoding="utf-8").splitlines()
    audit_path.write_text("\n".join(lines[:-1]) + "\n", encoding="utf-8")
    forged = copy.deepcopy(fixture["receipt"])
    forged_summary = forged["candidates"][0]
    forged_summary["post_save_audit"] = selector.artifact_binding(audit_path)
    forged_summary["summary_sha256"] = selector.self_hash_payload(
        forged_summary,
        field="summary_sha256",
    )
    forged["receipt_sha256"] = selector.self_hash_payload(
        forged,
        field="receipt_sha256",
    )

    with pytest.raises(selector.V15SelectionError, match="all Train32 rows"):
        selector.validate_selection_receipt(forged)


def test_protected_audit_binding_is_rejected_before_filesystem_access(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _selection_fixture(tmp_path, monkeypatch)
    forged = copy.deepcopy(fixture["receipt"])
    forged_summary = forged["candidates"][0]
    forged_summary["post_save_audit"] = {
        "path": str(
            tmp_path
            / "pairs_candidate64_failure32_holdout32_v1"
            / "holdout.jsonl"
        ),
        "bytes": 1,
        "sha256": "0" * 64,
    }
    forged_summary["summary_sha256"] = selector.self_hash_payload(
        forged_summary,
        field="summary_sha256",
    )
    forged["receipt_sha256"] = selector.self_hash_payload(
        forged,
        field="receipt_sha256",
    )

    def forbidden_filesystem_access(*_args: Any, **_kwargs: Any) -> Any:
        raise AssertionError("protected artifact reached filesystem access")

    monkeypatch.setattr(Path, "is_file", forbidden_filesystem_access)
    monkeypatch.setattr(Path, "stat", forbidden_filesystem_access)
    monkeypatch.setattr(selector, "sha256_file", forbidden_filesystem_access)

    with pytest.raises(selector.V15SelectionError, match="forbids validation/test/Hard32"):
        selector.validate_selection_receipt(forged)


def test_hard32_basename_is_rejected_before_filesystem_access(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    protected = tmp_path / "safe" / "hard32.jsonl"

    def forbidden_filesystem_access(*_args: Any, **_kwargs: Any) -> Any:
        raise AssertionError("Hard32 basename reached filesystem access")

    monkeypatch.setattr(Path, "is_file", forbidden_filesystem_access)
    monkeypatch.setattr(Path, "stat", forbidden_filesystem_access)
    monkeypatch.setattr(selector, "sha256_file", forbidden_filesystem_access)

    with pytest.raises(selector.V15SelectionError, match="forbids validation/test/Hard32"):
        selector.artifact_binding(protected)


@pytest.mark.parametrize(
    "role",
    (
        "post_save_audit",
        "checkpoint_root",
        "checkpoint_artifact",
        "rng_artifact",
        "launch_receipt",
        "completion_receipt",
        "selector_manifest",
    ),
)
def test_unlocked_artifact_binding_is_rejected_before_filesystem_access(
    role: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _selection_fixture(tmp_path, monkeypatch)
    forged = copy.deepcopy(fixture["receipt"])
    summary = forged["candidates"][0]
    checkpoint = summary["checkpoint"]
    arbitrary = tmp_path / "unlocked_ssd_artifacts"
    fake_binding = {
        "path": str(arbitrary / "innocent-looking.bin"),
        "bytes": 1,
        "sha256": "0" * 64,
    }
    if role == "post_save_audit":
        summary["post_save_audit"] = fake_binding
    elif role == "checkpoint_root":
        checkpoint["path"] = str(arbitrary / "trainer" / "checkpoint-1")
    elif role == "checkpoint_artifact":
        checkpoint["artifacts"]["delta_mem_adapter.pt"] = fake_binding
    elif role == "rng_artifact":
        checkpoint["rng_state_artifacts"]["rng_state.pth"] = fake_binding
    else:
        forged[role] = fake_binding
    summary["summary_sha256"] = selector.self_hash_payload(
        summary,
        field="summary_sha256",
    )
    forged["receipt_sha256"] = selector.self_hash_payload(
        forged,
        field="receipt_sha256",
    )

    def forbidden_filesystem_access(*_args: Any, **_kwargs: Any) -> Any:
        raise AssertionError("unlocked artifact reached filesystem access")

    monkeypatch.setattr(Path, "is_file", forbidden_filesystem_access)
    monkeypatch.setattr(Path, "stat", forbidden_filesystem_access)
    monkeypatch.setattr(selector, "sha256_file", forbidden_filesystem_access)

    with pytest.raises(selector.V15SelectionError, match="path differs|canonical training run"):
        selector.validate_selection_receipt(forged)


@pytest.mark.parametrize("document", ("selection_receipt", "candidate_lock"))
def test_selection_document_outside_locked_root_is_rejected_before_read(
    document: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _selection_fixture(tmp_path, monkeypatch)
    filename = (
        selector.SELECTION_RECEIPT_FILENAME
        if document == "selection_receipt"
        else selector.CANDIDATE_LOCK_FILENAME
    )
    arbitrary = tmp_path / "unlocked_selection" / filename

    def forbidden_read(*_args: Any, **_kwargs: Any) -> Any:
        raise AssertionError("unlocked selection document was read")

    monkeypatch.setattr(Path, "read_text", forbidden_read)
    with pytest.raises(selector.V15SelectionError, match="canonical selection run path"):
        if document == "selection_receipt":
            selector.validate_selection_receipt(arbitrary)
        else:
            selector.validate_candidate_lock(
                arbitrary,
                selection_receipt=fixture["receipt_path"],
            )


@pytest.mark.parametrize("document", ("selection_receipt", "candidate_lock"))
def test_selection_document_wrong_filename_is_rejected_before_read(
    document: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _selection_fixture(tmp_path, monkeypatch)
    wrong_name = fixture["receipt_path"].parent / f"wrong-{document}.json"

    def forbidden_read(*_args: Any, **_kwargs: Any) -> Any:
        raise AssertionError("misnamed selection document was read")

    monkeypatch.setattr(Path, "read_text", forbidden_read)
    with pytest.raises(selector.V15SelectionError, match="canonical selection run path"):
        if document == "selection_receipt":
            selector.validate_selection_receipt(wrong_name)
        else:
            selector.validate_candidate_lock(
                wrong_name,
                selection_receipt=fixture["receipt_path"],
            )


def test_post_save_audit_wrong_filename_is_rejected_before_filesystem_access(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _selection_fixture(tmp_path, monkeypatch)
    forged = copy.deepcopy(fixture["receipt"])
    summary = forged["candidates"][0]
    summary["post_save_audit"] = {
        "path": str(fixture["receipt_path"].parent / "train32-copy.jsonl"),
        "bytes": 1,
        "sha256": "0" * 64,
    }
    summary["summary_sha256"] = selector.self_hash_payload(
        summary,
        field="summary_sha256",
    )
    forged["receipt_sha256"] = selector.self_hash_payload(
        forged,
        field="receipt_sha256",
    )

    def forbidden_filesystem_access(*_args: Any, **_kwargs: Any) -> Any:
        raise AssertionError("misnamed audit reached filesystem access")

    monkeypatch.setattr(Path, "is_file", forbidden_filesystem_access)
    monkeypatch.setattr(Path, "stat", forbidden_filesystem_access)
    monkeypatch.setattr(selector, "sha256_file", forbidden_filesystem_access)
    with pytest.raises(selector.V15SelectionError, match="post-save audit path differs"):
        selector.validate_selection_receipt(forged)


@pytest.mark.parametrize("entrypoint", ("build", "validate"))
def test_candidate_lock_receipt_binding_is_rejected_before_filesystem_access(
    entrypoint: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _selection_fixture(tmp_path, monkeypatch)
    forged_binding = {
        "path": str(tmp_path / "unlocked_receipts" / selector.SELECTION_RECEIPT_FILENAME),
        "bytes": 1,
        "sha256": "0" * 64,
    }
    forged_lock = copy.deepcopy(fixture["lock"])
    forged_lock["selected_candidate"]["selection_receipt"] = forged_binding
    forged_lock["lock_sha256"] = selector.self_hash_payload(
        forged_lock,
        field="lock_sha256",
    )

    def forbidden_filesystem_access(*_args: Any, **_kwargs: Any) -> Any:
        raise AssertionError("unlocked receipt binding reached filesystem access")

    monkeypatch.setattr(Path, "is_file", forbidden_filesystem_access)
    monkeypatch.setattr(Path, "stat", forbidden_filesystem_access)
    monkeypatch.setattr(selector, "sha256_file", forbidden_filesystem_access)

    with pytest.raises(selector.V15SelectionError, match="selection receipt path differs"):
        if entrypoint == "build":
            selector.build_candidate_lock(
                selection_receipt=fixture["receipt"],
                selection_receipt_binding=forged_binding,
                hard32_output_dir=(
                    Path(selector.HARD32_OUTPUT_ROOT)
                    / f"{fixture['run_name']}_checkpoint-4"
                ),
            )
        else:
            selector.validate_candidate_lock(
                forged_lock,
                selection_receipt=fixture["receipt"],
            )


def test_symlinked_selection_run_is_rejected_before_receipt_read(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _selection_fixture(tmp_path, monkeypatch)
    selection_run = fixture["receipt_path"].parent
    redirected = tmp_path / "redirected_selection_run"
    selection_run.rename(redirected)
    selection_run.symlink_to(redirected, target_is_directory=True)

    def forbidden_read(*_args: Any, **_kwargs: Any) -> Any:
        raise AssertionError("receipt behind symlink was read")

    monkeypatch.setattr(Path, "read_text", forbidden_read)
    with pytest.raises(selector.V15SelectionError, match="symlink component"):
        selector.validate_selection_receipt(fixture["receipt_path"])


def test_selector_overwrite_is_rejected_before_any_runtime_access() -> None:
    args = selector._parse_args(
        [
            "--base-model",
            "/unused/base",
            "--run-dir",
            "/unused/run",
            "--launch-receipt",
            "/unused/launch.json",
            "--completion-receipt",
            "/unused/completion.json",
            "--output-dir",
            "/unused/selection",
            "--overwrite",
        ]
    )

    with pytest.raises(selector.V15SelectionError, match="forbids --overwrite"):
        selector._validate_runtime_args(args)


def test_partial_canonical_train32_audit_remains_resumable(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _selection_fixture(tmp_path, monkeypatch)
    records = [
        json.loads(line)
        for line in fixture["audit_paths"][0]
        .read_text(encoding="utf-8")
        .splitlines()[:3]
    ]

    completed = selector.validate_post_save_row_records(
        records,
        checkpoint_step=1,
        fingerprint=fixture["fingerprint"],
        rows=fixture["rows"],
        donor_by_ordinal=fixture["donors"],
        pair_by_ordinal=fixture["pair_entries"],
        require_complete=False,
    )

    assert list(completed) == [0, 1, 2]


@pytest.mark.parametrize("entrypoint", ("builder", "cli"))
def test_alternate_protected_output_cannot_mint_authorization(
    entrypoint: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _selection_fixture(tmp_path, monkeypatch)
    alternate = Path(selector.HARD32_OUTPUT_ROOT) / "alternate-checkpoint-4"

    with pytest.raises(
        selector.V15SelectionError,
        match="canonical selected-checkpoint path",
    ):
        if entrypoint == "builder":
            selector.build_candidate_lock(
                selection_receipt=fixture["receipt"],
                selection_receipt_binding=selector.artifact_binding(
                    fixture["receipt_path"]
                ),
                hard32_output_dir=alternate,
            )
        else:
            selector.validated_hard32_output_dir(
                alternate,
                run_name=fixture["run_name"],
                selected_checkpoint_step=4,
            )


def test_selector_manifest_runtime_mutation_cannot_be_resigned(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _selection_fixture(tmp_path, monkeypatch)
    manifest = json.loads(fixture["manifest_path"].read_text(encoding="utf-8"))
    manifest["fingerprint_payload"]["runtime"]["dtype"] = "float16"
    manifest["fingerprint"] = selector.canonical_sha256(
        manifest["fingerprint_payload"]
    )
    forged = _write_forged_manifest_receipt(
        fixture,
        manifest,
        receipt_updates={
            "fingerprint": manifest["fingerprint"],
            "runtime": manifest["fingerprint_payload"]["runtime"],
        },
    )

    with pytest.raises(selector.V15SelectionError, match="frozen runtime differs"):
        selector.validate_selection_receipt(forged)


def test_selector_manifest_fingerprint_mutation_does_not_reproduce(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _selection_fixture(tmp_path, monkeypatch)
    manifest = json.loads(fixture["manifest_path"].read_text(encoding="utf-8"))
    manifest["fingerprint"] = "0" * 64
    forged = _write_forged_manifest_receipt(
        fixture,
        manifest,
        receipt_updates={"fingerprint": "0" * 64},
    )

    with pytest.raises(
        selector.V15SelectionError,
        match="fingerprint does not reproduce",
    ):
        selector.validate_selection_receipt(forged)


@pytest.mark.parametrize(
    "identity",
    ("checkpoints", "launch_receipt", "completion_receipt", "base_model", "train_hash"),
)
def test_selector_manifest_fingerprint_identities_must_reproduce_receipt(
    identity: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _selection_fixture(tmp_path, monkeypatch)
    manifest = json.loads(fixture["manifest_path"].read_text(encoding="utf-8"))
    payload = manifest["fingerprint_payload"]
    if identity == "checkpoints":
        payload["checkpoints"][0]["artifacts"]["delta_mem_adapter.pt"]["sha256"] = (
            "9" * 64
        )
    elif identity == "launch_receipt":
        payload["launch_receipt"]["sha256"] = "9" * 64
    elif identity == "completion_receipt":
        payload["completion_receipt"]["sha256"] = "9" * 64
    elif identity == "base_model":
        payload["base_model"]["identity"] = False
    else:
        payload["train32_sha256"] = "9" * 64
    manifest["fingerprint"] = selector.canonical_sha256(payload)
    forged = _write_forged_manifest_receipt(
        fixture,
        manifest,
        receipt_updates={"fingerprint": manifest["fingerprint"]},
    )

    with pytest.raises(
        selector.V15SelectionError,
        match="fingerprint payload differs|manifest and selection receipt differ",
    ):
        selector.validate_selection_receipt(forged)


def test_selector_manifest_content_mutation_breaks_receipt_binding(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _selection_fixture(tmp_path, monkeypatch)
    manifest = json.loads(fixture["manifest_path"].read_text(encoding="utf-8"))
    manifest["benchmark_evidence_used"] = True
    selector.atomic_write_json(fixture["manifest_path"], manifest)

    with pytest.raises(selector.V15SelectionError, match="manifest artifact differs"):
        selector.validate_selection_receipt(fixture["receipt"])


def test_selector_manifest_alternate_path_is_rejected_before_filesystem_access(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _selection_fixture(tmp_path, monkeypatch)
    forged = copy.deepcopy(fixture["receipt"])
    forged["selector_manifest"] = {
        "path": str(fixture["manifest_path"].with_name("alternate-manifest.json")),
        "bytes": fixture["manifest_binding"]["bytes"],
        "sha256": fixture["manifest_binding"]["sha256"],
    }
    forged["receipt_sha256"] = selector.self_hash_payload(
        forged,
        field="receipt_sha256",
    )

    def forbidden_filesystem_access(*_args: Any, **_kwargs: Any) -> Any:
        raise AssertionError("alternate manifest path reached filesystem access")

    monkeypatch.setattr(Path, "is_file", forbidden_filesystem_access)
    monkeypatch.setattr(Path, "stat", forbidden_filesystem_access)
    monkeypatch.setattr(selector, "sha256_file", forbidden_filesystem_access)
    with pytest.raises(selector.V15SelectionError, match="selector_manifest path differs"):
        selector.validate_selection_receipt(forged)


def test_candidate_lock_binds_exact_selector_manifest_chain(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _selection_fixture(tmp_path, monkeypatch)
    forged = copy.deepcopy(fixture["lock"])
    forged["selected_candidate"]["selector_manifest"]["sha256"] = "9" * 64
    forged["lock_sha256"] = selector.self_hash_payload(
        forged,
        field="lock_sha256",
    )

    with pytest.raises(
        selector.V15SelectionError,
        match="selected checkpoint differs",
    ):
        selector.validate_candidate_lock(
            forged,
            selection_receipt=fixture["receipt_path"],
        )


def test_candidate_lock_manifest_path_is_rejected_before_filesystem_access(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _selection_fixture(tmp_path, monkeypatch)
    forged = copy.deepcopy(fixture["lock"])
    forged["selected_candidate"]["selector_manifest"] = {
        "path": str(tmp_path / "unlocked" / selector.SELECTOR_MANIFEST_FILENAME),
        "bytes": 1,
        "sha256": "9" * 64,
    }
    forged["lock_sha256"] = selector.self_hash_payload(
        forged,
        field="lock_sha256",
    )

    def forbidden_filesystem_access(*_args: Any, **_kwargs: Any) -> Any:
        raise AssertionError("unlocked candidate manifest reached filesystem access")

    monkeypatch.setattr(Path, "is_file", forbidden_filesystem_access)
    monkeypatch.setattr(Path, "stat", forbidden_filesystem_access)
    monkeypatch.setattr(selector, "sha256_file", forbidden_filesystem_access)
    with pytest.raises(selector.V15SelectionError, match="selector manifest path differs"):
        selector.validate_candidate_lock(
            forged,
            selection_receipt=fixture["receipt"],
        )


def test_provenance_keeps_exact_launcher_checkpoint_contracts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run_dir = tmp_path / "run"
    bundle = {
        "artifacts": {
            "source_manifest": {"sha256": "1" * 64},
            "pair_schedule": {"sha256": "2" * 64},
        }
    }
    raw_contracts: list[dict[str, Any]] = []
    completion_steps: list[int] = []
    monkeypatch.setattr(launch, "require_v15_run_path", lambda path, **_: path)
    monkeypatch.setattr(
        launch,
        "validate_data_contract",
        lambda: {"source_manifest_sha256": "1" * 64, "schedule_sha256": "2" * 64},
    )
    monkeypatch.setattr(launch, "validate_warm_start_contract", lambda: {"warm": True})
    monkeypatch.setattr(
        launch,
        "validate_base_model_contract",
        lambda path: {"path": str(path), "identity": True},
    )

    def checkpoint_contract(path: Path, **_: Any) -> dict[str, Any]:
        step = int(path.name.rsplit("-", 1)[1])
        contract = {
            "path": str(path),
            "checkpoint_step": step,
            "consumed_pair_presentations": step * 16,
            "artifacts": {"delta_mem_adapter.pt": {"sha256": str(step) * 64}},
            "rng_state_artifacts": {},
        }
        raw_contracts.append(contract)
        return contract

    monkeypatch.setattr(launch, "validate_checkpoint_contract", checkpoint_contract)

    def launch_receipt(path: Path, **kwargs: Any) -> dict[str, Any]:
        assert kwargs["checkpoint"].name == "checkpoint-4"
        assert kwargs["base_model_identity"]["identity"] is True
        return {"receipt_sha256": "3" * 64}

    monkeypatch.setattr(launch, "validate_launch_receipt", launch_receipt)

    def completion_receipt(path: Path, **kwargs: Any) -> dict[str, Any]:
        checkpoint = kwargs["checkpoint_contract"]
        assert checkpoint == raw_contracts[checkpoint["checkpoint_step"] - 1]
        completion_steps.append(checkpoint["checkpoint_step"])
        return {"receipt_sha256": "4" * 64}

    monkeypatch.setattr(launch, "validate_completion_receipt", completion_receipt)

    checkpoints, _, _, _ = selector._validate_provenance(
        run_dir=run_dir,
        launch_receipt=tmp_path / "launch.json",
        completion_receipt=tmp_path / "completion.json",
        bundle=bundle,
        base_model=tmp_path / "base",
    )

    assert checkpoints == raw_contracts
    assert completion_steps == [1, 2, 3, 4]
    assert all("memory_dir" not in checkpoint for checkpoint in checkpoints)
    assert "log_history" not in Path(selector.__file__).read_text(encoding="utf-8")
