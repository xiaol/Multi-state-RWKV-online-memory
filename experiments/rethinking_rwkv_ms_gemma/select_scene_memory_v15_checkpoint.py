#!/usr/bin/env python3
"""Select one V15 checkpoint using a post-save, Train32-only audit.

The selector reloads each saved checkpoint before measuring it.  It never uses
the Trainer log entry associated with the optimizer step and never reads a
validation, test, or Hard32 artifact.  Only the selected checkpoint may be
authorized for the separate frozen-Hard32 evaluator.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import tempfile
from typing import Any, Iterable, Mapping, Sequence


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = Path(__file__).resolve().parents[2]

from experiments.rethinking_rwkv_ms_gemma import (  # noqa: E402
    run_scene_train32_eval as train32,
)
from experiments.rethinking_rwkv_ms_gemma import (  # noqa: E402
    scene_memory_v15_data_contract as data_contract,
)
from experiments.rethinking_rwkv_ms_gemma.prepare_scene_memory_v7_data import (  # noqa: E402
    sha256_file,
)
from experiments.rethinking_rwkv_ms_gemma.run_scene_state_eval import (  # noqa: E402
    clear_model_memory,
    evaluate_condition,
    load_adapter_model,
    memory_architecture_contract,
    resolved_memory_layer_count,
    runtime_package_versions,
    utc_now,
)


SSD_ROOT = Path("/run/media/xiaol/B214449214445C0B")
RUN_ROOT = (
    SSD_ROOT
    / "delta_mem_outputs"
    / "novel_rwkv_ms_memory"
    / "scene_memory_v15"
)
SELECTION_ROOT = RUN_ROOT / "selection"
HARD32_OUTPUT_ROOT = RUN_ROOT / "hard32"
CHECKPOINT_STEPS = (1, 2, 3, 4)
TRAIN32_ROWS = 32
MAX_NEW_TOKENS = 128
IDENTITY_MARGIN = 1.0
CONDITION = "state_only"
HARD32_ACCESS_POLICY = "forbidden_not_resolved_opened_or_hashed"
HARD32_AUTHORIZATION_SCOPE = (
    "fixed_scene_v4_current_hard32_only_no_full170_no_test_no_other_benchmarks"
)

SELECTOR_CONTRACT = "scene_memory_v15_post_save_train32_selector"
SELECTION_POLICY = (
    "correct_state_exact_then_own_paired_identity_then_semantic_nll_v1"
)
ROW_SCHEMA = "rwkv_ms_scene_memory_v15_post_save_train32_row.v1"
CHECKPOINT_SUMMARY_SCHEMA = (
    "rwkv_ms_scene_memory_v15_post_save_checkpoint_summary.v1"
)
SELECTION_RECEIPT_SCHEMA = (
    "rwkv_ms_scene_memory_v15_train32_checkpoint_selection_receipt.v1"
)
SELECTOR_MANIFEST_SCHEMA = "rwkv_ms_scene_memory_v15_train32_selector_manifest.v1"
CANDIDATE_LOCK_SCHEMA = "rwkv_ms_scene_memory_v15_hard32_candidate_lock.v1"
HARD32_AUTHORIZATION_KIND = (
    "scene_memory_v15_train32_selected_checkpoint_lock"
)
SELECTION_RECEIPT_FILENAME = "selection_receipt.json"
SELECTOR_MANIFEST_FILENAME = "manifest.json"
CANDIDATE_LOCK_FILENAME = "scene_memory_v15_hard32_candidate_lock.json"

RANK_ORDER = (
    "correct_state_parsed_boundary_exact_rows_desc",
    "identity_own_beats_paired_rows_desc",
    "identity_margin_satisfied_rows_desc",
    "mean_identity_logit_margin_desc",
    "mean_identity_hinge_asc",
    "mean_correct_state_semantic_nll_asc",
    "checkpoint_step_asc",
)


class V15SelectionError(ValueError):
    """Raised when a V15 Train32 selection binding differs."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise V15SelectionError(message)


def canonical_sha256(value: Any) -> str:
    import hashlib

    return hashlib.sha256(
        json.dumps(
            value,
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()


def self_hash_payload(payload: Mapping[str, Any], *, field: str) -> str:
    unsigned = dict(payload)
    unsigned.pop(field, None)
    return canonical_sha256(unsigned)


def _validate_self_hash(payload: Mapping[str, Any], *, field: str) -> str:
    recorded = payload.get(field)
    _require(
        isinstance(recorded, str)
        and len(recorded) == 64
        and recorded == self_hash_payload(payload, field=field),
        f"V15 selector {field} differs",
    )
    return recorded


def _lexical_absolute(path: Path | str) -> Path:
    return Path(os.path.abspath(os.fspath(Path(path).expanduser())))


def require_ssd_path(
    path: Path | str,
    *,
    description: str,
    ssd_root: Path = SSD_ROOT,
) -> Path:
    candidate = _lexical_absolute(path)
    root = _lexical_absolute(ssd_root)
    _require(
        candidate == root or root in candidate.parents,
        f"V15 {description} must stay on the 2T SSD: {candidate}",
    )
    return candidate


def _reject_protected_path(
    path: Path | str,
    *,
    description: str,
    allow_candidate_lock_filename: bool = False,
) -> Path:
    candidate = _lexical_absolute(path)
    lowered = tuple(part.casefold() for part in candidate.parts)
    protected_names = {"val", "validation", "test", "hard32", "holdout"}
    protected_fragments = ("hard32", "holdout")
    candidate_lock_allowed = (
        allow_candidate_lock_filename
        and candidate.name.casefold() == CANDIDATE_LOCK_FILENAME.casefold()
    )

    def protected(part: str) -> bool:
        stem = Path(part).stem
        return (
            part in protected_names
            or stem in protected_names
            or any(fragment in part for fragment in protected_fragments)
        )

    _require(
        not any(protected(part) for part in lowered[:-1])
        and (candidate_lock_allowed or not protected(lowered[-1])),
        f"V15 selector forbids validation/test/Hard32 input for {description}",
    )
    return candidate


def _require_no_symlink_components(path: Path, *, description: str) -> None:
    current = Path(path.anchor)
    for component in path.parts[1:]:
        current /= component
        _require(
            not current.is_symlink(),
            f"V15 {description} contains a symlink component: {current}",
        )


def _binding_path_lexical(
    binding: Mapping[str, Any],
    *,
    description: str,
) -> Path:
    _require(isinstance(binding, Mapping), f"V15 {description} binding is missing")
    raw_path = binding.get("path")
    _require(isinstance(raw_path, str) and raw_path, f"V15 {description} path is invalid")
    return _reject_protected_path(raw_path, description=description)


def _require_binding_path(
    binding: Mapping[str, Any],
    *,
    expected_path: Path,
    description: str,
) -> Path:
    path = _binding_path_lexical(binding, description=description)
    _require(
        path == _lexical_absolute(expected_path),
        f"V15 {description} path differs",
    )
    return path


def _selection_document_path(
    path: Path | str,
    *,
    filename: str,
    description: str,
) -> tuple[Path, str]:
    candidate = _reject_protected_path(
        path,
        description=description,
        allow_candidate_lock_filename=filename == CANDIDATE_LOCK_FILENAME,
    )
    selection_root = _lexical_absolute(SELECTION_ROOT)
    _require(
        candidate.name == filename
        and candidate.parent.parent == selection_root
        and bool(candidate.parent.name)
        and Path(candidate.parent.name).name == candidate.parent.name,
        f"V15 {description} must use the canonical selection run path",
    )
    return candidate, candidate.parent.name


def _read_locked_json(path: Path, *, description: str) -> dict[str, Any]:
    _require_no_symlink_components(path, description=description)
    _require(
        path.is_file() and path.stat().st_size > 0,
        f"V15 {description} is missing or empty",
    )
    payload = json.loads(path.read_text(encoding="utf-8"))
    _require(isinstance(payload, dict), f"V15 {description} must be an object")
    return payload


def _required_checkpoint_artifact_names() -> tuple[str, ...]:
    from experiments.rethinking_rwkv_ms_gemma import (  # local startup dependency
        scene_memory_v15_launch_contract as launch,
    )

    return tuple(launch.REQUIRED_CHECKPOINT_ARTIFACTS)


def _preflight_selection_artifact_paths(
    payload: Mapping[str, Any],
    *,
    expected_run_name: str | None = None,
) -> str:
    """Reject every non-canonical binding before any file metadata or content access."""

    candidates = payload.get("candidates")
    _require(
        isinstance(candidates, list) and len(candidates) == len(CHECKPOINT_STEPS),
        "V15 selection candidates differ",
    )
    run_root = _lexical_absolute(RUN_ROOT)
    selection_root = _lexical_absolute(SELECTION_ROOT)
    run_name = expected_run_name
    if run_name is not None:
        _require(
            bool(run_name) and Path(run_name).name == run_name,
            "V15 selection run name is invalid",
        )
    required_artifacts = _required_checkpoint_artifact_names()

    for expected_step, summary in zip(CHECKPOINT_STEPS, candidates):
        _require(
            isinstance(summary, Mapping)
            and summary.get("checkpoint_step") == expected_step,
            "V15 selection candidate order differs",
        )
        checkpoint = summary.get("checkpoint")
        _require(isinstance(checkpoint, Mapping), "V15 selected checkpoint binding is missing")
        raw_checkpoint_path = checkpoint.get("path")
        _require(
            isinstance(raw_checkpoint_path, str) and raw_checkpoint_path,
            f"V15 checkpoint-{expected_step} path is invalid",
        )
        checkpoint_path = _reject_protected_path(
            raw_checkpoint_path,
            description=f"checkpoint-{expected_step}",
        )
        if run_name is None:
            _require(
                checkpoint_path.parent.name == "trainer"
                and checkpoint_path.parent.parent.parent == run_root,
                "V15 checkpoint path is outside the canonical training run",
            )
            run_name = checkpoint_path.parent.parent.name
            _require(
                bool(run_name) and Path(run_name).name == run_name,
                "V15 selection run name is invalid",
            )
        expected_checkpoint = (
            run_root / run_name / "trainer" / f"checkpoint-{expected_step}"
        )
        _require(
            checkpoint_path == expected_checkpoint
            and checkpoint.get("checkpoint_step") == expected_step,
            f"V15 checkpoint-{expected_step} path differs",
        )

        artifacts = checkpoint.get("artifacts")
        _require(
            isinstance(artifacts, Mapping)
            and set(artifacts) == set(required_artifacts),
            f"V15 checkpoint-{expected_step} artifact names differ",
        )
        for name in required_artifacts:
            _require_binding_path(
                artifacts[name],
                expected_path=expected_checkpoint / name,
                description=f"checkpoint-{expected_step} {name}",
            )
        rng_artifacts = checkpoint.get("rng_state_artifacts")
        _require(
            isinstance(rng_artifacts, Mapping),
            f"V15 checkpoint-{expected_step} RNG artifacts differ",
        )
        for name, binding in rng_artifacts.items():
            _require(
                isinstance(name, str)
                and Path(name).name == name
                and name.startswith("rng_state")
                and name.endswith(".pth"),
                f"V15 checkpoint-{expected_step} RNG artifact name differs",
            )
            _require_binding_path(
                binding,
                expected_path=expected_checkpoint / name,
                description=f"checkpoint-{expected_step} {name}",
            )

        _require_binding_path(
            summary.get("post_save_audit", {}),
            expected_path=(
                selection_root / run_name / f"checkpoint-{expected_step}.train32.jsonl"
            ),
            description=f"checkpoint-{expected_step} post-save audit",
        )

    if run_name is None:  # pragma: no cover - fixed non-empty candidates
        raise AssertionError("V15 selector checkpoint schedule is empty")
    log_root = run_root / "logs"
    _require_binding_path(
        payload.get("launch_receipt", {}),
        expected_path=log_root / f"{run_name}.launch.json",
        description="V15 launch_receipt",
    )
    _require_binding_path(
        payload.get("completion_receipt", {}),
        expected_path=log_root / f"{run_name}.completion.json",
        description="V15 completion_receipt",
    )
    _require_binding_path(
        payload.get("selector_manifest", {}),
        expected_path=selection_root / run_name / SELECTOR_MANIFEST_FILENAME,
        description="V15 selector_manifest",
    )
    return run_name


def artifact_binding(path: Path | str) -> dict[str, Any]:
    resolved = _reject_protected_path(path, description="artifact binding")
    _require_no_symlink_components(resolved, description="artifact binding")
    _require(
        resolved.is_file()
        and not resolved.is_symlink()
        and resolved.stat().st_size > 0,
        f"V15 selector artifact is missing, empty, or a symlink: {resolved}",
    )
    return {
        "path": str(resolved),
        "bytes": resolved.stat().st_size,
        "sha256": sha256_file(resolved),
    }


def validate_artifact_binding(
    binding: Mapping[str, Any],
    *,
    description: str,
    expected_path: Path | None = None,
) -> Path:
    # This lexical policy check must precede every exists/stat/open/hash operation.
    path = _binding_path_lexical(binding, description=description)
    if expected_path is not None:
        _require(
            path == _lexical_absolute(expected_path),
            f"V15 {description} path differs",
        )
    _require_no_symlink_components(path, description=description)
    _require(
        path.is_file() and not path.is_symlink() and path.stat().st_size > 0,
        f"V15 {description} is missing, empty, or a symlink",
    )
    _require(
        binding.get("bytes") == path.stat().st_size
        and binding.get("sha256") == sha256_file(path),
        f"V15 {description} artifact differs",
    )
    return path


def frozen_selector_runtime() -> dict[str, Any]:
    """Return the only runtime allowed to rank and authorize V15 checkpoints."""

    return {
        "condition": CONDITION,
        "max_new_tokens": MAX_NEW_TOKENS,
        "do_sample": False,
        "use_cache_generation": True,
        "prime_use_cache": False,
        "device": "cuda:0",
        "dtype": "bfloat16",
        "attn_implementation": "sdpa",
        "normal_fusion_profile": "native",
        "expected_memory_layer_count": 42,
        "identity_margin": IDENTITY_MARGIN,
        "packages": runtime_package_versions(),
    }


def _validate_frozen_selector_runtime(runtime: Mapping[str, Any]) -> dict[str, Any]:
    _require(
        isinstance(runtime, Mapping)
        and dict(runtime) == frozen_selector_runtime(),
        "V15 selector frozen runtime differs",
    )
    return dict(runtime)


def build_selector_fingerprint_payload(
    *,
    checkpoints: Sequence[Mapping[str, Any]],
    launch_receipt: Mapping[str, Any],
    completion_receipt: Mapping[str, Any],
    base_model: Mapping[str, Any],
    source_lock: Mapping[str, Any],
    runtime: Mapping[str, Any],
) -> dict[str, Any]:
    _require(
        len(checkpoints) == len(CHECKPOINT_STEPS)
        and all(isinstance(checkpoint, Mapping) for checkpoint in checkpoints)
        and [checkpoint.get("checkpoint_step") for checkpoint in checkpoints]
        == list(CHECKPOINT_STEPS),
        "V15 selector fingerprint checkpoints differ",
    )
    for description, value in (
        ("launch receipt", launch_receipt),
        ("completion receipt", completion_receipt),
        ("base model", base_model),
        ("source lock", source_lock),
    ):
        _require(
            isinstance(value, Mapping) and bool(value),
            f"V15 selector fingerprint {description} differs",
        )
    frozen_runtime = _validate_frozen_selector_runtime(runtime)
    return {
        "schema_version": 1,
        "contract": SELECTOR_CONTRACT,
        "selection_policy": SELECTION_POLICY,
        "saved_weight_timing": "reloaded_post_optimizer_update_checkpoint",
        "train32_sha256": data_contract.TRAIN32_SHA256,
        "train32_rows_sha256": data_contract.TRAIN32_ROWS_SHA256,
        "pair_manifest_sha256": data_contract.PAIR_MANIFEST_FILE_SHA256,
        "source_manifest_sha256": data_contract.SOURCE_MANIFEST_FILE_SHA256,
        "source_lock": dict(source_lock),
        "checkpoints": [dict(checkpoint) for checkpoint in checkpoints],
        "launch_receipt": dict(launch_receipt),
        "completion_receipt": dict(completion_receipt),
        "base_model": dict(base_model),
        "runtime": frozen_runtime,
        "hard32_access": HARD32_ACCESS_POLICY,
    }


def _validate_selector_fingerprint_payload(
    payload: Mapping[str, Any],
) -> dict[str, Any]:
    _require(
        isinstance(payload, Mapping),
        "V15 selector manifest fingerprint payload is missing",
    )
    checkpoints = payload.get("checkpoints")
    _require(
        isinstance(checkpoints, list)
        and all(isinstance(checkpoint, Mapping) for checkpoint in checkpoints),
        "V15 selector manifest fingerprint checkpoints differ",
    )
    expected = build_selector_fingerprint_payload(
        checkpoints=checkpoints,
        launch_receipt=payload.get("launch_receipt", {}),
        completion_receipt=payload.get("completion_receipt", {}),
        base_model=payload.get("base_model", {}),
        source_lock=payload.get("source_lock", {}),
        runtime=payload.get("runtime", {}),
    )
    _require(
        dict(payload) == expected,
        "V15 selector manifest fingerprint payload differs",
    )
    return expected


def build_selector_manifest(
    *,
    fingerprint_payload: Mapping[str, Any],
    created_at: str | None = None,
) -> dict[str, Any]:
    validated_payload = _validate_selector_fingerprint_payload(fingerprint_payload)
    return {
        "schema": SELECTOR_MANIFEST_SCHEMA,
        "created_at": utc_now() if created_at is None else created_at,
        "fingerprint": canonical_sha256(validated_payload),
        "fingerprint_payload": validated_payload,
        "benchmark_evidence_used": False,
        "hard32_access": HARD32_ACCESS_POLICY,
    }


def validate_selector_manifest_binding(
    binding: Mapping[str, Any],
    *,
    run_name: str,
) -> dict[str, Any]:
    expected_path = (
        _lexical_absolute(SELECTION_ROOT) / run_name / SELECTOR_MANIFEST_FILENAME
    )
    path = validate_artifact_binding(
        binding,
        description="V15 selector manifest",
        expected_path=expected_path,
    )
    manifest = _read_locked_json(path, description="V15 selector manifest")
    _require(
        set(manifest)
        == {
            "schema",
            "created_at",
            "fingerprint",
            "fingerprint_payload",
            "benchmark_evidence_used",
            "hard32_access",
        }
        and manifest.get("schema") == SELECTOR_MANIFEST_SCHEMA
        and isinstance(manifest.get("created_at"), str)
        and bool(manifest["created_at"])
        and manifest.get("benchmark_evidence_used") is False
        and manifest.get("hard32_access") == HARD32_ACCESS_POLICY,
        "V15 selector manifest contract differs",
    )
    fingerprint_payload = _validate_selector_fingerprint_payload(
        manifest.get("fingerprint_payload", {})
    )
    fingerprint = canonical_sha256(fingerprint_payload)
    _require(
        manifest.get("fingerprint") == fingerprint,
        "V15 selector manifest fingerprint does not reproduce",
    )
    result = dict(manifest)
    result["manifest_path"] = str(path)
    result["manifest_file_sha256"] = sha256_file(path)
    result["validated_fingerprint"] = fingerprint
    return result


def atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w",
        encoding="utf-8",
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        delete=False,
    ) as handle:
        temporary = Path(handle.name)
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def atomic_write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w",
        encoding="utf-8",
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        delete=False,
    ) as handle:
        temporary = Path(handle.name)
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True))
            handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _finite_number(value: Any, *, description: str) -> float:
    _require(
        not isinstance(value, bool)
        and isinstance(value, (int, float))
        and math.isfinite(float(value)),
        f"V15 {description} must be finite",
    )
    return float(value)


def _strict_exact(score: Mapping[str, Any]) -> bool:
    return bool(
        score.get("schema_valid") is True
        and score.get("fp") == 0
        and score.get("fn") == 0
    )


def build_post_save_row_record(
    *,
    checkpoint_step: int,
    sample: Mapping[str, Any],
    donor_sample: Mapping[str, Any],
    result: Mapping[str, Any],
    fingerprint: str,
) -> dict[str, Any]:
    """Normalize one correct-state generation and paired-target logit audit."""

    _require(checkpoint_step in CHECKPOINT_STEPS, "V15 audit checkpoint step differs")
    score = result.get("score_strict")
    semantic = result.get("semantic_decision_nll")
    _require(isinstance(score, Mapping), "V15 audit strict score is missing")
    _require(isinstance(semantic, Mapping), "V15 audit semantic NLL is missing")
    all_semantic = semantic.get("all_semantic")
    pair_target = semantic.get("pair_target")
    _require(
        isinstance(all_semantic, Mapping) and isinstance(pair_target, Mapping),
        "V15 audit semantic branches are missing",
    )
    margin = _finite_number(
        pair_target.get("selected_over_alternative_logprob_margin"),
        description="identity logit margin",
    )
    semantic_nll = _finite_number(
        all_semantic.get("mean_nll"),
        description="correct-state semantic NLL",
    )
    gold_content = sample.get("gold_content")
    raw_generation = result.get("raw_generation")
    _require(
        isinstance(gold_content, str) and isinstance(raw_generation, str),
        "V15 audit generation or gold text is invalid",
    )
    record: dict[str, Any] = {
        "schema": ROW_SCHEMA,
        "fingerprint": fingerprint,
        "checkpoint_step": checkpoint_step,
        "train_row_ordinal": int(sample["train_row_ordinal"]),
        "official_source_index": int(sample["official_source_index"]),
        "row_sha256": sample["row_sha256"],
        "label_sha256": sample["label_sha256"],
        "donor_train_row_ordinal": int(donor_sample["train_row_ordinal"]),
        "donor_row_sha256": donor_sample["row_sha256"],
        "condition": CONDITION,
        "saved_weight_timing": "reloaded_post_optimizer_update_checkpoint",
        "parsed_boundary_exact": _strict_exact(score),
        "raw_token_exact_telemetry": raw_generation.strip() == gold_content.strip(),
        "identity_logit_margin": margin,
        "identity_own_beats_paired": margin > 0.0,
        "identity_margin_satisfied": margin >= IDENTITY_MARGIN,
        "identity_hinge": max(0.0, IDENTITY_MARGIN - margin),
        "correct_state_semantic_nll": semantic_nll,
        "pair_target": {
            "target_mode": pair_target.get("target_mode"),
            "first_differing_semantic_ordinal": pair_target.get(
                "first_differing_semantic_ordinal"
            ),
            "selected_target_token_ids": pair_target.get(
                "selected_target_token_ids"
            ),
            "donor_target_token_ids": pair_target.get("donor_target_token_ids"),
            "causal_prefix_sha256": pair_target.get("causal_prefix_sha256"),
        },
        "generation": {
            "raw": raw_generation,
            "parsed_json": result.get("parsed_json"),
            "score_strict": dict(score),
            "output_tokens": result.get("output_tokens"),
            "hit_max_new_tokens": result.get("hit_max_new_tokens"),
            "input_rendered_sha256": result.get("input_rendered_sha256"),
        },
    }
    record["record_sha256"] = self_hash_payload(record, field="record_sha256")
    return record


def validate_post_save_row_records(
    records: Sequence[Mapping[str, Any]],
    *,
    checkpoint_step: int,
    fingerprint: str,
    rows: Sequence[Mapping[str, Any]],
    donor_by_ordinal: Mapping[int, int],
    require_complete: bool,
) -> dict[int, dict[str, Any]]:
    by_ordinal: dict[int, dict[str, Any]] = {}
    for source in records:
        record = dict(source)
        _require(record.get("schema") == ROW_SCHEMA, "V15 audit row schema differs")
        _validate_self_hash(record, field="record_sha256")
        ordinal = record.get("train_row_ordinal")
        _require(
            isinstance(ordinal, int)
            and not isinstance(ordinal, bool)
            and 0 <= ordinal < len(rows)
            and ordinal not in by_ordinal,
            "V15 audit row ordinal is invalid or duplicated",
        )
        row = rows[ordinal]
        donor_ordinal = donor_by_ordinal[ordinal]
        donor = rows[donor_ordinal]
        _require(
            record.get("fingerprint") == fingerprint
            and record.get("checkpoint_step") == checkpoint_step
            and record.get("official_source_index") == row["official_source_index"]
            and record.get("row_sha256") == row["row_sha256"]
            and record.get("label_sha256") == row["label_sha256"]
            and record.get("donor_train_row_ordinal") == donor_ordinal
            and record.get("donor_row_sha256") == donor["row_sha256"]
            and record.get("condition") == CONDITION
            and record.get("saved_weight_timing")
            == "reloaded_post_optimizer_update_checkpoint",
            "V15 audit row identity differs",
        )
        for field in (
            "parsed_boundary_exact",
            "raw_token_exact_telemetry",
            "identity_own_beats_paired",
            "identity_margin_satisfied",
        ):
            _require(isinstance(record.get(field), bool), f"V15 audit {field} differs")
        generation = record.get("generation")
        generation_score = (
            generation.get("score_strict")
            if isinstance(generation, Mapping)
            else None
        )
        raw_generation = (
            generation.get("raw") if isinstance(generation, Mapping) else None
        )
        gold_content = row.get("gold_content")
        _require(
            isinstance(generation_score, Mapping)
            and isinstance(raw_generation, str)
            and isinstance(gold_content, str)
            and record["parsed_boundary_exact"]
            == _strict_exact(generation_score)
            and record["raw_token_exact_telemetry"]
            == (raw_generation.strip() == gold_content.strip()),
            "V15 audit exact-generation claims do not reproduce",
        )
        margin = _finite_number(
            record.get("identity_logit_margin"),
            description="row identity margin",
        )
        hinge = _finite_number(record.get("identity_hinge"), description="row hinge")
        _require(
            record["identity_own_beats_paired"] == (margin > 0.0)
            and record["identity_margin_satisfied"] == (margin >= IDENTITY_MARGIN)
            and math.isclose(
                hinge,
                max(0.0, IDENTITY_MARGIN - margin),
                rel_tol=1e-9,
                abs_tol=1e-9,
            ),
            "V15 audit identity arithmetic differs",
        )
        _finite_number(
            record.get("correct_state_semantic_nll"),
            description="row semantic NLL",
        )
        by_ordinal[ordinal] = record
    if require_complete:
        _require(
            list(sorted(by_ordinal)) == list(range(TRAIN32_ROWS)),
            "V15 checkpoint audit must contain all Train32 rows",
        )
    return by_ordinal


def build_checkpoint_summary(
    *,
    checkpoint: Mapping[str, Any],
    records: Sequence[Mapping[str, Any]],
    records_binding: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build the metrics used by the fixed lexicographic selector."""

    step = checkpoint.get("checkpoint_step")
    _require(step in CHECKPOINT_STEPS, "V15 checkpoint summary step differs")
    _require(len(records) == TRAIN32_ROWS, "V15 checkpoint summary requires Train32")
    ordinals = [record.get("train_row_ordinal") for record in records]
    _require(
        ordinals == list(range(TRAIN32_ROWS)),
        "V15 checkpoint summary rows must be in Train32 order",
    )
    margins = [
        _finite_number(record.get("identity_logit_margin"), description="summary margin")
        for record in records
    ]
    hinges = [
        _finite_number(record.get("identity_hinge"), description="summary hinge")
        for record in records
    ]
    semantic_nlls = [
        _finite_number(
            record.get("correct_state_semantic_nll"),
            description="summary semantic NLL",
        )
        for record in records
    ]
    summary: dict[str, Any] = {
        "schema": CHECKPOINT_SUMMARY_SCHEMA,
        "checkpoint_step": step,
        "checkpoint": dict(checkpoint),
        "post_save_audit": (
            None if records_binding is None else dict(records_binding)
        ),
        "saved_weight_timing": "reloaded_post_optimizer_update_checkpoint",
        "condition": CONDITION,
        "rows": TRAIN32_ROWS,
        "correct_state_parsed_boundary_exact_rows": sum(
            record.get("parsed_boundary_exact") is True for record in records
        ),
        "raw_token_exact_rows_telemetry": sum(
            record.get("raw_token_exact_telemetry") is True for record in records
        ),
        "identity_own_beats_paired_rows": sum(margin > 0.0 for margin in margins),
        "identity_margin_satisfied_rows": sum(
            margin >= IDENTITY_MARGIN for margin in margins
        ),
        "mean_identity_logit_margin": sum(margins) / TRAIN32_ROWS,
        "mean_identity_hinge": sum(hinges) / TRAIN32_ROWS,
        "mean_correct_state_semantic_nll": sum(semantic_nlls) / TRAIN32_ROWS,
        "identity_margin": IDENTITY_MARGIN,
        "rank_order": list(RANK_ORDER),
        "donor_generation_used": False,
        "zero_state_generation_used": False,
        "benchmark_evidence_used": False,
        "hard32_access": HARD32_ACCESS_POLICY,
    }
    summary["summary_sha256"] = self_hash_payload(summary, field="summary_sha256")
    return summary


def checkpoint_rank_key(summary: Mapping[str, Any]) -> tuple[Any, ...]:
    """Return an ascending sort key for the frozen lexicographic policy."""

    _require(
        summary.get("schema") == CHECKPOINT_SUMMARY_SCHEMA,
        "V15 checkpoint summary schema differs",
    )
    _validate_self_hash(summary, field="summary_sha256")
    step = summary.get("checkpoint_step")
    _require(step in CHECKPOINT_STEPS, "V15 selector summary step differs")
    return (
        -int(summary["correct_state_parsed_boundary_exact_rows"]),
        -int(summary["identity_own_beats_paired_rows"]),
        -int(summary["identity_margin_satisfied_rows"]),
        -_finite_number(
            summary["mean_identity_logit_margin"],
            description="mean identity margin",
        ),
        _finite_number(summary["mean_identity_hinge"], description="mean hinge"),
        _finite_number(
            summary["mean_correct_state_semantic_nll"],
            description="mean semantic NLL",
        ),
        int(step),
    )


def select_checkpoint(
    summaries: Sequence[Mapping[str, Any]],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    _require(len(summaries) == len(CHECKPOINT_STEPS), "V15 selector requires checkpoints 1-4")
    copied = [dict(summary) for summary in summaries]
    _require(
        [summary.get("checkpoint_step") for summary in copied]
        == list(CHECKPOINT_STEPS),
        "V15 selector summaries must be checkpoints 1-4 in order",
    )
    ranked = sorted(copied, key=checkpoint_rank_key)
    return ranked[0], ranked


def _validate_manifest_receipt_chain(
    receipt: Mapping[str, Any],
    *,
    run_name: str,
) -> dict[str, Any]:
    manifest_binding = receipt.get("selector_manifest")
    _require(
        isinstance(manifest_binding, Mapping),
        "V15 selection selector manifest binding is missing",
    )
    manifest = validate_selector_manifest_binding(
        manifest_binding,
        run_name=run_name,
    )
    fingerprint_payload = manifest["fingerprint_payload"]
    candidates = receipt.get("candidates")
    train_inputs = receipt.get("train_inputs")
    _require(
        isinstance(candidates, list)
        and all(isinstance(summary, Mapping) for summary in candidates)
        and isinstance(train_inputs, Mapping),
        "V15 selection manifest receipt chain is incomplete",
    )
    expected_checkpoints = [dict(summary["checkpoint"]) for summary in candidates]
    _require(
        receipt.get("fingerprint") == manifest["validated_fingerprint"]
        and fingerprint_payload.get("checkpoints") == expected_checkpoints
        and fingerprint_payload.get("launch_receipt")
        == receipt.get("launch_receipt")
        and fingerprint_payload.get("completion_receipt")
        == receipt.get("completion_receipt")
        and fingerprint_payload.get("base_model") == receipt.get("base_model")
        and fingerprint_payload.get("runtime") == receipt.get("runtime")
        and fingerprint_payload.get("train32_sha256")
        == train_inputs.get("train32_sha256")
        and fingerprint_payload.get("train32_rows_sha256")
        == train_inputs.get("train32_rows_sha256")
        and fingerprint_payload.get("pair_manifest_sha256")
        == train_inputs.get("pair_manifest_sha256")
        and fingerprint_payload.get("source_manifest_sha256")
        == train_inputs.get("source_manifest_sha256")
        and fingerprint_payload.get("source_lock") == train_inputs.get("source_lock")
        and fingerprint_payload.get("hard32_access") == HARD32_ACCESS_POLICY,
        "V15 selector manifest and selection receipt differ",
    )
    _validate_frozen_selector_runtime(receipt.get("runtime", {}))
    return manifest


def build_selection_receipt(
    *,
    fingerprint: str,
    train_inputs: Mapping[str, Any],
    launch_receipt: Mapping[str, Any],
    completion_receipt: Mapping[str, Any],
    selector_manifest: Mapping[str, Any],
    base_model: Mapping[str, Any],
    summaries: Sequence[Mapping[str, Any]],
    runtime: Mapping[str, Any],
    created_at: str | None = None,
) -> dict[str, Any]:
    selected, ranked = select_checkpoint(summaries)
    receipt: dict[str, Any] = {
        "schema": SELECTION_RECEIPT_SCHEMA,
        "created_at": utc_now() if created_at is None else created_at,
        "status": "complete",
        "contract": SELECTOR_CONTRACT,
        "fingerprint": fingerprint,
        "selection_policy": SELECTION_POLICY,
        "rank_order": list(RANK_ORDER),
        "saved_weight_timing": "reloaded_post_optimizer_update_checkpoint",
        "checkpoint_steps": list(CHECKPOINT_STEPS),
        "train_inputs": dict(train_inputs),
        "launch_receipt": dict(launch_receipt),
        "completion_receipt": dict(completion_receipt),
        "selector_manifest": dict(selector_manifest),
        "base_model": dict(base_model),
        "runtime": dict(runtime),
        "candidates": [dict(summary) for summary in summaries],
        "ranked_checkpoint_steps": [
            int(summary["checkpoint_step"]) for summary in ranked
        ],
        "selected_checkpoint_step": int(selected["checkpoint_step"]),
        "selected_checkpoint": dict(selected["checkpoint"]),
        "selected_summary_sha256": selected["summary_sha256"],
        "selection_is_threshold_free": True,
        "donor_generation_used": False,
        "zero_state_generation_used": False,
        "benchmark_evidence_used": False,
        "hard32_access": HARD32_ACCESS_POLICY,
        "authorization": {
            "authorization_kind": HARD32_AUTHORIZATION_KIND,
            "scope": HARD32_AUTHORIZATION_SCOPE,
            "hard32_authorized": True,
            "selected_checkpoint_only": True,
            "full170_authorized": False,
            "test_authorized": False,
            "other_benchmarks_authorized": False,
        },
    }
    run_name = _preflight_selection_artifact_paths(receipt)
    _validate_manifest_receipt_chain(receipt, run_name=run_name)
    receipt["receipt_sha256"] = self_hash_payload(receipt, field="receipt_sha256")
    return receipt


def validate_selection_receipt(
    receipt: Path | str | Mapping[str, Any],
) -> dict[str, Any]:
    if isinstance(receipt, Mapping):
        payload = dict(receipt)
        receipt_path = None
        receipt_run_name = None
    else:
        receipt_path, receipt_run_name = _selection_document_path(
            receipt,
            filename=SELECTION_RECEIPT_FILENAME,
            description="selection receipt",
        )
        payload = _read_locked_json(receipt_path, description="selection receipt")
    receipt_sha256 = _validate_self_hash(payload, field="receipt_sha256")
    _require(
        payload.get("schema") == SELECTION_RECEIPT_SCHEMA
        and payload.get("status") == "complete"
        and payload.get("contract") == SELECTOR_CONTRACT
        and payload.get("selection_policy") == SELECTION_POLICY
        and payload.get("rank_order") == list(RANK_ORDER)
        and payload.get("saved_weight_timing")
        == "reloaded_post_optimizer_update_checkpoint"
        and payload.get("checkpoint_steps") == list(CHECKPOINT_STEPS)
        and payload.get("selection_is_threshold_free") is True
        and payload.get("donor_generation_used") is False
        and payload.get("zero_state_generation_used") is False
        and payload.get("benchmark_evidence_used") is False
        and payload.get("hard32_access") == HARD32_ACCESS_POLICY,
        "V15 selection receipt contract differs",
    )
    authorization = payload.get("authorization")
    _require(
        authorization
        == {
            "authorization_kind": HARD32_AUTHORIZATION_KIND,
            "scope": HARD32_AUTHORIZATION_SCOPE,
            "hard32_authorized": True,
            "selected_checkpoint_only": True,
            "full170_authorized": False,
            "test_authorized": False,
            "other_benchmarks_authorized": False,
        },
        "V15 selection authorization scope differs",
    )
    train_inputs = payload.get("train_inputs")
    _require(
        isinstance(train_inputs, Mapping)
        and train_inputs.get("source_split") == "train"
        and train_inputs.get("rows") == TRAIN32_ROWS
        and train_inputs.get("train32_sha256") == data_contract.TRAIN32_SHA256
        and train_inputs.get("train32_rows_sha256")
        == data_contract.TRAIN32_ROWS_SHA256
        and train_inputs.get("pair_manifest_sha256")
        == data_contract.PAIR_MANIFEST_FILE_SHA256
        and train_inputs.get("source_manifest_sha256")
        == data_contract.SOURCE_MANIFEST_FILE_SHA256
        and train_inputs.get("hard32_rows") == 0,
        "V15 selection Train32 input contract differs",
    )
    candidates = payload.get("candidates")
    _require(
        isinstance(candidates, list) and len(candidates) == len(CHECKPOINT_STEPS),
        "V15 selection candidates differ",
    )
    run_name = _preflight_selection_artifact_paths(
        payload,
        expected_run_name=receipt_run_name,
    )
    _validate_manifest_receipt_chain(payload, run_name=run_name)
    selected, ranked = select_checkpoint(candidates)
    _require(
        payload.get("ranked_checkpoint_steps")
        == [int(summary["checkpoint_step"]) for summary in ranked]
        and payload.get("selected_checkpoint_step") == selected["checkpoint_step"]
        and payload.get("selected_checkpoint") == selected["checkpoint"]
        and payload.get("selected_summary_sha256") == selected["summary_sha256"],
        "V15 selection result does not reproduce",
    )
    train_contract = _selector_train_contract()
    rows = train_contract["rows"]
    donor_by_ordinal = train_contract["pairing"]["donor_by_ordinal"]
    _require(len(rows) == TRAIN32_ROWS, "V15 selection input is not Train32")
    for summary in candidates:
        binding = summary.get("post_save_audit")
        _require(isinstance(binding, Mapping), "V15 post-save audit binding is missing")
        audit_path = validate_artifact_binding(
            binding,
            description=f"checkpoint-{summary['checkpoint_step']} post-save audit",
            expected_path=(
                _lexical_absolute(SELECTION_ROOT)
                / run_name
                / f"checkpoint-{summary['checkpoint_step']}.train32.jsonl"
            ),
        )
        records_by_ordinal = validate_post_save_row_records(
            _load_jsonl(audit_path, expected_path=audit_path),
            checkpoint_step=int(summary["checkpoint_step"]),
            fingerprint=str(payload.get("fingerprint", "")),
            rows=rows,
            donor_by_ordinal=donor_by_ordinal,
            require_complete=True,
        )
        checkpoint = summary.get("checkpoint")
        _require(isinstance(checkpoint, Mapping), "V15 selected checkpoint binding is missing")
        artifacts = checkpoint.get("artifacts")
        _require(isinstance(artifacts, Mapping), "V15 checkpoint artifacts are missing")
        checkpoint_path = (
            _lexical_absolute(RUN_ROOT)
            / run_name
            / "trainer"
            / f"checkpoint-{summary['checkpoint_step']}"
        )
        for name in _required_checkpoint_artifact_names():
            validate_artifact_binding(
                artifacts[name],
                description=f"checkpoint-{summary['checkpoint_step']} {name}",
                expected_path=checkpoint_path / name,
            )
        rng_artifacts = checkpoint.get("rng_state_artifacts")
        _require(isinstance(rng_artifacts, Mapping), "V15 checkpoint RNG artifacts are missing")
        for name, binding_record in rng_artifacts.items():
            validate_artifact_binding(
                binding_record,
                description=f"checkpoint-{summary['checkpoint_step']} {name}",
                expected_path=checkpoint_path / name,
            )
        rebuilt = build_checkpoint_summary(
            checkpoint=checkpoint,
            records=[records_by_ordinal[index] for index in range(TRAIN32_ROWS)],
            records_binding=binding,
        )
        _require(rebuilt == summary, "V15 checkpoint summary does not reproduce")
    validate_artifact_binding(
        payload["launch_receipt"],
        description="V15 launch_receipt",
        expected_path=_lexical_absolute(RUN_ROOT) / "logs" / f"{run_name}.launch.json",
    )
    validate_artifact_binding(
        payload["completion_receipt"],
        description="V15 completion_receipt",
        expected_path=(
            _lexical_absolute(RUN_ROOT) / "logs" / f"{run_name}.completion.json"
        ),
    )
    result = dict(payload)
    if receipt_path is not None:
        result["receipt_path"] = str(receipt_path)
        result["receipt_file_sha256"] = sha256_file(receipt_path)
    result["selection_run_name"] = run_name
    result["validated_receipt_sha256"] = receipt_sha256
    return result


def default_hard32_output_dir(
    *,
    run_name: str,
    selected_checkpoint_step: int,
) -> Path:
    _require(
        run_name and Path(run_name).name == run_name,
        "V15 run name is invalid",
    )
    _require(
        selected_checkpoint_step in CHECKPOINT_STEPS,
        "V15 selected checkpoint step differs",
    )
    return HARD32_OUTPUT_ROOT / f"{run_name}_checkpoint-{selected_checkpoint_step}"


def validated_hard32_output_dir(
    requested: Path | None,
    *,
    run_name: str,
    selected_checkpoint_step: int,
) -> Path:
    expected = _lexical_absolute(
        default_hard32_output_dir(
            run_name=run_name,
            selected_checkpoint_step=selected_checkpoint_step,
        )
    )
    if requested is not None:
        _require(
            _lexical_absolute(requested) == expected,
            "V15 --hard32-output-dir must equal the canonical selected-checkpoint path",
        )
    return expected


def build_candidate_lock(
    *,
    selection_receipt: Mapping[str, Any],
    selection_receipt_binding: Mapping[str, Any],
    hard32_output_dir: Path,
) -> dict[str, Any]:
    run_name = _preflight_selection_artifact_paths(selection_receipt)
    expected_receipt_path = (
        _lexical_absolute(SELECTION_ROOT) / run_name / SELECTION_RECEIPT_FILENAME
    )
    _require_binding_path(
        selection_receipt_binding,
        expected_path=expected_receipt_path,
        description="selection receipt",
    )
    validated = validate_selection_receipt(selection_receipt)
    selected_step = int(validated["selected_checkpoint_step"])
    selected_summary = next(
        summary
        for summary in validated["candidates"]
        if summary["checkpoint_step"] == selected_step
    )
    expected_receipt_sha = validated["validated_receipt_sha256"]
    receipt_path = validate_artifact_binding(
        selection_receipt_binding,
        description="selection receipt",
        expected_path=expected_receipt_path,
    )
    loaded = json.loads(receipt_path.read_text(encoding="utf-8"))
    _require(
        isinstance(loaded, dict)
        and loaded == dict(selection_receipt)
        and loaded.get("receipt_sha256") == expected_receipt_sha,
        "V15 candidate lock selection receipt payload differs",
    )
    output = _lexical_absolute(hard32_output_dir)
    expected_output = _lexical_absolute(
        default_hard32_output_dir(
            run_name=run_name,
            selected_checkpoint_step=selected_step,
        )
    )
    _require(
        output == expected_output,
        "V15 candidate Hard32 output must use the canonical selected-checkpoint path",
    )
    lock: dict[str, Any] = {
        "schema": CANDIDATE_LOCK_SCHEMA,
        "authorization_kind": HARD32_AUTHORIZATION_KIND,
        "selection_policy": SELECTION_POLICY,
        "candidate_count": 1,
        "rejected_checkpoint_steps": [
            step for step in CHECKPOINT_STEPS if step != selected_step
        ],
        "hard32_output_dir": str(output),
        "selected_candidate": {
            "checkpoint_step": selected_step,
            "checkpoint": dict(validated["selected_checkpoint"]),
            "post_save_audit": dict(selected_summary["post_save_audit"]),
            "summary_sha256": selected_summary["summary_sha256"],
            "selection_receipt": dict(selection_receipt_binding),
            "selection_receipt_payload_sha256": expected_receipt_sha,
            "selector_manifest": dict(validated["selector_manifest"]),
            "selector_manifest_fingerprint": validated["fingerprint"],
            "launch_receipt": dict(validated["launch_receipt"]),
            "completion_receipt": dict(validated["completion_receipt"]),
            "base_model": dict(validated["base_model"]),
        },
        "train_only_selection": True,
        "benchmark_evidence_used": False,
        "hard32_access_during_selection": HARD32_ACCESS_POLICY,
        "authorization": dict(validated["authorization"]),
    }
    lock["lock_sha256"] = self_hash_payload(lock, field="lock_sha256")
    return lock


def validate_candidate_lock(
    lock: Path | str | Mapping[str, Any],
    *,
    selection_receipt: Path | str | Mapping[str, Any],
) -> dict[str, Any]:
    if isinstance(lock, Mapping):
        payload = dict(lock)
        lock_path = None
        lock_run_name = None
    else:
        lock_path, lock_run_name = _selection_document_path(
            lock,
            filename=CANDIDATE_LOCK_FILENAME,
            description="candidate lock",
        )
        payload = _read_locked_json(lock_path, description="candidate lock")
    lock_sha256 = _validate_self_hash(payload, field="lock_sha256")
    if isinstance(selection_receipt, Mapping):
        receipt_payload = dict(selection_receipt)
        receipt_path = None
        receipt_run_name = None
    else:
        receipt_path, receipt_run_name = _selection_document_path(
            selection_receipt,
            filename=SELECTION_RECEIPT_FILENAME,
            description="candidate-lock selection receipt",
        )
        if lock_run_name is not None:
            _require(
                receipt_run_name == lock_run_name,
                "V15 candidate lock and selection receipt run names differ",
            )
        receipt_payload = _read_locked_json(
            receipt_path,
            description="candidate-lock selection receipt",
        )
    preflight_run_name = _preflight_selection_artifact_paths(
        receipt_payload,
        expected_run_name=receipt_run_name,
    )
    selected_for_preflight = payload.get("selected_candidate")
    _require(
        isinstance(selected_for_preflight, Mapping),
        "V15 candidate lock selected candidate is missing",
    )
    _require_binding_path(
        selected_for_preflight.get("selection_receipt", {}),
        expected_path=(
            _lexical_absolute(SELECTION_ROOT)
            / preflight_run_name
            / SELECTION_RECEIPT_FILENAME
        ),
        description="candidate-lock selection receipt",
    )
    _require_binding_path(
        selected_for_preflight.get("selector_manifest", {}),
        expected_path=(
            _lexical_absolute(SELECTION_ROOT)
            / preflight_run_name
            / SELECTOR_MANIFEST_FILENAME
        ),
        description="candidate-lock selector manifest",
    )
    validated_receipt = validate_selection_receipt(receipt_payload)
    _require(
        lock_run_name in (None, validated_receipt["selection_run_name"]),
        "V15 candidate lock run name differs",
    )
    selected = payload.get("selected_candidate")
    _require(
        payload.get("schema") == CANDIDATE_LOCK_SCHEMA
        and payload.get("authorization_kind") == HARD32_AUTHORIZATION_KIND
        and payload.get("selection_policy") == SELECTION_POLICY
        and payload.get("candidate_count") == 1
        and payload.get("train_only_selection") is True
        and payload.get("benchmark_evidence_used") is False
        and payload.get("hard32_access_during_selection") == HARD32_ACCESS_POLICY
        and payload.get("authorization") == validated_receipt["authorization"]
        and isinstance(selected, Mapping),
        "V15 candidate lock contract differs",
    )
    selected_step = validated_receipt["selected_checkpoint_step"]
    _require(
        payload.get("rejected_checkpoint_steps")
        == [step for step in CHECKPOINT_STEPS if step != selected_step]
        and selected.get("checkpoint_step") == selected_step
        and selected.get("checkpoint") == validated_receipt["selected_checkpoint"]
        and selected.get("selection_receipt_payload_sha256")
        == validated_receipt["validated_receipt_sha256"]
        and selected.get("selector_manifest")
        == validated_receipt["selector_manifest"]
        and selected.get("selector_manifest_fingerprint")
        == validated_receipt["fingerprint"]
        and selected.get("launch_receipt") == validated_receipt["launch_receipt"]
        and selected.get("completion_receipt")
        == validated_receipt["completion_receipt"]
        and selected.get("base_model") == validated_receipt["base_model"],
        "V15 candidate lock selected checkpoint differs",
    )
    if isinstance(selection_receipt, Mapping):
        validate_artifact_binding(
            selected.get("selection_receipt", {}),
            description="candidate-lock selection receipt",
            expected_path=(
                _lexical_absolute(SELECTION_ROOT)
                / validated_receipt["selection_run_name"]
                / SELECTION_RECEIPT_FILENAME
            ),
        )
    else:
        validate_artifact_binding(
            selected.get("selection_receipt", {}),
            description="candidate-lock selection receipt",
            expected_path=receipt_path,
        )
    hard32_output = _lexical_absolute(str(payload.get("hard32_output_dir", "")))
    expected_hard32_output = _lexical_absolute(
        default_hard32_output_dir(
            run_name=validated_receipt["selection_run_name"],
            selected_checkpoint_step=int(selected_step),
        )
    )
    _require(
        hard32_output == expected_hard32_output,
        "V15 candidate lock Hard32 output directory differs",
    )
    expected = build_candidate_lock(
        selection_receipt=receipt_payload,
        selection_receipt_binding=selected["selection_receipt"],
        hard32_output_dir=hard32_output,
    )
    _require(payload == expected, "V15 candidate lock is not canonical")
    result = dict(payload)
    if lock_path is not None:
        result["candidate_lock_path"] = str(lock_path)
        result["candidate_lock_file_sha256"] = sha256_file(lock_path)
    result["validated_lock_sha256"] = lock_sha256
    return result


def _load_jsonl(path: Path, *, expected_path: Path) -> list[dict[str, Any]]:
    path = _reject_protected_path(path, description="post-save audit")
    _require(
        path == _lexical_absolute(expected_path),
        "V15 post-save audit path differs",
    )
    _require_no_symlink_components(path, description="post-save audit")
    if not path.exists():
        return []
    records: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        _require(bool(line.strip()), f"V15 audit contains blank row {line_number}")
        value = json.loads(line)
        _require(isinstance(value, dict), f"V15 audit row {line_number} is not an object")
        records.append(value)
    return records


def _selector_train_contract() -> dict[str, Any]:
    root = data_contract.V7_ROOT
    return train32.validate_v7_contract(
        contract="scene_v7_train32_overfit",
        dataset_file=root / "train32.jsonl",
        row_manifest_file=root / "train32_rows.jsonl",
        pair_manifest_file=root / "train32_pair_manifest.json",
        source_manifest_file=root / "train32_source_manifest.json",
        expected_dataset_sha256=data_contract.TRAIN32_SHA256,
        expected_row_manifest_sha256=data_contract.TRAIN32_ROWS_SHA256,
        expected_pair_manifest_sha256=data_contract.PAIR_MANIFEST_FILE_SHA256,
        expected_source_manifest_sha256=data_contract.SOURCE_MANIFEST_FILE_SHA256,
        source_lock_file=train32.DEFAULT_SOURCE_LOCK,
    )


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-model", required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--launch-receipt", type=Path, required=True)
    parser.add_argument("--completion-receipt", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--hard32-output-dir", type=Path)
    parser.add_argument("--delta-mem-root", default=str(PROJECT_ROOT))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", default="bfloat16", choices=("bfloat16", "float16", "float32"))
    parser.add_argument("--attn-implementation", default="sdpa")
    parser.add_argument("--normal-fusion-profile", default="native", choices=("native",))
    parser.add_argument("--expected-memory-layer-count", type=int, default=42)
    parser.add_argument("--max-new-tokens", type=int, default=MAX_NEW_TOKENS)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args(argv)


def _validate_runtime_args(args: argparse.Namespace) -> None:
    _require(args.overwrite is False, "V15 selector forbids --overwrite")
    _require(args.device == "cuda:0", "V15 selector requires CUDA 0")
    _require(args.dtype == "bfloat16", "V15 selector requires bfloat16")
    _require(args.attn_implementation == "sdpa", "V15 selector requires SDPA")
    _require(args.normal_fusion_profile == "native", "V15 selector requires native fusion")
    _require(args.expected_memory_layer_count == 42, "V15 selector requires all 42 layers")
    _require(args.max_new_tokens == MAX_NEW_TOKENS, "V15 selector requires 128 new tokens")
    _require(
        _lexical_absolute(args.delta_mem_root) == PROJECT_ROOT,
        "V15 selector requires this checkout",
    )


def _validate_provenance(
    *,
    run_dir: Path,
    launch_receipt: Path,
    completion_receipt: Path,
    bundle: Mapping[str, Any],
    base_model: Path,
) -> tuple[
    list[dict[str, Any]],
    dict[str, Any],
    dict[str, Any],
    dict[str, Any],
]:
    """Validate all checkpoint and receipt artifacts through the V15 launcher."""

    from experiments.rethinking_rwkv_ms_gemma import (  # local to avoid startup cycles
        scene_memory_v15_launch_contract as launch,
    )

    validated_run = launch.require_v15_run_path(
        run_dir,
        description="v15_selector_training_run",
    )
    _require(validated_run == run_dir, "V15 selector training run path differs")
    data = launch.validate_data_contract()
    _require(
        data.get("source_manifest_sha256")
        == bundle.get("artifacts", {}).get("source_manifest", {}).get("sha256")
        and data.get("schedule_sha256")
        == bundle.get("artifacts", {}).get("pair_schedule", {}).get("sha256"),
        "V15 selector data provenance differs",
    )
    warm = launch.validate_warm_start_contract()
    base_model_identity = launch.validate_base_model_contract(base_model)
    checkpoints: list[dict[str, Any]] = []
    for step in CHECKPOINT_STEPS:
        checkpoint_path = run_dir / "trainer" / f"checkpoint-{step}"
        validated = launch.validate_checkpoint_contract(
            checkpoint_path,
            data=data,
            warm=warm,
            smoke=False,
        )
        _require(
            validated.get("checkpoint_step") == step
            and validated.get("path") == str(checkpoint_path),
            "V15 selector checkpoint contract differs",
        )
        checkpoints.append(dict(validated))
    launch_validation = launch.validate_launch_receipt(
        launch_receipt,
        checkpoint=Path(checkpoints[-1]["path"]),
        base_model_identity=base_model_identity,
    )
    completion_validation: dict[str, Any] | None = None
    for checkpoint in checkpoints:
        completion_validation = launch.validate_completion_receipt(
            completion_receipt,
            checkpoint=Path(checkpoint["path"]),
            checkpoint_contract=checkpoint,
            launch=launch_validation,
        )
    if completion_validation is None:  # pragma: no cover - frozen non-empty steps
        raise AssertionError("V15 selector checkpoint schedule is empty")
    return (
        checkpoints,
        launch_validation,
        completion_validation,
        base_model_identity,
    )


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    _validate_runtime_args(args)
    run_dir = _reject_protected_path(args.run_dir, description="training run")
    output_dir = _reject_protected_path(args.output_dir, description="selection output")
    _require(
        run_dir.parent == _lexical_absolute(RUN_ROOT)
        and bool(run_dir.name)
        and Path(run_dir.name).name == run_dir.name,
        "V15 training run must be one direct child of the locked run root",
    )
    _require(
        output_dir.parent == _lexical_absolute(SELECTION_ROOT),
        "V15 selection output must be one direct child of the locked selection root",
    )
    _require(output_dir.name == run_dir.name, "V15 selection output must use the run name")
    launch_receipt = _reject_protected_path(
        args.launch_receipt,
        description="launch receipt",
    )
    completion_receipt = _reject_protected_path(
        args.completion_receipt,
        description="completion receipt",
    )
    log_root = _lexical_absolute(RUN_ROOT) / "logs"
    _require(
        launch_receipt == log_root / f"{run_dir.name}.launch.json"
        and completion_receipt == log_root / f"{run_dir.name}.completion.json",
        "V15 launch/completion receipt paths differ",
    )
    for path, description in (
        (run_dir, "training run"),
        (output_dir, "selection output"),
        (launch_receipt, "launch receipt"),
        (completion_receipt, "completion receipt"),
    ):
        _require_no_symlink_components(path, description=description)
    bundle = data_contract.validate_bundle()
    train_contract = _selector_train_contract()
    rows = train_contract["rows"]
    donor_by_ordinal = train_contract["pairing"]["donor_by_ordinal"]
    _require(len(rows) == TRAIN32_ROWS, "V15 selector input is not Train32")
    base_model = require_ssd_path(args.base_model, description="base model")
    checkpoints, _, _, base_model_identity = _validate_provenance(
        run_dir=run_dir,
        launch_receipt=launch_receipt,
        completion_receipt=completion_receipt,
        bundle=bundle,
        base_model=base_model,
    )
    runtime = frozen_selector_runtime()
    launch_receipt_binding = artifact_binding(launch_receipt)
    completion_receipt_binding = artifact_binding(completion_receipt)
    fingerprint_payload = build_selector_fingerprint_payload(
        checkpoints=checkpoints,
        launch_receipt=launch_receipt_binding,
        completion_receipt=completion_receipt_binding,
        base_model=base_model_identity,
        source_lock=bundle["source_lock"],
        runtime=runtime,
    )
    fingerprint = canonical_sha256(fingerprint_payload)
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        step: output_dir / f"checkpoint-{step}.train32.jsonl"
        for step in CHECKPOINT_STEPS
    }
    paths["manifest"] = output_dir / SELECTOR_MANIFEST_FILENAME
    paths["receipt"] = output_dir / SELECTION_RECEIPT_FILENAME
    paths["lock"] = output_dir / CANDIDATE_LOCK_FILENAME
    if paths["manifest"].exists():
        existing = json.loads(paths["manifest"].read_text(encoding="utf-8"))
        existing_created_at = existing.get("created_at") if isinstance(existing, Mapping) else None
        expected_manifest = build_selector_manifest(
            fingerprint_payload=fingerprint_payload,
            created_at=(
                existing_created_at
                if isinstance(existing_created_at, str)
                else ""
            ),
        )
        _require(
            isinstance(existing, dict)
            and isinstance(existing_created_at, str)
            and bool(existing_created_at)
            and existing == expected_manifest,
            "V15 selector resume manifest differs",
        )
    else:
        atomic_write_json(
            paths["manifest"],
            build_selector_manifest(fingerprint_payload=fingerprint_payload),
        )
    summaries: list[dict[str, Any]] = []
    for checkpoint in checkpoints:
        step = int(checkpoint["checkpoint_step"])
        path = paths[step]
        completed = validate_post_save_row_records(
            _load_jsonl(path, expected_path=path),
            checkpoint_step=step,
            fingerprint=fingerprint,
            rows=rows,
            donor_by_ordinal=donor_by_ordinal,
            require_complete=False,
        )
        if len(completed) < TRAIN32_ROWS:
            args.memory_dir = Path(checkpoint["path"])
            args.base_model = str(base_model)
            expected_layers = resolved_memory_layer_count(
                args.memory_dir,
                args.expected_memory_layer_count,
            )
            architecture = memory_architecture_contract(args.memory_dir)
            _require(
                expected_layers == 42
                and architecture["target_layers"] == list(range(42))
                and architecture["delta_heads"] == ["q", "o"]
                and architecture["rank"] == 4
                and architecture["rwkv_ms_semantics_version"] == 2
                and architecture["memory_backend"] == "rwkv_ms",
                "V15 selector checkpoint architecture differs",
            )
            model, tokenizer, _ = load_adapter_model(args, expected_layers)
            try:
                train32.validate_runtime_prefixes(tokenizer, rows=rows)
                for ordinal, sample in enumerate(rows):
                    if ordinal in completed:
                        continue
                    donor = rows[donor_by_ordinal[ordinal]]
                    result = evaluate_condition(
                        model=model,
                        tokenizer=tokenizer,
                        sample=sample,
                        donor_sample=donor,
                        condition=CONDITION,
                        max_new_tokens=MAX_NEW_TOKENS,
                        device=args.device,
                        collect_semantic_nll=True,
                    )
                    completed[ordinal] = build_post_save_row_record(
                        checkpoint_step=step,
                        sample=sample,
                        donor_sample=donor,
                        result=result,
                        fingerprint=fingerprint,
                    )
                    atomic_write_jsonl(
                        path,
                        [completed[index] for index in sorted(completed)],
                    )
            finally:
                del model
                del tokenizer
                clear_model_memory()
        ordered = [completed[index] for index in range(TRAIN32_ROWS)]
        validate_post_save_row_records(
            ordered,
            checkpoint_step=step,
            fingerprint=fingerprint,
            rows=rows,
            donor_by_ordinal=donor_by_ordinal,
            require_complete=True,
        )
        summaries.append(
            build_checkpoint_summary(
                checkpoint=checkpoint,
                records=ordered,
                records_binding=artifact_binding(path),
            )
        )
    existing_receipt = (
        json.loads(paths["receipt"].read_text(encoding="utf-8"))
        if paths["receipt"].exists()
        else None
    )
    receipt = build_selection_receipt(
        fingerprint=fingerprint,
        train_inputs={
            "source_split": "train",
            "rows": TRAIN32_ROWS,
            "train32_sha256": data_contract.TRAIN32_SHA256,
            "train32_rows_sha256": data_contract.TRAIN32_ROWS_SHA256,
            "pair_manifest_sha256": data_contract.PAIR_MANIFEST_FILE_SHA256,
            "source_manifest_sha256": data_contract.SOURCE_MANIFEST_FILE_SHA256,
            "source_lock": bundle["source_lock"],
            "hard32_rows": 0,
        },
        launch_receipt=launch_receipt_binding,
        completion_receipt=completion_receipt_binding,
        selector_manifest=artifact_binding(paths["manifest"]),
        base_model=base_model_identity,
        summaries=summaries,
        runtime=runtime,
        created_at=(
            existing_receipt.get("created_at")
            if isinstance(existing_receipt, Mapping)
            else None
        ),
    )
    if existing_receipt is None:
        atomic_write_json(paths["receipt"], receipt)
    else:
        _require(existing_receipt == receipt, "V15 selection receipt resume differs")
    hard32_output = validated_hard32_output_dir(
        args.hard32_output_dir,
        run_name=run_dir.name,
        selected_checkpoint_step=int(receipt["selected_checkpoint_step"]),
    )
    lock = build_candidate_lock(
        selection_receipt=receipt,
        selection_receipt_binding=artifact_binding(paths["receipt"]),
        hard32_output_dir=hard32_output,
    )
    if paths["lock"].exists():
        existing_lock = json.loads(paths["lock"].read_text(encoding="utf-8"))
        _require(existing_lock == lock, "V15 candidate lock resume differs")
    else:
        atomic_write_json(paths["lock"], lock)
    print(
        json.dumps(
            {
                "status": "complete",
                "selected_checkpoint_step": receipt["selected_checkpoint_step"],
                "ranked_checkpoint_steps": receipt["ranked_checkpoint_steps"],
                "selection_receipt": str(paths["receipt"]),
                "candidate_lock": str(paths["lock"]),
                "hard32_output_dir": str(hard32_output),
                "hard32_executed": False,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
