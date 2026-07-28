from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from experiments.rethinking_rwkv_ms_gemma import analyze_novel_agent_eval as analysis


def metric_names(kind: str) -> tuple[str, str]:
    if kind == "attribution":
        return "strict_accuracy", "format_recovered_accuracy"
    if kind == "narrative":
        return "strict_accuracy", "format_recovered_unit_accuracy"
    return "strict_f1", "format_recovered_micro_f1"


def score_from_contribution(kind: str, contribution: tuple[int, ...]) -> dict[str, Any]:
    if kind == "attribution":
        return {"correct": bool(contribution[0])}
    if kind == "narrative":
        return {
            "correct_units": contribution[0],
            "gold_units": contribution[1],
        }
    return {
        "tp": contribution[0],
        "fp": contribution[1],
        "fn": contribution[2],
    }


def build_inputs(
    specs: tuple[analysis.TaskSpec, ...],
    base_contributions: dict[str, list[tuple[int, ...]]],
    normal_contributions: dict[str, list[tuple[int, ...]]],
    *,
    base_predictions: dict[str, list[Any]] | None = None,
    normal_predictions: dict[str, list[Any]] | None = None,
    base_hits: dict[str, list[bool]] | None = None,
    normal_hits: dict[str, list[bool]] | None = None,
) -> dict[str, Any]:
    strict_summary: dict[str, Any] = {
        "conditions": {"base": {}, "normal": {}}
    }
    metrics: dict[str, dict[str, Any]] = {"base": {}, "normal": {}}
    predictions: dict[str, dict[str, list[Any]]] = {"base": {}, "normal": {}}
    contributions = {
        "base": base_contributions,
        "normal": normal_contributions,
    }
    records_by_condition: dict[str, dict[str, dict[str, Any]]] = {
        "base": {},
        "normal": {},
    }
    bootstraps: dict[str, dict[str, Any]] = {}
    for spec in specs:
        strict_name, recovered_name = metric_names(spec.kind)
        task_base = base_contributions[spec.name]
        task_normal = normal_contributions[spec.name]
        if len(task_base) != spec.expected_rows or len(task_normal) != spec.expected_rows:
            raise AssertionError("Synthetic contribution count differs from task spec")
        task_base_predictions = (
            base_predictions[spec.name]
            if base_predictions is not None and spec.name in base_predictions
            else [object() for _ in task_base]
        )
        task_normal_predictions = (
            normal_predictions[spec.name]
            if normal_predictions is not None and spec.name in normal_predictions
            else [object() for _ in task_normal]
        )
        predictions["base"][spec.name] = task_base_predictions
        predictions["normal"][spec.name] = task_normal_predictions
        for condition, task_contributions in (
            ("base", task_base),
            ("normal", task_normal),
        ):
            primary_metric = analysis.metric_from_contributions(
                spec.kind,
                task_contributions,
            )
            strict_summary["conditions"][condition][spec.name] = {
                "primary_metric": primary_metric,
                "primary_metric_name": strict_name,
            }
            metrics[condition][spec.name] = {
                "primary_metric": primary_metric,
                "primary_metric_name": recovered_name,
            }
            hit_values = (
                base_hits.get(spec.name, [False] * spec.expected_rows)
                if condition == "base" and base_hits is not None
                else normal_hits.get(spec.name, [False] * spec.expected_rows)
                if condition == "normal" and normal_hits is not None
                else [False] * spec.expected_rows
            )
            for index, (contribution, hit_max) in enumerate(
                zip(task_contributions, hit_values, strict=True)
            ):
                records_by_condition[condition][f"{spec.name}:{index}"] = {
                    "score": score_from_contribution(spec.kind, contribution),
                    "hit_max_new_tokens": hit_max,
                }
        bootstraps[spec.name] = {
            "metric_name": recovered_name,
            "base": analysis.metric_from_contributions(spec.kind, task_base),
            "normal": analysis.metric_from_contributions(spec.kind, task_normal),
            **analysis.paired_bootstrap(spec.kind, task_base, task_normal),
        }
    return {
        "strict_summary": strict_summary,
        "metrics": metrics,
        "predictions": predictions,
        "contributions": contributions,
        "records_by_condition": records_by_condition,
        "all_row_bootstraps": bootstraps,
    }


def test_clean_narrative_selection_excludes_val_index_25(monkeypatch) -> None:
    monkeypatch.setattr(analysis, "BOOTSTRAP_REPLICATES", 100)
    specs = (
        analysis.TaskSpec("attribution-v3.2", "unused", "attribution", 2),
        analysis.TaskSpec("narrative-v3.2", "unused", "narrative", 26),
        analysis.TaskSpec("scene-v4-current", "unused", "scene", 2),
    )
    base = {
        "attribution-v3.2": [(1, 1), (0, 1)],
        "narrative-v3.2": [(1, 1)] * 26,
        "scene-v4-current": [(1, 0, 0)] * 2,
    }
    normal = {
        **base,
        "narrative-v3.2": [(1, 1)] * 25 + [(0, 1)],
    }
    inputs = build_inputs(specs, base, normal)

    result = analysis.build_selection_criterion(
        split="val",
        specs=specs,
        **inputs,
    )

    narrative = result["tasks"]["narrative-v3.2"]
    assert narrative["evaluated_rows"] == 26
    assert narrative["selection_rows"] == 25
    assert narrative["applied_excluded_zero_based_indices"] == [25]
    assert narrative["all_rows"]["strict"]["normal_minus_base"] == pytest.approx(-1 / 26)
    assert narrative["all_rows"]["recovered"]["normal_minus_base"] == pytest.approx(-1 / 26)
    assert narrative["clean_selection"]["strict"]["normal_minus_base"] == 0.0
    assert narrative["clean_selection"]["recovered"]["normal_minus_base"] == 0.0
    assert narrative["gates"]["recovered_metric_delta_floor"]["passed"] is True


def test_selection_marks_each_failed_gate_and_missing_tasks(monkeypatch) -> None:
    monkeypatch.setattr(analysis, "BOOTSTRAP_REPLICATES", 100)
    specs = (
        analysis.TaskSpec("attribution-v3.2", "unused", "attribution", 2),
    )
    base = {"attribution-v3.2": [(1, 1), (1, 1)]}
    normal = {"attribution-v3.2": [(0, 1), (0, 1)]}
    inputs = build_inputs(
        specs,
        base,
        normal,
        base_predictions={"attribution-v3.2": ["a", "b"]},
        normal_predictions={"attribution-v3.2": [None, None]},
        base_hits={"attribution-v3.2": [False, False]},
        normal_hits={"attribution-v3.2": [True, True]},
    )

    result = analysis.build_selection_criterion(
        split="val",
        specs=specs,
        **inputs,
    )

    attribution = result["tasks"]["attribution-v3.2"]
    assert attribution["status"] == "provisional_fail"
    assert all(gate["passed"] is False for gate in attribution["gates"].values())
    assert result["tasks"]["narrative-v3.2"]["status"] == "not_evaluated"
    assert result["tasks"]["scene-v4-current"]["status"] == "not_evaluated"
    assert result["status"] == "incomplete"
    assert result["complete"] is False
    assert result["overall_passed"] is False


def test_full_validation_selection_passes_equal_model_metrics(monkeypatch) -> None:
    monkeypatch.setattr(analysis, "BOOTSTRAP_REPLICATES", 100)
    specs = (
        analysis.TaskSpec("attribution-v3.2", "unused", "attribution", 30),
        analysis.TaskSpec("narrative-v3.2", "unused", "narrative", 39),
        analysis.TaskSpec("scene-v4-current", "unused", "scene", 170),
    )
    base = {
        "attribution-v3.2": [(1, 1)] * 30,
        "narrative-v3.2": [(1, 1)] * 39,
        "scene-v4-current": [(1, 0, 0)] * 170,
    }
    inputs = build_inputs(specs, base, dict(base))

    result = analysis.build_selection_criterion(
        split="val",
        specs=specs,
        **inputs,
    )

    assert result["status"] == "pass"
    assert result["complete"] is True
    assert result["all_gates_passed"] is True
    assert result["overall_passed"] is True
    assert all(
        result["tasks"][task_name]["criterion_passed"] is True
        for task_name in analysis.CORE_SELECTION_TASKS
    )


def scene_v6_contract(split: str, rows: int) -> dict[str, Any]:
    name = "scene_v6_validation" if split == "val" else "scene_v6_final_test"
    contract = {
        "name": name,
        "phase": "validation_selection" if split == "val" else "final_test",
        "split": split,
        "task": "scene-v4-current",
        "rows": rows,
        "conditions": ["base", "normal", "no_write"],
        "normal_fusion_profile": "native",
        "expected_memory_layer_count": 42,
        "memory_target_layers": list(range(42)),
        "memory_delta_heads": ["q", "o"],
        "memory_rank": 4,
        "rwkv_ms_semantics_version": 2,
        "memory_backend": "rwkv_ms",
        "official_dataset_revision": analysis.OFFICIAL_SCENE_V4_DATASET_REVISION,
        "official_dataset_sha256": analysis.OFFICIAL_SCENE_V4_SHA256[split],
        "overwrite_allowed": split == "val",
        "generation_policy": (
            "Append-only resumable records; completed keys are never regenerated."
        ),
    }
    if split == "test":
        contract.update(
            {
                "checkpoint_selection_forbidden": True,
                "test_once_enforcement_scope": (
                    "per_output_directory_and_fingerprint"
                ),
                "test_once_enforcement_caveat": (
                    "A new output directory can rerun inference; global single-use "
                    "enforcement is not provided. Checkpoint selection on test remains "
                    "forbidden."
                ),
            }
        )
    return contract


def scene_v6_gate_inputs(
    rows: int,
    *,
    normal_contribution: tuple[int, int, int] = (1, 0, 0),
    comparator_contribution: tuple[int, int, int] = (0, 0, 1),
    normal_predictions: list[Any] | None = None,
    normal_hits: list[bool] | None = None,
) -> dict[str, Any]:
    task_name = "scene-v4-current"
    contributions = {
        "base": {task_name: [comparator_contribution] * rows},
        "normal": {task_name: [normal_contribution] * rows},
        "no_write": {task_name: [comparator_contribution] * rows},
    }
    metrics = {
        condition: {
            task_name: {
                "primary_metric": analysis.metric_from_contributions(
                    "scene",
                    condition_contributions[task_name],
                ),
                "primary_metric_name": "format_recovered_micro_f1",
            }
        }
        for condition, condition_contributions in contributions.items()
    }
    predictions = {
        "base": {task_name: [set() for _ in range(rows)]},
        "normal": {
            task_name: (
                [set([1]) for _ in range(rows)]
                if normal_predictions is None
                else normal_predictions
            )
        },
        "no_write": {task_name: [set() for _ in range(rows)]},
    }
    records_by_condition = {condition: {} for condition in analysis.CONDITIONS}
    for condition in analysis.CONDITIONS:
        hits = normal_hits if condition == "normal" and normal_hits is not None else [False] * rows
        for index in range(rows):
            records_by_condition[condition][f"{task_name}:{index}"] = {
                "hit_max_new_tokens": hits[index]
            }
    samples = [
        analysis.DatasetSample(
            line_index=index,
            row_sha256=f"{index:064x}"[-64:],
            gold={"boundaries": [1]},
            candidates=(),
            paragraph_count=3,
        )
        for index in range(rows)
    ]
    return {
        "metrics": metrics,
        "predictions": predictions,
        "contributions": contributions,
        "records_by_condition": records_by_condition,
        "samples_by_task": {task_name: samples},
    }


def test_scene_v6_validation_gates_all_170_rows_and_paired_cis(monkeypatch) -> None:
    monkeypatch.setattr(analysis, "BOOTSTRAP_REPLICATES", 100)
    monkeypatch.setattr(analysis, "CONDITIONS", ("base", "normal", "no_write"))
    specs = (
        analysis.TaskSpec("scene-v4-current", "unused", "scene", 170),
    )
    inputs = scene_v6_gate_inputs(170)

    result = analysis.build_scene_v6_gate_analysis(
        contract=scene_v6_contract("val", 170),
        split="val",
        specs=specs,
        strict_summary={},
        **inputs,
    )

    assert result["status"] == "pass"
    assert result["all_official_rows_verified"] is True
    assert result["selection_authorized"] is True
    assert result["checkpoint_selection_forbidden"] is False
    assert result["aligned_qwen"]["status"] == "not_applicable"
    assert result["comparisons"]["base"]["ci_95_percentile"] == [1.0, 1.0]
    assert result["comparisons"]["no_write"]["ci_95_percentile"] == [1.0, 1.0]
    assert result["gates"]["normal_minus_no_write"]["passed"] is True


def test_scene_v6_final_test_uses_aligned_qwen_and_forbids_selection(monkeypatch) -> None:
    monkeypatch.setattr(analysis, "BOOTSTRAP_REPLICATES", 100)
    monkeypatch.setattr(analysis, "CONDITIONS", ("base", "normal", "no_write"))
    monkeypatch.setattr(
        analysis,
        "aligned_qwen_scene_reference",
        lambda strict_summary, samples, split: {
            "status": "aligned",
            "rows": len(samples),
            "source": "/aligned/qwen.json",
            "sha256": "a" * 64,
            "contributions": [(0, 0, 1)] * len(samples),
            "predictions": [set()] * len(samples),
            "micro_f1": 0.0,
        },
    )
    specs = (
        analysis.TaskSpec("scene-v4-current", "unused", "scene", 149),
    )

    result = analysis.build_scene_v6_gate_analysis(
        contract=scene_v6_contract("test", 149),
        split="test",
        specs=specs,
        strict_summary={},
        **scene_v6_gate_inputs(149),
    )

    assert result["status"] == "pass"
    assert result["selection_authorized"] is False
    assert result["checkpoint_selection_forbidden"] is True
    assert result["final_claim_authorized"] is True
    assert result["comparisons"]["aligned_qwen"]["ci_95_percentile"] == [1.0, 1.0]


def test_scene_v6_gate_fails_zero_ci_coverage_and_max_token_regression(monkeypatch) -> None:
    monkeypatch.setattr(analysis, "BOOTSTRAP_REPLICATES", 100)
    monkeypatch.setattr(analysis, "CONDITIONS", ("base", "normal", "no_write"))
    specs = (
        analysis.TaskSpec("scene-v4-current", "unused", "scene", 170),
    )
    normal_predictions = [None] * 10 + [set([1])] * 160
    normal_hits = [True] * 170

    result = analysis.build_scene_v6_gate_analysis(
        contract=scene_v6_contract("val", 170),
        split="val",
        specs=specs,
        strict_summary={},
        **scene_v6_gate_inputs(
            170,
            normal_contribution=(1, 0, 0),
            comparator_contribution=(1, 0, 0),
            normal_predictions=normal_predictions,
            normal_hits=normal_hits,
        ),
    )

    assert result["status"] == "fail"
    assert result["gates"]["normal_coverage"]["passed"] is False
    assert all(
        gate["passed"] is False
        for gate in result["gates"]["paired_ci_95_lower_strictly_positive"].values()
    )
    assert all(
        gate["passed"] is False
        for gate in result["gates"]["max_token_hit_rate_delta"].values()
    )


def test_aligned_qwen_scene_reference_validates_every_test_identity(
    tmp_path: Path,
) -> None:
    samples = [
        analysis.DatasetSample(0, "a" * 64, {"boundaries": [1]}, (), 3),
        analysis.DatasetSample(1, "b" * 64, {"boundaries": []}, (), 2),
    ]
    artifact = tmp_path / "scene_boundary_final.json"
    payload = {
        "v4-590": {
            "per_sample": [
                {"id": 0, "gold": [1], "pred": [1], "tp": 1, "fp": 0, "fn": 0, "paras": 3},
                {"id": 1, "gold": [], "pred": [1], "tp": 0, "fp": 1, "fn": 0, "paras": 2},
            ]
        }
    }
    artifact.write_text(json.dumps(payload), encoding="utf-8")
    digest = analysis.sha256_file(artifact)
    alignment = tmp_path / "qwen_alignment.json"
    alignment.write_text(
        json.dumps(
            {
                "schema": "scene_qwen_row_alignment.v1",
                "qwen_artifact_sha256": digest,
                "rows": [
                    {"id": index, "row_sha256": sample.row_sha256}
                    for index, sample in enumerate(samples)
                ],
            }
        ),
        encoding="utf-8",
    )
    strict_summary = {
        "references": {
            "scene-v4-current": {
                "artifact_source": str(artifact),
                "alignment_manifest_source": str(alignment),
                "alignment_manifest_sha256": analysis.sha256_file(alignment),
            },
            "source_hashes": {"scene_boundary_final.json": digest},
        }
    }

    result = analysis.aligned_qwen_scene_reference(
        strict_summary,
        samples,
        split="test",
    )

    assert result["status"] == "aligned"
    assert result["rows"] == 2
    assert result["contributions"] == [(1, 0, 0), (0, 1, 0)]

    payload["v4-590"]["per_sample"][1]["gold"] = [1]
    artifact.write_text(json.dumps(payload), encoding="utf-8")
    strict_summary["references"]["source_hashes"]["scene_boundary_final.json"] = (
        analysis.sha256_file(artifact)
    )
    with pytest.raises(ValueError, match="gold differs"):
        analysis.aligned_qwen_scene_reference(
            strict_summary,
            samples,
            split="test",
        )


def test_qwen_reference_without_row_hash_manifest_is_not_paired(
    tmp_path: Path,
) -> None:
    sample = analysis.DatasetSample(
        0,
        "a" * 64,
        {"boundaries": [1]},
        (),
        3,
    )
    artifact = tmp_path / "scene_boundary_final.json"
    artifact.write_text(
        json.dumps(
            {
                "v4-590": {
                    "per_sample": [
                        {
                            "id": 0,
                            "gold": [1],
                            "pred": [1],
                            "tp": 1,
                            "fp": 0,
                            "fn": 0,
                            "paras": 3,
                        }
                    ]
                }
            }
        ),
        encoding="utf-8",
    )
    result = analysis.aligned_qwen_scene_reference(
        {
            "references": {
                "scene-v4-current": {"artifact_source": str(artifact)},
                "source_hashes": {
                    "scene_boundary_final.json": analysis.sha256_file(artifact)
                },
            }
        },
        [sample],
        split="test",
    )

    assert result["status"] == "unverified_for_paired_ci"
    assert "source-row hashes" in result["reason"]


def test_strict_artifacts_bind_summary_references_to_fingerprint(
    tmp_path: Path,
) -> None:
    references = {"source_hashes": {"reference.json": "a" * 64}}
    contract = {"name": "scene_v6_validation"}
    payload = {
        "references": references,
        "evaluation_contract": contract,
        "split": "val",
        "normal_fusion_profile": "native",
    }
    fingerprint = analysis.fingerprint_payload_sha256(payload)
    manifest = {
        "fingerprint": fingerprint,
        "fingerprint_payload": payload,
        "references": references,
    }
    task_summaries = {
        spec.name: {"samples": spec.expected_rows}
        for spec in analysis.TASK_SPECS
    }
    summary = {
        "complete": True,
        "fingerprint": fingerprint,
        "references": references,
        "evaluation_contract": contract,
        "split": "val",
        "normal_fusion_profile": "native",
        "conditions": {
            condition: task_summaries for condition in analysis.CONDITIONS
        },
    }
    (tmp_path / "manifest.json").write_text(
        json.dumps(manifest),
        encoding="utf-8",
    )
    summary_path = tmp_path / "summary.json"
    summary_path.write_text(json.dumps(summary), encoding="utf-8")

    analysis.validate_strict_artifacts(tmp_path)
    summary_path.write_text(
        json.dumps({**summary, "references": {}}),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="references differs"):
        analysis.validate_strict_artifacts(tmp_path)
