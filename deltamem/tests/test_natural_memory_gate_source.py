from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from experiments.rethinking_rwkv_ms_gemma import prepare_natural_memory_gate as gate


def _row(system: str, user: str, answer: dict) -> dict:
    return {
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
            {"role": "assistant", "content": json.dumps(answer, ensure_ascii=False)},
        ]
    }


def _write(path: Path, rows: list[dict]) -> None:
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def _sources(tmp_path: Path) -> dict[str, Path]:
    paths = {
        "attribution": tmp_path / "attribution" / "train.jsonl",
        "narrative": tmp_path / "narrative" / "train.jsonl",
        "scene": tmp_path / "scene" / "train.jsonl",
    }
    attribution_rows = []
    narrative_rows = []
    scene_rows = []
    for index in range(100):
        unique_fill = hashlib.sha256(f"fixture-passage-{index}".encode()).hexdigest()
        passage = f"passage-{index}-{unique_fill}"
        attribution_rows.append(
            _row(
                "attribution system",
                "Candidates:\n- Alice\n- Bob\nContext:\n"
                f"[1] {passage}\n[2] attribution-context-{index}-{unique_fill}\n"
                "Target dialogue: who?",
                {"best_candidate": "Alice" if index % 2 == 0 else "Bob", "uncertain": False},
            )
        )
        narrative_rows.append(
            _row(
                "narrative system",
                f"[1] {passage}\n[2] narrative-unit-{index}-{unique_fill}",
                {
                    "labels": [
                        {"unit_id": "1", "type": "narration"},
                        {"unit_id": "2", "type": "action"},
                    ]
                },
            )
        )
        scene_rows.append(
            _row(
                "scene system",
                f"[P1] {passage}\n[P2] scene-middle-{index}-{unique_fill}\n"
                f"[P3] scene-end-{index}-{unique_fill}",
                {"boundaries": [1] if index % 2 == 0 else []},
            )
        )
    for path in paths.values():
        path.parent.mkdir()
    _write(paths["attribution"], attribution_rows)
    _write(paths["narrative"], narrative_rows)
    _write(paths["scene"], scene_rows)
    return paths


def test_load_items_preserves_native_granularity(tmp_path: Path) -> None:
    paths = _sources(tmp_path)
    items, audit = gate.load_items(paths, enforce_pinned_sources=False)

    assert len([item for item in items if item.task == "attribution"]) == 100
    assert len([item for item in items if item.task == "narrative"]) == 200
    assert len([item for item in items if item.task == "scene"]) == 200
    assert audit["row_count"] == 300
    assert audit["component_count"] < audit["row_count"]
    assert audit["signature_audit"]["signature_components_atomic"] is True
    assert audit["signature_audit"]["cross_component_signature_overlap_count"] == 0
    assert all(item.component_id for item in items)


def test_shared_passage_component_cannot_cross_splits(tmp_path: Path) -> None:
    paths = _sources(tmp_path)
    items, audit = gate.load_items(paths, enforce_pinned_sources=False)
    component_split = gate.assign_component_splits(audit["component_rows"], seed=7)
    split_audit = gate._split_audit(items, component_split)

    assert split_audit["passage_disjoint"] is True
    assert split_audit["cross_split_component_overlap"] == {}
    # The long shared sentence intentionally links attribution and narrative
    # source rows, proving that components are global rather than task-local.
    by_component: dict[str, set[str]] = {}
    for item in items:
        by_component.setdefault(item.component_id, set()).add(item.task)
    assert any(tasks == {"attribution", "narrative", "scene"} for tasks in by_component.values())


def test_episode_controls_are_four_slot_and_answer_free(tmp_path: Path) -> None:
    paths = _sources(tmp_path)
    output_dir = tmp_path / "gate-output"
    manifest = gate.build_dataset(
        output_dir=output_dir,
        source_paths=paths,
        model_id="local/test-model",
        model_revision="rev-a",
        hf_endpoint=gate.HF_MIRROR,
        episodes_per_task=1,
        enforce_pinned_sources=False,
    )

    assert manifest["schema"] == gate.SCHEMA
    assert manifest["hf_endpoint"] == gate.HF_MIRROR
    assert manifest["split_audit"]["passage_disjoint"] is True
    assert manifest["materialized_splits"] == ["train", "development"]
    assert not (output_dir / "sealed_validation.jsonl").exists()
    assert gate.verify_manifest_receipt(manifest) is True
    episodes = [
        json.loads(line)
        for line in (output_dir / "train.jsonl").read_text(encoding="utf-8").splitlines()
        if line
    ]
    assert episodes
    episode = episodes[0]
    assert len(episode["records"]) == 4
    assert len({record["component_id"] for record in episode["records"]}) == 4
    assert len(episode["queries"]) == 4
    assert set(episode["state_variants"]) == {
        "correct_state",
        "donor_state",
        "value_swap",
        "shuffled_slots",
        "no_state",
    }
    assert len(episode["state_variants"]["no_state"]["records"]) == 0
    correct_records = episode["state_variants"]["correct_state"]["records"]
    donor_records = episode["state_variants"]["donor_state"]["records"]
    swapped_records = episode["state_variants"]["value_swap"]["records"]
    assert [record["record_id"] for record in donor_records] == [
        record["record_id"] for record in correct_records
    ]
    assert all(
        donor["value_json"] != correct["value_json"]
        for correct, donor in zip(correct_records, donor_records, strict=True)
    )
    assert len(set(episode["donor_source_component_ids"])) == 4
    assert all(
        swapped["value_json"] != correct["value_json"]
        for correct, swapped in zip(correct_records, swapped_records, strict=True)
    )
    swap_sources = episode["value_swap_source_slot_by_destination_slot"]
    assert sorted(swap_sources) == list(range(4))
    assert all(source != destination for destination, source in enumerate(swap_sources))
    for query in episode["queries"]:
        assert query["answer_absent_from_read_prompt"] is True
        assert query["gold_json"] not in query["read_prompt"]
        assert "memory_value:" not in query["read_prompt"]
        slot = query["target_slot"]
        replacement = episode["query_counterfactual_records"][str(slot)][
            "target_slot_rewrite"
        ]
        assert replacement["replace_slot"] == slot
        rewrite_records = [dict(record) for record in episode["records"]]
        rewrite_records[slot] = replacement["replacement_record"]
        assert [r["record_id"] for r in rewrite_records] == [
            r["record_id"] for r in episode["records"]
        ]
        changed = [
            index
            for index, (left, right) in enumerate(
                zip(episode["records"], rewrite_records, strict=True)
            )
            if left["value_json"] != right["value_json"]
        ]
        assert changed == [slot]
        assert query["record_payload_sha256_by_condition"][
            "target_slot_rewrite"
        ] == replacement["result_record_payload_sha256"]
        assert set(query["expected_by_state"]) == {
            "correct_state",
            "donor_state",
            "value_swap",
            "target_slot_rewrite",
            "shuffled_slots",
            "no_state",
            "pristine_frozen_base",
        }


def test_generation_is_byte_deterministic(tmp_path: Path) -> None:
    paths = _sources(tmp_path)
    first = tmp_path / "first"
    second = tmp_path / "second"
    first_manifest = gate.build_dataset(
        output_dir=first,
        source_paths=paths,
        model_id="local/test-model",
        enforce_pinned_sources=False,
        episodes_per_task=2,
    )
    second_manifest = gate.build_dataset(
        output_dir=second,
        source_paths=paths,
        model_id="local/test-model",
        enforce_pinned_sources=False,
        episodes_per_task=2,
    )
    for split in gate.BUILD_PROFILES["development"]:
        assert (first / f"{split}.jsonl").read_bytes() == (second / f"{split}.jsonl").read_bytes()
    assert first_manifest["output_sha256"] == second_manifest["output_sha256"]
    assert first_manifest["manifest_receipt"] == second_manifest["manifest_receipt"]


def test_sealed_profile_requires_and_binds_frozen_lock_receipt(tmp_path: Path) -> None:
    paths = _sources(tmp_path)
    development = gate.build_dataset(
        output_dir=tmp_path / "development-package",
        source_paths=paths,
        model_id="local/test-model",
        model_revision="rev-a",
        episodes_per_task=1,
        enforce_pinned_sources=False,
    )
    with pytest.raises(ValueError, match="frozen lock receipt"):
        gate.build_dataset(
            output_dir=tmp_path / "sealed-without-lock",
            source_paths=paths,
            model_id="local/test-model",
            model_revision="rev-a",
            episodes_per_task=1,
            build_profile="sealed_validation",
            enforce_pinned_sources=False,
        )

    lock = {
        "schema": gate.SEALED_LOCK_SCHEMA,
        "configuration_frozen": True,
        "benchmark_contract_sha256": development["benchmark_contract_sha256"],
        "development_manifest_payload_sha256": development["manifest_receipt"][
            "payload_sha256"
        ],
        "runner_protocol_sha256": "a" * 64,
        "training_configuration_sha256": "b" * 64,
    }
    lock_path = tmp_path / "sealed-lock.json"
    lock_path.write_text(json.dumps(lock), encoding="utf-8")
    sealed_dir = tmp_path / "sealed-package"
    sealed = gate.build_dataset(
        output_dir=sealed_dir,
        source_paths=paths,
        model_id="local/test-model",
        model_revision="rev-a",
        episodes_per_task=1,
        build_profile="sealed_validation",
        sealed_lock_receipt=lock_path,
        enforce_pinned_sources=False,
    )

    assert sealed["materialized_splits"] == ["sealed_validation"]
    assert sealed["benchmark_contract_sha256"] == development[
        "benchmark_contract_sha256"
    ]
    assert sealed["sealed_lock"]["receipt"] == lock
    assert (sealed_dir / "sealed_validation.jsonl").is_file()
    assert not (sealed_dir / "train.jsonl").exists()
    assert not (sealed_dir / "development.jsonl").exists()


def test_formal_build_requires_pinned_gemma4_weights(tmp_path: Path) -> None:
    paths = _sources(tmp_path)
    assert gate.DEFAULT_MODEL_ID == "google/gemma-4-E4B-it"
    assert gate.DEFAULT_MODEL_REVISION.startswith("a4c2d58")
    assert gate.DEFAULT_EPISODES_PER_TASK == 32
    with pytest.raises(ValueError, match="model-path weight binding"):
        gate.build_dataset(
            output_dir=tmp_path / "unbound-formal",
            source_paths=paths,
            model_path=None,
            episodes_per_task=1,
            enforce_pinned_sources=True,
        )


def test_mirror_and_protected_source_contracts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    paths = _sources(tmp_path)
    with pytest.raises(ValueError, match="HF mirror"):
        gate.build_dataset(
            output_dir=tmp_path / "bad-hf",
            source_paths=paths,
            hf_endpoint="https://huggingface.co",
            enforce_pinned_sources=False,
        )
    monkeypatch.setenv("HF_ENDPOINT", "https://huggingface.co")
    with pytest.raises(ValueError, match="HF_ENDPOINT"):
        gate.require_hf_mirror(gate.HF_MIRROR)
    monkeypatch.setenv("HF_ENDPOINT", gate.HF_MIRROR)

    protected = tmp_path / "val" / "train.jsonl"
    protected.parent.mkdir()
    protected.write_text("{}\n", encoding="utf-8")
    bad_paths = dict(paths)
    bad_paths["scene"] = protected
    with pytest.raises(ValueError, match="Only train.jsonl"):
        gate.load_items(bad_paths, enforce_pinned_sources=False)

    alias = tmp_path / "alias" / "train.jsonl"
    alias.parent.mkdir()
    alias.symlink_to(paths["scene"])
    symlink_paths = dict(paths)
    symlink_paths["scene"] = alias
    with pytest.raises(ValueError, match="Symbolic-link"):
        gate.load_items(symlink_paths, enforce_pinned_sources=False)


def test_local_model_binding_hashes_runtime_and_weight_artifacts(tmp_path: Path) -> None:
    model_path = tmp_path / "model"
    model_path.mkdir()
    for name in (*gate.MODEL_RUNTIME_ARTIFACTS, "generation_config.json", "model.safetensors"):
        (model_path / name).write_bytes(f"fixture:{name}\n".encode("ascii"))

    binding = gate._model_binding(
        "local/test-model",
        "rev-a",
        model_path,
        gate.HF_MIRROR,
    )

    assert binding["weights_bound"] is True
    assert binding["binding_scope"] == "hf_identity_and_local_runtime_artifacts"
    assert set(binding["local_artifacts"]) == {
        *gate.MODEL_RUNTIME_ARTIFACTS,
        "generation_config.json",
        "model.safetensors",
    }
    assert all(len(record["sha256"]) == 64 for record in binding["local_artifacts"].values())
    assert len(binding["binding_sha256"]) == 64
