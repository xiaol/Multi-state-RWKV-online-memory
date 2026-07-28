from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from experiments.rethinking_rwkv_ms_gemma import prepare_scene_failure_pairs as pairs
from experiments.rethinking_rwkv_ms_gemma import run_scene_train_base_eval as producer


def compact_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def scene_row(label: str, boundaries: list[int]) -> str:
    return compact_json(
        {
            "messages": [
                {"role": "system", "content": "Detect scene boundaries as JSON."},
                {
                    "role": "user",
                    "content": f"[P1] {label} one.\n[P2] {label} two.\n[P3] {label} three.",
                },
                {
                    "role": "assistant",
                    "content": compact_json({"boundaries": boundaries}),
                },
            ]
        }
    )


def write_jsonl(path: Path, rows: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(rows) + "\n", encoding="utf-8")


def make_base_model(root: Path) -> Path:
    base_model = root / "base-model"
    base_model.mkdir()
    artifacts = {
        "config.json": '{"model_type":"gemma"}\n',
        "generation_config.json": '{"do_sample":false}\n',
        "tokenizer_config.json": '{"chat_template":"test"}\n',
        "tokenizer.json": '{"version":"1.0"}\n',
        "chat_template.jinja": "{{ messages }}\n",
    }
    for name, content in artifacts.items():
        (base_model / name).write_text(content, encoding="utf-8")
    (base_model / "model-00001-of-00002.safetensors").write_bytes(b"weight-shard-1")
    nested = base_model / "weights"
    nested.mkdir()
    (nested / "model-00002-of-00002.safetensors").write_bytes(b"weight-shard-2")
    return base_model


def make_args(
    *,
    dataset_file: Path,
    base_model: Path,
    output_dir: Path,
    candidate_count: int = 2,
) -> SimpleNamespace:
    return SimpleNamespace(
        base_model=base_model,
        dataset_file=dataset_file,
        expected_dataset_sha256=producer.sha256_file(dataset_file),
        output_dir=output_dir,
        candidate_count=candidate_count,
        selection_seed=19,
        max_new_tokens=128,
        device="cpu",
        dtype="float32",
        attn_implementation="sdpa",
        prepare_only=True,
        overwrite=False,
    )


def test_candidate_selection_is_deterministic_and_label_independent() -> None:
    def rows(gold_offset: int, *, count: int = 8) -> list[pairs.SourceRow]:
        result = []
        for index in range(count):
            prompt_hash = producer.sha256_text(f"prompt-{index}")
            result.append(
                pairs.SourceRow(
                    split="train",
                    line_index=index,
                    raw_line=f"raw-{index}-gold-{gold_offset}",
                    row_sha256=producer.sha256_text(
                        f"raw-{index}-gold-{gold_offset}"
                    ),
                    prompt_sha256=prompt_hash,
                    messages=[],
                    gold={"boundaries": [gold_offset]},
                    paragraph_count=3,
                )
            )
        return result

    first = producer.select_candidate_rows(rows(1), count=3, seed=71)
    second = producer.select_candidate_rows(rows(2), count=3, seed=71)

    assert [row.line_index for row in first] == [row.line_index for row in second]
    assert [row.line_index for row in first] == sorted(row.line_index for row in first)
    candidates = rows(1, count=80)
    forward_64 = producer.select_candidate_rows(candidates, count=64, seed=71)
    reverse_64 = producer.select_candidate_rows(
        list(reversed(candidates)),
        count=64,
        seed=71,
    )
    assert producer.DEFAULT_CANDIDATE_COUNT == 64
    assert producer.MAX_CANDIDATE_ROWS == 64
    assert len(forward_64) == 64
    assert {row.line_index for row in forward_64} == {
        row.line_index for row in reverse_64
    }
    with pytest.raises(ValueError, match="capped at 64"):
        producer.select_candidate_rows(rows(1) * 9, count=65, seed=71)


def test_prepare_only_writes_locked_selection_without_loading_model(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dataset_file = tmp_path / "official" / "train.jsonl"
    write_jsonl(
        dataset_file,
        [scene_row(f"train-{index}", [1] if index % 2 else []) for index in range(4)],
    )
    base_model = make_base_model(tmp_path)
    output_dir = tmp_path / "prepared"
    args = make_args(
        dataset_file=dataset_file,
        base_model=base_model,
        output_dir=output_dir,
    )

    monkeypatch.setattr(producer, "parse_args", lambda: args)

    def fail_if_loaded(**kwargs):
        raise AssertionError("prepare-only must not load a model")

    monkeypatch.setattr(producer, "load_model_and_tokenizer", fail_if_loaded)
    producer.main()

    selection = json.loads(
        (output_dir / "candidate_selection.json").read_text(encoding="utf-8")
    )
    manifest = json.loads((output_dir / "manifest.json").read_text(encoding="utf-8"))
    progress = json.loads((output_dir / "progress.json").read_text(encoding="utf-8"))

    assert selection["split"] == "train"
    assert selection["selection_uses_gold_labels"] is False
    assert selection["selection_uses_model_output"] is False
    assert len(selection["rows"]) == 2
    assert all(len(row["row_sha256"]) == 64 for row in selection["rows"])
    assert manifest["fingerprint_payload"]["condition"] == "base"
    assert manifest["fingerprint_payload"]["split"] == "train"
    assert len(manifest["fingerprint_payload"]["base_model_artifacts"]["weights"]) == 2
    assert "transformers" in manifest["fingerprint_payload"]["generation_runtime_versions"]
    assert manifest["output"]["builder_argument"] == "--base-train-eval-jsonl"
    assert progress["model_loaded"] is False
    assert not (output_dir / "base.jsonl").exists()


def test_prepare_rejects_non_train_path_and_dataset_hash_drift(tmp_path: Path) -> None:
    dataset_file = tmp_path / "official" / "train.jsonl"
    write_jsonl(dataset_file, [scene_row("train", [1])])
    base_model = make_base_model(tmp_path)

    wrong_split = tmp_path / "official" / "val.jsonl"
    wrong_split.write_bytes(dataset_file.read_bytes())
    wrong_args = make_args(
        dataset_file=wrong_split,
        base_model=base_model,
        output_dir=tmp_path / "wrong-split",
        candidate_count=1,
    )
    with pytest.raises(ValueError, match="official train.jsonl"):
        producer.prepare_run(wrong_args)

    drift_args = make_args(
        dataset_file=dataset_file,
        base_model=base_model,
        output_dir=tmp_path / "hash-drift",
        candidate_count=1,
    )
    drift_args.expected_dataset_sha256 = "0" * 64
    with pytest.raises(ValueError, match="Dataset SHA-256 differs"):
        producer.prepare_run(drift_args)


def test_base_model_fingerprint_binds_all_weights_and_runtime_artifacts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    base_model = make_base_model(tmp_path)
    monkeypatch.setattr(
        producer.metadata,
        "version",
        lambda distribution: f"locked-{distribution}",
    )

    first = producer.local_base_model_artifacts(base_model)
    versions = producer.generation_runtime_versions()
    weight_paths = {row["relative_path"] for row in first["weights"]}
    runtime_paths = {row["relative_path"] for row in first["runtime_artifacts"]}

    assert weight_paths == {
        "model-00001-of-00002.safetensors",
        "weights/model-00002-of-00002.safetensors",
    }
    assert {
        "config.json",
        "generation_config.json",
        "tokenizer_config.json",
        "tokenizer.json",
        "chat_template.jinja",
    }.issubset(runtime_paths)
    assert versions["torch"] == "locked-torch"
    assert versions["huggingface-hub"] == "locked-huggingface-hub"

    (base_model / "weights" / "model-00002-of-00002.safetensors").write_bytes(
        b"changed-weight-shard-2"
    )
    second = producer.local_base_model_artifacts(base_model)
    assert second["aggregate_sha256"] != first["aggregate_sha256"]


def test_overwrite_rejects_unmanaged_jsonl_without_deleting_it(tmp_path: Path) -> None:
    dataset_file = tmp_path / "official" / "train.jsonl"
    write_jsonl(dataset_file, [scene_row("train", [1])])
    base_model = make_base_model(tmp_path)
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    stale_path = output_dir / "base_full.jsonl"
    stale_path.write_text('{"stale":true}\n', encoding="utf-8")
    args = make_args(
        dataset_file=dataset_file,
        base_model=base_model,
        output_dir=output_dir,
        candidate_count=1,
    )
    args.overwrite = True

    with pytest.raises(ValueError, match="unrelated stale JSONL"):
        producer.prepare_run(args)
    assert stale_path.read_text(encoding="utf-8") == '{"stale":true}\n'


def test_resume_validation_binds_fingerprint_and_source_hash(tmp_path: Path) -> None:
    dataset_file = tmp_path / "train.jsonl"
    write_jsonl(dataset_file, [scene_row("train", [2])])
    source = producer.load_source_split(dataset_file, split="train")[0]
    generation = {
        "status": "ok",
        "raw_generation": '{"boundaries":[1]}',
        "parsed_json": {"boundaries": [1]},
        "input_tokens": 10,
        "output_tokens": 4,
        "hit_max_new_tokens": False,
        "elapsed_seconds": 0.1,
        "peak_cuda_memory_bytes": None,
        "memory_trace": [],
    }
    record = producer.build_base_record(
        source,
        generation,
        fingerprint="f" * 64,
        max_new_tokens=128,
        completed_at="2026-01-01T00:00:00+00:00",
    )

    assert producer.validate_resume_records(
        [record],
        selected_by_index={0: source},
        fingerprint="f" * 64,
    ) == {0: record}
    with pytest.raises(ValueError, match="contract differs"):
        producer.validate_resume_records(
            [{**record, "fingerprint": "e" * 64}],
            selected_by_index={0: source},
            fingerprint="f" * 64,
        )
    with pytest.raises(ValueError, match="contract differs"):
        producer.validate_resume_records(
            [{**record, "row_sha256": "0" * 64}],
            selected_by_index={0: source},
            fingerprint="f" * 64,
        )
