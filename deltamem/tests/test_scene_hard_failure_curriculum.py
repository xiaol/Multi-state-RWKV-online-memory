from __future__ import annotations

from argparse import Namespace
from collections import Counter
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import torch

from deltamem.train.delta_sft_experimental import (
    _scene_state_generation_pairing_binding,
)
from experiments.rethinking_rwkv_ms_gemma import (
    prepare_scene_hard_failure_curriculum as builder,
)
from experiments.rethinking_rwkv_ms_gemma.prepare_scene_failure_pairs import (
    BaseRecord,
    SourceRow,
)
from experiments.rethinking_rwkv_ms_gemma.prepare_scene_memory_v7_data import (
    Candidate,
)


SYSTEM = "Detect scene boundaries and output JSON only."


def compact(value: Any) -> str:
    return json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":"))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=True, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def write_jsonl(path: Path, lines: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def source_line(index: int, boundaries: list[int]) -> str:
    paragraphs = "\n".join(
        f"[P{paragraph}] sample-{index}-{'x' * (index % 11)}-{paragraph}"
        for paragraph in range(1, 8)
    )
    return compact(
        {
            "messages": [
                {"role": "system", "content": SYSTEM},
                {"role": "user", "content": paragraphs},
                {
                    "role": "assistant",
                    "content": compact({"boundaries": boundaries}),
                },
            ]
        }
    )


def make_dynamic_producer_bundle(
    root: Path,
    *,
    train_file: Path,
    rows: list[str],
) -> dict[str, Any]:
    source_rows = builder.load_source_split(train_file, split="train")
    selection_rows = [
        {
            "source_index": source.line_index,
            "row_sha256": source.row_sha256,
            "user_prompt_sha256": source.prompt_sha256,
        }
        for source in source_rows
    ]
    selection = {
        "schema": builder.PRODUCER_SELECTION_SCHEMA,
        "task": builder.TASK,
        "split": "train",
        "dataset_file": str(train_file.resolve()),
        "dataset_sha256": builder.sha256_file(train_file),
        "candidate_count": len(rows),
        "selection_seed": 99,
        "selection_basis": "sha256(selection_seed + NUL + user_prompt_sha256)",
        "selection_uses_gold_labels": False,
        "selection_uses_model_output": False,
        "rows": selection_rows,
    }
    selection_path = root / "candidate_selection.json"
    write_json(selection_path, selection)
    fingerprint_payload = {
        "schema": builder.PRODUCER_SCHEMA,
        "task": builder.TASK,
        "task_kind": "scene",
        "condition": "base",
        "split": "train",
        "base_model": str((root / "frozen-base").resolve()),
        "base_model_artifacts": {
            "root": str((root / "frozen-base").resolve()),
            "weights": [{"relative_path": "model.safetensors", "sha256": "1" * 64}],
            "runtime_artifacts": [{"relative_path": "config.json", "sha256": "2" * 64}],
            "aggregate_sha256": "3" * 64,
        },
        "dataset_file": str(train_file.resolve()),
        "dataset_sha256": builder.sha256_file(train_file),
        "selection_sha256": builder.canonical_sha256(selection),
        "selected_rows": selection_rows,
        "candidate_count": len(rows),
        "selection_seed": 99,
    }
    fingerprint = builder.canonical_sha256(fingerprint_payload)
    records = []
    for index, raw in enumerate(rows):
        gold = json.loads(json.loads(raw)["messages"][2]["content"])
        parsed = {"wrong_schema": []}
        records.append(
            compact(
                {
                    "key": f"{builder.TASK}:{index}",
                    "condition": "base",
                    "task": builder.TASK,
                    "task_kind": "scene",
                    "split": "train",
                    "line_index": index,
                    "row_sha256": hashlib.sha256(raw.encode("utf-8")).hexdigest(),
                    "gold": gold,
                    "status": "ok",
                    "parsed_json": parsed,
                    "raw_generation": compact(parsed),
                    "fingerprint": fingerprint,
                    "score": {},
                }
            )
        )
    base_path = root / "base.jsonl"
    write_jsonl(base_path, records)
    manifest_path = root / "manifest.json"
    write_json(
        manifest_path,
        {
            "schema": builder.PRODUCER_SCHEMA,
            "fingerprint": fingerprint,
            "fingerprint_payload": fingerprint_payload,
            "selection": {
                "path": str(selection_path.resolve()),
                "sha256": builder.sha256_file(selection_path),
                "rows": len(rows),
                "uses_gold_labels": False,
                "uses_model_output": False,
            },
            "output": {"base_records": str(base_path.resolve())},
        },
    )
    summary_path = root / "summary.json"
    write_json(
        summary_path,
        {
            "schema": builder.PRODUCER_SCHEMA,
            "fingerprint": fingerprint,
            "complete": True,
            "completed": len(rows),
            "expected": len(rows),
            "condition": "base",
            "task": builder.TASK,
            "split": "train",
        },
    )
    return {
        "base": base_path,
        "base_sha256": builder.sha256_file(base_path),
        "manifest_sha256": builder.sha256_file(manifest_path),
        "selection_sha256": builder.sha256_file(selection_path),
        "summary_sha256": builder.sha256_file(summary_path),
    }


class FakeTokenizer:
    name_or_path = "fake-scene-tokenizer"
    chat_template = "fake"

    def apply_chat_template(
        self,
        messages: list[dict[str, str]],
        *,
        tokenize: bool = False,
        add_generation_prompt: bool = False,
        **_: Any,
    ) -> str:
        assert tokenize is False
        rendered = "".join(
            f"<{message['role']}>{message['content']}</{message['role']}>"
            for message in messages
        )
        if add_generation_prompt:
            rendered += "<assistant>"
        return rendered

    def __call__(
        self,
        rendered: str,
        *,
        add_special_tokens: bool,
        return_offsets_mapping: bool = False,
        return_tensors: str | None = None,
    ) -> SimpleNamespace:
        assert add_special_tokens is False
        ids = [ord(character) for character in rendered]
        if return_tensors is None:
            return SimpleNamespace(input_ids=ids)
        assert return_tensors == "pt" and return_offsets_mapping is True
        return SimpleNamespace(
            input_ids=torch.tensor([ids], dtype=torch.long),
            attention_mask=torch.ones((1, len(ids)), dtype=torch.long),
            offset_mapping=torch.tensor(
                [[[index, index + 1] for index in range(len(ids))]],
                dtype=torch.long,
            ),
        )


def dummy_candidate(
    index: int,
    *,
    boundaries: list[int],
    write_tokens: int,
) -> Candidate:
    messages = [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": f"[P1] row {index}\n[P2] end"},
        {"role": "assistant", "content": compact({"boundaries": boundaries})},
    ]
    raw = compact({"messages": messages})
    row_hash = hashlib.sha256(raw.encode()).hexdigest()
    prompt_hash = hashlib.sha256(messages[1]["content"].encode()).hexdigest()
    source = SourceRow(
        split="train",
        line_index=index,
        raw_line=raw,
        row_sha256=row_hash,
        prompt_sha256=prompt_hash,
        messages=messages,
        gold={"boundaries": boundaries},
        paragraph_count=2,
    )
    record = BaseRecord(
        eval_line_index=index,
        raw_record_sha256=hashlib.sha256(f"record-{index}".encode()).hexdigest(),
        row_sha256=row_hash,
        key=f"{builder.TASK}:{index}",
        source_line_index=index,
        producer_fingerprint="a" * 64,
        parsed_json={"wrong": []},
        gold={"boundaries": boundaries},
    )
    candidate = Candidate(
        source=source,
        base_record=record,
        base_payload={},
        strict_score={"schema_valid": False, "fp": 0, "fn": len(boundaries)},
        failure_stratum="invalid_schema",
        boundary_count=len(boundaries),
        label_sha256=builder.canonical_sha256(boundaries),
        paragraph_hashes=(),
        selection_sha256=prompt_hash,
    )
    candidate.token_metadata = {"write_token_count": write_tokens}
    return candidate


def test_balanced_pair_selection_is_reciprocal_safe_and_deterministic() -> None:
    candidates = [
        *(dummy_candidate(index, boundaries=[], write_tokens=100 + index * 10) for index in range(4)),
        *(
            dummy_candidate(
                index,
                boundaries=[index - 3],
                write_tokens=100 + (index - 4) * 10,
            )
            for index in range(4, 10)
        ),
        *(
            dummy_candidate(
                index,
                boundaries=[1, index - 8],
                write_tokens=105 + (index - 10) * 10,
            )
            for index in range(10, 16)
        ),
    ]

    first, audit = builder.select_balanced_pairs(candidates, max_pairs=8)
    second, second_audit = builder.select_balanced_pairs(candidates, max_pairs=8)

    assert first == second
    assert audit == second_audit
    assert audit["pair_strata"] == {
        "presence": 4,
        "same_cardinality_value": 4,
        "cross_cardinality_value": 0,
    }
    assert audit["same_cardinality"]["represented_cardinalities"] == [1, 2]
    assert len({index for pair in first for index in pair}) == 16
    for left, right in first:
        left_row = candidates[left]
        right_row = candidates[right]
        assert left_row.label_sha256 != right_row.label_sha256
        if left_row.boundary_count == 0 or right_row.boundary_count == 0:
            assert (left_row.boundary_count == 0) != (right_row.boundary_count == 0)
        else:
            assert left_row.boundary_count == right_row.boundary_count


@pytest.fixture
def dynamic_inputs(tmp_path: Path) -> dict[str, Any]:
    # Sixteen candidates deliberately proves that the builder does not retain
    # the historical fixed-64 producer restriction.
    labels = [
        [],
        [],
        [],
        [],
        [1],
        [2],
        [3],
        [4],
        [5],
        [6],
        [1, 2],
        [1, 3],
        [2, 4],
        [2, 5],
        [3, 5],
        [4, 6],
    ]
    rows = [source_line(index, boundaries) for index, boundaries in enumerate(labels)]
    train = tmp_path / "official" / "train.jsonl"
    write_jsonl(train, rows)
    producer = make_dynamic_producer_bundle(
        tmp_path / "base-producer",
        train_file=train,
        rows=rows,
    )
    tokenizer = tmp_path / "tokenizer"
    tokenizer.mkdir()
    (tokenizer / "tokenizer.json").write_text("{}\n", encoding="utf-8")
    (tokenizer / "chat_template.jinja").write_text("fake\n", encoding="utf-8")
    protected = {
        name: tmp_path / "protected" / name
        for name in ("val.jsonl", "test.jsonl", "hard32.jsonl", "hard32_selection.json")
    }
    for name, path in protected.items():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"protected {name}\n", encoding="utf-8")
    return {
        "rows": rows,
        "train": train,
        "producer": producer,
        "tokenizer": tokenizer,
        "protected": protected,
    }


def build_dynamic(
    dynamic_inputs: dict[str, Any],
    output: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> dict[str, Any]:
    monkeypatch.setattr(builder, "load_tokenizer", lambda _: FakeTokenizer())
    producer = dynamic_inputs["producer"]
    tokenizer = dynamic_inputs["tokenizer"]
    return builder.prepare_scene_hard_failure_curriculum(
        train_file=dynamic_inputs["train"],
        expected_train_sha256=builder.sha256_file(dynamic_inputs["train"]),
        base_eval=producer["base"],
        expected_base_eval_sha256=producer["base_sha256"],
        expected_base_manifest_sha256=producer["manifest_sha256"],
        expected_base_selection_sha256=producer["selection_sha256"],
        expected_base_summary_sha256=producer["summary_sha256"],
        tokenizer_path=tokenizer,
        expected_tokenizer_json_sha256=builder.sha256_file(tokenizer / "tokenizer.json"),
        expected_chat_template_sha256=builder.sha256_file(tokenizer / "chat_template.jinja"),
        output_dir=output,
        max_pairs=8,
    )


def test_dynamic_curriculum_preserves_train_rows_and_passes_trainer_contract(
    dynamic_inputs: dict[str, Any],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output = tmp_path / "curriculum"
    manifest = build_dynamic(dynamic_inputs, output, monkeypatch)

    assert manifest["base_evaluation"]["candidate_count"] == 16
    assert manifest["candidate_pool"]["eligible_strict_failures"] == 16
    assert manifest["selection"]["selected_pairs"] == 8
    assert manifest["selection"]["selected_rows"] == 16
    assert manifest["selection"]["pair_strata"] == {
        "presence": 4,
        "same_cardinality_value": 4,
        "cross_cardinality_value": 0,
    }
    emitted = (output / "train.jsonl").read_text(encoding="utf-8").splitlines()
    assert set(emitted) == set(dynamic_inputs["rows"])
    assert all(list(json.loads(row)) == ["messages"] for row in emitted)
    assert all(
        [message["role"] for message in json.loads(row)["messages"]]
        == ["system", "user", "assistant"]
        for row in emitted
    )

    pair_manifest = json.loads((output / "pair_manifest.json").read_text())
    directed = {
        entry["train_row_ordinal"]: entry
        for entry in pair_manifest["directed_pairs"]
    }
    assert len(directed) == 16
    assert Counter(entry["target_stratum"] for entry in directed.values()) == {
        "presence": 8,
        "same_cardinality_value": 8,
    }
    for ordinal, entry in directed.items():
        reverse = directed[entry["donor_train_row_ordinal"]]
        assert reverse["donor_train_row_ordinal"] == ordinal
        assert entry["source_label_sha256"] != entry["donor_label_sha256"]

    source_path = output / "source_manifest.json"
    args = Namespace(
        scene_state_source_manifest=source_path,
        expected_scene_state_source_manifest_sha256=builder.sha256_file(source_path),
        train_file=output / "train.jsonl",
    )
    binding = _scene_state_generation_pairing_binding(args)
    assert binding["source_identity"]["train_rows"] == 16
    assert binding["quotas"] == {
        "presence": 8,
        "same_cardinality_value": 8,
        "cross_cardinality_value": 0,
    }

    schedule = [
        json.loads(line)
        for line in (output / "pair_schedule.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    assert len(schedule) == 32
    canonical_pairs = {
        tuple(sorted((ordinal, entry["donor_train_row_ordinal"])))
        for ordinal, entry in directed.items()
    }
    for cycle_index in range(1, 5):
        cycle = schedule[(cycle_index - 1) * 8 : cycle_index * 8]
        assert {tuple(entry["canonical_pair_ordinals"]) for entry in cycle} == canonical_pairs
        assert sorted(
            ordinal
            for entry in cycle
            for ordinal in entry["canonical_pair_ordinals"]
        ) == list(range(16))
        assert [entry["optimizer_step"] for entry in cycle] == list(
            range((cycle_index - 1) * 8 + 1, cycle_index * 8 + 1)
        )
    for entry in schedule:
        unsigned = dict(entry)
        digest = unsigned.pop("entry_sha256")
        assert digest == builder.canonical_sha256(unsigned)
        assert len(entry["members"]) == 2
        assert entry["members"][0]["donor_row_sha256"] == entry["members"][1][
            "source_row_sha256"
        ]
        assert entry["members"][1]["donor_row_sha256"] == entry["members"][0][
            "source_row_sha256"
        ]

    schedule_manifest = json.loads(
        (output / "pair_schedule_manifest.json").read_text(encoding="utf-8")
    )
    assert schedule_manifest["curriculum"]["gradient_accumulation_steps"] == 1
    assert schedule_manifest["curriculum"]["optimizer_steps"] == 32
    assert schedule_manifest["curriculum"]["generation_endpoint_steps"] == [
        8,
        16,
        24,
        32,
    ]
    source_manifest = json.loads(source_path.read_text(encoding="utf-8"))
    train_schedule = source_manifest["hard_failure_curriculum"]["train_schedule"]
    assert train_schedule["pair_schedule"]["sha256"] == builder.sha256_file(
        output / "pair_schedule.jsonl"
    )
    assert train_schedule["optimizer_steps"] == 32


def test_pair_schedule_is_byte_deterministic(
    dynamic_inputs: dict[str, Any],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    first = tmp_path / "first"
    second = tmp_path / "second"
    build_dynamic(dynamic_inputs, first, monkeypatch)
    build_dynamic(dynamic_inputs, second, monkeypatch)

    first_entries = [
        json.loads(line)
        for line in (first / "pair_schedule.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    second_entries = [
        json.loads(line)
        for line in (second / "pair_schedule.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert first_entries == second_entries
    assert builder.canonical_sha256(first_entries) == builder.canonical_sha256(
        second_entries
    )


def test_curriculum_never_reads_protected_validation_or_test_paths(
    dynamic_inputs: dict[str, Any],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original_open = Path.open
    original_read_text = Path.read_text
    read_paths: set[Path] = set()

    def tracked_open(path: Path, *args, **kwargs):
        mode = args[0] if args else kwargs.get("mode", "r")
        if "r" in mode:
            read_paths.add(Path(path).absolute())
        return original_open(path, *args, **kwargs)

    def tracked_read_text(path: Path, *args, **kwargs):
        read_paths.add(Path(path).absolute())
        return original_read_text(path, *args, **kwargs)

    monkeypatch.setattr(Path, "open", tracked_open)
    monkeypatch.setattr(Path, "read_text", tracked_read_text)
    manifest = build_dynamic(dynamic_inputs, tmp_path / "curriculum", monkeypatch)

    assert not ({path.absolute() for path in dynamic_inputs["protected"].values()} & read_paths)
    protected = manifest["protected_evaluation"]
    assert protected == builder.protected_evaluation_bindings()
    assert protected["hard32"]["data_sha256"] == builder.HARD32_FILE_SHA256
    assert protected["hard32"]["selection_sha256"] == builder.HARD32_SELECTION_SHA256
    assert protected["official_test"]["sha256"] == builder.TEST_FILE_SHA256
    assert all(
        item["path"] is None
        for key, item in protected.items()
        if key != "policy"
    )


def test_dynamic_base_bundle_hash_mismatch_fails_closed(
    dynamic_inputs: dict[str, Any],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(builder, "load_tokenizer", lambda _: FakeTokenizer())
    producer = dynamic_inputs["producer"]
    tokenizer = dynamic_inputs["tokenizer"]
    with pytest.raises(builder.ContractError, match="frozen-base evaluation SHA-256 differs"):
        builder.prepare_scene_hard_failure_curriculum(
            train_file=dynamic_inputs["train"],
            expected_train_sha256=builder.sha256_file(dynamic_inputs["train"]),
            base_eval=producer["base"],
            expected_base_eval_sha256="0" * 64,
            expected_base_manifest_sha256=producer["manifest_sha256"],
            expected_base_selection_sha256=producer["selection_sha256"],
            expected_base_summary_sha256=producer["summary_sha256"],
            tokenizer_path=tokenizer,
            expected_tokenizer_json_sha256=builder.sha256_file(tokenizer / "tokenizer.json"),
            expected_chat_template_sha256=builder.sha256_file(tokenizer / "chat_template.jinja"),
            output_dir=tmp_path / "should-not-build",
            max_pairs=8,
        )
