from __future__ import annotations

import ast
import json
import math

from experiments.rethinking_rwkv_ms_gemma import (
    prepare_natural_memory_native_scene_state_retrieval as retrieval,
)
from experiments.rethinking_rwkv_ms_gemma import (
    analyze_natural_memory_native_scene_state_retrieval as analysis,
)
from experiments.rethinking_rwkv_ms_gemma import (
    run_natural_memory_native_scene_state_retrieval as runner,
)


def test_state_retrieval_protocol_receipts_are_bound() -> None:
    protocol, amendment = retrieval.validate_protocol()

    assert protocol["authorization"]["publisher_validation_predictions_allowed_as_input"] is False
    assert protocol["authorization"]["publisher_test_authorized"] is False
    assert protocol["authorization"]["hard32_authorized"] is False
    assert amendment["authorization_changed"] is False
    assert amendment["gates_changed"] is False


def test_state_retrieval_mapper_has_no_json_literal_names() -> None:
    tree = ast.parse(retrieval.Path(retrieval.__file__).read_text(encoding="utf-8"))

    assert not {
        node.id for node in ast.walk(tree) if isinstance(node, ast.Name)
    } & {"false", "true", "null"}


def test_state_retrieval_executables_have_no_json_literal_names() -> None:
    for module in (runner, analysis):
        tree = ast.parse(module.Path(module.__file__).read_text(encoding="utf-8"))
        assert not {
            node.id for node in ast.walk(tree) if isinstance(node, ast.Name)
        } & {"false", "true", "null"}


def test_state_retrieval_runner_hash_is_bound() -> None:
    assert runner.sha256_file(runner.Path(runner.__file__)) == analysis.EXPECTED_RUNNER_SHA256


def test_state_retrieval_holdout_requires_signed_selection(tmp_path) -> None:
    selection = tmp_path / "selection.json"
    selection.write_text("{}\n", encoding="utf-8")

    try:
        runner.validate_selection(selection, runner_sha256=analysis.EXPECTED_RUNNER_SHA256)
    except ValueError as error:
        assert "receipt is missing" in str(error)
    else:
        raise AssertionError("Unsigned holdout selection was accepted")


def test_state_retrieval_selection_binds_single_method(tmp_path) -> None:
    analyzer_sha256 = runner.sha256_file(
        runner.SCRIPT_DIR / "analyze_natural_memory_native_scene_state_retrieval.py"
    )
    value = {
        "schema": runner.SELECTION_SCHEMA,
        "protocol_payload_sha256": retrieval.PROTOCOL_PAYLOAD_SHA256,
        "amendment_payload_sha256": retrieval.AMENDMENT_PAYLOAD_SHA256,
        "mapping_payload_sha256": runner.MAPPING_PAYLOAD_SHA256,
        "runner_sha256": analysis.EXPECTED_RUNNER_SHA256,
        "analyzer_sha256": analyzer_sha256,
        "selected_method": "char_tfidf_nearest",
        "phase_one_passed": True,
        "holdout_authorized": True,
    }
    value["receipt"] = {
        "payload_sha256": runner.canonical_sha256(value),
    }
    selection = tmp_path / "selection.json"
    selection.write_text(json.dumps(value), encoding="utf-8")

    loaded = runner.validate_selection(
        selection,
        runner_sha256=analysis.EXPECTED_RUNNER_SHA256,
    )

    assert loaded["selected_method"] == "char_tfidf_nearest"


def test_state_retrieval_partition_hashes_are_bound() -> None:
    root = retrieval.PROJECT_ROOT / (
        "experiments/rethinking_rwkv_ms_gemma/local_artifacts/"
        "natural_memory_native_development_v1"
    )
    rows = retrieval.load_prompt_rows(
        root / retrieval.TARGET_RELATIVE_PATH,
        expected_sha256=retrieval.TARGET_SHA256,
        expected_rows=retrieval.EXPECTED_TARGET_ROWS,
    )

    payload = retrieval.validate_partitions(rows)

    assert len(payload) == 357
    assert sum(record["partition"] == "fit" for record in payload) == 289
    assert sum(record["partition"] == "holdout" for record in payload) == 68


def test_state_retrieval_text_excludes_system_and_assistant() -> None:
    messages = [
        {"role": "system", "content": "SECRET SYSTEM"},
        {"role": "user", "content": "Ａ  B\nC"},
        {"role": "assistant", "content": "SECRET LABEL"},
    ]

    assert retrieval.normalize_user_text(messages) == "a b c"


def test_state_bank_tfidf_uses_sublinear_tf_and_cosine() -> None:
    idf, inverted = retrieval.fit_tfidf_index(["aaaa", "ab"])
    scores = retrieval.cosine_scores(
        "aaaa",
        idf=idf,
        inverted=inverted,
        document_count=2,
    )

    assert math.isclose(scores[0], 1.0, rel_tol=0.0, abs_tol=1e-12)
    assert 0.0 <= scores[1] < scores[0]


def test_state_retrieval_empty_vector_tie_breaks_lowest_index() -> None:
    idf, inverted = retrieval.fit_tfidf_index(["abcd", "efgh"])
    scores = retrieval.cosine_scores(
        "x",
        idf=idf,
        inverted=inverted,
        document_count=2,
    )

    assert scores == [0.0, 0.0]
    assert retrieval.best_index(scores) == 0


def test_hash_random_is_stable() -> None:
    row_hash = "ab" * 32
    first = int(
        retrieval.hashlib.sha256(
            f"{retrieval.RANDOM_NAMESPACE}{row_hash}".encode("ascii")
        ).hexdigest()[:16],
        16,
    ) % retrieval.EXPECTED_BANK_ROWS
    second = int(
        retrieval.hashlib.sha256(
            f"{retrieval.RANDOM_NAMESPACE}{row_hash}".encode("ascii")
        ).hexdigest()[:16],
        16,
    ) % retrieval.EXPECTED_BANK_ROWS

    assert first == second
