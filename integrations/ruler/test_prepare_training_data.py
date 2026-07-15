from __future__ import annotations

import hashlib
import json

from prepare_training_data import (
    canonical_prompt,
    format_answer,
    prompt_hashes,
    split_document_query,
)
from task_config import GEMMA4_PROMPT_PREFIX, GEMMA4_PROMPT_SUFFIX


def test_niah_prompt_split_and_answer() -> None:
    prompt = (
        GEMMA4_PROMPT_PREFIX
        + "Memorize this.\nLong document.\nWhat is the hidden number?"
        + GEMMA4_PROMPT_SUFFIX
    )
    document, query = split_document_query("niah_single_1", canonical_prompt(prompt))
    assert document == "Memorize this.\nLong document."
    assert query == "What is the hidden number?"
    assert format_answer(
        "niah_single_1",
        {"answer_prefix": " Answer:", "outputs": ["1234567"]},
    ) == "1234567"


def test_cwe_uses_last_question_delimiter() -> None:
    body = "Few shot.\nQuestion: old question\nAnswer: old answer\nWords.\nQuestion: final question"
    document, query = split_document_query("cwe", body)
    assert document.endswith("Words.")
    assert query == "Question: final question"
    assert format_answer(
        "cwe",
        {"answer_prefix": "Answer:", "outputs": ["alpha", "beta"]},
    ) == "alpha, beta"


def test_prompt_hashes_reads_requested_eval_subset(tmp_path) -> None:
    prompt = GEMMA4_PROMPT_PREFIX + "Document.\nWhat is hidden?" + GEMMA4_PROMPT_SUFFIX
    task_dir = tmp_path / "4096" / "niah_single_1"
    task_dir.mkdir(parents=True)
    (task_dir / "validation.jsonl").write_text(
        json.dumps({"input": prompt}) + "\n",
        encoding="utf-8",
    )

    assert prompt_hashes(tmp_path, (4096,), ("niah_single_1",), "validation") == {
        hashlib.sha256("Document.\nWhat is hidden?".encode("utf-8")).hexdigest()
    }
