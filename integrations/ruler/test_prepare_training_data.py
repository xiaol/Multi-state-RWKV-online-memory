from __future__ import annotations

from prepare_training_data import canonical_prompt, format_answer, split_document_query
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
