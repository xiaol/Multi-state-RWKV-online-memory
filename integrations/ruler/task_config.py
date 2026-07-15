from __future__ import annotations


EVAL_TASKS = (
    "niah_single_1",
    "niah_single_2",
    "niah_single_3",
    "niah_multikey_1",
    "niah_multikey_2",
    "niah_multikey_3",
    "niah_multivalue",
    "niah_multiquery",
    "vt",
    "cwe",
    "fwe",
    "qa_1",
    "qa_2",
)

TRAIN_TASKS = EVAL_TASKS[:-2]

GEMMA4_CHAT_TEMPLATE_NAME = "gemma4-chat"
GEMMA4_CHAT_TEMPLATE = "<bos><|turn>user\n{task_template}<turn|>\n<|turn>model\n"
GEMMA4_PROMPT_PREFIX = "<bos><|turn>user\n"
GEMMA4_PROMPT_SUFFIX = "<turn|>\n<|turn>model\n"

DEFAULT_EVAL_LENGTHS = (4096, 8192, 16384, 32768)
DEFAULT_TRAIN_LENGTHS = (2048, 4096, 8192)
DEFAULT_TRAIN_SEEDS = (100000, 100100, 100200)
OFFICIAL_EVAL_SEED = 42

TASK_TOKEN_BUDGETS = {
    "niah_single_1": 128,
    "niah_single_2": 128,
    "niah_single_3": 128,
    "niah_multikey_1": 128,
    "niah_multikey_2": 128,
    "niah_multikey_3": 128,
    "niah_multivalue": 128,
    "niah_multiquery": 128,
    "vt": 30,
    "cwe": 120,
    "fwe": 50,
    "qa_1": 32,
    "qa_2": 32,
}


def parse_csv_strings(raw: str) -> tuple[str, ...]:
    values = tuple(value.strip() for value in raw.split(",") if value.strip())
    if not values:
        raise ValueError("Expected at least one comma-separated value")
    return values


def parse_csv_ints(raw: str) -> tuple[int, ...]:
    values = tuple(int(value.strip()) for value in raw.split(",") if value.strip())
    if not values or any(value <= 0 for value in values):
        raise ValueError("Expected positive comma-separated integers")
    return values
