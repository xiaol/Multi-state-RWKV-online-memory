"""Write / clear / read task data.

Every example has three parts:
  passage  -- text that is written into memory and then removed from the context
  question -- asked with the passage absent
  answer   -- short target, only answerable from the passage

Two sources:
  synthetic -- K facts about invented people; distinct entity/attribute pairs per passage,
               fresh entities and values for every example, disjoint name pools for
               train and eval so nothing can be memorised in the weights.
  squad     -- SQuAD v1.1 context / question / answer span.
"""

from __future__ import annotations

import random
import re
import string
from collections import Counter
from dataclasses import dataclass
from pathlib import Path


@dataclass
class Example:
    passage: str
    question: str
    answer: str
    aliases: tuple[str, ...] = ()


FIRST_NAMES_TRAIN = (
    "Alice Bruno Carla Dmitri Elena Farid Greta Hiro Ines Jonas Keiko Lars Mira Nadia Oskar "
    "Priya Quentin Rosa Samir Tessa Umar Vera Wen Xiomara Yusuf Zara Anouk Bashir Celine Dario"
).split()
FIRST_NAMES_EVAL = (
    "Amara Benedikt Chiara Dev Esme Felix Gwen Hamza Ivo Jana Kofi Leila Matteo Noor Olive "
    "Pavel Rafael Sana Tomas Uli Viktor Wanda Ximena Yara Zeno Aiko Bram Cosima Dani Eyal"
).split()
SYLLABLES = "ba be bi bo bu da de di do du ka ke ki ko ku la le li lo lu ma me mi mo mu na ne ni no nu ra re ri ro ru sa se si so su ta te ti to tu va ve vi vo vu".split()

ATTRIBUTES: dict[str, tuple[str, ...]] = {
    "favorite color": tuple("red blue green yellow purple orange black white pink brown gray teal".split()),
    "hometown": tuple("Lisbon Oslo Nairobi Lima Kyoto Prague Denver Perth Quito Zagreb Tunis Bergen Dublin Havana Manila".split()),
    "pet": tuple("dog cat parrot rabbit hamster turtle goldfish ferret canary gecko".split()),
    "car": tuple("Toyota Honda Ford Volvo Fiat Mazda Subaru Kia Peugeot Skoda".split()),
    "birth month": tuple("January February March April May June July August September October November December".split()),
    "job": tuple("baker nurse pilot lawyer farmer chef teacher plumber dentist tailor florist welder".split()),
    "lucky number": tuple(str(n) for n in range(2, 60)),
    "favorite fruit": tuple("apple mango banana cherry grape peach plum kiwi lemon melon pear fig".split()),
    "instrument": tuple("piano guitar violin drums flute cello trumpet harp banjo oboe".split()),
    "sport": tuple("tennis soccer chess rugby golf boxing rowing judo cycling fencing".split()),
    "favorite drink": tuple("coffee tea cocoa cider lemonade soda milk juice".split()),
    "door code": tuple(f"{n:04d}" for n in range(1000, 9999, 37)),
}


def _surname(rng: random.Random) -> str:
    return "".join(rng.choice(SYLLABLES) for _ in range(rng.randint(2, 3))).capitalize()


def synthetic_example(
    rng: random.Random, *, facts: int, entities: int, split: str
) -> Example:
    names_pool = FIRST_NAMES_TRAIN if split == "train" else FIRST_NAMES_EVAL
    if entities > facts:
        entities = facts
    people = []
    seen = set()
    while len(people) < entities:
        name = f"{rng.choice(names_pool)} {_surname(rng)}"
        if name not in seen:
            seen.add(name)
            people.append(name)
    attrs = list(ATTRIBUTES)
    pairs: list[tuple[str, str]] = []
    used = set()
    while len(pairs) < facts:
        person = people[len(pairs) % entities] if len(pairs) < entities else rng.choice(people)
        attr = rng.choice(attrs)
        if (person, attr) in used:
            continue
        used.add((person, attr))
        pairs.append((person, attr))
    rng.shuffle(pairs)
    lines = []
    values = {}
    for person, attr in pairs:
        value = rng.choice(ATTRIBUTES[attr])
        values[(person, attr)] = value
        lines.append(f"{person}'s {attr} is {value}.")
    passage = "Here are some facts to remember.\n" + "\n".join(lines)
    person, attr = rng.choice(pairs)
    question = f"What is {person}'s {attr}?"
    return Example(passage=passage, question=question, answer=values[(person, attr)])


def synthetic_examples(
    n: int, *, seed: int, facts: int, entities: int, split: str
) -> list[Example]:
    rng = random.Random(f"{seed}-{split}-{facts}-{entities}")
    return [synthetic_example(rng, facts=facts, entities=entities, split=split) for _ in range(n)]


def squad_examples(root: Path, split: str, *, max_context_chars: int = 1500, limit: int | None = None, seed: int = 0) -> list[Example]:
    import pyarrow.parquet as pq

    name = "train-00000-of-00001.parquet" if split == "train" else "validation-00000-of-00001.parquet"
    table = pq.read_table(root / name).to_pylist()
    rng = random.Random(seed)
    rng.shuffle(table)
    out = []
    for row in table:
        context = row["context"]
        if len(context) > max_context_chars:
            continue
        answers = row["answers"]["text"]
        if not answers:
            continue
        out.append(
            Example(
                passage=context,
                question=row["question"],
                answer=answers[0],
                aliases=tuple(dict.fromkeys(answers)),
            )
        )
        if limit is not None and len(out) >= limit:
            break
    return out


SYNTHETIC_INSTRUCTION = "Answer with only the value, nothing else."
SQUAD_INSTRUCTION = "Answer with a short phrase, nothing else."


def read_prompt(example: Example, *, in_context: bool, dataset: str) -> str:
    instruction = SYNTHETIC_INSTRUCTION if dataset == "synthetic" else SQUAD_INSTRUCTION
    if in_context:
        return f"{example.passage}\n\n{instruction}\nQuestion: {example.question}"
    return f"{instruction}\nQuestion: {example.question}"


def normalize_answer(text: str) -> str:
    text = text.lower()
    text = "".join(ch for ch in text if ch not in set(string.punctuation))
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    return " ".join(text.split())


def exact_match(prediction: str, example: Example) -> float:
    pred = normalize_answer(prediction.strip().splitlines()[0] if prediction.strip() else "")
    golds = (example.answer,) + tuple(example.aliases)
    return float(any(pred == normalize_answer(g) for g in golds))


def f1_score(prediction: str, example: Example) -> float:
    pred = normalize_answer(prediction.strip().splitlines()[0] if prediction.strip() else "").split()
    best = 0.0
    for gold in (example.answer,) + tuple(example.aliases):
        gold_tokens = normalize_answer(gold).split()
        common = Counter(pred) & Counter(gold_tokens)
        overlap = sum(common.values())
        if overlap == 0:
            continue
        precision = overlap / len(pred)
        recall = overlap / len(gold_tokens)
        best = max(best, 2 * precision * recall / (precision + recall))
    return best
