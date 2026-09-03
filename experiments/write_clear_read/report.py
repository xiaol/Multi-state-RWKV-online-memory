#!/usr/bin/env python
"""Render RESULTS.md tables from runs/*/log.jsonl (latest evaluation of every run)."""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
RUNS = ROOT / "runs"
COND = ("memory_correct", "memory_donor", "memory_zero", "base_in_context")


def load(run: Path) -> dict | None:
    log = run / "log.jsonl"
    if not log.is_file():
        return None
    records = [json.loads(l) for l in log.read_text().splitlines() if l.strip()]
    config = next((r for r in records if r.get("event") == "config"), None)
    evals = [r for r in records if r.get("event") == "eval"]
    trains = [r for r in records if r.get("event") == "train"]
    if config is None or not evals:
        return None
    last_step = max(r["step"] for r in evals)
    latest = {r["set"]: r for r in evals if r["step"] == last_step}
    done = any(r.get("event") == "done" for r in records)
    return {
        "name": run.name,
        "config": config,
        "latest": latest,
        "step": last_step,
        "train_step": trains[-1]["step"] if trains else 0,
        "done": done,
        "history": evals,
    }


def describe(cfg: dict) -> str:
    model = Path(cfg["model"]).name
    mem = cfg.get("memory", "delta")
    read = cfg.get("read_mode", "bank")
    routing = cfg["routing"]
    slots = cfg["n_states"] * cfg["slots_per_state"] if read == "bank" else f"{cfg['n_states']}/pos"
    data = cfg["dataset"] if cfg["dataset"] != "synthetic" else f"synthetic K={cfg['facts']}"
    return f"{model} | {mem} | read={read} | routing={routing} | slots={slots} | train={data}"


def main() -> None:
    rows = [r for r in (load(p) for p in sorted(RUNS.iterdir())) if r and "dead" not in r["name"]]
    out = ["# Write / clear / read results", "",
           "One frozen base model, evaluated under four read conditions on the same rows. The passage is",
           "never in the read context for the three memory conditions; only the state survives.", "",
           "* `correct`: question + memory slots from the state written from **this row's** passage.",
           "* `donor`: question + slots from the state written from **another row's** passage (previous row",
           "  in the eval batch). Same task and format, different names and values. The control: the gap",
           "  `correct - donor` is the row-specific content the state carries; what they share is format.",
           "* `zero`: question only, no slots. The frozen base with no information (chance level).",
           "* `in-context`: passage **and** question in the prompt, no slots, no adapter. The upper bound:",
           "  the frozen model reading the passage with its own attention.", "",
           "Expected ordering for a working memory: `zero <= donor < correct <= in-context`.", "",
           "**EM**: greedy generation, first line normalised (lower-case, no punctuation or articles),",
           "must equal the gold value (any alias for SQuAD); fraction of rows. Primary metric.",
           "**CE**: teacher-forced mean per-token cross-entropy of the gold answer tokens; lower is better.",
           "Comparable across the three memory conditions (same adapter), not against `in-context`,",
           "because the trained adapter learns the answer style (SQuAD: in-context CE 3.40 but EM 0.52).",
           "**mem mass**: mean attention probability on the memory slots in the wrapped layers during the",
           "`correct` read; diagnostic only. **step**: optimizer updates. **eval set**: `synthetic_kN` = N",
           "facts per passage (training used K=8 unless the setup says otherwise); `squad_val` = SQuAD v1.1",
           "validation. Regenerate with `python report.py`.", ""]
    out += ["| run | setup | step | eval set | correct EM | donor EM | zero EM | in-context EM | correct CE | donor CE | zero CE | mem mass |",
            "|---|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for r in rows:
        status = "" if r["done"] else f" (running, train step {r['train_step']})"
        for name, e in sorted(r["latest"].items()):
            mass = sum(e["mem_mass"].values()) / max(1, len(e["mem_mass"]))
            out.append(
                f"| {r['name']}{status} | {describe(r['config'])} | {r['step']} | {name} | "
                f"{e['memory_correct']['em']:.3f} | {e['memory_donor']['em']:.3f} | {e['memory_zero']['em']:.3f} | "
                f"{e['base_in_context']['em']:.3f} | {e['memory_correct']['ce']:.3f} | {e['memory_donor']['ce']:.3f} | "
                f"{e['memory_zero']['ce']:.3f} | {mass:.2f} |"
            )
    out += ["", "## Learning curves (correct EM / donor EM on the training-distribution eval set)", ""]
    for r in rows:
        key = next((k for k in r["latest"] if k.endswith(f"k{r['config']['facts']}") or k == "squad_val"), None)
        if key is None:
            continue
        curve = [(e["step"], e["memory_correct"]["em"], e["memory_donor"]["em"]) for e in r["history"] if e["set"] == key]
        out.append(f"- **{r['name']}** ({key}): " + ", ".join(f"{s}: {c:.2f}/{d:.2f}" for s, c, d in curve))
    (ROOT / "RESULTS.md").write_text("\n".join(out) + "\n")
    print("\n".join(out))


if __name__ == "__main__":
    main()
