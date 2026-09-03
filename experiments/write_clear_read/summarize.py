#!/usr/bin/env python
"""Print a compact table of every run's latest evaluation (and training position)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent / "runs"
COND = ("memory_correct", "memory_donor", "memory_zero", "base_in_context")


def main() -> None:
    only_last = "--all" not in sys.argv
    for run in sorted(ROOT.iterdir()):
        log = run / "log.jsonl"
        if not log.is_file():
            continue
        records = [json.loads(l) for l in log.read_text().splitlines() if l.strip()]
        trains = [r for r in records if r.get("event") == "train"]
        evals = [r for r in records if r.get("event") == "eval"]
        ooms = sum(1 for r in records if r.get("event") == "oom_retry")
        last_step = trains[-1]["step"] if trains else 0
        last_loss = trains[-1]["loss"] if trains else float("nan")
        print(f"== {run.name}  step {last_step}  loss {last_loss:.3f}  oom_retries {ooms}")
        by_step: dict[int, list[dict]] = {}
        for r in evals:
            by_step.setdefault(r["step"], []).append(r)
        steps = sorted(by_step)
        if only_last:
            steps = [s for s in steps if s == steps[-1] or s == 0][-1:]
        for step in steps:
            for r in by_step[step]:
                cells = []
                for c in COND:
                    if c in r:
                        cells.append(f"{c.split('_', 1)[1]:>10} ce {r[c]['ce']:.3f} em {r[c]['em']:.3f}")
                mass = sum(r["mem_mass"].values()) / max(1, len(r["mem_mass"]))
                print(f"   step {step:5d} {r['set']:>14}  " + " | ".join(cells) + f" | mass {mass:.2f}")
                if r["set"].endswith("k8") or r["set"] == "squad_val" or r["set"].endswith("k2"):
                    print("          samples: " + ", ".join(f"{s['gold']}->{s['pred'][:14]!r}" for s in r["samples"][:6]))


if __name__ == "__main__":
    main()
