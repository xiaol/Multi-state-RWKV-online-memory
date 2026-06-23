# Multi-State RWKV Online Memory

Mechanism-level experiments for comparing Dynamic Linear Attention (DLA) with
RWKV-style online memory under controlled state and boundary policies.

This repository starts from the Log-Linear Attention codebase and adds a
CPU-only proof of concept in `dla_poc.py`. It reproduces the core DLA mechanism
from arXiv 2606.10650 and adds HRM-Text-inspired memory baselines:

- `rwkv_mem(delta_rule)`: single online delta-rule associative memory.
- `rwkv_mem(rwkv7)`: single read-before-write RWKV-7 state.
- `rwkv_mem(rwkv7 multi-state)`: same RWKV-7 state update, but one state per
  adaptive memory block.
- State-only ablation: fixes the exact same boundaries for linear/DLA states
  and RWKV-7 states, so the comparison isolates the state update.

## Quick Start

```bash
python3.12 -m venv .venv
PATH="$PWD/.venv/bin:$PATH" bash run.sh
```

If the environment is already set up:

```bash
.venv/bin/python dla_poc.py
```

Outputs are written to:

```text
EVAL.md
.openresearch/artifacts/dla_summary.json
.openresearch/artifacts/dla_trials.jsonl
.openresearch/artifacts/dla_comparison.png
.openresearch/artifacts/run_log.txt
```

## Current Result

The main DLA reproduction still passes:

- DLA lowers the Theorem 3.1 deviation bound in every tested config.
- DLA beats fixed Log-Linear blocking on needle recall at matched state count.
- The repo Log-Linear attention smoke test passes on CPU.

Mechanism recall comparison:

| needles | filler/seg | K | states | fixed | rwkv_mem(delta_rule) | rwkv_mem(rwkv7) | rwkv_mem(rwkv7 multi-state) | DLA |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 6 | 8 | 16 | 12.0 | 0.920 | 0.229 | 0.797 | 1.000 | 1.000 |
| 10 | 6 | 24 | 20.0 | 0.934 | 0.122 | 0.682 | 1.000 | 1.000 |
| 8 | 10 | 20 | 16.0 | 0.887 | 0.046 | 0.626 | 1.000 | 1.000 |

State-update-only comparison, with identical boundaries for both states:

| boundary policy | needles | filler/seg | K | states | linear/DLA state | RWKV-7 state | RWKV - linear |
|---|---:|---:|---:|---:|---:|---:|---:|
| oracle | 8 | 12 | 16 | 16.0 | 1.000 | 1.000 | +0.000 |
| dla | 8 | 12 | 16 | 16.0 | 1.000 | 1.000 | +0.000 |
| fixed | 8 | 12 | 16 | 16.0 | 0.848 | 0.980 | +0.133 |
| noisy_dla | 8 | 12 | 16 | 16.0 | 0.874 | 0.987 | +0.112 |
| low_k_dla | 8 | 12 | 16 | 8.0 | 0.640 | 0.952 | +0.313 |
| oracle | 12 | 10 | 16 | 24.0 | 1.000 | 1.000 | +0.000 |
| dla | 12 | 10 | 16 | 16.0 | 0.792 | 0.982 | +0.190 |
| fixed | 12 | 10 | 16 | 16.0 | 0.763 | 0.991 | +0.228 |
| noisy_dla | 12 | 10 | 16 | 16.0 | 0.691 | 0.973 | +0.282 |
| low_k_dla | 12 | 10 | 16 | 8.0 | 0.516 | 0.889 | +0.373 |
| oracle | 16 | 8 | 12 | 32.0 | 1.000 | 1.000 | +0.000 |
| dla | 16 | 8 | 12 | 12.0 | 0.556 | 0.827 | +0.272 |
| fixed | 16 | 8 | 12 | 12.0 | 0.649 | 0.972 | +0.324 |
| noisy_dla | 16 | 8 | 12 | 12.0 | 0.509 | 0.819 | +0.311 |
| low_k_dla | 16 | 8 | 12 | 6.0 | 0.371 | 0.669 | +0.299 |

This table fixes the exact same token blocks for both methods. `linear/DLA state`
uses the standard block sum `sum k_t v_t^T`; `RWKV-7 state` uses the RWKV-7
recurrence inside each same block. Therefore each row compares state
update/readout only, not boundary quality.

Interpretation:

- With perfect or near-perfect boundaries, linear/DLA state and RWKV-7 state tie
  on this synthetic recall task.
- When boundaries are fixed, noisy, or compressed to low K, RWKV-7 state is more
  robust in this task.
- DLA's main advantage is adaptive boundary/state allocation; RWKV-7's advantage
  appears in the state update when boundaries are held fixed and imperfect.

Full tables are in `EVAL.md`.

## What Is Compared

`dla_poc.py` runs four groups of checks.

1. Codebase smoke test
   - Loads the original Log-Linear Attention pure PyTorch path directly.
   - Avoids CUDA-only Triton/Mamba dependencies.

2. DLA deviation-bound check
   - Implements Algorithm 1: information-aware dynamic state merging.
   - Implements Algorithm 2: capacity-bounded adjacent state merging.
   - Compares DLA blocks against fixed contiguous blocks at matched state count.

3. Needle associative recall
   - Uses synthetic rare needle tokens mixed with redundant filler tokens.
   - Compares fixed blocks, DLA blocks, delta-rule memory, RWKV-7 memory, and
     multi-state RWKV-7 memory.

4. State-update-only ablation
   - Uses the same block boundaries for both linear/DLA and RWKV-7 state update.
   - Boundary policies: `oracle`, `dla`, `fixed`, `noisy_dla`, `low_k_dla`.
   - This isolates whether the state update/readout is stronger, independent of
     boundary selection.

## Scope

This is a training-free mechanism reproduction. It does not reproduce 50B-token
pretraining, downstream language-model evaluations, or trained HRM-Text
checkpoints.

The HRM/RWKV baselines are self-contained ports of the memory recurrence ideas,
not full imports of HRM-Text:

- `rwkv_mem(delta_rule)` follows the read-before-write delta-rule associative
  state from HRM-Text's `models/rwkv_memory.py`.
- `rwkv_mem(rwkv7)` follows the latest read-before-write RWKV-7 recurrence from
  HRM-Text's `models/rwkv7.py`, specialized to the synthetic key/value stream.

## Repository Layout

```text
dla_poc.py                         # Main reproduction and comparisons
run.sh                             # CPU dependency install + run
EVAL.md                            # Generated report from latest run
.openresearch/artifacts/           # JSONL, JSON, figure, run log
hattention/                        # Log-Linear Attention implementation used for smoke test
figs/                              # Original figure asset
integrations/delta_mem_rwkv_ms/    # delta-Mem RWKV-MS adapter patch and launcher
```

## Delta-Mem Adapter

The practical RWKV-MS memory adapter for delta-Mem is packaged in
`integrations/delta_mem_rwkv_ms/`. It includes the patch, the minimal
HRM-Text-derived RWKV-7 core, a matched delta-rule/RWKV-MS training launcher, and
a temporary-clone checker for applying the patch safely. The patch supports
Qwen3, SmolLM3, and Gemma4 text attention; for `google/gemma-4-E4B-it` it wraps
the non-KV-shared attention layers and skips the KV-shared tail layers.

### Gemma tau2 status

The current Gemma + RWKV-MS tau2 investigation is documented in
`GEMMA_RWKV_MS_TAU2_PLAN.md`. The matching local delta-Mem integration commit is:

```text
bec8330 Add RWKV-MS memory backend for Gemma tau2
```

Local tau2 telecom findings so far:

- The best useful adapter is the 2-layer `q,o` run:
  `manual_telecom_success20_rank4_layers0_1`.
- A `q,k,v,o` ablation did not improve the held-out `refuel_data` call.
- Training `q,o` on every eligible Gemma4 layer wrapped 24 non-KV-shared layers
  and skipped 18 KV-shared tail layers, but did not fix the first-turn missing
  arguments.
- The failure is not a wrong line ID: the model selects `L1002` correctly but
  often omits `customer_id` and `gb_amount`.
- Data inspection points to sparse schema supervision and prompt schema mismatch:
  the 20-row successful telecom train split has only one complete assistant
  `refuel_data(...)` action.
- A multi-turn validation retry works when the tool-error turn explicitly names
  the required signature and known values, producing:
  `refuel_data(customer_id="C1001", line_id="L1002", gb_amount=1.0)`.

Practical next step for tau2 evaluation is to run the agent with a tool-call
validator that stops/parses at `[/ACTION]` and retries malformed calls with an
explicit `[TOOL_ERROR]` message.

## Acknowledgement

This work builds on the Log-Linear Attention repository and uses local
HRM-Text/RWKV memory ideas as mechanism baselines. The added experiments are
intended for controlled research exploration, not as a trained-model benchmark.
