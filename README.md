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

The active Gemma + RWKV-MS tau2 recipe is documented in
`GEMMA_RWKV_MS_TAU2_TRAINING_PLAN_V2.md`. The matching local delta-Mem
integration commit used by the benchmark artifacts is:

```text
bec8330 Add RWKV-MS memory backend for Gemma tau2
```

Current best learned no-rule adapter:

```text
/run/media/xiaol/B214449214445C0B/delta_mem_outputs/gemma_rwkv_ms_tau2/v2ruleplanner_mobile_focusedtools_turns_formatrefresh_continue200_len192_layers0_5_qo_r8/checkpoints/step-100
```

The table below keeps learned adapters separate from rule-assisted diagnostic
runs. "No-rule" means no eval-time `--mobile-data-rule-planner` and no parser
format-repair patch.

| Run / condition | Layers / rank / length | pass^1 | Takeaway |
| --- | --- | ---: | --- |
| Base Gemma, focused tools + line verify + autostop | none | 4/20 (0.20) | Current low baseline |
| Base Gemma, checklist prompt | none | 7/20 (0.35) | Prompt-only improvement, below learned best |
| Original 82-row Phase 1 | `0,1` / r8 / len256 | 1/20 (0.05) | Dataset/format mismatch; reject |
| Generated action SFT | `0,1` / r8 / len256 | 9/20 (0.45) | 2 layers help but are not enough |
| Generated action SFT | `0-5` / r8 / len256 | 10/20 (0.50) | Shallow 6-layer band is better |
| Generated action SFT | all eligible / r4 / len256 | 1/20 (0.05) | All-layer adapter over-perturbs |
| Format-refresh continuation, final | `0-5` / r8 / len192 | 12/20 (0.60) | Good final checkpoint |
| Format-refresh continuation, `step-100` | `0-5` / r8 / len192 | **14/20 (0.70)** | Best learned no-rule checkpoint |
| Eval-time rule planner + float formatting fix | base and RWKV-MS | 20/20 (1.00) | Diagnostic ceiling, not a learned-model result |

Adapter size from saved checkpoints:

| Adapter shape | Trainable adapter params |
| --- | ---: |
| 2 layers, r8 `q,o` | 257,744 |
| 6 layers, r8 `q,o` | 797,808 |
| 24 eligible layers, r4 `q,o` | 1,594,080 |

Status interpretation:

- The original tau2 data was the problem: the 82-row run trained for 656
  optimizer steps and its loss moved, but the benchmark collapsed to 1/20.
- Generated mobile-data action SFT transfers better, and the 6-layer shallow
  adapter is the current useful capacity point.
- The 200-step format-refresh continuation overtrains relative to its
  `step-100` checkpoint, so checkpoint selection matters.
- The next benchmark should run the `step-100` checkpoint on at least 50 tasks,
  preferably the full telecom split, before treating 14/20 as robust.

## Acknowledgement

This work builds on the Log-Linear Attention repository and uses local
HRM-Text/RWKV memory ideas as mechanism baselines. The added experiments are
intended for controlled research exploration, not as a trained-model benchmark.
