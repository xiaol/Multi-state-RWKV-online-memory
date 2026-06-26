# Gemma + RWKV-MS Tau2 Training Plan

> **Active recipe:** training and benchmarking are now driven by
> `GEMMA_RWKV_MS_TAU2_TRAINING_PLAN_V2.md` (written after the first 20-task
> benchmark and the RWKV-7 layer-depth init fix). This file remains the
> reference for objective, dataset, environment, and V1 results.

## Objective

Train a Gemma model with the delta-Mem RWKV-MS backend and benchmark it only on
tau2-bench, focused first on the telecom domain.

The primary experiment is not a general coding benchmark and not a long-document
QA benchmark. The goal is to improve interactive tool-agent reliability in a
customer-support simulation.

The current working hypothesis is that coding/terminal-agentic training can
improve tau2 telecom because the telecom tasks share the same loop as debugging:
inspect state, diagnose the fault, apply a fix, then verify. The public
`yuxinlu1/gemma-4-12B-agentic-fable5-composer2.5-v2-3.5x-tau2-GGUF` card reports
about 15% for base `gemma-4-12B-it` and about 55% for the agentic/coding
fine-tune on a local 20-task tau2 telecom run with Q8_0 and the same harness.
Treat that as directional evidence, not a directly comparable leaderboard
number.

## Target Comparison

Train and evaluate:

| System | Purpose |
| --- | --- |
| Base Gemma | Reference baseline |
| Gemma + RWKV-MS | Main trained system |

Optional later comparison:

| System | Purpose |
| --- | --- |
| Gemma + delta-rule delta-Mem | Ablation against the original memory backend |

For the first run, keep scope narrow: train only Gemma + RWKV-MS and compare it
against base Gemma on tau2-bench.

## Model Format

Use full Hugging Face / safetensors model weights for training.

Do not train from GGUF. GGUF is only for final local inference or quantized
serving after training is complete.

Required input:

```bash
BASE_MODEL_PATH=/run/media/xiaol/B214449214445C0B/models/gemma/gemma-4-E4B-it
```

The full Gemma safetensors model is stored on the 2 TB SSD, mounted at:

```text
/run/media/xiaol/B214449214445C0B
```

The confirmed training model directory is:

```text
/run/media/xiaol/B214449214445C0B/models/gemma/gemma-4-E4B-it
```

Confirmed files include `config.json`, `model.safetensors`, `tokenizer.json`,
`tokenizer_config.json`, `generation_config.json`, `processor_config.json`, and
`chat_template.jinja`.

The local Hugging Face cache entries under
`/home/xiaol/.cache/huggingface/hub/models--google--gemma-4-*` are incomplete
metadata-only entries and should not be used as `BASE_MODEL_PATH`.

## Codebase

RWKV-MS integration lives in:

```text
/home/xiaol/X/Multi-state-RWKV-online-memory/integrations/delta_mem_rwkv_ms
```

Training code lives in:

```text
/home/xiaol/X/delta-Mem
```

The delta-Mem checkout already appears patched for RWKV-MS and Gemma4 support.
The integration patch checker passes against the delta-Mem repository:

```bash
cd /home/xiaol/X/Multi-state-RWKV-online-memory
bash integrations/delta_mem_rwkv_ms/check_delta_mem_patch.sh /home/xiaol/X/delta-Mem
```

## Dataset Choice

Use tau2-style trajectories first, then add a small coding/terminal-agentic
warm-up slice if the goal is to reproduce the kind of tau2 telecom gain reported
by the Gemma4-12B v2 agentic/coding model.

### First Dataset: Tau2 SFT

Use:

```text
Jarrodbarnes/tau2-sft-final
```

Why:

- Directly aligned with tau2-style tool-agent behavior.
- Contains airline, retail, and telecom trajectories.
- Telecom subset is the most relevant part for tau2-bench telecom.
- Apache 2.0 license.
- Better aligned than Fable-5 for customer-support tool use.

Use telecom examples first. If the telecom subset is too small, include airline
and retail as general support-agent warm-up, but keep telecom weighted highest.

### Optional Warm-Up: Coding / Terminal-Agentic Data

The v2 Gemma model card suggests the tau2 telecom improvement came from
agentic/terminal trajectories plus verified coding traces. For our RWKV-MS run,
that means the best optional non-tau2 data is not generic code completion; it is
multi-step tool-use data where the model reads state, reasons, acts, and
verifies.

Recommended optional slice:

```text
10-30% coding / terminal-agentic trajectories
70-90% tau2-style trajectories
```

Candidate datasets:

| Dataset | Use |
| --- | --- |
| `Glint-Research/Fable-5-traces` | Optional small warm-up; useful style match, but license-sensitive |
| `Glint-Research/Complete-FABLE.5-traces-2M` | Optional larger Fable source; requires aggressive filtering |

Filter coding/agentic examples for:

- explicit read/inspect before action
- tool calls with observations
- verification after changes
- no malformed tool syntax
- no benchmark leakage into tau2 eval tasks

### Dataset To Avoid As The Only First Run

Do not use `Glint-Research/Fable-5-traces` as the only first dataset.

Reason:

- It is coding-agent telemetry, not tau2 telecom/customer-support data.
- Tool schema differs from tau2 tools.
- It is AGPL-3.0.
- It may help general tool-use style, but it cannot replace tau2-style telecom
  trajectories for a tau2-only benchmark.

Optional later use:

```text
5-10% Fable-style traces as general agent/tool warm-up
```

Only add this after the tau2 pipeline is working, unless we explicitly decide to
run a two-stage recipe:

```text
Stage A: coding / terminal-agentic warm-up
Stage B: tau2 telecom SFT
Stage C: RWKV-MS adapter training / refresh on tau2 trajectories
```

## Training Data Preparation

Convert tau2 trajectories into the format expected by
`deltamem.train.delta_sft_experimental`.

Target JSONL structure should preserve:

- system/domain policy
- user turns
- assistant reasoning if present
- tool calls/actions
- tool observations/results
- final assistant response
- task/domain metadata

Required filtering:

- Keep successful trajectories first.
- Prefer telecom domain.
- Drop malformed or partial trajectories.
- Keep tool call syntax consistent.
- Keep held-out eval tasks out of training.

Suggested split:

```text
train: tau2-sft telecom train trajectories plus optional non-telecom warm-up
eval: official tau2-bench telecom benchmark tasks only
```

Suggested training mixes:

```text
Conservative:
90% tau2-style trajectories
10% coding / terminal-agentic warm-up

Agentic-heavy:
70% tau2-style trajectories
30% coding / terminal-agentic warm-up

Telecom-only final polish:
100% tau2 telecom successful trajectories
```

## Environment Setup

In delta-Mem:

```bash
cd /home/xiaol/X/delta-Mem
PYTHON_BIN=python3.11 INSTALL_FLASH_ATTN=0 bash scripts/setup_uv_env.sh
source .venv/bin/activate
```

If training on CUDA and FlashAttention is available:

```bash
INSTALL_FLASH_ATTN=1 bash scripts/setup_uv_env.sh
```

Run basic checks:

```bash
cd /home/xiaol/X/delta-Mem
source .venv/bin/activate
PYTHONPATH=. python -m compileall -q deltamem
PYTHONPATH=. python -m pytest -q deltamem/tests/test_delta_mem_regressions.py -k "gemma4 or rwkv_ms"
```

## Smoke Test

Before real training, run a tiny Gemma + RWKV-MS smoke test:

- Load Gemma from `BASE_MODEL_PATH`.
- Attach delta-Mem with `memory_backend="rwkv_ms"`.
- Verify Gemma non-KV-shared attention layers are wrapped.
- Verify Gemma KV-shared layers are skipped.
- Run one short forward pass.
- Verify RWKV-MS state exists and has expected shape.
- Save and reload a tiny adapter checkpoint.

Do not start full training until this passes.

## Training Configuration

Start with the existing launcher:

```bash
cd /home/xiaol/X/delta-Mem

BASE_MODEL_PATH=/run/media/xiaol/B214449214445C0B/models/gemma/gemma-4-E4B-it \
TRAIN_FILE=/path/to/tau2_train.jsonl \
OUTPUT_ROOT=/home/xiaol/X/outputs/gemma_rwkv_ms_tau2 \
NPROC_PER_NODE=1 \
bash scripts/run_rwkv_ms_delta_rule_comparison.sh
```

For first run, either edit the script or create a smaller RWKV-MS-only launcher.
The comparison script trains both delta-rule and RWKV-MS, but this project phase
only requires RWKV-MS.

Local status:

- A RWKV-MS-only launcher was added at
  `/home/xiaol/X/delta-Mem/scripts/run_gemma_rwkv_ms_tau2.sh`.
- The Hugging Face `Trainer` path currently runs forward/backward but did not
  update the tiny BF16 RWKV-MS adapter weights in the local Transformers 5.12.1
  environment. Use the manual loop added at
  `/home/xiaol/X/delta-Mem/scripts/train_gemma_rwkv_ms_tau2_manual.py` for
  current tau2 training runs.
- The manual loop reuses delta-Mem model attach/freeze, dataset tokenization,
  episode collator, and `DeltaMemTrainer.compute_loss`, but performs
  `loss.backward()` and `AdamW.step()` explicitly.

Core RWKV-MS options:

```bash
--memory-backend rwkv_ms
--rwkv-ms-num-states 4
--rwkv-ms-chunk-size 1024
--rwkv-ms-boundary-mode fixed_chunk
--rwkv-ms-erase-gate 1.0
--rwkv-ms-read-top-k 0
```

Suggested first training run:

```text
examples: 100-500
epochs: 1
max-length: 512
max-write-length: 2048 or 4096
batch size: 1
gradient accumulation: 4
rank: 8
alpha: 16
delta heads: q,o
```

Suggested full run:

```text
examples: all filtered tau2 trajectories plus generated successful telecom data
epochs: 1-3
max-length: 512
max-write-length: 8192
batch size: 1
gradient accumulation: 4-16
rank: 8 or 16
alpha: 16 or 32
delta heads: q,o first; q,k,v,o later only if needed
```

## Benchmark

Use only tau2-bench.

Primary domain:

```text
telecom
```

Report:

| Metric | Meaning |
| --- | --- |
| pass^1 | One-run success reliability |
| pass^2 | Success in two independent runs, stricter |
| pass^4 | Optional if budget allows |
| average turns | Conversation efficiency |
| tool error rate | Invalid or failed tool calls |
| policy violation rate | Violations of domain policy |
| failure categories | Manual or scripted error taxonomy |

Primary table:

| Model | tau2 telecom pass^1 | pass^2 | avg turns | tool errors |
| --- | ---: | ---: | ---: | ---: |
| Base Gemma | 0.30 | TBD | TBD | TBD |
| Gemma + RWKV-MS (2-layer q,o) | 0.35 | TBD | TBD | TBD |
| Gemma + RWKV-MS (all-eligible q,o) | 0.15 | TBD | TBD | TBD |

Source: 20-task telecom solo-mode run, greedy decode (`do_sample=false`,
`max_new_tokens=384`, seed 300). Base + 2-layer in
`tau2_solo_benchmark_small20`; all-eligible q,o in
`tau2_solo_benchmark_small20_full_qo` (base not re-run there — base is
deterministic at 0.30 under identical settings). pass^2 / avg turns / tool
errors not yet extracted from the per-simulation results.

Adapters:

- 2-layer: `manual_telecom_success20_rank4_layers0_1` (layers 0,1).
- all-eligible: `manual_telecom_success20_rank4_all_eligible_qo` (24 non-KV-shared
  layers 0-23 wrapped, KV-shared tail 24-41 skipped).

Key finding: wrapping all eligible layers **hurts** tau2 telecom (0.15, half of
base), while the narrow 2-layer adapter slightly helps (0.35 vs 0.30 base). This
is consistent with the earlier qualitative result that the all-eligible adapter
emitted incomplete / malformed tool calls. Recommend keeping RWKV-MS narrow
(few early layers) and treating the all-eligible config as a negative result.

Optional later table:

| Model | tau2 telecom pass^1 | pass^2 | avg turns | tool errors |
| --- | ---: | ---: | ---: | ---: |
| Base Gemma | TBD | TBD | TBD | TBD |
| Gemma + delta-rule | TBD | TBD | TBD | TBD |
| Gemma + RWKV-MS | TBD | TBD | TBD | TBD |

## Evaluation Rules

Keep evaluation clean:

- Do not train on official tau2-bench eval tasks.
- Do not tune prompts repeatedly on the final eval split.
- Use identical decoding settings for base Gemma and Gemma + RWKV-MS.
- Record model path, adapter checkpoint, prompt template, tau2-bench commit, and
  decoding parameters.

Suggested decoding:

```text
temperature: 0 or low deterministic setting for first comparison
top_p: 1.0
max turns: tau2 default
max tokens: tau2 default or fixed across models
```

## Milestones

### Milestone 1: Confirm Inputs

- [x] Full Gemma safetensors path confirmed:
  `/run/media/xiaol/B214449214445C0B/models/gemma/gemma-4-E4B-it`.
- [x] delta-Mem environment installed on the 2 TB SSD and symlinked from
  `/home/xiaol/X/delta-Mem/.venv`.
- [x] tau2-sft dataset downloaded to the SSD Hugging Face cache.
- [ ] tau2-bench repository installed or runnable.

### Milestone 2: Data Conversion

- [x] Inspect tau2-sft schema.
- [x] Filter telecom successful trajectories.
- [x] Convert to delta-Mem JSONL training format.
- [x] Validate tokenization and sequence lengths.

Local converted data:

```text
/run/media/xiaol/B214449214445C0B/delta_mem_data/tau2/tau2_telecom_smoke8.jsonl
/run/media/xiaol/B214449214445C0B/delta_mem_data/tau2/tau2_telecom_success.jsonl
/run/media/xiaol/B214449214445C0B/delta_mem_data/tau2/tau2_telecom_all_valid.jsonl
```

Counts from `tau2_sft_final.jsonl`:

```text
telecom success: 20
telecom all valid: 82
all valid rows: 416
```

### Milestone 3: Smoke Test

- [ ] Gemma loads.
- [ ] RWKV-MS adapter attaches.
- [ ] Forward pass works.
- [ ] Tiny checkpoint saves and reloads.

Local smoke status:

- Gemma E4B loads from the SSD path.
- RWKV-MS attaches to Gemma4 non-KV-shared layers.
- A Gemma4 sliding-window 4D-mask edge case was fixed in
  `deltamem/core/delta_impl.py`; without this, 512-token write phases marked all
  tokens invalid and kept RWKV-MS state at zero.
- Corrected manual smoke output:

```text
/run/media/xiaol/B214449214445C0B/delta_mem_outputs/gemma_rwkv_ms_tau2/manual_smoke8_rank2_layer0
```

### Milestone 4: Tiny Training Run

- [x] Train on available successful telecom examples.
- [x] Confirm nonzero gradients and adapter tensor updates.
- [x] Confirm adapter checkpoint can be loaded and prompted.

First completed local tiny run:

```text
/run/media/xiaol/B214449214445C0B/delta_mem_outputs/gemma_rwkv_ms_tau2/manual_telecom_success20_rank4_layers0_1
```

Run configuration:

```text
train file: tau2_telecom_success.jsonl
examples: 20
steps: 20
target layers: 0,1
rank: 4
alpha: 8
max_length: 256
max_write_length: 512
attention: sdpa
optimizer: manual AdamW
```

Qualitative held-out prompt comparison:

```text
script: /home/xiaol/X/delta-Mem/scripts/compare_gemma_rwkv_ms_prompt.py
report: /run/media/xiaol/B214449214445C0B/delta_mem_outputs/gemma_rwkv_ms_tau2/prompt_compare_heldout_row4.json
valid row index: 4
task_id: [telecom][mobile_data_issue]data_usage_exceeded|user_abroad_roaming_enabled_off[PERSONA:Easy]
decode: greedy, max_new_tokens=768
```

Result summary:

- Base Gemma diagnosed the exceeded data limit and asked the user how much data
  to refuel.
- Gemma + RWKV-MS emitted the trained `[ACTION] ... [/ACTION]` style and chose
  `refuel_data`, but the generated call was incomplete for this tool schema:
  `refuel_data(line_id="L1002")`.
- The row is held out by `task_id` and `tau2_task_id`, but its
  `tool_sequence_hash` appears in the training set, so treat this as a
  qualitative format/behavior check rather than a clean generalization
  benchmark.

Follow-up local ablations:

- The training loop was fixed so a final partial gradient accumulation window
  performs an optimizer step and records `optimizer_steps`.
- The best useful checkpoint remains the 2-layer `q,o` run:
  `/run/media/xiaol/B214449214445C0B/delta_mem_outputs/gemma_rwkv_ms_tau2/manual_telecom_success20_rank4_layers0_1`.
- A `q,k,v,o` run on layers `0,1` did not improve the held-out row; it often
  asked for confirmation or emitted malformed completion text.
- An all-eligible-layer `q,o` run wrapped 24 Gemma4 non-KV-shared layers
  (`0-23`) and skipped the 18 KV-shared tail layers (`24-41`), but still
  produced an incomplete first-turn `refuel_data(line_id="L1002")` call.
- Data inspection found only 1 complete assistant `refuel_data(...)` call in
  the 20-row successful telecom training set. The broader all-valid telecom
  file contains 12 complete `refuel_data(...)` calls. This points to sparse
  schema supervision and prompt schema mismatch as the likely issue, not a line
  ID bug or RWKV-MS recurrence bug.
- Greedy decoding was used for the main comparisons:
  `do_sample=false`, `temperature=1.0`, `top_p=1.0`, `top_k=0`.
  Decode sweeps did not fix the missing arguments.

Multi-turn repair result:

- A lightweight repair harness was added in delta-Mem at
  `scripts/test_gemma_rwkv_ms_multiturn_repair.py`.
- If the first model turn emits the incomplete `refuel_data(line_id="L1002")`,
  a second `[TOOL_ERROR]` turn can repair it.
- For the 2-layer `q,o` adapter, a signature-only repair prompt produced all
  required arguments by 384 tokens, but used `gb_amount=1` instead of the exact
  gold `1.0`; repair prompts that supplied the known values produced the exact
  gold call.
- For the all-eligible-layer `q,o` adapter, signature-only repair asked the user
  how much data to add, but repair prompts that supplied the known values
  produced the exact gold call:
  `refuel_data(customer_id="C1001", line_id="L1002", gb_amount=1.0)`.
- Practical tau2 wrapper fix: validate tool-call arguments, stop/parse at
  `[/ACTION]`, and retry with an explicit tool-error turn that names the
  required signature and the known values from the conversation.

Current delta-Mem implementation commit:

```text
bec8330 Add RWKV-MS memory backend for Gemma tau2
```

### Milestone 5: Tau2 Baseline

- [ ] Run base Gemma on tau2 telecom.
- [ ] Record pass^1 and failure modes.

### Milestone 6: Full RWKV-MS Training

- [ ] Train RWKV-MS adapter on filtered tau2 data.
- [ ] Save final checkpoint.
- [ ] Run tau2 telecom benchmark.

### Milestone 7: Report

- [ ] Produce benchmark table.
- [ ] Include training config.
- [ ] Include eval config.
- [ ] Include qualitative failure examples.
- [ ] Decide whether to add generated telecom trajectories or delta-rule ablation.

## Open Decisions

- Which Gemma base: E4B for fast iteration or 12B for stronger final result.
- Whether to train telecom-only or telecom-heavy mixed-domain.
- Whether to add a coding/terminal-agentic warm-up stage before tau2 SFT.
- Whether to generate additional successful tau2 telecom trajectories.
- Whether to include delta-rule ablation after the first RWKV-MS result.
- Whether to export GGUF after training.

## References

- tau2-bench: https://github.com/sierra-research/tau2-bench
- Gemma4-12B v2 agentic/coding model card:
  https://huggingface.co/yuxinlu1/gemma-4-12B-agentic-fable5-composer2.5-v2-3.5x-tau2-GGUF
- Tau2 SFT bootstrap:
  https://huggingface.co/datasets/Jarrodbarnes/tau2-sft-final
- Fable-5 traces:
  https://huggingface.co/datasets/Glint-Research/Fable-5-traces
