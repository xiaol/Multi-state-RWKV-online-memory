# Gemma + RWKV-MS Tau2 Training Plan — V2

Supersedes the training/benchmark sections of `GEMMA_RWKV_MS_TAU2_PLAN.md`.
Keep that file for the original objective, dataset, and environment notes; this
file is the active recipe after the first benchmark round and the RWKV-7 init
fix.

## What V1 established (and why we are revising)

First 20-task telecom solo-mode benchmark (greedy, `max_new_tokens=384`, seed
300):

| Model | pass^1 |
| --- | ---: |
| Base Gemma | 0.30 (6/20) |
| RWKV-MS, 2-layer `q,o` (layers 0,1) | 0.35 (7/20) |
| RWKV-MS, all-eligible `q,o` (layers 0-23) | 0.15 (3/20) |

Two problems made these numbers unreliable:

1. **Undertraining.** Both adapters were trained for only 20 steps on 20
   examples. Training loss never decreased (oscillated ~2.6-4.5 and ended where
   it started). The benchmark mostly measured the *initialization* of RWKV-MS,
   not a learned skill. The 2-layer "+0.05" is a single extra task — noise at
   n=20. The all-eligible "-0.15" (3/20) is the only real signal, and it is
   confounded by undertraining.

2. **Init bug (now fixed).** The RWKV-7 core in `deltamem/core/hrm_rwkv7.py`
   dropped the official RWKV-7 layer-depth-dependent init: every wrapped layer
   was initialized as if it were the shallowest layer (`x_*` mix and `w0` decay
   used `ratio=1` / linear), instead of the official depth-graded schedules from
   RWKV-LM v7 (`train_temp/src/model.py`). It also used a single truncated-normal
   scale for r/k/v instead of the official uniform scheme with the key 10x
   smaller. This is plausibly a real contributor to the all-eligible
   degradation: all 24 cores got an identical shallow-layer memory horizon.

   The recurrent state init itself (zeros per sequence, no cross-sequence carry)
   and the core output projection (`output.weight = 0`, exact identity at init)
   already matched the official recipe and were not changed.

## Init fix (applied)

In `deltamem/core/hrm_rwkv7.py` (`HRMRWKV7LowRankCore`):

- `__init__` now takes `layer_id` and `n_layer` (defaults `0` / `1`, which
  reproduce the old shallow-layer init for backward compatibility).
- `reset_parameters` restored the official depth schedules:
  - `ratio_0_to_1 = layer_id / (n_layer - 1)`,
    `ratio_1_to_almost0 = 1 - layer_id / n_layer`.
  - `x_r/x_w/x_k/x_v/x_a/x_g = 1 - ddd^(p * ratio_1_to_almost0)`.
  - `w0` decay exponent `1 + ratio_0_to_1^0.3` (deeper layers → longer horizon).
  - r/v `~uniform(±0.5/√dim)`, k `~uniform(±0.05/√dim)`.
- `a0`, `k_k`, `k_a`, the `w/a/g` LoRAs, and `output.weight = 0` are unchanged
  (they already matched).

In `deltamem/core/delta_impl.py`:

- Added `_resolve_num_hidden_layers(config)` (handles flat and `text_config`-
  nested configs).
- The core is now constructed with `layer_id=self.layer_idx` and
  `n_layer=_resolve_num_hidden_layers(self.config)`.

Verification done: `compileall` clean; depth grading confirmed (shallow vs deep
`x_w`/`w0` differ; key 10x smaller than receptance; output zero); defaults match
old init; full `deltamem/tests/test_delta_mem_regressions.py` = 59 passed. The
integration mirror `integrations/delta_mem_rwkv_ms/hrm_rwkv7.py` was synced to
the fixed version.

> Note: `integrations/delta_mem_rwkv_ms/delta_mem_rwkv_ms.patch` is stale (the
> RWKV-MS feature is already committed in delta-Mem at `bec8330`, so the patch
> checker reports "already applied"). Regenerating that patch is out of scope
> for this plan.

## Fixed inputs

```text
BASE_MODEL   = /run/media/xiaol/B214449214445C0B/models/gemma/gemma-4-E4B-it
OUTPUT_ROOT  = /run/media/xiaol/B214449214445C0B/delta_mem_outputs/gemma_rwkv_ms_tau2
TOK_ROOT     = /run/media/xiaol/B214449214445C0B/delta_mem_tokenized/tau2_manual
TRAINER      = /home/xiaol/X/delta-Mem/scripts/train_gemma_rwkv_ms_tau2_manual.py
BENCH        = /home/xiaol/X/delta-Mem/scripts/run_tau2_gemma_local_benchmark.py
```

Training data (telecom, already converted):

```text
tau2_telecom_success.jsonl     20 rows  (successful telecom only)
tau2_telecom_all_valid.jsonl   82 rows  (all valid telecom trajectories)
```

Decision for V2: train on `tau2_telecom_all_valid.jsonl` (82) as the primary
set — 20 rows is too few to train past noise. Keep `tau2_telecom_success.jsonl`
as an optional final polish. If 82 is still too few to drop loss, generate more
successful telecom trajectories (see V1 "Open Decisions") before widening
layers.

## Current status — 2026-06-25

The original V2 recipe has now been tested. The 82-row `all_valid` Phase 1 run
did reach 656 optimizer steps and its loss moved, but it did not transfer to the
tau2 mobile-data benchmark. The useful path is the later generated
mobile-data/action SFT data, especially the format-refresh continuation.

Accepted learned online-memory comparisons below mean **no eval-time
`--mobile-data-rule-planner` and no parser format-repair patch**. The benchmark
is still only a 20-task screen, so treat it as model selection, not a final
claim.

Best learned checkpoint:

```text
/run/media/xiaol/B214449214445C0B/delta_mem_outputs/gemma_rwkv_ms_tau2/v2ruleplanner_mobile_focusedtools_turns_formatrefresh_continue200_len192_layers0_5_qo_r8/checkpoints/step-100
```

HF online-memory repo:

```text
xiaol/gemma-4-e4B-hybrid-rnn-mem-rwkv-fable5-gpt5.5-v1
```

Benchmark summary:

```text
/run/media/xiaol/B214449214445C0B/delta_mem_outputs/gemma_rwkv_ms_tau2/tau2_bench_ruleplanner_sft_formatrefresh_step100_len192_layers0_5_norule_20/rwkv_ms_ruleplanner_sft_formatrefresh_step100_len192_layers0_5_norule_20/summary.json
```

Configuration: Gemma4 E4B, RWKV-MS `q,o`, layers `0-5`, rank 8 / alpha 16,
`max_length=192`, resumed from the 6-layer len256 generated-action SFT run.
Result: **14/20**, `pass_hat_1=0.70`, `avg_reward=0.70`,
`infra_error_count=0`.

Accepted 20-task benchmark setup for the current table: solo/dummy-user mode,
greedy decoding (`do_sample=false`, `temperature=1.0`, `top_p=1.0`, `top_k=0`),
`max_new_tokens=96`, `max_steps=40`, `max_errors=4`, seed `300`.

The final checkpoint from the same 200-step continuation scored **12/20**. This
is now a clear early-stopping/checkpoint-selection issue: more steps helped the
training loss but not the task score.

Actual generated training files used after rejecting the 82-row data:

```text
/run/media/xiaol/B214449214445C0B/delta_mem_data/tau2/tau2_telecom_mobile_data_rule_planner_train_focusedtools_nopolicy_turns.jsonl
/run/media/xiaol/B214449214445C0B/delta_mem_data/tau2/tau2_telecom_mobile_data_rule_planner_train_focusedtools_nopolicy_turns_formatrefresh_balanced.jsonl
```

The corrective balanced dataset
`tau2_telecom_oracle_action_mobile_train_focused_diag_finalspeed_focusedtools_nopolicy_sigtools_turns_corrective_balanced_v2.jsonl`
was generated and load-checked, but continuation on it regressed to 10/20, so it
is not the default path.

| Run / online-memory path | Layers / rank / length | Data or eval condition | pass^1 | Status |
| --- | --- | --- | ---: | --- |
| Base checkpoint `google/gemma-4-E4B-it`, focused tools + line verify + autostop | none | base-only baseline for accepted setup | 4/20 (0.20) | Lower current base checkpoint benchmark |
| Base checkpoint `google/gemma-4-E4B-it`, checklist prompt | none | prompt-only checklist baseline | 7/20 (0.35) | Prompt helps, still below learned best |
| Phase 1 `all_valid` memory path | `0,1` / r8 / len256 | 82 original tau2 valid rows, 656 steps | 1/20 (0.05) | Reject; loss moved but data/format did not transfer |
| Generated action SFT | `0,1` / r8 / len256 | 3,519 turn rows, 656 steps | 9/20 (0.45) | 2 layers are useful but not best |
| Generated action SFT | `0-5` / r8 / len256 | same data, 656 steps | 10/20 (0.50) | Shallow 6-layer band beats 2 layers |
| Generated action SFT | all eligible / r4 / len256 | 24 non-KV-shared layers, 656 steps | 1/20 (0.05) | Reject; overfits/over-perturbs despite low train loss |
| Format-refresh continuation, final | `0-5` / r8 / len192 | 5,027 turn rows, 200 more steps | 12/20 (0.60) | Good but not best checkpoint |
| Format-refresh continuation, checkpoint `step-100` | `0-5` / r8 / len192 | same run, early checkpoint | **14/20 (0.70)** | Current best learned no-rule online memory |
| Len96 continuation from 12/20 memory path | `0-5` / r8 / len96 | shorter-context continuation | 11/20 (0.55) | No improvement |
| Len128 continuation from 12/20 memory path | `0-5` / r8 / len128 | shorter-context continuation | 10/20 (0.50) | No improvement |
| Corrective/oracle balanced continuation | `0-5` / r8 / len192 | corrective final-speed data | 10/20 (0.50) | Reject as default |
| Checklist eval on format-refresh memory path | `0-5` / r8 / len192 | checklist prompt at eval | 5/20 (0.25) | Reject; checklist hurts learned memory behavior |

The eval-time mobile-data rule planner / float-format fix is excluded from the
comparison table because it is benchmark-specific control logic, not model
behavior. It is useful only as an internal diagnostic ceiling for task mechanics.

Parameter counts from the saved online-memory checkpoints:

| Memory-path shape | Trainable memory params |
| --- | ---: |
| 2 layers, r8 `q,o` | 257,744 |
| 6 layers, r8 `q,o` | 797,808 |
| 24 eligible layers, r4 `q,o` | 1,594,080 |

Interpretation:

- The original 82-row tau2 data was the dataset problem. It could move loss but
  did not teach the exact mobile-data action format/sequence needed by tau2.
- The generated mobile-data action data is better aligned. The 6-layer shallow
  band is currently the best capacity point.
- Two layers are smaller and cheaper, but not enough for the current best
  result. More layers mean more per-layer memory modules and more total online state;
  the per-layer RWKV-MS state shape is unchanged.
- All eligible layers are still bad in this task even after the init fix. The
  online-memory path can drive training loss very low but damages task behavior.
- Next required step is a larger confirmation benchmark (`--num-tasks >= 50`,
  ideally full telecom) for the `step-100` checkpoint before calling the result
  robust.

### HF release framing and local cost notes

First-line model-card quote:

```text
"One can have both the fish and the bear's paw."
```

Release message: keep the original Gemma capability by freezing the base model,
then add a tiny learned RWKV-MS online-memory path for stateful local agent
behavior. The current 6-layer path has `797,808` trainable memory parameters,
so the learned checkpoint is small; the base model still dominates VRAM.

All costs are local and variable. The accepted path used CUDA bf16 on an RTX
4090 24 GB setup, 3,519 turn rows for the generated-action SFT stage, then
5,027 rows for the 200-step format-refresh continuation. Wall time and memory
depend on local model storage, attention backend, sequence length, wrapped
layers, rank, cache state, and allocator fragmentation. The intended recipe is
to keep the base frozen and adapt the online memory to each user's own data.

## Phase 0 — Init-fix controls (cheap, do first)

Goal: separate "init perturbation" from "training damage", and confirm the fix
changed the deep-layer behavior. Eval only; no/loops of training.

1. **Untrained all-eligible eval (pre-fix question).** Attach all-eligible `q,o`
   with the *new* init, zero training steps, eval on the 20-task set.
   - ≈0.30 → the depth-graded init alone recovers base-level behavior at init.
   - ≪0.30 → residual init perturbation across depth; revisit `online_gain`
     scaling by `1/num_layers`.
2. Record alongside the V1 all-eligible 0.15 to show the init delta at step 0.

This is the fastest way to attribute the V1 0.15 result.

## Training diagnosis (why V1/V2-first looked flat)

Ran an **overfit control**: 2 examples, 2 layers (`0,1`), rank 4, lr 1e-3,
accum 1, 240 optimizer steps. Findings:

- Loss **did move** (example A 5.70 → 3.11) — gradients flow, training is **not
  structurally broken** (no dead-gradient / masking bug).
- But `grad_norm` starts **~6e-4** and stays tiny for ~the first 100 steps, then
  grows to ~0.4–3 (with an 18.5 spike) by step ~200. The adapter is initialized
  **near-identity** (beta gate σ≈0.18, `online_gain=0.05`, core output zero), so
  it contributes ~nothing at first → near-zero gradients → a **~100-step slow
  warmup** before learning begins.
- This explains the flat runs: V1 did **20** optimizer steps, the first V2 run
  **62** — both entirely inside the warmup regime. The loss wasn't stuck; it
  hadn't started.
- Secondary: **no gradient clipping** (the 18.5 spike → instability), and even a
  2-example overfit only reaches ~3.1 (rank-4 q,o on 2 layers has limited
  capacity).

### Fixes

1. **Open the gates at init** so the adapter has leverage from step 0:
   `--beta-bias-init 0.0 --online-gain 0.2` (was −1.5 / 0.05).
2. **Higher LR**: `--learning-rate 1e-3` (was 2e-4).
3. **Gradient clipping** (new flag added to the manual trainer):
   `--max-grad-norm 1.0`. Implemented in
   `scripts/train_gemma_rwkv_ms_tau2_manual.py` (clips global trainable-param
   grad norm before every optimizer step, including the final partial window;
   `0` disables).
4. **Many more optimizer steps**: use `--gradient-accumulation-steps 1` and
   enough epochs to get **hundreds** of steps (82 rows × 8 epochs ≈ 656 steps).

### Memory envelope (hard constraint)

Base Gemma E4B is multimodal and occupies ~21 GiB by itself on the 24 GiB GPU.
`--max-length 512 --max-write-length 2048` OOMs. Use **`--max-length 256
--max-write-length 512`** (V1-proven). Full-layer (all-eligible, 24 layers) must
use **rank 4** to fit; `≤6` layers can use rank 8. Always export
`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`.

## Phase 1 — Train until the loss actually moves

Start with the 2-layer config (cheapest, best in V1), 82-row data, **all fixes**:

```bash
cd /home/xiaol/X/delta-Mem && source .venv/bin/activate
export PYTHONPATH=. TOKENIZERS_PARALLELISM=false PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

python scripts/train_gemma_rwkv_ms_tau2_manual.py \
  --model-path /run/media/xiaol/B214449214445C0B/models/gemma/gemma-4-E4B-it \
  --train-file /run/media/xiaol/B214449214445C0B/delta_mem_data/tau2/tau2_telecom_all_valid.jsonl \
  --tokenized-dataset-root /run/media/xiaol/B214449214445C0B/delta_mem_tokenized/tau2_manual \
  --output-dir /run/media/xiaol/B214449214445C0B/delta_mem_outputs/gemma_rwkv_ms_tau2/v2p1_layers0_1_qo_r8 \
  --target-layers "0,1" --delta-heads q,o --rank 8 --alpha 16 \
  --max-length 256 --max-write-length 512 \
  --per-device-train-batch-size 1 --gradient-accumulation-steps 1 \
  --learning-rate 1e-3 --beta-bias-init 0.0 --online-gain 0.2 --max-grad-norm 1.0 \
  --num-train-epochs 8 --max-steps -1 \
  --logging-steps 10 --save-steps 200 --seed 42
```

Optional fast validation first (should now collapse toward ~0, unlike before):
swap `--train-file` to `tau2_telecom_overfit2.jsonl` and `--num-train-epochs
120`. If the 2-example loss does **not** drop well below 1, raise LR or gate
openness further before the full run.

Acceptance gate (training, not benchmark):

- Train loss shows a clear downward trend **after the ~first 100 steps**
  (compare last-quarter mean to the post-warmup minimum, not to step 0).
- `optimizer_steps` reaches the hundreds.
- `grad_norm` rises out of the ~1e-4 floor into the ~0.1–1 range (clipped at 1.0).
- Adapter tensors change (growing `delta_*_proj` norm in `training_summary.json`).

If loss still won't drop: raise `--learning-rate` to 3e-3, push
`--beta-bias-init` to ~0.5 / `--online-gain` to ~0.3, or add more telecom data.
Do not proceed to the layer sweep until the 2-layer run trains cleanly.

## Phase 2 — Layer-count sweep with the fixed init

Only after Phase 1 trains cleanly. Same data, epochs, rank; vary wrapped layers.
The depth-aware init now gives deep layers the correct memory horizon, so this
is the real test of the V1 "all-layers worse" finding.

Reuse the **exact Phase 1 command and fix flags** (lr 1e-3, gates open, clip 1.0,
256/512, accum 1, 8 epochs); only change `--target-layers`, `--output-dir`, and
**rank** (memory):

| Run name | `--target-layers` | rank/alpha | Notes |
| --- | --- | --- | --- |
| `v2p1_layers0_1_qo_r8` | `0,1` | 8 / 16 | V1 best, Phase 1 run |
| `v2p1_layers0_5_qo_r8` | `0,1,2,3,4,5` | 8 / 16 | shallow band |
| `v2p1_all_qo_r4` | `""` (empty = all eligible) | **4 / 8** | re-test all 24, fixed init; rank 4 to fit 24 GiB |

Empty `--target-layers ""` resolves to all eligible non-KV-shared layers
(0-23 on Gemma4 E4B; KV-shared 24-41 are skipped automatically). The all-layer
run **must** drop to rank 4 — rank 8 over 24 layers leaves <2 GiB free and the
allocator thrashes (looks hung). Watch `nvidia-smi`: keep ≥2 GiB free.

## Phase 3 — Benchmark

Identical decoding to V1 so numbers are comparable. Base is deterministic and
already 0.30 — do not re-run it; reuse the stored value (or run once to
re-confirm).

```bash
PYTHONPATH=. python scripts/run_tau2_gemma_local_benchmark.py \
  --adapter-dir "$OUTPUT_ROOT/<run_name>" \
  --output-root "$OUTPUT_ROOT/tau2_bench_<run_name>" \
  --num-tasks 20 \
  --systems rwkv_ms \
  --rwkv-system-name <run_name>
```

Decoding (fixed across all systems): greedy (`do_sample` off), `temperature=1.0`,
`top_p=1.0`, `top_k=0`, `max_new_tokens=384`, `seed=300`, solo mode.

After the 20-task screen, run the best 1-2 adapters on a **larger task set**
(`--num-tasks` ≥ 50 or the full telecom split) before claiming a win — n=20
confidence intervals are ±~0.1-0.2.

## Success criteria

- A trained RWKV-MS adapter beats base Gemma (0.30) on the 20-task screen **and**
  holds the gain on the larger set, with a training run whose loss demonstrably
  decreased.
- Layer-count question answered: with the fixed init, does all-eligible still
  underperform the narrow band? Report the V1-vs-V2 delta for all-eligible.

## Open levers if results are still flat

- Learned initial state `S0` instead of hard zeros (currently impossible to
  express; would need a parameter in `_ensure_state`).
- Scale `online_gain` / per-layer read gate by `1/num_layers` for wide configs.
- More telecom training data (generate successful trajectories).
- Two-stage recipe (coding/terminal warm-up → telecom SFT) per V1 Stage A/B/C.

## Milestones

- [ ] Phase 0: untrained all-eligible fixed-init eval started, but the 20-task
  run was terminated after a partial result. Do not use it as a reported score.
- [x] Phase 1: 2-layer 82-row run completed 656 optimizer steps. Loss moved, but
  benchmark result was 1/20, so the original 82-row data is rejected.
- [x] Phase 2: generated-action layer sweep trained (`0,1`; `0-5`;
  all-eligible). The 6-layer shallow band is best; all-eligible is rejected.
- [x] Phase 3: 20-task benchmark screen completed for the useful adapters.
- [ ] Phase 4: run the `step-100` format-refresh checkpoint on at least 50 tasks
  or the full telecom split.
- [x] Report: V1 vs V2/current table, training-loss interpretation, decoding
  status, and current recommendation recorded here.
