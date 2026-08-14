# Multi-State RWKV Online Memory

Mechanism-level experiments for comparing Dynamic Linear Attention (DLA) with
RWKV-style online memory under controlled state and boundary policies.

HF checkpoint:
[`xiaol/gemma-4-e4B-hybrid-rnn-mem-rwkv-fable5-gpt5.5-v1`](https://huggingface.co/xiaol/gemma-4-e4B-hybrid-rnn-mem-rwkv-fable5-gpt5.5-v1)

## Latest Trained-Model Result

The frozen Gemma4 + RWKV-MS outer-memory system now passes its preregistered
native publisher-validation gate. The decoder and all task routing rules were
locked on publisher-TRAIN-derived development data before the validation split
was opened. Evaluation used the identical frozen
`google/gemma-4-E4B-it` comparator, greedy decoding, write-then-read online
memory, the HF mirror, and four A100 GPUs.

| Native task | Frozen Gemma base | Locked outer-memory system | Delta |
| --- | ---: | ---: | ---: |
| Attribution candidate accuracy (29 rows) | 0.8966 | 0.8966 | +0.0000 |
| Narrative unit accuracy (39 rows, 1,449 units) | 0.6432 | 0.6467 | +0.0035 |
| Scene-boundary micro-F1 (170 rows) | 0.1820 | 0.2727 | **+0.0907** |

All three tasks met the `>=0.95` coverage floor, no task regressed, and two
tasks improved. The scene result is the main effect: memory reduced false
positives from 698 to 171, while true positives changed from 87 to 54 and false
negatives from 84 to 117. Narrative gained five correct units. Attribution is
preserved exactly by using the frozen-base candidate-likelihood scorer.

This is a system-level result for the locked decoder, not a claim that raw
memory improves every task. Its fixed task policy is:

- attribution: frozen-base candidate likelihood;
- narrative: use memory only for the preregistered
  `base=narration, memory=scene_description` label pair;
- scene: use the RWKV-MS memory generation directly.

The reported scope is 238 rows. Attribution source row 0 was excluded in the
protocol before the final run because it had already been touched by historical
runtime diagnostics. Publisher test and Hard32 remain unopened.

Reproducibility evidence:

- [Locked publisher-validation protocol](experiments/rethinking_rwkv_ms_gemma/natural_memory_native_publisher_validation_protocol_v1.json)
- [Signed validation decision](experiments/rethinking_rwkv_ms_gemma/local_artifacts/natural_memory_native_publisher_validation_v1/decision.json)
- [Signed metrics and artifact hashes](experiments/rethinking_rwkv_ms_gemma/local_artifacts/natural_memory_native_publisher_validation_v1/result.json)
- [Validation runner](experiments/rethinking_rwkv_ms_gemma/run_natural_memory_native_publisher_validation.py)
  and [hash-bound analyzer](experiments/rethinking_rwkv_ms_gemma/analyze_natural_memory_native_publisher_validation.py)
- Independent replication seeds R12 and R13 each passed all 52 sealed checks:
  [R12](experiments/rethinking_rwkv_ms_gemma/local_artifacts/natural_memory_gate_replication_r12_sealed_run_split20260825_seed53/evaluation.json)
  and [R13](experiments/rethinking_rwkv_ms_gemma/local_artifacts/natural_memory_gate_replication_r13_sealed_run_split20260826_seed54/evaluation.json)

### Post-Validation Mechanism Study

A preregistered 357-row publisher-TRAIN-derived study has now tested three
online-state counterfactuals and 16 conservative scene routers. Correct state
scored 0.2901 scene micro-F1, versus 0.1909 with zero state and 0.1923 after
cyclically moving complete state bundles across the 42 wrapped layers. This is
strong evidence that structured online state matters. The stricter causal gate
did not pass, however: a different-gold, write-length-matched donor state scored
0.3001, beating the row-correct state by 0.0101. We therefore do not claim that
row-specific episodic content uniquely causes the scene gain.

The router screen selected `memory_plus_small_base_2`: union frozen-base and
memory boundaries only when the base predicts at most two. It improved fit
micro-F1 from 0.2844 to 0.3190 and held-out development micro-F1 from 0.3093 to
0.3103. That held-out gain is real but only +0.0011, so this router is a future
replication candidate and does not replace the accepted validation decoder.

A subsequent preregistered label-free state-retrieval study also failed its
materiality gate. Four rules selected external states from a 1,443-row
TRAIN-derived bank. Deterministic hash-random state won the 289-row fit screen,
improving 0.2795 to 0.3093 micro-F1, but improved the exactly-once 68-row
intervention holdout only from 0.3333 to 0.3352 (+0.0019, below the required
+0.005). Semantic character-TF-IDF and token-length retrieval did not win. This
points to weak generic state-induced regularization, not reliable semantic
state retrieval.

Effective Q/O memory-strength calibration has now also completed. A signed
excluded-row preflight proved that `0x`, `0.5x`, and `1x` produce different
outputs, then four A100s generated 1,136 candidates on the 284-row fit
partition. Full strength remained best at 0.2915 micro-F1. The `0.5x` and
`0.75x` candidates scored 0.2853; lower strengths scored 0.2195 and 0.2112.
No intermediate strength passed the `+0.005` gate, so no selection was written
and the 73-row intervention holdout remains unopened. This rules out a fixed
global amplitude as the next boundary, not memory learning itself.

Evidence:

- [Locked causal/router protocol](experiments/rethinking_rwkv_ms_gemma/natural_memory_native_scene_causal_router_protocol_v1.json)
- [Signed causal/router decision](experiments/rethinking_rwkv_ms_gemma/local_artifacts/natural_memory_native_scene_causal_router_v1/decision.json)
- [Signed result and raw-artifact hashes](experiments/rethinking_rwkv_ms_gemma/local_artifacts/natural_memory_native_scene_causal_router_v1/result.json)
- [Signed state-retrieval failure](experiments/rethinking_rwkv_ms_gemma/local_artifacts/natural_memory_native_scene_state_retrieval_v1/decision.json)
- [State-retrieval fit and holdout receipts](experiments/rethinking_rwkv_ms_gemma/local_artifacts/natural_memory_native_scene_state_retrieval_v1/holdout_result.json)
- [Effective-strength protocol](experiments/rethinking_rwkv_ms_gemma/natural_memory_native_scene_strength_controller_protocol_v2.json)
- [Signed effective-strength decision](experiments/rethinking_rwkv_ms_gemma/local_artifacts/natural_memory_native_scene_strength_controller_v2_r2/decision.json)
- [Effective-strength fit receipt](experiments/rethinking_rwkv_ms_gemma/local_artifacts/natural_memory_native_scene_strength_controller_v2_r2/fit_result.json)

### Contrast-Trained Scene Memory

The recommended training intervention has now succeeded on publisher-TRAIN-
derived scene data. Only the 126 shared Q/O content-gate tensors were updated;
all other adapter tensors remained bit-identical. Training used correct/no-
state positives and different-gold, write-length-matched donor negatives on
four A100 GPUs. A locked 64-row checkpoint probe selected step 16; step 32 was
rejected because it no longer beat the donor control.

Checkpoint 16 then generalized to all 220 remaining open fit rows. On the
combined 284-row fit partition it reached `0.3197` scene micro-F1, versus
`0.2915` for frozen V9, `0.3058` for matched-donor state, and `0.1980` for zero
state. The output changed on 25.7% of rows versus V9, and every preregistered
coverage, native-gain, and causal-control gate passed. This is stronger than a
generic state effect: the row-correct state now beats the matched donor.

This result does not replace the accepted publisher-validation number. It is a
TRAIN-derived candidate result, and no validation predictions were used for
checkpoint selection or analysis.

Evidence:

- [Locked contrast-training protocol](experiments/rethinking_rwkv_ms_gemma/natural_memory_native_scene_contrast_dropout_protocol_v1.json)
- [Signed training result](experiments/rethinking_rwkv_ms_gemma/local_artifacts/natural_memory_native_scene_contrast_dropout_train_v1/result.json)
- [Signed checkpoint-16 selection](experiments/rethinking_rwkv_ms_gemma/local_artifacts/natural_memory_native_scene_contrast_probe_v1/selection.json)
- [Locked full-fit progression](experiments/rethinking_rwkv_ms_gemma/natural_memory_native_scene_contrast_progression_protocol_v1.json)
- [Signed full-fit result](experiments/rethinking_rwkv_ms_gemma/local_artifacts/natural_memory_native_scene_contrast_progression_v1/result.json)

The subsequent multitask-preservation gate passed scene progression and exact
attribution reuse, but failed its narrative comparator narrowly. Checkpoint 16
reached `0.5987` routed narrative unit accuracy on the untouched 114-row
remainder, above frozen base (`0.5847`) but below V9's routed comparator
(`0.6007`) by `0.0020`. The complete signed failure is archived in the
[preservation result](experiments/rethinking_rwkv_ms_gemma/local_artifacts/natural_memory_native_multitask_preservation_v1/result.json).

### Hybrid Validation Replication

The failure is isolated to 12 narrative unit-level disagreements, so replacing
V9 narrative behavior with checkpoint-16 output is rejected. The signed hybrid
candidate keeps the proven task-wise contracts: frozen-base candidate
likelihood for attribution, exact V9 routed output for narrative, and
checkpoint-16 correct-state generation for scene. On the combined open
TRAIN-derived fit rows this gives attribution accuracy `0.6966`, V9 routed
narrative accuracy `0.6007`, and checkpoint-16 scene micro-F1 `0.3197`.

The TRAIN-derived hybrid gates passed without opening any protected split and
authorized one separately preregistered publisher-validation replication. That
replication generated every condition again from raw validation rows on four
A100 GPUs; it did not read or reuse prior validation predictions.

| Native task | Fresh frozen base | Hybrid candidate | Delta |
| --- | ---: | ---: | ---: |
| Attribution candidate accuracy | 0.8966 | 0.8966 | +0.0000 |
| Narrative unit accuracy | 0.6432 | 0.6467 | +0.0035 |
| Scene-boundary micro-F1 | 0.1820 | 0.2711 | +0.0891 |

The candidate improved two tasks over base, but the stricter training-gain gate
failed: freshly regenerated V9 reached `0.2727` scene micro-F1, so checkpoint 16
was lower by `0.0016` instead of exceeding V9 by the required `0.005`. The
contrast-trained checkpoint therefore does not replace V9. The accepted V9
publisher-validation result at the top of this README remains authoritative.
No publisher test, Hard32, or unused strength holdout evaluation is authorized.

Evidence:

- [Locked hybrid protocol](experiments/rethinking_rwkv_ms_gemma/natural_memory_native_multitask_hybrid_protocol_v1.json)
- [Signed hybrid result](experiments/rethinking_rwkv_ms_gemma/local_artifacts/natural_memory_native_multitask_hybrid_v1.json)
- [Hash-bound hybrid analyzer](experiments/rethinking_rwkv_ms_gemma/analyze_natural_memory_native_multitask_hybrid.py)
- [Locked fresh-validation protocol](experiments/rethinking_rwkv_ms_gemma/natural_memory_native_hybrid_publisher_validation_protocol_v1.json)
- [Signed fresh-validation failure](experiments/rethinking_rwkv_ms_gemma/local_artifacts/natural_memory_native_hybrid_publisher_validation_v1/result.json)
- [Fresh-validation runner](experiments/rethinking_rwkv_ms_gemma/run_natural_memory_native_hybrid_publisher_validation.py)
  and [analyzer](experiments/rethinking_rwkv_ms_gemma/analyze_natural_memory_native_hybrid_publisher_validation.py)

### Cross-Fitted Scene Router

A subsequent TRAIN-only study tested a different method without reopening any
protected split. Eight fixed V9/checkpoint-16 set-combination rules were
selected independently inside five hash folds and scored only on each held-out
fold. Four folds selected `v9_if_subset_else_checkpoint`; one selected raw
checkpoint 16.

The cross-fitted router reached `0.3191` scene micro-F1, above frozen V9
(`0.2915`) but just below checkpoint 16 (`0.3197`) by `0.0006`. It removed four
false positives but also lost one true positive. The preregistered gate required
a `+0.005` gain over both inputs, so this method failed and no external
replication is authorized. Simple set routing is therefore not the next
boundary; further work must improve training robustness or expose calibrated
token-level confidence while remaining publisher-TRAIN-derived.

Evidence:

- [Locked cross-fit protocol](experiments/rethinking_rwkv_ms_gemma/natural_memory_native_scene_crossfit_router_protocol_v1.json)
- [Signed cross-fit failure](experiments/rethinking_rwkv_ms_gemma/local_artifacts/natural_memory_native_scene_crossfit_router_v1/result.json)
- [Cross-fit analyzer](experiments/rethinking_rwkv_ms_gemma/analyze_natural_memory_native_scene_crossfit_router.py)

### Checkpoint-Soup Failure

A second publisher-TRAIN-only study tested weight-space averaging instead of
another output router. Seven convex recipes mixed only the 126 learned content-
gate tensors from frozen V9 and contrast checkpoints 8, 16, and 32. The recipe
bytes were signed and pushed before any new generation. Five hash folds then
selected among those recipes with unchanged checkpoint 16 available as a
fallback, using only the other four folds for each held-out decision.

The best single recipe, `trajectory_centered` (`25%` step 8, `50%` step 16,
`25%` step 32), reached `0.3175` scene micro-F1 on the 220 post-probe fit rows.
That is only `+0.0021` over checkpoint 16 (`0.3154`), below the locked `+0.005`
requirement. Fold selection was also unstable: the five winners were
`trajectory_centered`, `s16_75_s32_25`, checkpoint 16,
`trajectory_centered`, and `v9_25_s16_75`. Their combined out-of-fold score was
`0.3012`, or `-0.0142` versus checkpoint 16, despite remaining `+0.0108` above
V9 (`0.2904`). The method therefore failed and authorizes no external
replication. Neither prediction-set routing nor convex checkpoint averaging is
the next boundary; the next study must change training robustness itself.

Evidence:

- [Locked checkpoint-soup protocol](experiments/rethinking_rwkv_ms_gemma/natural_memory_native_scene_checkpoint_soup_protocol_v1.json)
- [Signed candidate materialization](experiments/rethinking_rwkv_ms_gemma/local_artifacts/natural_memory_native_scene_checkpoint_soup_materialization_v1/result.json)
- [Signed checkpoint-soup failure](experiments/rethinking_rwkv_ms_gemma/local_artifacts/natural_memory_native_scene_checkpoint_soup_v1/result.json)
- [Four-GPU runner](experiments/rethinking_rwkv_ms_gemma/run_natural_memory_native_scene_checkpoint_soup.py)
  and [cross-fit analyzer](experiments/rethinking_rwkv_ms_gemma/analyze_natural_memory_native_scene_checkpoint_soup.py)

### Independent-Seed Robust Training Failure

The next publisher-TRAIN-only study changed training rather than selecting
among outputs. Three independently hashed 256-row schedules trained the same
126 content-gate tensors on four A100 GPUs with global batch 16, half the
original learning rate, and an explicit `0.995` post-step pull toward V9. All
three runs were numerically clean and tightly matched in endpoint delta norm
(`0.1001`, `0.1039`, and `0.0973`). Their pairwise delta cosines were
`0.611`-`0.659`. The only preregistered candidate was the equal mean of the
three signed V9-relative deltas, fixed and pushed before generation.

That candidate reached `0.3059` scene micro-F1 on the same 220 open fit rows.
It remained above V9 (`0.2904`) by `+0.0155`, but fell below checkpoint 16
(`0.3154`) by `-0.0094`. Relative to checkpoint 16 it gained one true positive
but added 21 false positives, so the locked `+0.005` gate failed. The averaged
delta was only `0.0873` from V9 versus `0.1847` for checkpoint 16 and had
cosine `0.566` with checkpoint 16's delta. The lower-rate V9-centered ensemble
therefore stabilized an underpowered direction rather than preserving the
single checkpoint's precision. It authorizes no external replication. A next
training study should anchor small independent residual updates at checkpoint
16 instead of pulling every run back toward V9.

Evidence:

- [Locked seed-ensemble protocol](experiments/rethinking_rwkv_ms_gemma/natural_memory_native_scene_seed_ensemble_protocol_v1.json)
- [Signed seed-delta materialization](experiments/rethinking_rwkv_ms_gemma/local_artifacts/natural_memory_native_scene_seed_ensemble_materialization_v1/result.json)
- [Signed TRAIN-only failure](experiments/rethinking_rwkv_ms_gemma/local_artifacts/natural_memory_native_scene_seed_ensemble_v1/result.json)
- [Four-GPU trainer](experiments/rethinking_rwkv_ms_gemma/run_natural_memory_native_scene_seed_ensemble.py),
  [evaluation runner](experiments/rethinking_rwkv_ms_gemma/run_natural_memory_native_scene_seed_ensemble_eval.py),
  and [analyzer](experiments/rethinking_rwkv_ms_gemma/analyze_natural_memory_native_scene_seed_ensemble.py)

### Checkpoint-16 Residual Training Failure

A follow-up publisher-TRAIN-only study preserved checkpoint 16 as the anchor
instead of restarting from V9. Three four-GPU runs used disjoint sets of 128
previously unused rows, eight global-batch-16 updates, learning rate `2.5e-5`,
and a `0.995` post-step pull toward checkpoint 16. The runs were numerically
clean, changed no frozen parameters, and ended only `0.0251`, `0.0269`, and
`0.0258` from the anchor. Their residual directions were weakly aligned,
however: pairwise cosines were `0.049`-`0.189`. The locked equal residual mean
therefore had norm `0.0169` and was fixed and pushed before generation.

The candidate reached `0.3129` scene micro-F1 on the same 220 open fit rows.
It preserved checkpoint 16's 79 true positives and 161 false negatives, but
added four false positives, reducing micro-F1 by `-0.0025` from checkpoint 16
(`0.3154`). It still remained `+0.0225` above V9 (`0.2904`) and changed `6.36%`
of checkpoint-16 outputs, but failed the preregistered `+0.005` improvement
gate. It authorizes no external replication, and publisher validation, test,
Hard32, and the unused 73-row holdout remain unopened. Repeated endpoint
averaging is no longer the useful boundary: the next training intervention
must directly suppress false-positive scene labels while preserving
checkpoint 16's true positives.

Evidence:

- [Locked checkpoint-16 residual protocol](experiments/rethinking_rwkv_ms_gemma/natural_memory_native_scene_c16_residual_protocol_v1.json)
- [Signed residual materialization](experiments/rethinking_rwkv_ms_gemma/local_artifacts/natural_memory_native_scene_c16_residual_materialization_v1/result.json)
- [Signed TRAIN-only failure](experiments/rethinking_rwkv_ms_gemma/local_artifacts/natural_memory_native_scene_c16_residual_v1/result.json)
- [Four-GPU residual trainer](experiments/rethinking_rwkv_ms_gemma/run_natural_memory_native_scene_c16_residual.py),
  [evaluation runner](experiments/rethinking_rwkv_ms_gemma/run_natural_memory_native_scene_c16_residual_eval.py),
  and [analyzer](experiments/rethinking_rwkv_ms_gemma/analyze_natural_memory_native_scene_c16_residual.py)

### Precision-Unlikelihood Training Failure

The next publisher-TRAIN-only intervention started from checkpoint 16 and used
256 previously untouched eligible rows. One four-A100 run made 16 global-batch-
16 updates at learning rate `1.5e-5`, with a `0.995` post-step pull toward the
starting checkpoint. Each row combined unit-weight gold teacher-forced CE with
weight-`0.5` unlikelihood on only the decimal token or tokens of one inserted
false boundary under the same correct online state. JSON syntax was never a
negative target. Of the 256 negatives, 199 inserted boundary `1`; the loss
penalized 270 false-boundary digit tokens in total. The run changed only the
126 content-gate tensors, ended `0.0499` L2 from checkpoint 16, and was fixed
and pushed before generation.

| TRAIN-derived scene candidate | TP | FP | FN | Micro-F1 |
| --- | ---: | ---: | ---: | ---: |
| Precision-unlikelihood endpoint | 76 | 202 | 164 | 0.2934 |
| Checkpoint 16 | 79 | 182 | 161 | 0.3154 |
| Frozen V9 | 80 | 231 | 160 | 0.2904 |

The candidate lost `0.0219` micro-F1 from checkpoint 16 and missed the locked
gain over V9 by `0.0019`. It changed 23 of 220 outputs: nine additions, seven
removals, and seven substitutions. Those changes added 30 false boundaries
while removing only 10, lost four true boundaries while adding one, and
included one unstable row that added all 15 boundaries from `2` through `16`.
Boundary `1` itself was not calibrated: five false instances were removed but
five new false instances appeared, while two true instances were lost and one
was added. Teacher-forced digit-only unlikelihood therefore did not transfer
to greedy-set precision; combined gold CE also failed to preserve the local
checkpoint-16 decision surface. The endpoint is archived without external
replication. Publisher validation, publisher test, Hard32, and the unused
73-row strength holdout remain unopened.

Evidence:

- [Locked precision-unlikelihood protocol](experiments/rethinking_rwkv_ms_gemma/natural_memory_native_scene_precision_unlikelihood_protocol_v1.json)
- [Signed training endpoint](experiments/rethinking_rwkv_ms_gemma/local_artifacts/natural_memory_native_scene_precision_unlikelihood_train_v1/result.json)
- [Signed materialization](experiments/rethinking_rwkv_ms_gemma/local_artifacts/natural_memory_native_scene_precision_unlikelihood_materialization_v1/result.json)
- [Signed TRAIN-only failure](experiments/rethinking_rwkv_ms_gemma/local_artifacts/natural_memory_native_scene_precision_unlikelihood_v1/result.json)
- [Four-GPU trainer](experiments/rethinking_rwkv_ms_gemma/run_natural_memory_native_scene_precision_unlikelihood.py),
  [evaluation runner](experiments/rethinking_rwkv_ms_gemma/run_natural_memory_native_scene_precision_unlikelihood_eval.py),
  and [analyzer](experiments/rethinking_rwkv_ms_gemma/analyze_natural_memory_native_scene_precision_unlikelihood.py)

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

## Mechanism-Level Result

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

The top-level DLA comparison is a training-free mechanism reproduction. It does
not reproduce 50B-token pretraining or trained HRM-Text checkpoints. The
separate Gemma4 + RWKV-MS experiments documented above and under
`experiments/rethinking_rwkv_ms_gemma/` do include trained adapters and native
downstream evaluation.

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
deltamem/                          # bundled patched HF online-memory runtime
integrations/delta_mem_rwkv_ms/    # launchers, docs, GGUF tools, optional upstream patch
integrations/delta_mem_rwkv_ms/gguf/ # GGUF sidecar, fixture, and parity helpers
```

## HOLA Hippocampus on RWKV-7 Multi-State

`experiments/hola_hippocampus/` replaces the neocortex of HOLA (arXiv 2607.02303,
semiparametric memory = compressive state + bounded exact-KV cache) with this repo's
read-before-write RWKV-7 multi-state and re-tests HOLA's design claims on the
state-only ablation grid. The mapping is exact: for unit keys the RWKV-7 correction
term makes the update a delta rule, so HOLA's surprise score beta*||e|| becomes the
write magnitude `m_t = ||Delta_t||_F` already computed by the recurrence.

Result summary (5 seeds; full tables in `experiments/hola_hippocampus/REPORT.md`):

- The weakest state-only cell (16 needles, `low_k_dla`, 0.669 above) rises to
  **0.880** with a 16-slot surprise cache; a matched recency cache stays at 0.665.
- HOLA's two claims reproduce on RWKV-7: recency caching is dead weight for far
  needles, and a flat softmax read (0.83*cos) equals no cache at all.
- One correction was required: raw surprise admission fails with an untrained
  constant gate; an online CLS-style consolidation rule (demote cache entries whose
  key the state later predicts well) plus a read-confidence gate makes the cache
  strictly non-harmful. Hypothesis ledger and run provenance live in `.keel/`.

Run: `.venv/bin/python experiments/hola_hippocampus/hola_rwkv_ms.py`

## Delta-Mem RWKV-MS Online Memory

The practical RWKV-MS online-memory integration is self-contained in this
repository. The patched Python runtime is bundled at top-level `deltamem/`, so
normal Qwen/Gemma HF training and inference do not require another delta-Mem
checkout. `integrations/delta_mem_rwkv_ms/` contains HF inference and verified
manual training-smoke entry points, a matched delta-rule/RWKV-MS launcher, GGUF
tools, and an optional upstream patch export. The runtime supports Qwen3,
Qwen3.5/Qwen3.6, SmolLM3, and Gemma4 text attention;
for `google/gemma-4-E4B-it` it wraps the non-KV-shared attention layers and
skips the KV-shared tail layers.

Fresh RWKV-MS configs use semantics v2: FP32 recurrent matrices, bounded
per-head write sources, RWKV decay without a second lambda decay, cosine slot
routing, and a bias-free empty-state readout. Checkpoints without an explicit
`rwkv_ms_semantics_version` load as legacy v1 and must not be resumed as v2;
start a fresh run with `--rwkv-ms-semantics-version 2`.

Transformers exposes Qwen3.6 as `qwen3_5`. Its 64-layer hybrid stack has 16
full-attention layers at physical indices `3,7,11,...,63`; the other 48 Gated
DeltaNet layers are not wrapped. Use layer `3` for a smoke run or
`3,7,11,15,19,23` for the six early eligible layers. This Qwen path is the HF
integration and is separate from the Gemma-only GGUF sidecar runtime.

The bundled `deltamem/` package provides the wrapper/session machinery:
attaching online-memory modules to a Transformers model, loading
`delta_mem_adapter.pt`, keeping RWKV-MS state synchronized with the KV cache,
and applying the chat template. The optional
`integrations/delta_mem_rwkv_ms/delta_mem_rwkv_ms.patch` exports these changes
for upstream delta-Mem revision `5cd5d9153c7f408764728d953565201e198c39e2`;
it is not needed for normal use of this repository.
See [bundled runtime provenance](integrations/delta_mem_rwkv_ms/BUNDLED_RUNTIME.md)
for the source snapshot and local integration revision.

For HF workflows, install the bundled package from the repository root before
running the commands below:

```bash
pip install -r requirements.txt
pip install -e .
```

### Gemma tau2 status

The active Gemma + RWKV-MS tau2 recipe is documented in
`GEMMA_RWKV_MS_TAU2_TRAINING_PLAN_V2.md`. For reproducibility, the benchmark
artifacts record this historical source integration commit (it is not a current
external runtime dependency):

```text
bec8330 Add RWKV-MS memory backend for Gemma tau2
```

Current best learned no-rule online-memory checkpoint:

```text
xiaol/gemma-4-e4B-hybrid-rnn-mem-rwkv-fable5-gpt5.5-v1
```

Local source checkpoint:

```text
/run/media/xiaol/B214449214445C0B/delta_mem_outputs/gemma_rwkv_ms_tau2/v2ruleplanner_mobile_focusedtools_turns_formatrefresh_continue200_len192_layers0_5_qo_r8/checkpoints/step-100
```

The table below keeps learned online-memory runs separate from rule-assisted diagnostic
runs. "No-rule" means no eval-time `--mobile-data-rule-planner` and no parser
format-repair patch.

Release framing: "One can have both the fish and the bear's paw." The base
Gemma checkpoint remains frozen to preserve original behavior, while the learned
RWKV-MS path adds a small recurrent memory surface that can be adapted to local
domain data.

| Run / condition | Layers / rank / length | pass^1 | Takeaway |
| --- | --- | ---: | --- |
| Base checkpoint `google/gemma-4-E4B-it`, focused tools + line verify + autostop | none | 4/20 (0.20) | Current base-only baseline for the accepted setup |
| Base checkpoint `google/gemma-4-E4B-it`, checklist prompt | none | 7/20 (0.35) | Prompt-only baseline, still below learned best |
| Original 82-row Phase 1 | `0,1` / r8 / len256 | 1/20 (0.05) | Dataset/format mismatch; reject |
| Generated action SFT | `0,1` / r8 / len256 | 9/20 (0.45) | 2 layers help but are not enough |
| Generated action SFT | `0-5` / r8 / len256 | 10/20 (0.50) | Shallow 6-layer band is better |
| Generated action SFT | all eligible / r4 / len256 | 1/20 (0.05) | All-layer memory path over-perturbs |
| Format-refresh continuation, final | `0-5` / r8 / len192 | 12/20 (0.60) | Good final checkpoint |
| Format-refresh continuation, `step-100` | `0-5` / r8 / len192 | **14/20 (0.70)** | Best learned no-rule checkpoint |

Memory-path size from saved checkpoints:

| Memory-path shape | Trainable memory params |
| --- | ---: |
| 2 layers, r8 `q,o` | 257,744 |
| 6 layers, r8 `q,o` | 797,808 |
| 24 eligible layers, r4 `q,o` | 1,594,080 |

Local training-cost notes:

- Experiments were local CUDA bf16 runs on an RTX 4090 24 GB setup.
- Generated mobile-data action SFT used 3,519 turn rows for 656 optimizer steps.
- Format-refresh continuation used 5,027 turn rows for 200 optimizer steps.
- Exact wall time and VRAM vary with local hardware, sequence length, layer
  count, rank, cache location, and fragmentation; adapt the frozen-base
  online-memory recipe to your own data.

Status interpretation:

- The original tau2 data was the problem: the 82-row run trained for 656
  optimizer steps and its loss moved, but the benchmark collapsed to 1/20.
- Generated mobile-data action SFT transfers better, and the 6-layer shallow
  online-memory path is the current useful capacity point.
- The 200-step format-refresh continuation overtrains relative to its
  `step-100` checkpoint, so checkpoint selection matters.
- The eval-time rule planner / float-format fix is excluded from the comparison
  table because it is benchmark-specific control logic, not model behavior.
- The next benchmark should run the `step-100` checkpoint on at least 50 tasks,
  preferably the full telecom split, before treating 14/20 as robust.

Recommended HF online-memory inference command:

```bash
python integrations/delta_mem_rwkv_ms/inference.py \
  --memory-repo xiaol/gemma-4-e4B-hybrid-rnn-mem-rwkv-fable5-gpt5.5-v1 \
  --base-model google/gemma-4-E4B-it \
  --device cuda:0 \
  --dtype bfloat16 \
  --attn-implementation sdpa
```

## Gemma4 GGUF First Step

A base Gemma4 E4B GGUF has been downloaded for llama.cpp testing on the 2 TB SSD:

```text
/run/media/xiaol/B214449214445C0B/models/gguf/gemma-4-E4B-it/gemma-4-E4B-it-Q8_0.gguf
/run/media/xiaol/B214449214445C0B/models/gguf/gemma-4-E4B-it/mmproj-gemma-4-E4B-it-Q8_0.gguf
/run/media/xiaol/B214449214445C0B/models/gguf/gemma-4-E4B-it/gemma-4-E4B-it-rwkv-ms-memory.gguf
```

The first two files are normal base-model inference artifacts. The RWKV-MS
memory file is a GGUF sidecar containing the adapter tensors and metadata. The
local llama.cpp branch can now consume that sidecar in an experimental Gemma4
runtime path: model load owns the sidecar tensors in CPU buffers, the Gemma4
graph applies RWKV-MS `q,o` deltas on target layers `0-5`, and a mutable
RWKV-MS state buffer is updated during prompt/generation scans. The current
runtime is intentionally constrained to one sequence. The server/UI path keeps
physical microbatches serial (`-ub 1`) for the best-tested state behavior; the
CLI graph can build experimental graph-unrolled multi-token prompt scans, but
that path still needs stronger state-level parity coverage before it should be
treated as production-ready. See
`GGUF_EXTERNAL_MEMORY_FEASIBILITY.md` and
`integrations/delta_mem_rwkv_ms/GGUF_PORT_PLAN.md`.
At llama.cpp model load time, the sidecar path now performs semantic validation
before runtime use: it verifies `delta_mem.base_gguf_sha256` against the exact
loaded base GGUF file, then rejects unsupported `num_state_heads != 1`,
duplicate compact tensor names, missing required tensors, and wrong ggml-order
tensor shapes.

Patched llama.cpp fork:

```text
https://github.com/xiaol/llama.cpp-online-memory
branch: main
commit: 85da0c63b Add Gemma4 RWKV-MS GGUF sidecar runtime
base upstream: ggml-org/llama.cpp 1ec44d1
```

Current sidecar identity:

```text
sha256: 0c646a776b5b12c9d3657ffd2e5e581be1eb46e858f1f404afeaa7077c02974e
bound base GGUF sha256: fb8f0c032de00b18c710824af3c7e5777c71e5fb60b13f13575f0a9e92ddecd0
size: 1,663,840 bytes
tensor name format: compact_with_source_name_manifest
tensors: 186 BF16
```

Start a recent llama.cpp server:

```bash
LLAMA_SERVER_BIN=/run/media/xiaol/B214449214445C0B/tools/llama.cpp/build-cuda/bin/llama-server \
LLAMA_REASONING=off \
bash tools/llama_server_gemma4.sh
```

For the experimental RWKV-MS sidecar runtime through `llama-server`, use the
patched llama.cpp build and the constrained sidecar mode:

```bash
mkdir -p .openresearch/artifacts/gguf_ui
LLAMA_SERVER_BIN=/run/media/xiaol/B214449214445C0B/tools/llama.cpp/build-cuda/bin/llama-server \
LLAMA_PORT=18083 \
LLAMA_RWKV_MS=1 \
LLAMA_REASONING=off \
bash tools/llama_server_gemma4.sh 2>&1 | tee .openresearch/artifacts/gguf_ui/llama_server_rwkv_ms.log
```

The helper sets `--rwkv-ms-sidecar`, `--batch-size 2`, `--ubatch-size 1`,
`--parallel 1`, disables continuous batching/context shift/prompt-cache reuse,
disables server prompt-cache RAM and context checkpoints, enables a slot-save
directory for manual slot 0 save/restore, and uses text-only mode for the
current one-sequence runtime.
The patched llama.cpp context rejects sidecar runs with more than one sequence.
The server/helper keep `--ubatch-size 1`, reject speculative decoding, and
preflight unsafe slot/cache/batch overrides before starting `llama-server`;
model load also rejects malformed or unsupported sidecars before any RWKV-MS
graph consumes their tensors. A sidecar exported for a different base GGUF now
fails model load with a hash mismatch instead of running against the wrong
weights.

For the best-tested experimental runtime path, pass the sidecar and use serial
physical microbatches:

```bash
/run/media/xiaol/B214449214445C0B/tools/llama.cpp/build-cuda/bin/llama-completion \
  -m /run/media/xiaol/B214449214445C0B/models/gguf/gemma-4-E4B-it/gemma-4-E4B-it-Q8_0.gguf \
  --rwkv-ms-sidecar /run/media/xiaol/B214449214445C0B/models/gguf/gemma-4-E4B-it/gemma-4-E4B-it-rwkv-ms-memory.gguf \
  -p "Hi" -n 32 -c 96 -b 2 -ub 1 -ngl 99 --no-warmup --no-display-prompt \
  --no-perf -no-cnv -s 123 --temp 0 --top-k 1
```

With the same seed and greedy sampling, the base and sidecar paths now diverge.
The sidecar path produced `! I'm excited to chat with you. What's on your mind
today? ...`, while the base path continued `! I'm excited to chat with you. I'm
here to help ...`. Treat this as a smoke signal consistent with the sidecar path;
confirm runtime use with server logs and the reference-trace health check.

The local CUDA build is from the online-memory fork commit `85da0c63b`, based
on upstream llama.cpp `1ec44d1`, and detects the RTX 4090 as `CUDA0`. CUDA 13.1
plus GCC 15 needed a local header shim during build; the resulting binary is
under the SSD tool directory above.

Then launch the local testing UI:

```bash
python3.12 -m venv .venv-ui
.venv-ui/bin/pip install -r requirements-ui.txt
LLAMA_BASE_URL=http://127.0.0.1:18083/v1 \
LLAMA_RWKV_MS=1 \
LLAMA_MODEL=gemma-4-e4b-it-rwkv-ms-q8 \
GGUF_RWKV_MS_SIDECAR_PATH=/run/media/xiaol/B214449214445C0B/models/gguf/gemma-4-E4B-it/gemma-4-E4B-it-rwkv-ms-memory.gguf \
LLAMA_SERVER_LOG=.openresearch/artifacts/gguf_ui/llama_server_rwkv_ms.log \
GGUF_RWKV_MS_HEALTH_OUTPUT=.openresearch/artifacts/gguf_ui/rwkv_ms_runtime_health.json \
GGUF_UI_REQUIRE_RWKV_MS_HEALTH=1 \
GGUF_UI_PORT=7861 \
.venv-ui/bin/python tools/gemma_gguf_ui.py
```

Before comparing prompts, verify that the endpoint is really the patched
sidecar runtime. The UI exposes the same check through its RWKV-MS runtime
button, writes the health file, and blocks sidecar chat/trace comparison while
the selected endpoint/model/sidecar/log do not match a recent successful check.

```bash
.venv-ui/bin/python tools/check_rwkv_ms_gguf_runtime.py \
  --base-url http://127.0.0.1:18083/v1 \
  --server-log .openresearch/artifacts/gguf_ui/llama_server_rwkv_ms.log \
  --output .openresearch/artifacts/gguf_ui/rwkv_ms_runtime_health.json
```

The check requires the server log because API output alone cannot prove that
llama.cpp loaded the RWKV-MS sidecar. It verifies model listing, a chat smoke
request, the saved reference trace, slot 0 save/restore with exact-prefix
continuation, corrupted slot restore rejection, and log evidence for RWKV-MS
activation, one server slot, disabled prompt cache, disabled context
checkpoints, and exact-prefix slot reuse. The sidecar server also rejects
speculative decoding options.

For repeatable prompt checks against the same server:

```bash
.venv-ui/bin/python tools/eval_gguf_prompts.py configs/gguf_rwkv_ms_prompt_suite.jsonl \
  --base-url http://127.0.0.1:18083/v1 \
  --model gemma-4-e4b-it-rwkv-ms-q8 \
  --rwkv-ms \
  --temperature 0 \
  --seed 42
```

For the RWKV-MS side of the future port, inspect the PyTorch memory checkpoint
into a tensor/config manifest:

```bash
.venv/bin/python integrations/delta_mem_rwkv_ms/gguf/inspect_memory_checkpoint.py \
  --memory-dir /run/media/xiaol/B214449214445C0B/models/delta_mem/gemma-4-e4B-hybrid-rnn-mem-rwkv-fable5-gpt5.5-v1 \
  --output .openresearch/artifacts/gguf_memory_manifest.json
```

To regenerate and validate the GGUF memory sidecar:

```bash
.venv/bin/python integrations/delta_mem_rwkv_ms/gguf/export_memory_gguf.py \
  --manifest-output .openresearch/artifacts/rwkv_ms_memory_sidecar_manifest.json
.venv/bin/python integrations/delta_mem_rwkv_ms/gguf/inspect_memory_gguf.py \
  --memory-dir /run/media/xiaol/B214449214445C0B/models/delta_mem/gemma-4-e4B-hybrid-rnn-mem-rwkv-fable5-gpt5.5-v1
.venv/bin/python integrations/delta_mem_rwkv_ms/gguf/materialize_memory_gguf.py --force
.venv/bin/python integrations/delta_mem_rwkv_ms/gguf/compare_memory_checkpoints.py
```

To generate and validate the isolated RWKV-MS math fixture from the
sidecar-rebuilt checkpoint:

```bash
.venv/bin/python integrations/delta_mem_rwkv_ms/gguf/generate_rwkv_ms_math_fixture.py \
  --output .openresearch/artifacts/rwkv_ms_math_fixture.json
.venv/bin/python integrations/delta_mem_rwkv_ms/gguf/validate_rwkv_ms_math_fixture.py \
  --fixture .openresearch/artifacts/rwkv_ms_math_fixture.json \
  --json
```

The current fixture uses real layer-0 adapter tensors, covers projection,
read-before-write state update, readout, and active `q,o` delta heads, and
validates with `max_abs_diff: 0.0`. It is a PyTorch golden math fixture for a
future GGML port, not stock llama.cpp memory execution.

The local llama.cpp checkout has an isolated C++ fixture for the compact
sidecar:

```bash
/run/media/xiaol/B214449214445C0B/tools/cmake-4.3.3/bin/cmake \
  --build /run/media/xiaol/B214449214445C0B/tools/llama.cpp/build-cuda \
  --target test-rwkv-ms-fixture -j 8

/run/media/xiaol/B214449214445C0B/tools/llama.cpp/build-cuda/bin/test-rwkv-ms-fixture \
  .openresearch/artifacts/rwkv_ms_math_fixture.json \
  /run/media/xiaol/B214449214445C0B/models/gguf/gemma-4-E4B-it/gemma-4-E4B-it-rwkv-ms-memory.gguf \
  1e-5 1e-5
```

Current strict sidecar result: `{"ok":true,"compared":51,"sidecar":true,"max_abs_diff":1.37090683e-06}`.
The no-sidecar run also passes with `compared=11` and `max_abs_diff=5.96046448e-08`.
This covers `tests/test-rwkv-ms-fixture.cpp` in llama.cpp parsing the compact
sidecar, computing memory projections, `HRMRWKV7LowRankCore` feature
projections, driving a second C++ read-before-write scan from those
sidecar/GGML tensors, graph readout from the scan `raw_reads` plus graph
`feature_g`, and `delta_q`/`delta_o` from the graph-produced readout. This
fixture remains the isolated math parity check; the separate `llama-completion`
smoke above is the Gemma4 generation runtime check.

The local llama.cpp checkout also has `tests/test-rwkv-ms-state.cpp` for the
RWKV-MS recurrent state payload. It checks v2 state metadata, deterministic
sidecar fingerprint validation, staged sidecar-local restore, and rejection for
metadata/fingerprint/length mismatches. The fingerprint now includes the bound
base GGUF hash, so slot files created before that binding should be regenerated.
Full and sequence state restore now snapshot the current context before
RWKV-MS-enabled loads and roll back that snapshot if the normal memory portion
loads but the RWKV-MS sub-state fails. Failed server slot restore still clears
the affected slot/context state after the library rollback and returns the
exact state-load error.
Context-owned memory mutation now uses llama.cpp `llama_context_memory_*`
wrappers in the patched paths: clear and supported full-sequence removal keep
RWKV-MS state synchronized, while unsupported sequence copy, keep, shift, and
division fail explicitly under RWKV-MS instead of mutating only KV cache.

To generate the first PyTorch golden trace from the sidecar-rebuilt checkpoint:

```bash
.venv/bin/python \
  integrations/delta_mem_rwkv_ms/gguf/generate_reference_trace.py \
  --max-new-tokens 64 \
  --output .openresearch/artifacts/gguf_reference_trace_from_sidecar_64.json \
  --save-snapshot-dir .openresearch/artifacts/gguf_reference_snapshot_from_sidecar_64
```

To compare the running GGUF backend against that reference trace:

```bash
LLAMA_RWKV_MS=1 \
LLAMA_MODEL=gemma-4-e4b-it-rwkv-ms-q8 \
.venv-ui/bin/python tools/compare_gguf_to_reference_trace.py \
  --output .openresearch/artifacts/gguf_ui/trace_compare_reasoning_off.jsonl
```

With `LLAMA_REASONING=off`, the comparison harness can log either base-GGUF or
RWKV-MS-sidecar runs. Stock llama.cpp still does not execute RWKV-MS memory; the
sidecar mode requires the local patched branch.

## Acknowledgement

This work builds on the Log-Linear Attention repository and uses local
HRM-Text/RWKV memory ideas as mechanism baselines. The added experiments are
intended for controlled research exploration, not as a trained-model benchmark.
