# Multi-State RWKV Online Memory: V1-V16 Progression Report

Snapshot: 2026-07-27 18:57 CST

## Executive conclusion

V16 is the strongest version on the narrow in-sample causal-memory probe, but it
is not a successful model version yet. On its 32 training pairs it improves the
exact-pair first-distinguishing-token gap to `+0.07031`, while its completed
attribution validation result is only `16/30`, versus `25/30` for the Gemma base
and `19/30` for V15. Training is complete; this report does not wait for the
still-running full V16 dataset benchmark.

The project made three real advances:

1. Training loss fell from a final-epoch mean of `6.864` on the V2 loss probe to
   `1.626` in V6 after switching to the content-control probe, semantics v2, and
   24 layers. These are training-set results on two different 32-row probes.
2. The trainable memory path expanded from 6 layers to all 42 layers under the
   available VRAM budget.
3. V16 produces a measurable causal effect at the first history-dependent
   answer token on the same 32 pairs used for training.

It has not yet produced the result that matters: positive performance on a
disjoint memory-specific validation set plus a positive structured-task
validation delta over the same base model.

## Comparison rules

- V1 used 41,266 novel-writing episodes. V2-V3 used
  `novel_memory_loss_probe_seed20260724_n32.jsonl`; V4-V16 used the distinct
  `novel_memory_content_control_probe_seed20260724_n32.jsonl`. Their losses are
  training-set measurements, not measures of disjoint generalization.
- V2-V9 are primarily CE/training-mechanics experiments. Their task CE is
  reasonably comparable when the data and objective match.
- V10-V16 add contrast and representation terms. Total loss after V10 is not
  directly comparable with CE-only V2-V9; correct-history CE is listed
  separately.
- V10-V15 are independent 32-step objective forks from V8 checkpoint 384, not
  a sequential V10 -> V11 -> ... -> V15 training chain.
- V16 is a fresh 32-step optimizer run warm-started from V14 checkpoint 416. It
  imports V14 adapter weights, adds 42 residual-gain parameters, and does not
  import optimizer, scheduler, trainer, or RNG state.
- Dataset scores below use conservative schema recovery because Gemma emits
  semantically valid aliases such as `character` instead of
  `best_candidate`. Strict official-schema accuracy is therefore misleadingly
  zero for some tasks.
- V1-V8 describe conceptual experiment progression, not cross-version
  checkpoint continuation. Each was a fresh version family; continuation
  checkpoints stayed within that version.

## Checkpoint lineage

```text
Fresh version families: V1, V2, V3, V4, V5, V6, V7, V8

V8 checkpoint 128
`-- V9: post-attention-norm placement, 32 new steps

V8 checkpoint 384
|-- V10: contrast v1, 32 new steps
|-- V11: contrast v2, 32 new steps
|-- V12: contrast v3, 32 new steps
|-- V13: contrast v4, 32 new steps
|-- V14: contrast v5, 32 new steps
|   `-- V16: adapter-only warm start from V14-416, fresh optimizer, 32 steps
`-- V15: contrast v6 W8, 32 new steps
```

## Version progression

| Version | Main change | Layers / data / updates | Verified result | Interpretation |
|---|---|---:|---:|---|
| V1 | Original rank-8 `q,o` RWKV-MS training, legacy recurrent semantics | 6 layers; 41,266 rows; 1 epoch | Mean loss `3.6310`; final `3.6972`; minimum `3.1448` | This is the earlier six-layer approximately `3.6` result. Recurrent states grew extremely large and the structured-task transfer benchmark regressed. |
| V2 | Fixed training plumbing; 6-layer loss probe | 6; 32 rows; 128 updates | Final-epoch mean logged loss `6.864` | The memory path was not learning the target well. |
| V3 | Canonical full-context teacher protocol; KL-isolation experiments | 6; 32; 128 | KL0 final-epoch mean CE `5.551` | Corrected objective accounting, but improvement was still small. |
| V4 | Content-control probe; no read/write dropout, no KL | 6; 32; 128 | Final-epoch mean CE `4.861` | Confirmed that the corrected CE path could move. |
| V5 | Expanded from 6 to 24 eligible layers; rank 4 | 24; 32; 128 | Final-epoch mean CE `3.686`; final `3.578` | Layer coverage materially improved fitting. Its numerical similarity to V1 is coincidental: V5 used 24 layers and only 32 content-control rows. |
| V6 | RWKV-MS semantics v2 with FP32 recurrent matrices; extended CE training | 24; 32; 672 (21 passes) | Final-epoch mean CE `1.626`; target `1.7` reached | Strongest pure training-loss result, but it is in-sample memorization on 32 rows, not proof of memory specificity or transfer. |
| V7 | `o`-head content-gated addition | 24; 32; checkpoint 432 (partial run) | Last 32-update mean CE `2.509` | Gating/fusion changed optimization; no validation benefit was established. |
| V8 | All 42 layers, rank 4, frozen-MLP activation checkpointing | 42; 32; 384 | Final 32-update mean CE `2.047` | Solved the VRAM/layer-coverage problem, but CE alone still did not prove that the state content controlled the answer. |
| V9 | Move fusion to post-attention norm | 42; 32; checkpoint 160 (32 new) | New-step mean CE `4.070` | Clear loss regression; placement was not a direct improvement. |
| V10 | Content contrast v1: correct versus shuffled history | 42; 32 new updates from V8-384 | Total loss `2.130`; correct CE `1.802` | No-write CE was `5.588`, but shuffled-versus-correct CE differed by only `0.00268`: state presence mattered much more than state identity. |
| V11 | Exact sequential contrast / negative priming | 42; 32 from V8-384 | Total `2.123`; correct CE `1.794` | Fixed execution fidelity, but did not create a strong identity-dependent gap. |
| V12 | Previous-source gradient plus valid-context mask | 42; 32 from V8-384 | Total `4.440`; correct CE `4.111` | Regressed sharply under this formulation. |
| V13 | Causal read mask including supervised predictors | 42; 32 from V8-384 | Total `2.124`; correct CE `1.796` | Restored fit, but still lacked strong end-task evidence. |
| V14 | Representation contrast at the first supervised predictor | 42; 32 from V8-384 | Total `2.230`; correct CE `1.786`; mean representation distance `0.0354` | Memory vectors separated, but correct/shuffled fused deltas remained almost collinear (mean cosine `0.9989`) and causal CE effects were small. |
| V15 | Contrast v6 on the first differing supervised span, W8 target | 42; 32 from V8-384 | Total `2.251`; correct CE `1.804`; mean training targeted gap `-0.00118` | Better-aligned objective, but the full validation benchmark failed all three core tasks. |
| V16 | V15 objective plus post-attention residual hybrid (`gamma=0.01`, cap `0.02`) | 42; 32 fresh updates from V14-416 | Total `2.042`; correct CE `1.605`; in-sample exact-pair W1 gap `+0.07031`; W8 gap `+0.00692` | Best causal-token training signal so far. Dataset attribution nevertheless falls to `16/30`; full benchmark is pending. |

The CE columns above are means over the final 32 logged updates unless otherwise
stated. V1 lists individual logged endpoints because its 41,266-row trainer
state is reported from its completed checkpoint aggregate.

## Causal-memory progression

The key diagnostic question changed over time. Unless stated otherwise, these
diagnostics use the same 32 rows used for training and are not held out:

1. V2-V6 asked, "Can the adapter lower answer CE?" The answer became yes.
2. V10 asked, "Does the answer depend on which history was written?" The
   answer was barely: shuffled-minus-correct full-answer CE was `+0.00268`.
3. V14 showed that history vectors differed but mostly in directions that had
   little causal effect. Across layers, correct/shuffled fused-delta cosine was
   `0.9989`; the mean single-layer bidirectional CE effect was `0.00355`.
4. The V14 layer-state diagnostic gave a correct-history full CE of `1.53193`,
   a W1 donor-minus-correct gap of `-0.01758`, and a W8 gap of `+0.00761`.
5. A residual-hybrid inference sweep on V14 showed that geometry could expose
   more of the state signal. At scale `0.01`, W1 gap was `+0.03320` and W8 gap
   was `+0.02177`. This motivated V16.
6. V16 checkpoint 32 reaches correct-history full CE `1.38570`, W1
   donor-minus-correct gap `+0.07031` (median `+0.06649`, positive on `21/32`
   rows), and W8 gap `+0.00692`.

V16 therefore improves a narrow in-sample causal mechanism metric. The small W8
effect and negative attribution validation result show that this is not yet
sufficient for useful task behavior.

## Dataset benchmark

This benchmark is a structured-task transfer/preservation gate, not direct
proof of online-memory learning. It resets RWKV state for every row, leaves the
full prompt visible to normal attention, and currently has no donor-history or
no-write condition. Passage-level disjointness is also not fully established:
the V1 audit found source-passage content overlap on 155 of 253 rows, and the
current V16 manifest records that an overlap audit was not performed.

### Completed results

| Checkpoint / split | Attribution | Narrative unit accuracy | Scene boundary F1 | Decision |
|---|---:|---:|---:|---|
| V1 original adapter, test | `0.9333 -> 0.9333` (`+0.0000`) | `0.6481 -> 0.6278` (`-0.0203`) | v3.2: `0.3088 -> 0.1837` (`-0.1252`); v4: `0.1943 -> 0.1470` (`-0.0473`) | Regressed overall |
| V15 checkpoint 416, validation | `0.8333 -> 0.6333` (`-0.2000`) | `0.6363 -> 0.5894` (`-0.0469`) | v4: `0.1874 -> 0.1495` (`-0.0379`) | Failed all three core gates |

Each cell is `base -> memory (memory - base)`. V1 used the test split and four
tasks; V15 used the current three-task validation selection protocol, so their
absolute scores should not be compared across rows.

### V16 in progress

- Protocol: validation, 239 rows per condition, paired base and native V16,
  478 generations total.
- Snapshot progress: `283/478`; base is complete and V16 has completed 30
  attribution plus 14 narrative rows.
- Completed attribution: base `25/30 = 83.3%`; V15 `19/30 = 63.3%`; V16
  `16/30 = 53.3%`.
- V16 attribution delta is `-30` percentage points versus base and `-10`
  points versus V15. V16 already fails this task; narrative and scene remain
  pending.
- V15 and V16 base generations are byte-for-byte identical on the compared
  rows, so the attribution comparison is paired and directly comparable.

The bundled Qwen3-8B Novel Base SFT plus task-LoRA reference scores are useful
context only. They use a different base model, task-specific training, and the
test split; they are not the checkpoint-selection comparison for this work.

## What has and has not been established

Established:

- The trainer can optimize all 42 memory layers within VRAM limits.
- CE can be driven below `1.7` on the 32-row training probe.
- State presence can strongly affect prediction.
- On its 32 training pairs, V16 makes the first distinguishing answer token
  measurably depend on the correct versus donor state.
- The step-32 V16 training log does not show a closed gate: its logged mean is
  `0.933`, and `77.7%` of the logged gate values exceed `0.99`. These are not
  global benchmark aggregates.

Not established:

- That the memory encodes enough history identity rather than generic answer
  structure.
- That lowering the 32-row training CE improves disjoint validation behavior.
- That the all-layer memory correction is injected in a task-useful direction.
- That any trained version improves the novel-agent benchmark over Gemma base.

## Recommended selection policy

1. Make a disjoint memory-specific validation set the primary criterion. Score
   correct versus exact-donor W1/W8 tokens and normal state versus no-write or
   no-state; the prompt must not expose the written facts through normal
   attention.
2. Use the native dataset validation benchmark as a secondary structured-task
   transfer gate against the identical frozen Gemma base. The goal is positive
   task improvement, not merely the current preservation tolerance.
3. Evaluate fixed small slices at every candidate checkpoint. Run all 239 native
   validation rows only after both the memory-specific slice and at least two
   structured tasks improve.
4. Do not select on loss or donor gaps from the same 32 training rows.
5. Treat V16 as a useful mechanism experiment, not as the new best checkpoint.
   Run the untouched test split only after selecting a checkpoint on validation.

## Recommended V17 proof run

The next run should test benchmark-relevant memory learning directly instead of
adding another contrast-loss variant:

- Build a deterministic, balanced 96-row training manifest: 32 attribution, 32
  narrative, and 32 scene rows.
- Set `episode_recent_messages=0`. For the dataset's standard
  `[system, user, assistant]` rows this writes `[system, user]` and reads/trains
  on `[system, assistant]`; reusing V16's value of 1 on these three-message rows
  would leave the write history empty. V16's own five-message probe rows did
  have a non-empty write history.
- Train all 42 layers with direct `context_dropout_ce`, a frozen fusion gate,
  no KL, no contrast term, and no read/write dropout.
- Use a fresh optimizer at learning rate `5e-4`; save steps 16 and 32, and only
  extend to steps 64 and 96 after validation improves.
- Evaluate both normal state and the same adapter with writes disabled. A useful
  checkpoint must improve the fixed validation slice, improve at least two of
  the three tasks, and beat its no-state control.

This requires a generic topology-exact adapter-only warm start. The current
warm-start mode is deliberately restricted to the V14-to-V16 residual-hybrid
ablation and should not be reused as if it were generic.

## Repository provenance

- All V1-V16 source, launchers, diagnostics, benchmark tooling, and this report
  are maintained in `/home/xiaol/X/Multi-state-RWKV-online-memory`.
- `/home/xiaol/X/delta-Mem/.venv/bin/python` is only the existing Python
  environment. Runs set `PYTHONPATH` or `--delta-mem-root` to the canonical
  Multi-state repository, so no source is imported from the sibling checkout.
- The sibling `delta-Mem` commit `bec8330` is an older Tau2 application line and
  is superseded by the current Multi-state implementation; it should not be
  cherry-picked into this history.
- Models, datasets, checkpoints, and generated benchmark outputs remain on the
  2 TB SSD. They are evidence artifacts rather than Git source and are linked
  below by their exact paths.

## Primary evidence

All paths below are relative to
`/run/media/xiaol/B214449214445C0B/delta_mem_outputs/novel_rwkv_ms_memory`:

- V1: `full_l0_5_r8_read192_write512_chunk128_suffix112/trainer/checkpoint-41266/trainer_state.json`
- V2: `v2_loss_probe_layers0_5_seed20260724_n32_e4_write512/trainer/checkpoint-128/trainer_state.json`
- V3: `v3_loss_probe_layers0_5_seed20260724_n32_e4_write512_recent1_kl0_isolation/trainer/checkpoint-128/trainer_state.json`
- V4: `v4_content_control_l0_5_n32_recent1_readwrite0_kl0_beta0_gain02_lr1e3_e4/checkpoint-128_answer_token_ce_ablation.json`
- V5: `v5_all_eligible_l0_23_r4_n32_recent1_readwrite0_ceonly_kl0_gain02_lr1e3_e4/loss_probe_analysis.json`
- V6: `v6_semantics2_all_l0_23_r4_n32_ceonly_gain02_lr1e3_e21_cont640/loss_analysis.json`
- V7: `v7_semantics2_all_l0_23_r4_n32_ceonly_o_contentgate01_gain02_lr1e3_e16_cont384/trainer/checkpoint-432/trainer_state.json`
- V8: `v8_semantics2_all42_l0_41_r4_n32_ceonly_o_contentgate01_gain02_lr1e3_e12_from256_mlpckpt/trainer/checkpoint-384/trainer_state.json`
- V9: `v9_semantics2_all42_l0_41_r4_n32_ceonly_o_postnorm_contentgate01_gain02_lr1e3_e5_from128_mlpckpt/trainer/checkpoint-160/trainer_state.json`
- V10: `v10_semantics2_all42_l0_41_r4_n32_contentcontrast025_margin05_o_contentgate01_gain02_lr1e3_e13_from384_mlpckpt/checkpoint-416_answer_token_ce_ablation.json`
- V11: `v11_semantics2_all42_l0_41_r4_n32_contentcontrastv2_seqexact025_margin05_o_contentgate01_gain02_lr1e3_e13_from384_mlpckpt/trainer/checkpoint-416/trainer_state.json`
- V12: `v12_semantics2_all42_l0_41_r4_n32_contentcontrastv3_seqexact025_ctxmask_prevgrad_margin05_o_contentgate01_gain02_lr1e3_e13_from384_mlpckpt/trainer/checkpoint-416/trainer_state.json`
- V13: `v13_semantics2_all42_l0_41_r4_n32_contentcontrastv4_seqexact025_causalread_prevgrad_margin05_o_contentgate01_gain02_lr1e3_e13_from384_mlpckpt/trainer/checkpoint-416/trainer_state.json`
- V14: `v14_semantics2_all42_l0_41_r4_n32_contentcontrastv5_repr01_margin01_seqexact025_causalread_prevgrad_margin05_o_contentgate01_gain02_lr1e3_e13_from384_mlpckpt/representation_diagnostic/checkpoint-416_writer_reader_representation.json`
- V15: `eval_v15_checkpoint416_val_core_full_v1/format_recovered_summary.json`
- V16 diagnostic: `v16_semantics2_all42_l0_41_r4_n32_contentcontrastv6_w8_repr01_margin05_seqexact025_causalread_prevgrad_o_contentgate01_gain02_residhybrid_gamma0p01_cap0p02_lr1e3_fresh32_fromv14/exact_pair_diagnostic/checkpoint-32_w1_w8.json`
- V16 live benchmark: `eval_v16_checkpoint32_val_core_full_v1/progress.json`
