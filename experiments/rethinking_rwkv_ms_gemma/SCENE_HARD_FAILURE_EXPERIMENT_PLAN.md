# Scene Hard-Failure Memory Experiment

## Question

Can a frozen Gemma model use RWKV-MS state to recover scene-boundary cases that
the same frozen base model fails?

Training loss is diagnostic. The experiment passes only when state improves the
dataset's native scene-boundary benchmark metric and the improvement depends on
the contents of the correct state.

## Why This Benchmark

`scene-v4-current` is the hardest of the three Novel Agent validation tasks for
the frozen Gemma base:

| Task | Base validation score |
| --- | ---: |
| Attribution | 0.8333 accuracy |
| Narrative parsing | 0.6363 unit accuracy |
| Scene boundary | 0.1874 micro-F1 |

The only held-out screen in scope is the frozen 32-row scene validation slice
known as Hard32. It contains 31 base failures and one base-success sentinel.
Attribution, narrative, complete validation, and test are out of scope.

## Existing Evidence

The historical 24-layer V6 checkpoint reached a final training loss of
`1.8330` and a minimum logged loss of `1.1140`. On Hard32, however, its native
strict scene F1 was `0.0` for base, no-write, and normal memory. Under the more
permissive recovery diagnostic, base and no-write scored `0.1379`, while normal
memory scored `0.0`.

The newer all-42-layer V6 identity proof also failed Hard32 at steps 16 and 32.
Step 32 made correct-state NLL better than donor and zero state on `20/32` rows,
but native strict F1 remained `0.0` in every condition. A small token-level
state signal therefore did not become useful benchmark generation.

These results rule out loss and same-row logit gaps as checkpoint-selection
criteria.

## Frozen Architecture

- Frozen Gemma-4 E4B base model.
- Fresh RWKV-MS adapter; no warm start from a trained adapter.
- All 42 layers, rank 4, alpha 8, Q+O delta paths.
- Four recurrent states, semantics v2, 128-token chunk routing.
- Reset state for every row.
- Write exact `[system, user]`; read/generate with `[system, assistant]` and
  read-side writes disabled.
- No KL, base-preservation, or unrelated benchmark objective.

## Data Contract

1. Mine failures only from the official `scene-v4-current` training split using
   frozen-base outputs generated under the benchmark prompt.
2. Preserve source `[system, user, assistant]` serialization byte-for-byte.
3. Pair label-distinct rows symmetrically, minimizing write-token length
   difference.
4. Balance empty/nonempty pairs and nonempty same-cardinality pairs. The latter
   prevents state identity from being solved only by predicting list length.
5. Bind source rows, base records, pair schedule, tokenizer, and official
   dataset revision by SHA-256.
6. Training must not resolve or open validation, Hard32, or test files. It may
   bind their already-published hashes only to prove exclusion.

Hard32 remains held out. "Pair the dataset to the benchmark" means identical
task, prompt, answer schema, and metric, not training on Hard32 answers.

## Training Sequence

### 1. Smoke

Run one reciprocal pair through one real optimizer update. Require finite loss
and gradients, changes in all 22 first-step-reachable trainable tensor families
across all 42 layers, optimizer state for all 27 trainable families, no changes
outside the adapter, and sufficient VRAM headroom. The five zero-seeded delayed
families cannot change on the first AdamW update; treating that structural zero
gradient as a smoke failure would make the smoke contract impossible.

### 2. Train-Only Overfit Proof

Train a bounded curriculum of base-failed training rows, save every optimizer
update, and evaluate checkpoints only on those training rows. Use the existing
cached-prefix boundary repair plus correct-state identity objective; do not add
a new loss family in this experiment.

Present all 16 reciprocal pairs for four deterministic cycles. Unlike V15,
perform one optimizer update per symmetric pair instead of accumulating an
entire 16-pair cycle into one update. This gives 64 optimizer updates from the
same 64 pair presentations that gave V15 only four updates. Save every update
for auditability, but run generation screening only at cycle endpoints 16, 32,
48, and 64. Stop only at the first checkpoint that both passes the train-only
gate and has current-byte changes in all 1,134 trainable layer-family tensors.
An earlier gate pass with partial coverage is recorded but does not authorize
selection, so screening continues to the next endpoint.
The schedule may be extended only by complete, deterministically bound cycles
after all four initial cycle endpoints fail.

The overfit proof must show all of the following:

- The selected checkpoint changes all 27 trainable tensor families in all 42
  layers relative to the fresh seeded adapter (1,134 layer-family combinations).
- `normal_full` native scene F1 is higher than both frozen base and no-write.
- `state_only` native scene F1 is higher than donor, shuffled, and zero/no-write
  state.
- At least one nonempty, same-cardinality reciprocal pair switches to the
  correct boundary values in both directions.
- Canonical `{"boundaries": [...]}` coverage is at least 95%.
- Predicted boundary density is no more than twice gold density.

If this proof fails, stop. Do not spend Hard32 or broader benchmark compute.

#### Fail-Closed Endpoint Screening

Run the endpoint driver only after the production launcher has written its
64-step completion receipt and all 64 checkpoint audits. The driver accepts no
model, dataset, condition, endpoint, or held-out path overrides:

```bash
cd /home/xiaol/X/Multi-state-RWKV-online-memory
PYTHONPATH="$PWD" /home/xiaol/X/delta-Mem/.venv/bin/python \
  experiments/rethinking_rwkv_ms_gemma/run_scene_hard_failure_endpoint_screen.py \
  --run-root /run/media/xiaol/B214449214445C0B/delta_mem_outputs/novel_rwkv_ms_memory/scene_hard_failure/scene_hard_failure_four_cycle_pair64_hard_failure_train32_v1_step64 \
  --completion-receipt /run/media/xiaol/B214449214445C0B/delta_mem_outputs/novel_rwkv_ms_memory/scene_hard_failure/logs/scene_hard_failure_four_cycle_pair64_hard_failure_train32_v1_step64.completion.json
```

Before claiming any output path, the command runs the exact Train32 evaluator
arguments with `--preflight-only` for steps 16, 32, 48, and 64. All four must
return zero without creating the screen root or any endpoint output directory.
Only then does the command create the fresh `train32_endpoint_screen` directory
inside the run root. It generates and recomputes the Train32 gate at each step
in order and stops at the first gate pass with full current-byte coverage. Each
evaluated endpoint records the benchmark-gate result, exact adapter coverage,
and combined selection eligibility, and contains
`manifest.json`, `summary.json`, `progress.json`, the seven condition JSONL
files, and `focused_recovery_gate.json`. `screening_protocol.json` binds the
production completion receipt and all preflight commands/results before
generation begins. The final `train32_checkpoint_selection_receipt.json` binds
that protocol, the same preflight evidence, every executed command, every
evaluated endpoint, and the production receipt.

Exit status `0` means the first gate-plus-coverage eligible endpoint is
authorized only for the separate Hard32 command. Exit status `1` means all four
endpoints were ineligible and an unauthorized deterministic fallback receipt
was written; this includes a gate pass whose current-byte coverage is partial.
Exit status `2` is a contract or runtime failure. Do not use `--overwrite`,
rename outputs to a held-out term, or manually substitute evaluator arguments.

### 3. Single Held-Out Screen

Select exactly one checkpoint from train-only evidence, then run Hard32 once in
these conditions:

1. `base_full`
2. `no_write_full`
3. `normal_full`
4. `state_only`
5. `state_only_donor`
6. `state_only_shuffled`
7. `state_only_no_write`

The primary metric is the dataset-native literal `boundaries` micro-F1. The
format-recovered F1 is secondary diagnostic evidence and cannot replace the
native score.

The held-out checkpoint passes only when:

- `normal_full` native F1 exceeds both base and no-write by at least `0.05`.
- `state_only` native F1 exceeds donor, shuffled, and zero/no-write by at least
  `0.05`.
- Correct state recovers at least three more frozen-base failures than no-write.
- At least 31/32 outputs are canonical and recoverable.
- Boundary density remains at most twice gold density.

Only a passing Hard32 receipt can authorize a later complete scene validation
run. It never authorizes attribution, narrative, or test evaluation.

## Decision Logic

```text
smoke fails
  -> fix implementation

train-only state-causality or F1 gate fails
  -> fix training objective/topology; do not run Hard32

train-only gate passes, Hard32 fails
  -> the adapter memorized train failures; increase disjoint scene data or
     change the learning mechanism before another held-out run

Hard32 passes
  -> run complete scene validation only
```
