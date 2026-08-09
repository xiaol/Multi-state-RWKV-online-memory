# Next Goal: Native Transfer V1

## Decision

The next goal is to make the proven synthetic online-memory mechanism change and
improve decisions on the Novel Agent validation tasks. The current adapter
passes its sealed associative-memory benchmark, but its native validation
generations are identical to frozen Gemma on all 239 rows.

Do not retune the passed sealed candidate or the terminally invalid R3 split.
Train a new candidate on official training rows only and keep `test` and
`Hard32` sealed.

## Hypothesis

A task-aligned write/read curriculum can transfer the multi-state memory skill
to natural novel prompts without changing the frozen Gemma backbone. The
curriculum must make the answer depend on recalled state, while retaining the
author-compatible output schema.

## Training Contract

- Freeze Gemma and train only the outer online-memory adapter.
- Use the official attribution, narrative, and scene training splits.
- Convert each training example into an explicit write episode and a later read
  episode; include donor-state, shuffled-state, and no-write negatives.
- Distill the base model on non-memory tokens so memory does not perturb
  ordinary language behavior.
- Add author-schema loss or constrained training targets; do not count format
  recovery as the primary training objective.
- Use four DDP ranks. Keep global batch size fixed at 16 for the first candidate
  so GPU count changes throughput rather than optimization. Adjust per-rank
  batching or accumulation only to preserve that global batch.
- Screen data/seed candidates only for generator feasibility before freezing
  the preregistration. After freezing, prohibit seed substitution.

## Validation Gate

Evaluate only the fixed validation rows with frozen greedy decoding.

1. Memory is at least base on all three recovered task metrics.
2. Memory is strictly better than base on at least two tasks.
3. At least one improved task has a paired 95% bootstrap interval whose upper
   bound is positive and no task exceeds the preregistered regression floor.
4. Normal memory beats no-write on each improved task, and shuffled or donor
   state removes at least half of the gain.
5. Author-schema validity is at least 95% on attribution and narrative.
6. Memory decoding is no slower than 2.0 times base after profiling and kernel
   optimization.
7. The result is reproduced by two frozen seeds before any final test is opened.

## Directions, In Priority Order

1. **Task-aligned write/read curriculum.** This directly targets the observed
   failure: the adapter runs, but changes zero native predictions.
2. **Explicit router supervision.** Train which spans write, which query reads,
   and which of the four states owns each fact.
3. **Ablation-enforced causality.** Require normal to beat no-write, shuffled,
   donor, and zero-state conditions rather than accepting adapter-only gains.
4. **Schema-preserving decoding.** Train exact author schemas or add constrained
   JSON decoding so semantic quality is not hidden by aliases.
5. **Memory efficiency.** Fuse recurrent decoding kernels, cache static adapter
   projections, and batch independent validation rows; the current adapter is
   roughly three times slower than base.
6. **Capacity experiments.** Only after transfer appears, compare state count,
   key dimension, eviction, hierarchical slots, and global/local partitions.
7. **Parameter-matched baselines.** Compare against LoRA and retrieval/cache
   baselines with equal trainable parameters and training tokens.

## Stop Rule

If a correctly instrumented candidate still produces fewer than 5% prediction
disagreements from base on development validation, stop the run before sealed
evaluation. That outcome means the training signal still does not control the
native decision path.
