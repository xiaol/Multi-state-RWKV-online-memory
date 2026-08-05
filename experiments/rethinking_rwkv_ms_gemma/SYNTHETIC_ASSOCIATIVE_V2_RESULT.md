# Synthetic Associative Retrieval V2

This reference experiment tests whether the projected-KV outer memory behaves
as a two-slot associative store after fitting four synthetic training rows.

## Result

The step-64 checkpoint learned a causal, state-dependent whole-mapping code,
but it did not learn semantic query-to-slot addressing:

- correct, donor, and value-swapped states selected their corresponding answer
  in 4/4 rows;
- intended route selections were 84/168 across the evaluated trajectory;
- changing only the query while preserving byte-identical state changed the
  selected slot in 0/84 comparisons;
- repeating the same query across compatible mappings selected the same slot
  in 70/84 comparisons.

The v2 result therefore rejects the claim that output accuracy on these four
training rows demonstrates associative addressing. The next experiment must
use record-local writes, separate key/value spans, direct route supervision,
and held-out compositions.

## Reproduction Evidence

- Locked source: `local_artifacts/synthetic_associative_retrieval_canary_v2/`
- Evaluation receipt:
  `local_artifacts/synthetic_associative_retrieval_runs/synthetic_associative_projected_kv_s2_k32_t16_u1_b4_lr2e4_seed42/trajectory_eval_v2.json`
- Receipt SHA-256:
  `1131b988cff84ba114a45ae7447f0cdf3fab28d204cbe483f3c5c89ee16fe33f`
- Focused evaluator tests: 19 passed
- Protected evaluation and Hard32 access: none

The checkpoint and adapter tensors are intentionally not committed. Their
cryptographic bindings remain in the evaluation receipt.
