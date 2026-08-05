# Synthetic Compositional Associative Retrieval V3: Seed 42

This is the first acceptance-eligible V3 proof run. It trained the frozen-Gemma
projected-KV outer-memory adapter on all 384 training rows, then evaluated all
192 untouched synthetic held-out rows with greedy decoding.

## Result

Seed 42 proves semantic query-to-slot routing, but it does not yet prove exact
value transport:

- correct-memory semantic routing was 8007/8064 (99.293%);
- query counterfactual routing under byte-identical memory was 8007/8064
  (99.293%), with 48/48 memory-state families byte-identical;
- shuffled-slot semantic routing was 8006/8064 (99.281%);
- correct-memory greedy exact answers were 10/192 (5.208%), while
  teacher-forced token accuracy was 1564/1768 (88.462%);
- donor-memory greedy exact answers were 10/192 (5.208%);
- value-swap greedy exact answers were 9/192 (4.688%);
- no-write greedy exact answers were 0/192, token accuracy fell to 778/1768
  (44.005%), and routes were absent in every module-row;
- four-slot occupancy and forced write routes were both exact, all router
  gradients were finite and nonzero, and source/model immutability passed.

The acceptance gate therefore failed only the correct, donor, and value-swap
answer criteria. This differs from V2: the router now changes correctly with
the query and exceeds the locked 95% routing threshold, while the remaining
bottleneck is decoding the selected value. Over the final 32 training steps,
mean route accuracy was 99.498%, but whole-answer exact accuracy was only
3.125% and answer CE was 0.3457. The current projected-KV value code has rank
4 per layer, so the next controlled experiment widens only that code and uses
train-only evaluation before another held-out proof.

Seeds 43 and 44 are intentionally not launched because seed 42 did not pass
the conjunctive gate.

## Reproduction Evidence

- Runner commit: `6589648e13de6785ae6caa897fbdab956fee8969`
- Source partition SHA-256:
  `6c92f1e6e651a321bbddb53704995f70a9570a842aaedf4e1c2e82b991f71b93`
- Source manifest SHA-256:
  `e3835feb524527a554ab8215afa3746254cf875e813637925d5d904aa3dee79c`
- Split manifest SHA-256:
  `53d6f8badcdbd7c2fa27a02916a83605e824a2e13880d5d63186d4563d5f9a9a`
- Run directory:
  `local_artifacts/v3_proof_seed42_all_l42_b4_s384_r1/`
- Evaluation canonical SHA-256:
  `91cd35ff2577ec4e60385e677d202f0c98c3a0b5ca92d9ecc63db0dbfc0a3e3e`
- Run receipt canonical SHA-256:
  `b9cebe1ce3c89608e8cd56ebffcb321b45ee015d9e0a2d22c1a59761c7311a84`
- Run receipt file SHA-256:
  `1d0d0d75ceea98baa9a68dfa12247b3339191b6789c3a5d429e627b4c1c08970`
- Receipt validation: `valid: true` with model hashes rechecked
- Focused V3/V2 contract suite after rank parameterization: 51 passed
- Protected evaluation and Hard32 access: none

The adapter tensor is intentionally not committed. Its SHA-256 and exact
configuration remain bound into the committed run receipt.
