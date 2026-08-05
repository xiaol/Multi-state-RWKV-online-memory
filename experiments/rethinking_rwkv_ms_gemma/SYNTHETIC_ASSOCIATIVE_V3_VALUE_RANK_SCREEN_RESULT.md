# Synthetic Associative V3 Value-Rank Train Screen: Seed 42

## Scope

These screens test whether projected-KV value width and training duration explain
the failed seed-42 V3 value readout. They are selection runs, not acceptance
proofs:

- all 384 training rows were trained and evaluated with `eval_split=train`;
- the untouched 192-row heldout split was not evaluated;
- evaluation was teacher-forced (`--no-greedy`) under `correct`, `donor`,
  `value_swap`, `shuffled_slots`, and `no_write` interventions;
- all 42 memory layers were trained with batch size 4, learning rate `2e-4`,
  answer/route weights `1.0/1.0`, key dimension 32, and temperature 16;
- ranks 8, 16, and 32 ran for 384 steps; the rank-4 duration control ran for
  768 steps; and
- `alpha=2*rank`, so the configured alpha/rank scale remained constant.

The frozen model identity was
`554deae79721a9f6e85623cc4e2f8d20b88facd1f0276556fb0cbd8ee05cc478`.
All runs recorded `HF_ENDPOINT=https://hf-mirror.com`.

## Post-training evaluation

Whole-answer exact and token accuracy are teacher-forced. The donor column is
reported for receipt completeness but is not independent causal evidence; see
the donor audit below.

| Rank | Steps | Correct exact | Donor exact | Value-swap exact | Correct tokens | Value-swap tokens |
|---:|---:|---:|---:|---:|---:|---:|
| 4 | 768 | 282/384 (73.438%) | 282/384 (73.438%) | 270/384 (70.312%) | 3434/3536 (97.115%) | 3422/3536 (96.776%) |
| 8 | 384 | 49/384 (12.760%) | 49/384 (12.760%) | 45/384 (11.719%) | 3196/3536 (90.385%) | 3187/3536 (90.130%) |
| 16 | 384 | 140/384 (36.458%) | 140/384 (36.458%) | 131/384 (34.115%) | 3284/3536 (92.873%) | 3274/3536 (92.590%) |
| 32 | 384 | 295/384 (76.823%) | 295/384 (76.823%) | 287/384 (74.740%) | 3447/3536 (97.483%) | 3439/3536 (97.257%) |

| Rank | Correct route | Query-CF route | Value-swap route | Shuffled route | Tail-32 answer CE | Tail-32 exact | Tail-32 route |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 4 | 16042/16128 (99.467%) | 16042/16128 (99.467%) | 16035/16128 (99.423%) | 16040/16128 (99.454%) | 0.091083 | 73.438% | 99.628% |
| 8 | 16094/16128 (99.789%) | 16094/16128 (99.789%) | 16090/16128 (99.764%) | 16094/16128 (99.789%) | 0.314212 | 9.375% | 99.665% |
| 16 | 16072/16128 (99.653%) | 16072/16128 (99.653%) | 16066/16128 (99.616%) | 16072/16128 (99.653%) | 0.245694 | 24.219% | 99.572% |
| 32 | 16085/16128 (99.733%) | 16085/16128 (99.733%) | 16076/16128 (99.678%) | 16085/16128 (99.733%) | 0.151836 | 52.344% | 99.721% |

Every screen also had:

- no-write exact `0/384`, token accuracy `1562/3536` (44.174%), and routes
  absent in `16128/16128` module-rows;
- exact full occupancy (`16128/16128`) and forced write routing
  (`64512/64512`) in each answer-bearing condition;
- byte-identical runtime query-counterfactual states in `96/96` memory-state
  families;
- finite, nonzero router gradients in every targeted module; and
- passing split-leakage, source/model immutability, and receipt-integrity checks.

The train-only gates are marked `acceptance_eligible: false` by design because
they do not evaluate heldout greedy answers.

## Integrity receipts

`evaluation` and `receipt` below are canonical JSON hashes. The file columns are
the byte-level SHA-256 values bound by or returned from receipt validation.

| Rank/steps | Evaluation | Evaluation file | Receipt | Receipt file |
|---|---|---|---|---|
| 4/768 | `80089700d09332bce96bbe3188fddee7182974f812c071e7b41f23d109a0b922` | `39250a1b05dc36fa903378c220f9b37c06b58ea8d67acb2cd4a5c3987fe3bd9c` | `c29a5e228fe1e9c59da6dc56b6a7f7e9f77ef469282d187947eb50060fd54b74` | `5797704e4efe2508bd89dae4b0b7a70f56052e5fe35bfacd51fbf8e0fe8e1405` |
| 8/384 | `10d8c530afefb652e5cc03e89fe5bd96f25766da156980a8780247475e836f88` | `aef8cd09a4f88300a4c002b131cc19baca63e54e1e6eb2b5603a66c08dc0eb07` | `5ec32b98235ff5656d3675adef72fcf8a7b4b35342b7940192f995f9e81eed29` | `ae8ccb1543ded1de7d3dd10a24c00a956b4895969e62baafe52cbd4c5ed6b3b4` |
| 16/384 | `7d4a0f82bad853f46a63e7358e72383c9dc3bb8963287d172c8b5a64764a8149` | `6abf0bdeefc332188bb92e344cd30d051ed1efc3d162beae9bdfd561476bebbc` | `ed5001110f3bee886265dff0b5b8fb3938d4f96521b6bb51122fd26b1251c1d6` | `b05e3bcaa80d3b532dff266209ec36c665ccd5fe7d55dc923d1e2a7a40cca1b4` |
| 32/384 | `5699b5b73c41d0de81457ca0c0c5cdc7649dc969d3395172641729a7561ced9a` | `5feabe4750c415b877a4bd2bfc4b51c54e836f315829e21e84348ee0032832c6` | `6f5bb33a6b4333717dd34737fbdc2e224b4d49b9b168cc873c0092d52537b549` | `54546428849f555c1c80c9af0d708e2c296391d85c60e9cf1ac7efe8bbee28c9` |

All four receipts were independently validated with `--verify-model-hashes` and
returned `valid: true`. Common source provenance was:

- partition SHA-256:
  `6c92f1e6e651a321bbddb53704995f70a9570a842aaedf4e1c2e82b991f71b93`;
- canonical source-manifest SHA-256:
  `e3835feb524527a554ab8215afa3746254cf875e813637925d5d904aa3dee79c`;
- split-manifest SHA-256:
  `53d6f8badcdbd7c2fa27a02916a83605e824a2e13880d5d63186d4563d5f9a9a`.

No adapter tensor is part of this document.

## Donor redundancy audit

The current aggregate donor intervention is a reindexing of the correct
intervention. `donor_example` takes the donor row's records, read features,
labels, route target, and expected value. The generator pairs the two mapping
offsets bijectively and reciprocally at the same target slot. Therefore, over a
complete train or heldout partition, donor examples form the same multiset as
correct examples. This is why correct and donor counts are identical at every
rank.

Consequences:

- donor must still meet the existing numerical threshold for compatibility with
  the locked runner, but it supplies no additional causal evidence;
- `value_swap`, which keeps the query/key layout and reassigns values, is the
  decisive intervention in these receipts; and
- before an acceptance claim, the evaluator must add a paired same-query,
  different-memory test whose greedy answer or first-answer-token logits switch
  to the memory-selected value. Aggregate donor accuracy cannot substitute for
  that paired flip.

## Interpretation and selection

Routing is solved for this screen: all semantic, counterfactual-query, and
shuffled-slot route accuracies remain above 99.4%. No further key/query/router
work is justified unless a later run regresses below 95%.

Value width materially improves convergence at the matched 384-step budget:
correct exact rises from 12.760% at rank 8 to 36.458% at rank 16 and 76.823% at
rank 32. Duration also matters. Within the rank-4 control, the final-32-step
answer CE/exact changed from `0.344465`/4.688% at steps 353-384 to
`0.091083`/73.438% at steps 737-768 while routing stayed saturated.

The selected next experiment is therefore rank 32 for 768 steps, train-only,
with every other variable held fixed. This combines the best width with the
duration shown to be necessary. It is still projected-KV outer-online-memory
training; it is not another routing experiment and it is not yet novel-dataset
training.

## Pre-heldout gate

Do not spend heldout evaluation unless the full 384-row rank-32/768 train screen
satisfies all of the following:

- teacher-forced whole-answer exact is at least 95% for `correct` and
  `value_swap`; the redundant `donor` aggregate must also be at least 95% for
  compatibility;
- correct, query-counterfactual, value-swap, and shuffled-slot semantic routing
  are each at least 95%;
- no-write whole-answer exact is at most 35% and no-write routes are absent in
  every possible module-row;
- occupancy and forced writes are exact, runtime counterfactual states are
  byte-identical, all router gradients are finite and nonzero, and split/source/
  model integrity passes; and
- the heldout proof protocol includes the paired same-query memory-flip metric
  described above.

If the train gate passes, run one acceptance-eligible rank-32 seed-42 heldout
proof with greedy decoding, then seeds 43 and 44 only after seed 42 passes. If
the train gate fails, do not access heldout: classify whether only value swap
fails (paired value-binding training) or correct readout also fails (oracle-slot
value-path localization).

## Selected command

```bash
HF_ENDPOINT=https://hf-mirror.com python -m \
  experiments.rethinking_rwkv_ms_gemma.run_synthetic_compositional_associative_retrieval_v3 \
  --source-manifest experiments/rethinking_rwkv_ms_gemma/local_artifacts/synthetic_compositional_associative_canary_v3/source_manifest.json \
  --model-path /root/X/.cache/hf/gemma-4-E4B-it-a4c2d58 \
  --profile microfit \
  --seed 42 \
  --eval-split train \
  --batch-size 4 \
  --eval-batch-size 8 \
  --learning-rate 2e-4 \
  --answer-weight 1.0 \
  --route-weight 1.0 \
  --max-grad-norm 1.0 \
  --device cuda:0 \
  --dtype bfloat16 \
  --attn-implementation sdpa \
  --target-layers all \
  --key-dim 32 \
  --temperature 16.0 \
  --rank 32 \
  --epochs 8 \
  --max-steps 768 \
  --no-greedy \
  --output-dir experiments/rethinking_rwkv_ms_gemma/local_artifacts/v3_train_screen_seed42_rank32_s768_r1
```
