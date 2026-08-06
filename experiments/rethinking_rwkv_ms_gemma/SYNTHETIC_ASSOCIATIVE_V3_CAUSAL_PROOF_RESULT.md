# Synthetic Compositional Associative Retrieval V3: Three-Seed Result

The locked Revision 5 aggregate proof **failed** because only two of the three
required held-out seeds passed. Seeds 42 and 44 passed every causal criterion;
seed 43 retained near-perfect semantic routing but missed the 95% exact-answer
threshold. No three-seed proof-set certificate was created.

This is strong evidence that the frozen-Gemma projected-KV outer memory can
learn compositional addressing and value transport, but it is not the required
three-seed robustness proof.

## Locked Protocol

- Frozen Gemma model with projected-KV memory adapters on all 42 attention
  layers; all 42 base MLPs remained frozen.
- Rank 32 value projection, key dimension 32, temperature 16.
- 17,420,200 trainable adapter parameters in 1,176 tensors.
- BF16 SDPA, batch size 4, learning rate `2e-4`, 768 optimizer updates.
- 384 synthetic training rows and 192 held-out rows.
- Greedy whole-answer exact accuracy was the acceptance metric.
- Required held-out seeds were fixed in advance as 42, 43, and 44.
- Positive answer and routing thresholds were 95%; no-write answer accuracy
  had to remain at or below 35%, with routes absent on every row.

The prerequisite seed-42 train screen passed all metric criteria with
`train_screen_passed: true`. Its top-level `gate.passed: false` and
`acceptance_eligible: false` are expected because a train-split screen cannot
be an acceptance proof.

## Held-Out Result

All counts below are greedy whole-answer exact results over 192 rows. Routing
is the semantic query-to-slot accuracy across all evaluated memory layers.

| Seed | Correct / donor / shuffled | Value swap | Target rewrite | Paired rewrite flip | Positive routing | No write | Gate |
|---:|---:|---:|---:|---:|---:|---:|:---:|
| 42 | 190/192 (98.96%) | 189/192 (98.44%) | 192/192 (100%) | 190/192 (98.96%) | 100% | 0/192 | pass |
| 43 | 181/192 (94.27%) | 183/192 (95.31%) | 182/192 (94.79%) | 171/192 (89.06%) | 99.74%-99.83% | 0/192 | **fail** |
| 44 | 191/192 (99.48%) | 192/192 (100%) | 192/192 (100%) | 191/192 (99.48%) | 99.96%-100% | 0/192 | pass |

The exact 95% threshold requires at least 183 correct rows. Seed 43 therefore
missed correct, donor, and shuffled-slot accuracy by two rows and target-rewrite
accuracy by one row. Its value-swap answer gate passed. The paired rewrite
criterion failed more clearly because the baseline and rewritten errors were
on different rows.

Every seed passed the structural and causal checks:

- query counterfactual routing was 100%, 99.78%, and 99.96% for seeds 42-44;
- runtime states were byte-identical across each query-counterfactual family;
- target rewrites changed only the selected binding, and every replacement
  binding was absent from all training bindings;
- shuffled physical slots preserved semantic routing;
- all positive conditions had full four-slot occupancy and exact forced writes;
- no-write answers were 0/192 and memory routes were absent on every row;
- input, source, and frozen model identities were unchanged.

The result localizes the remaining variance to exact value readout, not
semantic addressing. It would be invalid to lower the threshold, replace seed
43, or tune on this now-observed held-out partition and still call it the
pre-registered proof.

## Integrity And Reproduction Evidence

- Proof-run commit: `58795989648fe192a90ad9de3e0da7e7f1033dda`
- Independent stricter verifier: `332edfc0af3622e0c054b4b3ee6cd54631d782b3`
- Source manifest canonical SHA-256:
  `e3835feb524527a554ab8215afa3746254cf875e813637925d5d904aa3dee79c`
- Source partitions SHA-256:
  `6c92f1e6e651a321bbddb53704995f70a9570a842aaedf4e1c2e82b991f71b93`
- Split manifest SHA-256:
  `53d6f8badcdbd7c2fa27a02916a83605e824a2e13880d5d63186d4563d5f9a9a`
- Frozen model identity SHA-256:
  `554deae79721a9f6e85623cc4e2f8d20b88facd1f0276556fb0cbd8ee05cc478`
- Frozen model weights SHA-256:
  `cfbd3d2f1cd71bd471c37fe2bf8546d5028d41e5736f64e1ca6c6b8893125503`

| Run | Receipt canonical | Receipt file | Evaluation canonical | Adapter weights |
|:---|:---|:---|:---|:---|
| train gate | `1a834e997e3ce3fe070360f882575c578970a544aa7f2a25f89684afe53acaa1` | `08f7ec374af9beba385d01a2a9e98395c67913de1aadfd6135825ae3c9a9dd54` | `8e967e691a6e972846a53ef25d14038f7a1fbacbfa6010e62d51424e19731cb7` | `a9c638c7aefd651b376ee51062ac80ea348a57ce61b532b189e3259d2af1ec51` |
| seed 42 | `ea0bf42890425c8160888014d319ec46fefa1066127a21f1769a7bfc2f6da4a4` | `04f6c597a6e419824a6905e74dc03b7494f663c6b3ac140ca409858c7d730711` | `163549120ed3174ff8ef56fc64889ea5d3d7e75002a2f0789402ff93abc0d0e5` | `ec65b0e4b7b18456076a583381870612b4976f3266ce412401fe388d5c2191dc` |
| seed 43 | `386d5903b28ce71db3afd4b45f1e7d090f6b77a018bbd60595ba432e244988b9` | `69cd966f47cd2702ebe8ede3900951360d8a2c06a836f5c405ec13bb16c08caa` | `6453b094d2adf4e5f925173d729d93ee576aae8c2f4c2f571971d077439a92a6` | `7578462bf1994a404b3f30f7be4c3e09b51c8e153d71c608d980a551a2296032` |
| seed 44 | `570841b7c130de4dd083267def1cd874443efeb29e1650a223b20b1edb0590e6` | `1fc8b1c8e260b67aa2f27a3f633cc2324ea9cd27373eafc1103eb2e0bf38c96a` | `003f35746f211957858c70e1e236efe559b7c8aba5b81dfcbcc2d6ec92dc82df` | `922a71a2383d0416a478b92c137447b78e90d0f57ff24745ff1e38705b9448be` |

Every receipt is semantically valid under the launch commit and the independent
verifier with full model hashes rechecked. The stricter verifier also binds
held-out and train-screen code provenance, model before/after identity, source
identity, and the exact frozen-base adapter attachment.

The source partitions, configs, evaluations, receipts, and training traces are
committed. Adapter `.pt` tensors are intentionally omitted from Git but retained
locally and bound by SHA-256. Because receipts record absolute local paths and
full validation loads the omitted tensors, this commit is durable hash-bound
evidence rather than a portable one-command replay from a fresh checkout.

No protected evaluation data or Hard32 data was accessed. Hugging Face access
used `https://hf-mirror.com`.

## Decision And Next Goal

Do not spend another cycle tuning against this exposed synthetic held-out set.
Use it as development evidence: semantic addressing is stable, while exact
readout has seed variance.

The next primary goal should be a passage-disjoint, benchmark-shaped causal
memory gate on the novel data:

1. Train adapter weights offline while keeping Gemma frozen; at inference,
   write history into the outer state online without online weight updates.
2. Remove answer-bearing facts from the visible read prompt and discard ordinary
   KV history, so the answer can only arrive through outer memory.
3. Evaluate matched `correct-state`, `donor-state`, and `no-write/no-state`
   conditions on a sealed validation split.
4. Require correct state to beat both controls in every seed and improve at
   least two of attribution, narrative, and scene over frozen Gemma base.
5. Run all four GPUs as independent seeds/configurations; first probe batch 6
   for memory and throughput, then freeze batch size before formal runs.

If correct routing survives but answers fail, localize the oracle-slot value
path. If routing fails, work on the address encoder. Only after causal natural
transfer passes should work move to learned write boundaries, longer-horizon
capacity/interference, or a native recurrent RWKV-state comparison.
