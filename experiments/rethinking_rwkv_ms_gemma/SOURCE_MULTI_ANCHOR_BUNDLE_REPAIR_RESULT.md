# Multi-anchor RWKV value-bundle repair result

This open-development run tested a Full-Bandwidth-inspired value path from
three native RWKV reads at layers `(5, 11, 17)`. The three-anchor bundle was
the only residual value; the Gemma hidden query supplied a multiplicative gate.
There was no hidden-only, query-only, terminal-read, or projected-value
bypass, and an all-zero bundle was required to reproduce provider-off exactly.
This is a causal development result, not a reproduction of
[Full-Bandwidth Transformer](https://arxiv.org/abs/2608.08888) and not native
benchmark evidence.

## Frozen setup

- Fresh component-disjoint reservation: 40 reciprocal pairs, 80 rows.
- Training: 48 rows; held out: 32 rows.
- Excluded components: historical 94, parent 160, protected 64, cumulative
  development 64, prior multi-anchor 80, and weighted-renewal 80.
- Trainable tensors: `state_down`, `query_gate`, and `output_up` only
  (`166,912` parameters).
- Exactly four A100s with `HF_ENDPOINT=https://hf-mirror.com`.
- Both first-answer-token and first target/donor-divergent-token views were
  evaluated once after the locked 48-update run.

## Held-out result

| view | target selected | gain vs provider-off | donor-both margin | layer-both margin | decision |
| --- | ---: | ---: | ---: | ---: | --- |
| first answer token | `0.90625` | mean `-0.017416`, positive `0.40625` | mean `+0.025600`, positive `0.4375` | mean `-0.008605`, positive `0.4375` | failed |
| divergent token | `0.90625` | mean `+0.024898`, positive `0.50000` | mean `+0.003757`, positive `0.4375` | mean `+0.034956`, positive `0.53125` | failed |

Zero/provider-off exactness and lifecycle/gradient contracts passed. The
donor-positive fraction and the complete held-out gate did not. The exact
bundle family is therefore retired without tuning batch size, gain, learning
rate, duration, or thresholds. Protected mechanics, generation, and native
benchmark data remained closed.

Result: `local_artifacts/natural_memory_native_rwkv_source_multi_anchor_bundle_repair_development_train_v1/result.json`  
Status: `open_heldout_failed_not_promoted`  
Result payload SHA-256: `bae833672db803011c24eb9ada0f7f44690359325a1b7b0c489a3aefd863667c`

## Interpretation and next boundary

The paper's asymmetric GLU successfully motivates mandatory state use and
depth renewal, but this run again shows that a separable state-value/query-gate
bundle does not bind the RWKV value to the source identity. A matched donor can
still pass the same gate. Full-Bandwidth-style temporal feedback is therefore
deferred until identity-specific mechanics and a causal donor gate pass.

The next independent family is an explicit address-derived virtual key paired
with an RMS-normalized RWKV state as a virtual value. It tests identity through
attention logits while keeping the real cache, projected carrier, and RWKV
writer unchanged. Only a mechanics pass followed by a fresh causal endpoint can
authorize a later mandatory deep-to-shallow GLU and Jacobi-style feedback test.
