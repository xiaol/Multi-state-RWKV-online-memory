# Address-keyed feedback narrative result

Two open-only, four-A100 narrative identity screens were completed with
`HF_ENDPOINT=https://hf-mirror.com`. Neither route passed its predeclared
held-out causal gate, so neither opened protected mechanics, protected causal,
or native benchmark data.

## Full-Bandwidth Transformer boundary

The experiments transfer the useful structural clue from arXiv `2608.08888v1`:
the memory state is a mandatory value path and the current hidden state is only
a multiplicative gate. They do not reproduce the paper's layer-0 temporal
feedback, non-detached Jacobi multi-pass training, or training scale. The paper
also does not solve matched-donor source identity. The exact paper review is in
`FULL_BANDWIDTH_RWKV_REVIEW.md`.

## Open results

| route | target selection | divergent donor mean / positive fraction | divergent correct-vs-off mean / positive fraction | decision |
| --- | ---: | ---: | ---: | --- |
| address/query-gated recurrent value | `0.78125` | `+0.011029 / 0.40625` | `+0.010785 / 0.56250` | failed, not promoted |
| direct address-modulated recurrent value | `0.78125` | `-0.011634 / 0.37500` | `-0.000870 / 0.46875` | failed, not promoted |

The first-token view also failed. The original address/query-gated route had a
donor mean of `-0.009414` and correct-vs-off mean of `-0.007593`. Direct address
modulation made both worse: `-0.016043` and `-0.012811`, respectively. Exact
zero controls and finite/bounded residual controls passed in both routes.

The sealed result bindings are:

- v1 result SHA-256: `bac46a5091aea9e1c35e88465f53ef54f040180918295999fbbcbb185106605a`
- v1 result receipt: `2582c8ef305a04a230c6fe19ea51fca6c1496bb7616ca1fdfbdc252d7734aef0`
- v2 result SHA-256: `4b02dda608e35e0a8703fe40aeb7e86316531dd32b909424ee549ffbb098d416`
- v2 result receipt: `7a7afc41bbc899c04647cbd019952069a880e2c30dec0ccaa2fc631e2a3d62ca`

## Decision

The current bottleneck is source identity before value injection, not missing
value-path capacity. Both architectures inherited the same `0.78125` source
selection rate, while a stronger direct address transformation degraded causal
margins. The next route should therefore train a small identity scoring head on
source/donor/negative address contrast and freeze it before attaching any RWKV
value feedback. Repeating FFN or temporal-feedback variants behind the current
selector is not a clean next test.
