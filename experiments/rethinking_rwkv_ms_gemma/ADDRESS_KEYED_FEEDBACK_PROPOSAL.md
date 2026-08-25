# Address-Keyed RWKV Feedback Cell

This is the next open-development route motivated by Full-Bandwidth Transformer
(arXiv:2608.08888v1), but it is not a reproduction of that paper.

## What carries over

The paper's useful architectural lesson is that the recurrent state should be a
mandatory value path and the current token should only gate that value. The new
cell follows that rule: the selected native RWKV reads are projected into a
state value, the selected source address and current hidden query form the
identity gate, and the output is produced only from the resulting recurrent
state.

## Cell

For anchor reads `m[t]`, selected mapped addresses `a[t]`, and hidden query
`q[t]`, the cell computes:

```text
u[t] = tanh(W_m RMS(m[t]))
c[t] = W_a RMS(a[t]) + W_q RMS(q[t])
p[t] = u[t] * sigmoid(c[t])
z[t] = decay * z[t-1] + (1 - decay) * p[t]
d[t] = tanh(W_out(SiLU(z[t]) * 2 sigmoid(W_g RMS(q[t]))))
```

The scan is causal over the answer-token sequence. If the selected state or
address is zero, the output and recurrent state are exactly zero. There is no
projected-value or hidden-only bypass.

Implementation: `SourceBoundAddressKeyedFeedbackFFN` in
`deltamem/core/cumulative_rwkv_residual.py`.

## Evaluation firewall

The route needs a fresh open split and protocol. The first screen should test:

- exact provider-off and zero-state behavior;
- matched target versus donor state/address swaps;
- state-only and address-only swaps;
- cyclic anchor-layer swaps;
- prompt-latched source and confidence across every intervention;
- finite causal scan and contraction on the final feedback updates.

Require target selection at least `0.875`, donor and layer positive-row
fractions at least `0.75`, and donor mean margin at least `0.02` before any
protected mechanics, causal, or native benchmark access. A pass would be an
open causal route result only; it would not justify a Full-Bandwidth or SOTA
claim.

## Data reservation boundary

The currently sealed source has `708` passage components. The low-rank route's
manifest records `622` components already excluded and `80` newly selected,
leaving only `6` component slots. That is not enough for another paired
32-row/64-component screen. The next executable goal therefore needs a fresh
HF-mirror source/manifest (or an explicitly approved reservation policy) before
training this cell. Reusing the failed low-rank bundle would invalidate the
identity claim.
